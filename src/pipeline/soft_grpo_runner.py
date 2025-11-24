"""Stage 2 SofT-GRPO runner orchestration."""

from __future__ import annotations

import datetime as _dt
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .soft_grpo_config import (
    format_runner_arg,
    get_soft_grpo_paths,
    load_soft_grpo_config,
    merge_runner_args,
)
from .soft_grpo_dataset import SoftGrpoDatasetBuilder, SoftGrpoDatasetInfo
from .soft_grpo_manifest import (
    append_stage2_run,
    list_markets_with_stage1,
    manifest_path_for_market,
    load_manifest,
)
from .soft_grpo_validation import validate_soft_grpo_requirements, print_validation_report

logger = logging.getLogger(__name__)


class SoftGrpoRunnerError(RuntimeError):
    """Raised when Stage 2 runner fails."""


@dataclass
class SoftGrpoRunResult:
    """Outcome of a Stage 2 execution attempt."""

    success: bool
    command: Optional[str]
    log_file: Optional[str]


def run_soft_grpo_stage2(
    market: str,
    test_mode: bool = False,
    dataset_only: bool = False,
    config_path: Optional[str] = None,
    logger_obj: Optional[logging.Logger] = None,
) -> SoftGrpoRunResult:
    """Public entry point for CLI integration."""
    log = logger_obj or logger
    config = load_soft_grpo_config(config_path)
    project_root = Path.cwd()
    paths = get_soft_grpo_paths(config, project_root)
    manifest_path = manifest_path_for_market(market, paths["model_root"])
    manifest = load_manifest(manifest_path)
    stage1 = manifest.get("stage1")
    if not stage1:
        raise SoftGrpoRunnerError(
            f"No Stage 1 metadata found for market {market}. Complete Phase 3 Stage 1 first."
        )

    training_enabled = not dataset_only
    checks = validate_soft_grpo_requirements(
        config=config,
        project_root=project_root,
        stage1=stage1,
        training_enabled=training_enabled,
    )
    print_validation_report(checks)

    dataset_builder = SoftGrpoDatasetBuilder(config=config, project_root=project_root)
    dataset_info = dataset_builder.build(market, manifest=manifest, test_mode=test_mode)

    if dataset_only:
        log.info("[SofT-GRPO] Dataset export complete for %s. Skipping training per request.", market)
        return SoftGrpoRunResult(True, None, None)

    runner = _SoftGrpoCommandRunner(
        config=config,
        market=market,
        dataset_info=dataset_info,
        stage1=stage1,
        test_mode=test_mode,
        project_root=project_root,
    )
    return runner.run()


class _SoftGrpoCommandRunner:
    """Build and launch the SofT-GRPO training command."""

    def __init__(
        self,
        config: Dict[str, Any],
        market: str,
        dataset_info: SoftGrpoDatasetInfo,
        stage1: Dict[str, Any],
        test_mode: bool,
        project_root: Path,
    ):
        self.config = config
        self.market = market
        self.dataset_info = dataset_info
        self.stage1 = stage1
        self.test_mode = test_mode
        self.project_root = project_root
        self.paths = get_soft_grpo_paths(config, project_root)
        self.runner_cfg = config.get("runner", {})

    def run(self) -> SoftGrpoRunResult:
        mode = self.runner_cfg.get("mode", "python").lower()
        if mode not in {"python", "script"}:
            raise SoftGrpoRunnerError(f"Unsupported runner mode: {mode}")

        cmd = self._build_command(mode)
        log_file = self._prepare_log_file()
        env = self._build_env()

        logger.info("[SofT-GRPO] Launching Stage 2 for %s", self.market)
        logger.info("[SofT-GRPO] Command: %s", " ".join(cmd))
        logger.info("[SofT-GRPO] Log file: %s", log_file)

        with open(log_file, "w", encoding="utf-8") as log_fh:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self._working_directory(mode),
                env=env,
                text=True,
            )
            if process.stdout:
                for line in process.stdout:
                    log_fh.write(line)
                    log_fh.flush()
                    logger.info("[SofT-GRPO] %s", line.rstrip())
            return_code = process.wait()

        status = "success" if return_code == 0 else "failed"
        append_stage2_run(
            manifest_path_for_market(self.market, self.paths["model_root"]),
            {
                "status": status,
                "command": " ".join(cmd),
                "log_file": str(log_file),
                "test_mode": self.test_mode,
                "return_code": return_code,
                "dataset": self.dataset_info.to_manifest_dict(),
            },
        )

        if return_code != 0:
            raise SoftGrpoRunnerError(
                f"SofT-GRPO Stage 2 failed for {self.market}. See logs at {log_file}."
            )

        return SoftGrpoRunResult(True, " ".join(cmd), str(log_file))

    def _build_command(self, mode: str) -> list:
        # Get base model path from Stage 1 manifest
        # The manifest stores it as stage1["base_model"]["path"] or stage1["base_model_path"]
        base_model = self.stage1.get("base_model", {})
        base_model_path = base_model.get("local_path") or base_model.get("path", "")
        
        # Handle relative paths - prepend /workspace if not absolute
        if base_model_path and not base_model_path.startswith("/"):
            base_model_path = f"/workspace/{base_model_path}"
        
        placeholders = {
            "train_dataset": self.dataset_info.train_path,
            "val_dataset": self.dataset_info.val_path,
            "market": self.market,
            "output_dir": str(Path(self.paths["model_root"]) / self.market / "soft_grpo"),
            "base_model_path": str(base_model_path) if base_model_path else "",
            "lora_adapter_path": str(self.stage1.get("lora_adapter_path", "")),
        }

        args = merge_runner_args(self.runner_cfg.get("args", {}))
        if self.test_mode:
            args = merge_runner_args(args, self.runner_cfg.get("test_overrides"))

        if mode == "python":
            module = self.runner_cfg.get("python_module", "verl.trainer.main_ppo")
            if not module:
                raise SoftGrpoRunnerError("runner.python_module is required for python mode.")

            cmd = ["python", "-m", module]
            for key, value in args.items():
                formatted = format_runner_arg(value, placeholders)
                cmd.append(f"{key}={formatted}")
            return cmd

        script_path = self.runner_cfg.get("script_path")
        if not script_path:
            raise SoftGrpoRunnerError("runner.script_path is required for script mode.")
        script_resolved = Path(script_path)
        if not script_resolved.is_absolute():
            script_resolved = (self.project_root / script_resolved).resolve()
        if not script_resolved.exists():
            raise SoftGrpoRunnerError(f"SofT-GRPO script not found at {script_resolved}")

        cmd = [str(script_resolved)]
        for key, value in args.items():
            formatted = format_runner_arg(value, placeholders)
            cmd.append(f"{key}={formatted}")
        return cmd

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        placeholders = {
            "train_dataset": self.dataset_info.train_path,
            "val_dataset": self.dataset_info.val_path,
            "market": self.market,
            "output_dir": str(Path(self.paths["model_root"]) / self.market / "soft_grpo"),
            "base_model_path": str(self.stage1.get("base_model", {}).get("local_path", "")),
            "lora_adapter_path": str(self.stage1.get("lora_adapter_path", "")),
        }

        for key, value in (self.runner_cfg.get("env") or {}).items():
            env[key] = format_runner_arg(value, placeholders)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Ensure repo_root is in PYTHONPATH so 'verl' is importable
        repo_root = str(self.paths["repo_root"])
        current_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{current_path}" if current_path else repo_root

        return env

    def _prepare_log_file(self) -> str:
        logs_dir = Path(self.paths["logs_dir"])
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"soft_grpo_{self.market}_{timestamp}.log"
        return str(log_file)

    def _working_directory(self, mode: str) -> str:
        if mode == "python":
            return str(self.paths["repo_root"] or self.project_root)
        script_path = self.runner_cfg.get("script_path")
        if script_path:
            return str(Path(script_path).resolve().parent)
        return str(self.project_root)


__all__ = [
    "run_soft_grpo_stage2",
    "SoftGrpoRunnerError",
    "SoftGrpoRunResult",
    "list_markets_with_stage1",
]
