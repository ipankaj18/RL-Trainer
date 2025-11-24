"""Dataset exporter for SofT-GRPO Stage 2 training."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .soft_grpo_config import get_soft_grpo_paths, load_soft_grpo_config
from .soft_grpo_manifest import load_manifest, manifest_path_for_market, update_stage2_dataset_info

logger = logging.getLogger(__name__)


class SoftGrpoDatasetError(RuntimeError):
    """Raised when dataset generation fails."""


@dataclass
class SoftGrpoDatasetInfo:
    """Metadata describing exported datasets."""

    market: str
    total_samples: int
    train_samples: int
    val_samples: int
    train_path: str
    val_path: str
    jsonl_path: str
    metadata_path: str
    experience_path: str

    def to_manifest_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


class SoftGrpoDatasetBuilder:
    """Build SofT-GRPO compatible datasets from Stage 1 experience buffers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, project_root: Optional[Path] = None):
        self.config = config or load_soft_grpo_config()
        self.project_root = Path(project_root or Path.cwd())
        self.paths = get_soft_grpo_paths(self.config, self.project_root)
        self.dataset_cfg = self.config.get("dataset", {})
        self.soft_cfg = self.config.get("soft_grpo", {})

    def build(self, market: str, manifest: Optional[Dict[str, Any]] = None, test_mode: bool = False) -> SoftGrpoDatasetInfo:
        """Create train/val parquet files for Stage 2."""
        manifest = manifest or self._load_manifest(market)
        stage1 = manifest.get("stage1") or {}
        experience_path = stage1.get("experience_path")
        if not experience_path:
            experience_path = str(
                Path(self.paths["dataset_root"])
                / market
                / self.soft_cfg.get("experience_filename", "stage1_experience.jsonl")
            )

        exp_path = Path(experience_path)
        if not exp_path.exists():
            raise SoftGrpoDatasetError(
                f"Experience buffer not found at {exp_path}. Run Phase 3 Stage 1 first."
            )

        experiences = self._load_experiences(exp_path)
        filtered = self._filter_experiences(experiences, test_mode=test_mode)
        if not filtered:
            raise SoftGrpoDatasetError("No completed experiences with outcomes available for dataset export.")

        random_seed = int(self.dataset_cfg.get("random_seed", 42))
        random.Random(random_seed).shuffle(filtered)

        val_split = float(self.dataset_cfg.get("val_split", 0.1))
        val_count = max(1, int(len(filtered) * val_split)) if len(filtered) > 1 else 0
        val_records = filtered[:val_count]
        train_records = filtered[val_count:] or filtered

        market_dir = Path(self.paths["dataset_root"]) / market
        market_dir.mkdir(parents=True, exist_ok=True)
        train_path = market_dir / self.soft_cfg.get("train_filename", "stage2_train.parquet")
        val_path = market_dir / self.soft_cfg.get("val_filename", "stage2_val.parquet")
        jsonl_path = market_dir / self.soft_cfg.get("jsonl_snapshot", "stage2_samples.jsonl")
        metadata_path = market_dir / self.soft_cfg.get("metadata_filename", "dataset_metadata.json")

        self._write_records(train_path, train_records)
        self._write_records(val_path, val_records or train_records[:1])
        self._write_jsonl(jsonl_path, filtered[: min(len(filtered), 200)])
        self._write_metadata(metadata_path, len(filtered), len(train_records), len(val_records or train_records[:1]))

        dataset_info = SoftGrpoDatasetInfo(
            market=market,
            total_samples=len(filtered),
            train_samples=len(train_records),
            val_samples=len(val_records or train_records[:1]),
            train_path=str(train_path),
            val_path=str(val_path),
            jsonl_path=str(jsonl_path),
            metadata_path=str(metadata_path),
            experience_path=str(exp_path),
        )

        update_stage2_dataset_info(
            manifest_path_for_market(market, self.paths["model_root"]),
            {
                **dataset_info.to_manifest_dict(),
                "generated_at": self._now_iso(),
                "test_mode": bool(test_mode),
            },
        )

        logger.info(
            "[SofT-GRPO][Dataset] Generated %d samples for %s (train=%d, val=%d)",
            dataset_info.total_samples,
            market,
            dataset_info.train_samples,
            dataset_info.val_samples,
        )

        return dataset_info

    def _load_manifest(self, market: str) -> Dict[str, Any]:
        manifest_path = manifest_path_for_market(market, self.paths["model_root"])
        return load_manifest(manifest_path)

    def _load_experiences(self, path: Path) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("[SofT-GRPO][Dataset] Failed to decode line in %s: %s", path, exc)
        return items

    def _filter_experiences(self, experiences: Sequence[Dict[str, Any]], test_mode: bool) -> List[Dict[str, Any]]:
        min_reward = float(self.dataset_cfg.get("min_reward", -1.0))
        min_success_pnl = float(self.dataset_cfg.get("min_success_pnl", 0.0))
        include_failures = bool(self.dataset_cfg.get("include_failures", True))
        max_samples = self.dataset_cfg.get("max_samples")
        if test_mode and max_samples:
            max_samples = min(int(max_samples), 512)

        completed = [exp for exp in experiences if exp.get("outcome") is not None]
        if not completed:
            raise SoftGrpoDatasetError(
                "No experiences with recorded outcomes were found. "
                "Ensure hybrid_agent.llm_advisor.update_outcome() is called after trades complete."
            )

        filtered: List[Dict[str, Any]] = []
        for exp in completed:
            # Validate observation dimension (CRITICAL for Phase 3 compatibility)
            obs = exp.get("observation")
            if isinstance(obs, list) and len(obs) != 261:
                raise SoftGrpoDatasetError(
                    f"Found observation with {len(obs)} dimensions, expected 261. "
                    "This indicates outdated Stage 1 data. "
                    "Please re-run Phase 3 Stage 1 training."
                )

            outcome = exp.get("outcome") or {}
            reward = outcome.get("reward", 0.0)
            pnl = outcome.get("pnl", 0.0)
            success = pnl >= min_success_pnl

            if reward < min_reward and not include_failures:
                continue
            if not success and not include_failures:
                continue

            filtered.append(self._experience_to_record(exp, outcome, success))
            if max_samples and len(filtered) >= int(max_samples):
                break

        if not filtered:
            raise SoftGrpoDatasetError(
                "No experiences passed the dataset filters.\n"
                f"  Completed experiences: {len(completed)}\n"
                f"  min_reward: {min_reward}\n"
                f"  min_success_pnl: {min_success_pnl}\n"
                f"  include_failures: {include_failures}\n"
                "Tip: relax filter thresholds or allow failures to be included."
            )

        return filtered

    def _experience_to_record(self, exp: Dict[str, Any], outcome: Dict[str, Any], success: bool) -> Dict[str, Any]:
        action_names = self.dataset_cfg.get("action_names", {})
        action_id = int(exp.get("action", 0))
        action_name = action_names.get(action_id, f"ACTION_{action_id}")

        system_prompt = (
            exp.get("market_context", {}).get("system_prompt")
            or exp.get("prompts", {}).get("system")
            or ""
        )
        prompt_text = exp.get("prompt", "")

        prompt_messages = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.append({"role": "user", "content": prompt_text})

        response_text = exp.get("response", "")

        record = {
            "data_source": self.dataset_cfg.get("data_source", "phase3_stage1"),
            "prompt": prompt_messages,
            "ability": self.dataset_cfg.get("ability_label", "trading"),
            "reward_model": {
                "ground_truth": response_text,
                "style": "trade_reasoning",
                "action": action_name,
                "reward": outcome.get("reward"),
                "pnl": outcome.get("pnl"),
                "success": bool(success),
            },
            "extra_info": {
                "action_id": action_id,
                "action_name": action_name,
                "timestamp": exp.get("timestamp"),
                "market": exp.get("market_context", {}).get("market"),
                "observation": exp.get("observation"),
                "position_state": exp.get("position_state"),
                "market_context": exp.get("market_context"),
            },
        }
        return record

    def _write_records(self, path: Path, records: Sequence[Dict[str, Any]]) -> None:
        df = pd.DataFrame(records)
        df.to_parquet(path, index=False)

    def _write_jsonl(self, path: Path, records: Sequence[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_metadata(self, path: Path, total: int, train: int, val: int) -> None:
        payload = {
            "total_samples": total,
            "train_samples": train,
            "val_samples": val,
            "generated_at": self._now_iso(),
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @staticmethod
    def _now_iso() -> str:
        import datetime

        return datetime.datetime.now(datetime.timezone.utc).isoformat()


__all__ = [
    "SoftGrpoDatasetBuilder",
    "SoftGrpoDatasetInfo",
    "SoftGrpoDatasetError",
]
