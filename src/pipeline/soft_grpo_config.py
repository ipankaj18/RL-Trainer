"""Helpers for loading and normalizing SofT-GRPO Stage 2 configuration."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

DEFAULT_CONFIG_PATH = Path("config/phase3_soft_grpo.yaml")


class SoftGRPOConfigError(RuntimeError):
    """Raised when the SofT-GRPO configuration is invalid."""


def load_soft_grpo_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration for the SofT-GRPO pipeline."""
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        raise SoftGRPOConfigError(
            f"SofT-GRPO config not found at {config_path}. "
            "Create config/phase3_soft_grpo.yaml before running Stage 2."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    # Ensure expected top-level sections exist
    data.setdefault("soft_grpo", {})
    data.setdefault("dataset", {})
    data.setdefault("runner", {})
    repo_override = os.environ.get("SOFT_GRPO_REPO_ROOT")
    if repo_override:
        data_path["repo_root"] = repo_override
    data_path = data["soft_grpo"]
    # Normalize repeatable keys
    data_path.setdefault("model_root", "./models/phase3_hybrid")
    data_path.setdefault("dataset_root", "./data/soft_grpo")
    data_path.setdefault("logs_dir", "./logs/soft_grpo")
    data_path.setdefault("tensorboard_dir", "./tensorboard_logs/phase3/soft_grpo")
    data_path.setdefault("experience_filename", "stage1_experience.jsonl")
    data_path.setdefault("train_filename", "stage2_train.parquet")
    data_path.setdefault("val_filename", "stage2_val.parquet")
    data_path.setdefault("jsonl_snapshot", "stage2_samples.jsonl")
    data_path.setdefault("metadata_filename", "dataset_metadata.json")
    data_path.setdefault("lora_subdir", "stage1_lora")
    data_path.setdefault("repo_root", "./SofT-GRPO-master-main")
    data["runner"].setdefault("test_overrides", {})

    return data


def resolve_path(path_like: Optional[str], base_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve relative paths against the provided base or project root."""
    if not path_like:
        return None

    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate

    if base_dir is None:
        base_dir = Path.cwd()

    return (base_dir / candidate).resolve()


def get_soft_grpo_paths(config: Dict[str, Any], project_root: Optional[Path] = None) -> Dict[str, Path]:
    """Return resolved path helpers from configuration."""
    base = Path(project_root or Path.cwd())
    sg_cfg = config.get("soft_grpo", {})

    def _resolve(key: str) -> Path:
        return resolve_path(sg_cfg.get(key), base)  # type: ignore

    return {
        "model_root": _resolve("model_root"),
        "dataset_root": _resolve("dataset_root"),
        "logs_dir": _resolve("logs_dir"),
        "tensorboard_dir": _resolve("tensorboard_dir"),
        "repo_root": _resolve("repo_root"),
    }


def merge_runner_args(
    base_args: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a shallow copy of args with overrides applied."""
    merged = copy.deepcopy(base_args or {})
    if overrides:
        for key, value in overrides.items():
            merged[key] = value
    return merged


def format_runner_arg(value: Any, placeholders: Dict[str, str]) -> str:
    """
    Replace placeholders (e.g., {train_dataset}) and coerce to string.

    Verl command-line arguments expect `key=value` strings.
    """
    if isinstance(value, (dict, list)):
        value_str = json.dumps(value)
    else:
        value_str = str(value)

    for key, replacement in placeholders.items():
        value_str = value_str.replace(f"{{{key}}}", replacement)
    return value_str


__all__ = [
    "SoftGRPOConfigError",
    "DEFAULT_CONFIG_PATH",
    "load_soft_grpo_config",
    "resolve_path",
    "get_soft_grpo_paths",
    "merge_runner_args",
    "format_runner_arg",
]
