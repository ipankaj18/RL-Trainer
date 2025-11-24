"""Manifest helpers for Phase 3 Stage 2 (SofT-GRPO) pipeline."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

MANIFEST_FILENAME = "phase3_stage_manifest.json"
MANIFEST_VERSION = 1


class SoftGrpoManifestError(RuntimeError):
    """Raised when Stage 2 manifest operations fail."""


def _utcnow_iso() -> str:
    """Return current UTC time as ISO format string."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def manifest_path_for_market(market: str, model_root: Path) -> Path:
    """Return manifest path for the given market."""
    return Path(model_root) / market / MANIFEST_FILENAME


def load_manifest(path: Path) -> Dict[str, Any]:
    """Load manifest JSON from disk (or return baseline template)."""
    if not path.exists():
        return {
            "version": MANIFEST_VERSION,
            "stage1": {},
            "stage2": {},
        }

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    data.setdefault("stage1", {})
    data.setdefault("stage2", {})
    return data


def save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    """Persist manifest data atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def record_stage1_completion(
    path: Path,
    market: str,
    model_checkpoint: str,
    vecnorm_path: str,
    total_timesteps: int,
    llm_config: Dict[str, Any],
    llm_stats: Dict[str, Any],
    hybrid_stats: Dict[str, Any],
    lora_path: str,
    experience_path: str,
    experience_count: int,
    test_mode: bool,
) -> Dict[str, Any]:
    """Update manifest with Stage 1 (LoRA fine-tune) metadata."""
    manifest = load_manifest(path)
    manifest["version"] = MANIFEST_VERSION
    manifest["market"] = market
    manifest["updated_at"] = _utcnow_iso()
    stage1 = {
        "completed_at": _utcnow_iso(),
        "model_checkpoint": model_checkpoint,
        "vecnormalize_path": vecnorm_path,
        "total_timesteps": total_timesteps,
        "test_mode": bool(test_mode),
        "base_model": {
            "name": llm_config.get("name"),
            "local_path": llm_config.get("local_path"),
        },
        "lora_adapter_path": lora_path,
        "experience_path": experience_path,
        "experience_samples": experience_count,
        "adapter_source": llm_config.get("adapter_path"),
        "llm_stats": llm_stats or {},
        "hybrid_stats": hybrid_stats or {},
        "prompts": llm_config.get("prompts", {}),
    }
    manifest["stage1"] = stage1
    save_manifest(path, manifest)
    return manifest


def update_stage2_dataset_info(path: Path, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """Persist dataset metadata produced for Stage 2."""
    manifest = load_manifest(path)
    stage2 = manifest.setdefault("stage2", {})
    stage2["dataset"] = dataset_info
    stage2.setdefault("runs", [])
    manifest["updated_at"] = _utcnow_iso()
    save_manifest(path, manifest)
    return manifest


def append_stage2_run(path: Path, run_info: Dict[str, Any]) -> Dict[str, Any]:
    """Append Stage 2 execution metadata (success/failure/logs)."""
    manifest = load_manifest(path)
    stage2 = manifest.setdefault("stage2", {})
    runs = stage2.setdefault("runs", [])
    run_info = dict(run_info)
    run_info.setdefault("timestamp", _utcnow_iso())
    runs.append(run_info)
    manifest["updated_at"] = _utcnow_iso()
    save_manifest(path, manifest)
    return manifest


def list_markets_with_stage1(model_root: Path) -> List[Dict[str, Any]]:
    """Return markets where Stage 1 metadata exists."""
    markets = []
    if not model_root.exists():
        return markets

    for child in sorted(model_root.iterdir()):
        if not child.is_dir():
            continue
        manifest_path = child / MANIFEST_FILENAME
        if not manifest_path.exists():
            continue
        data = load_manifest(manifest_path)
        if data.get("stage1"):
            markets.append(
                {
                    "market": child.name,
                    "manifest_path": manifest_path,
                    "stage1": data["stage1"],
                }
            )
    return markets


__all__ = [
    "SoftGrpoManifestError",
    "MANIFEST_FILENAME",
    "manifest_path_for_market",
    "load_manifest",
    "save_manifest",
    "record_stage1_completion",
    "update_stage2_dataset_info",
    "append_stage2_run",
    "list_markets_with_stage1",
]
