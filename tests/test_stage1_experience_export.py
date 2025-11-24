"""
Quick health check for Phase 3 Stage 1 exports (experience + manifest).

Purpose:
- Detect when Stage 1 exported zero experiences or only pending entries.
- Verify the manifest points to an existing LoRA adapter directory.

Run with: pytest -q tests/test_stage1_experience_export.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


MODEL_ROOT = Path("models/phase3_hybrid")
MANIFEST_NAME = "phase3_stage_manifest.json"


def _iter_manifests() -> list[Path]:
    if not MODEL_ROOT.exists():
        return []
    manifests = []
    for child in MODEL_ROOT.iterdir():
        if not child.is_dir():
            continue
        candidate = child / MANIFEST_NAME
        if candidate.exists():
            manifests.append(candidate)
    return manifests


@pytest.mark.parametrize("manifest_path", _iter_manifests())
def test_stage1_experience_has_completed_outcomes(manifest_path: Path) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage1 = data.get("stage1") or {}

    experience_path = Path(stage1.get("experience_path", ""))
    assert experience_path.is_file(), (
        f"Stage 1 experience file missing at {experience_path} "
        f"(manifest: {manifest_path})"
    )

    records = []
    with experience_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                pytest.fail(f"Invalid JSONL entry in {experience_path}: {exc}")

    assert records, (
        f"No experiences exported (0 lines) in {experience_path}. "
        "Ensure llm_reasoning._add_to_experience_buffer() is called and experiences are persisted."
    )

    completed = [r for r in records if r.get("outcome")]
    assert completed, (
        "No experiences have outcomes recorded. "
        "Call hybrid_agent.update_llm_outcome(...) (or llm_advisor.update_outcome) when trades settle "
        "so Stage 2 can filter completed samples."
    )


@pytest.mark.parametrize("manifest_path", _iter_manifests())
def test_stage1_manifest_points_to_lora_dir(manifest_path: Path) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage1 = data.get("stage1") or {}
    lora_path = Path(stage1.get("lora_adapter_path", ""))
    assert lora_path.exists(), f"LoRA adapter path does not exist: {lora_path} (manifest: {manifest_path})"
    adapter_config = lora_path / "adapter_config.json"
    assert adapter_config.exists(), (
        f"LoRA adapter config missing at {adapter_config}. "
        "Ensure save_lora_adapters() writes adapters alongside the final checkpoint."
    )


def test_has_manifests_present() -> None:
    manifests = _iter_manifests()
    if not manifests:
        pytest.skip("No Phase 3 Stage 1 manifests found under models/phase3_hybrid")

