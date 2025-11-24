# Phase 3 Stage 2 – SofT-GRPO Integration

This document explains how the Phase 3 LoRA fine-tune (Stage 1) hands off to SofT-GRPO reasoning (Stage 2).

## Overview

- **Stage 1** (existing Phase 3) trains the hybrid PPO + LoRA agent and now exports:
  - LoRA adapters (`models/phase3_hybrid/<market>/stage1_lora/`)
  - Experience buffer dump (`data/soft_grpo/<market>/stage1_experience.jsonl`)
  - Manifest (`models/phase3_hybrid/<market>/phase3_stage_manifest.json`)
- **Stage 2** converts the dump into a SofT-GRPO dataset and launches Verl using the provided SofT-GRPO repository.

All Stage 2 settings live in `config/phase3_soft_grpo.yaml`. Adjust dataset budgets, repo paths, Verl arguments, or environment variables there.

## Prerequisites

1. Complete Phase 3 Stage 1 for the desired market (produces manifest + adapters).
2. Clone/download `SofT-GRPO-master-main` into the project root (already in repo).
3. Install Verl and SofT-GRPO dependencies (follow the upstream README; torch 2.6 + flash-attn required).
4. Ensure the Meta Llama base weights referenced in `config/llm_config.yaml` exist locally.

## Running Stage 2

```
python main.py
# Menu → Training → Phase 3 Stage 2: SofT-GRPO Reasoning
```

Options:

- **Test mode**: applies `runner.test_overrides` (tiny batch sizes, 1 epoch, n=1 rollouts).
- **Dataset only**: exports parquet/jsonl files and updates the manifest without launching Verl.

The CLI prompts for the market (auto-detected via manifest files) and prints the log path on completion.

### Preflight checks

Every Stage 2 run now executes safety checks:

- Verl import + SofT-GRPO repo structure (override with `SOFT_GRPO_REPO_ROOT` if the repo lives elsewhere on a Pod)
- Base Meta-Llama weights referenced by the Stage 1 manifest
- CUDA + flash-attn availability (skipped when running in dataset-only mode)

Example override for RunPod:

```
export SOFT_GRPO_REPO_ROOT=/workspace/AI-Trainer/SofT-GRPO-master-main
```

### Dataset export guardrails

The exporter aborts if no experiences have recorded outcomes. When filters remove all samples, the error now prints the thresholds so you can relax `min_reward` or set `include_failures: true` in `config/phase3_soft_grpo.yaml`.

## Directory Layout

```
data/soft_grpo/<market>/
  ├── stage1_experience.jsonl   # exported buffer
  ├── stage2_train.parquet
  ├── stage2_val.parquet
  ├── stage2_samples.jsonl
  └── dataset_metadata.json

models/phase3_hybrid/<market>/
  ├── stage1_lora/              # adapter snapshot
  ├── phase3_stage_manifest.json
  └── soft_grpo/                # Verl output (set via config)

logs/soft_grpo/*.log            # runner stdout/stderr mirrors
```

## Manifest Fields

`models/phase3_hybrid/<market>/phase3_stage_manifest.json` contains:

- `stage1`: checkpoint paths, adapter locations, buffer stats, LLM/hybrid metrics
- `stage2.dataset`: dataset counts + file references
- `stage2.runs`: timestamped entries with command, log file, return code, and dataset snapshot

This makes it easy to trace which dataset + config produced a SofT-GRPO adapter bundle.

## Validation Checklist

1. Run Stage 1 (Phase 3) to completion. Confirm manifest + adapters exist.
2. `python main.py` → Training → Phase 3 Stage 2:
   - First run with **dataset only** to validate export.
   - Re-run with full training (or **Test mode**).
3. Inspect `logs/soft_grpo/<timestamp>.log` and `tensorboard_logs/phase3/soft_grpo`.
4. Archive any resulting adapters/checkpoints inside `models/phase3_hybrid/<market>/soft_grpo/`.

Add these bullets to the PR testing notes (on top of normal `pytest` + `python main.py --test`).*** End Patch
