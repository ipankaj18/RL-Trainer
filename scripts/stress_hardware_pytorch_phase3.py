#!/usr/bin/env python3
"""
PyTorch Phase 3 (Hybrid RL + LLM) Hardware Stress Test and Auto-tuner

Auto-tunes parameters for Phase 3 hybrid RL+LLM training stack.
Optimizes vectorized_envs, batch_size, and timesteps_reduction for
maximum GPU utilization while maintaining training stability.

NOTE: For JAX training (Phases 1 & 2), use scripts/stress_hardware_jax.py instead.

Runs successive hardware-maximized validation passes while varying
vectorized env counts, batch sizes, and timesteps reduction until
improvement plateaus. The best-performing parameters can be saved
as a reusable profile.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from testing_framework import TestingFramework, create_test_config  # type: ignore  # noqa: E402
from model_utils import detect_available_markets  # type: ignore  # noqa: E402


SearchCombo = Tuple[int, int, float]  # (vectorized_envs, batch_size, timesteps_reduction)


def detect_market(market_arg: str | None) -> str:
    """Resolve market from argument or available data."""
    if market_arg:
        return market_arg.upper()
    detected = detect_available_markets(str(PROJECT_ROOT / "data"))
    if not detected:
        raise ValueError("No market data found under data/. Run data processing first.")
    if len(detected) == 1:
        return detected[0]["market"].upper()
    symbols = ", ".join(item["market"] for item in detected)
    raise ValueError(f"Multiple markets detected ({symbols}); pass --market to select one.")


def build_search_space() -> List[SearchCombo]:
    """Return an ordered search space from lighter to heavier loads."""
    envs = [8, 12, 16, 24, 32]
    batches = [256, 512, 768, 1024]
    reductions = [0.05, 0.1, 0.15]
    combos: List[SearchCombo] = []
    for e in envs:
        for b in batches:
            for r in reductions:
                combos.append((e, b, r))
    # Sort by rough ascending load so early exits are faster
    combos.sort(key=lambda x: (x[0], x[1], x[2]))
    return combos


def score_run(summary: Dict[str, float]) -> float:
    """
    Compute a scalar score for a run.
    Emphasize GPU utilization and penalize memory saturation.
    """
    gpu_util = summary.get("avg_gpu_utilization", 0.0)
    gpu_mem = summary.get("peak_gpu_memory_gb", 0.0)
    sys_mem = summary.get("peak_system_memory_gb", 0.0)
    duration = summary.get("duration_seconds", 1.0)

    # Penalize excessive system memory; GPU mem used as soft guidance only.
    mem_penalty = max(0.0, sys_mem - 28.0) * 2.0  # assume >28GB system mem usage is risky
    duration_penalty = min(duration / 600.0, 2.0) * 2.0  # up to -4 for runs longer than 10 min

    return gpu_util - mem_penalty - duration_penalty - (0.1 * gpu_mem)


def save_profile(name: str, summary: Dict[str, float]) -> Path:
    """Persist the best parameters to config/hardware_profiles/<name>.yaml."""
    profiles_dir = PROJECT_ROOT / "config" / "hardware_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    path = profiles_dir / f"{name}.yaml"

    payload = {
        "mode": "hardware_maximized",
        "vectorized_envs": int(summary.get("vectorized_envs", 16)),
        "batch_size": int(summary.get("batch_size", 512)),
        "timesteps_reduction": float(summary.get("timesteps_reduction", 0.1)),
        "device": summary.get("device", "auto"),
        "notes": "Auto-tuned via stress_hardware_pytorch_phase3.py",
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def run_combo(market: str, combo: SearchCombo) -> Dict[str, float]:
    """Execute a single validation run for the combo."""
    vec_envs, batch_size, reduction = combo
    config = create_test_config(
        mode="hardware_maximized",
        market=market,
        vectorized_envs=vec_envs,
        batch_size=batch_size,
        timesteps_reduction=reduction,
        device="auto",
    )
    framework = TestingFramework(config)
    summary = framework.run_hardware_maximized_validation()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test hardware and auto-tune hybrid RL/LLM parameters.")
    parser.add_argument("--market", help="Market symbol (e.g., NQ, ES). If omitted, auto-detects.")
    parser.add_argument("--max-runs", type=int, default=6, help="Maximum number of combos to evaluate.")
    parser.add_argument("--patience", type=int, default=2, help="Stop after this many non-improving runs.")
    parser.add_argument("--min-gain", type=float, default=0.5, help="Minimum score improvement to reset patience.")
    parser.add_argument("--save-name", help="Optional profile name to save the best parameters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (PROJECT_ROOT / "Phi-3-mini-4k-instruct").exists():
        print("[WARN] Phi-3-mini-4k-instruct not found in project root; GPU LLM performance may degrade.")

    if not os.environ.get("CUDA_ROOT"):
        print("[WARN] CUDA_ROOT not set; ensure your venv activation exports it for GPU runs.")

    market = detect_market(args.market)
    search_space = build_search_space()

    best_summary: Dict[str, float] | None = None
    best_score = float("-inf")
    no_gain_streak = 0
    evaluated = 0

    print(f"[INFO] Starting hardware stress test on {market}")
    print(f"[INFO] Will evaluate up to {args.max_runs} combos (patience={args.patience}, min_gain={args.min_gain})")

    for combo in search_space:
        if evaluated >= args.max_runs:
            break

        vec_envs, batch_size, reduction = combo
        print(f"[INFO] Combo {evaluated+1}: envs={vec_envs}, batch={batch_size}, reduction={reduction}")
        summary = run_combo(market, combo)
        score = score_run(summary)
        evaluated += 1

        print(
            f"[RESULT] avg_gpu_util={summary.get('avg_gpu_utilization', 0):.1f}% | "
            f"peak_gpu_mem={summary.get('peak_gpu_memory_gb', 0):.2f}GB | "
            f"duration={summary.get('duration_seconds', 0):.1f}s | score={score:.2f}"
        )

        if score > best_score + args.min_gain:
            best_score = score
            best_summary = summary
            no_gain_streak = 0
            print(f"[UPDATE] New best configuration found (score={best_score:.2f})")
        else:
            no_gain_streak += 1
            if no_gain_streak >= args.patience:
                print("[STOP] Improvement plateau detected; stopping search.")
                break

    if not best_summary:
        print("[ERROR] No runs completed; cannot save profile.")
        return

    print(
        f"[BEST] envs={best_summary.get('vectorized_envs')} "
        f"batch={best_summary.get('batch_size')} "
        f"reduction={best_summary.get('timesteps_reduction')} "
        f"avg_gpu_util={best_summary.get('avg_gpu_utilization', 0):.1f}% "
        f"peak_gpu_mem={best_summary.get('peak_gpu_memory_gb', 0):.2f}GB "
        f"duration={best_summary.get('duration_seconds', 0):.1f}s "
        f"score={best_score:.2f}"
    )

    if args.save_name:
        path = save_profile(args.save_name, best_summary)
        print(f"[SAVE] Profile saved to {path}")
    else:
        print("[INFO] Pass --save-name <profile> to persist the best parameters.")


if __name__ == "__main__":
    main()
