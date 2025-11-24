#!/usr/bin/env python3
"""
Diagnostic Evaluation Script

Analyzes evaluation results to understand failure modes.
Loads evaluation NPZ files and provides detailed statistics.

Usage:
    python scripts/diagnose_evaluation.py --phase 1
    python scripts/diagnose_evaluation.py --phase 2
    python scripts/diagnose_evaluation.py --eval-file logs/phase2/eval_20251115_120000/evaluations.npz
"""

import numpy as np
import argparse
from pathlib import Path
import sys


def analyze_evaluation(npz_file: str):
    """Analyze a single evaluation NPZ file."""
    if not Path(npz_file).exists():
        print(f"❌ File not found: {npz_file}")
        return

    print(f"\n{'='*80}")
    print(f"EVALUATION ANALYSIS: {npz_file}")
    print(f"{'='*80}\n")

    data = np.load(npz_file)

    # Extract data
    timesteps = data['timesteps']
    results = data['results']  # Shape: (n_checkpoints, n_eval_episodes)
    ep_lengths = data['ep_lengths']  # Shape: (n_checkpoints, n_eval_episodes)

    n_checkpoints = len(timesteps)
    n_episodes = results.shape[1]

    print(f"📊 SUMMARY")
    print(f"  Checkpoints evaluated: {n_checkpoints}")
    print(f"  Episodes per checkpoint: {n_episodes}")
    print(f"  Timesteps: {timesteps[0]:,} -> {timesteps[-1]:,}")

    # Analyze each checkpoint
    for i, ts in enumerate(timesteps):
        rewards = results[i]
        lengths = ep_lengths[i]

        print(f"\n{'─'*80}")
        print(f"CHECKPOINT #{i+1} - Timestep {ts:,}")
        print(f"{'─'*80}")

        print(f"\n  📈 REWARDS:")
        print(f"     Mean:     {rewards.mean():>10.2f}")
        print(f"     Std:      {rewards.std():>10.2f}")
        print(f"     Min:      {rewards.min():>10.2f}")
        print(f"     Max:      {rewards.max():>10.2f}")
        print(f"     Variance: {rewards.var():>10.4f}")

        print(f"\n  📏 EPISODE LENGTHS:")
        print(f"     Mean:     {lengths.mean():>10.1f} bars")
        print(f"     Std:      {lengths.std():>10.1f} bars")
        print(f"     Min:      {lengths.min():>10.0f} bars")
        print(f"     Max:      {lengths.max():>10.0f} bars")
        print(f"     Variance: {lengths.var():>10.2f}")

        # Check for deterministic behavior (zero variance)
        if rewards.var() < 0.001:
            print(f"\n  ⚠️  WARNING: Reward variance is near-zero (deterministic failure)")
            print(f"      All episodes have nearly identical rewards!")

        if lengths.var() < 1.0:
            print(f"\n  ⚠️  WARNING: Episode length variance is near-zero")
            print(f"      All episodes terminate at the same step!")
            print(f"      This suggests evaluation is completely deterministic.")

        # Reward distribution
        print(f"\n  📊 REWARD DISTRIBUTION:")
        print(f"     Values: {np.array2string(rewards, precision=2, separator=', ')}")

        print(f"\n  📊 LENGTH DISTRIBUTION:")
        print(f"     Values: {lengths.tolist()}")

    # Overall analysis
    print(f"\n{'='*80}")
    print(f"OVERALL ANALYSIS")
    print(f"{'='*80}\n")

    all_rewards = results.flatten()
    all_lengths = ep_lengths.flatten()

    if all_rewards.mean() < 0:
        print(f"  ❌ PROBLEM: Mean reward across all checkpoints is NEGATIVE ({all_rewards.mean():.2f})")
        print(f"     This indicates the agent is consistently losing money.")
    elif all_rewards.mean() > 0:
        print(f"  ✅ SUCCESS: Mean reward is POSITIVE ({all_rewards.mean():.2f})")
    else:
        print(f"  ⚠️  NEUTRAL: Mean reward is near zero ({all_rewards.mean():.2f})")

    if all_lengths.var() < 1.0:
        print(f"\n  ❌ CRITICAL: Episode length is deterministic across ALL checkpoints!")
        print(f"     Variance: {all_lengths.var():.4f}")
        print(f"     This means evaluation is not representative of real performance.")
        print(f"     Recommendation: Add multiple evaluation slices with different start points.")

    # Check for improvement over time
    if n_checkpoints > 1:
        first_mean = results[0].mean()
        last_mean = results[-1].mean()
        improvement = last_mean - first_mean

        print(f"\n  📈 PROGRESS:")
        print(f"     First checkpoint: {first_mean:>10.2f}")
        print(f"     Last checkpoint:  {last_mean:>10.2f}")
        print(f"     Improvement:      {improvement:>10.2f} ({improvement/abs(first_mean)*100:+.1f}%)")

        if improvement > 0:
            print(f"     ✅ Model is improving!")
        elif improvement < 0:
            print(f"     ❌ Model is DEGRADING!")
        else:
            print(f"     ⚠️  Model is stagnant")

    print(f"\n{'='*80}\n")


def find_latest_eval_file(phase: int) -> str:
    """Find the most recent evaluation file for a phase."""
    eval_dir = Path(f"logs/phase{phase}")

    if not eval_dir.exists():
        return None

    # Look for eval_* subdirectories
    eval_dirs = list(eval_dir.glob("eval_*"))
    if eval_dirs:
        # Get most recent
        latest = max(eval_dirs, key=lambda p: p.name)
        npz_file = latest / "evaluations.npz"
        if npz_file.exists():
            return str(npz_file)

    # Fall back to direct evaluations.npz
    direct_file = eval_dir / "evaluations.npz"
    if direct_file.exists():
        return str(direct_file)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose evaluation results")
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                       help="Phase to analyze (1, 2, or 3)")
    parser.add_argument('--eval-file', type=str,
                       help="Specific evaluation NPZ file to analyze")
    args = parser.parse_args()

    if args.eval_file:
        analyze_evaluation(args.eval_file)
    elif args.phase:
        eval_file = find_latest_eval_file(args.phase)
        if eval_file:
            print(f"Found latest evaluation file for Phase {args.phase}: {eval_file}")
            analyze_evaluation(eval_file)
        else:
            print(f"❌ No evaluation file found for Phase {args.phase}")
            print(f"   Looked in: logs/phase{args.phase}/")
            sys.exit(1)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python scripts/diagnose_evaluation.py --phase 2")
        print("  python scripts/diagnose_evaluation.py --eval-file logs/phase2/eval_20251115_120000/evaluations.npz")
