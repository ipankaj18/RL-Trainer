#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action Distribution Analyzer

Analyzes action distributions from training logs to diagnose:
- Which actions the agent is taking
- Invalid action rates
- Action masking effectiveness
- Position entry/exit patterns

Usage:
    python src/diagnostics/analyze_action_distribution.py --logdir logs/phase2
    python src/diagnostics/analyze_action_distribution.py --tensorboard tensorboard_logs/phase2
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("[WARNING] TensorBoard not installed. Install with: pip install tensorboard")


class ActionDistributionAnalyzer:
    """Analyzes action distributions from training logs."""

    ACTION_NAMES_PHASE1 = {
        0: "HOLD",
        1: "BUY",
        2: "SELL"
    }

    ACTION_NAMES_PHASE2 = {
        0: "HOLD",
        1: "BUY",
        2: "SELL",
        3: "MOVE_SL_TO_BE",
        4: "ENABLE_TRAIL",
        5: "DISABLE_TRAIL"
    }

    def __init__(self, phase: int = 2):
        self.phase = phase
        self.action_names = self.ACTION_NAMES_PHASE2 if phase == 2 else self.ACTION_NAMES_PHASE1

    def analyze_evaluations_npz(self, npz_path: str) -> Dict:
        """
        Analyze evaluation results from evaluations.npz file.

        Args:
            npz_path: Path to evaluations.npz

        Returns:
            Dictionary with analysis results
        """
        if not os.path.exists(npz_path):
            return {"error": f"File not found: {npz_path}"}

        try:
            data = np.load(npz_path)

            # Extract data
            timesteps = data.get('timesteps', [])
            results = data.get('results', [])
            ep_lengths = data.get('ep_lengths', [])

            analysis = {
                'num_evaluations': len(timesteps),
                'timesteps': timesteps.tolist() if len(timesteps) > 0 else [],
                'mean_rewards': [],
                'std_rewards': [],
                'mean_ep_lengths': [],
                'reward_progression': []
            }

            for i, (ts, rewards) in enumerate(zip(timesteps, results)):
                analysis['mean_rewards'].append(float(np.mean(rewards)))
                analysis['std_rewards'].append(float(np.std(rewards)))
                if i < len(ep_lengths):
                    analysis['mean_ep_lengths'].append(float(np.mean(ep_lengths[i])))

            # Detect issues
            analysis['issues'] = []

            # Check for zero rewards
            if len(analysis['mean_rewards']) > 0:
                if all(r == 0.0 for r in analysis['mean_rewards']):
                    analysis['issues'].append("All evaluation rewards are exactly 0.0 - possible logging bug")

                # Check for negative rewards
                negative_pct = sum(1 for r in analysis['mean_rewards'] if r < 0) / len(analysis['mean_rewards']) * 100
                if negative_pct > 80:
                    analysis['issues'].append(f"{negative_pct:.0f}% of evaluations have negative rewards - agent is losing money")

                # Check for reward improvement
                if len(analysis['mean_rewards']) >= 3:
                    early_mean = np.mean(analysis['mean_rewards'][:3])
                    late_mean = np.mean(analysis['mean_rewards'][-3:])
                    improvement = late_mean - early_mean
                    analysis['reward_improvement'] = float(improvement)

                    if improvement < 0:
                        analysis['issues'].append(f"Reward decreased by {abs(improvement):.3f} - possible overtraining")
                    elif improvement < 0.1:
                        analysis['issues'].append(f"Minimal reward improvement ({improvement:.3f}) - slow learning")

            return analysis

        except Exception as e:
            return {"error": f"Failed to load {npz_path}: {e}"}

    def analyze_tensorboard_logs(self, logdir: str) -> Dict:
        """
        Analyze action distributions from TensorBoard logs.

        Args:
            logdir: Path to TensorBoard log directory

        Returns:
            Dictionary with action distribution analysis
        """
        if not TENSORBOARD_AVAILABLE:
            return {"error": "TensorBoard not installed"}

        if not os.path.exists(logdir):
            return {"error": f"Directory not found: {logdir}"}

        analysis = {
            'action_distribution': {},
            'invalid_action_rate': None,
            'rollout_metrics': {},
            'checkpoint_metrics': {}
        }

        try:
            # Find event files
            event_files = list(Path(logdir).rglob("events.out.tfevents.*"))
            if not event_files:
                return {"error": f"No TensorBoard event files found in {logdir}"}

            # Load TensorBoard data
            ea = event_accumulator.EventAccumulator(str(event_files[0]))
            ea.Reload()

            # Get available tags
            scalar_tags = ea.Tags()['scalars']

            # Extract action distribution
            for action_idx in range(len(self.action_names)):
                tag = f'action_dist/action_{action_idx}_pct'
                if tag in scalar_tags:
                    events = ea.Scalars(tag)
                    values = [e.value for e in events]
                    analysis['action_distribution'][self.action_names[action_idx]] = {
                        'mean': float(np.mean(values)) if values else 0.0,
                        'latest': float(values[-1]) if values else 0.0,
                        'history': values
                    }

            # Extract invalid action rate
            if 'action_dist/invalid_action_pct' in scalar_tags:
                events = ea.Scalars('action_dist/invalid_action_pct')
                values = [e.value for e in events]
                analysis['invalid_action_rate'] = {
                    'mean': float(np.mean(values)) if values else 0.0,
                    'latest': float(values[-1]) if values else 0.0,
                    'max': float(np.max(values)) if values else 0.0
                }

            # Extract rollout metrics
            rollout_metrics = ['rollout/ep_rew_mean', 'rollout/ep_len_mean',
                             'train/explained_variance', 'train/approx_kl']
            for metric in rollout_metrics:
                if metric in scalar_tags:
                    events = ea.Scalars(metric)
                    values = [e.value for e in events]
                    analysis['rollout_metrics'][metric] = {
                        'mean': float(np.mean(values)) if values else 0.0,
                        'latest': float(values[-1]) if values else 0.0,
                        'history': values[-10:] if len(values) > 10 else values  # Last 10 values
                    }

            # Extract checkpoint metrics
            checkpoint_metrics = ['checkpoint/val_reward', 'checkpoint/sharpe_ratio',
                                'checkpoint/win_rate', 'checkpoint/total_return']
            for metric in checkpoint_metrics:
                if metric in scalar_tags:
                    events = ea.Scalars(metric)
                    values = [e.value for e in events]
                    analysis['checkpoint_metrics'][metric] = {
                        'mean': float(np.mean(values)) if values else 0.0,
                        'latest': float(values[-1]) if values else 0.0,
                        'history': values
                    }

            # Diagnose issues
            analysis['issues'] = self._diagnose_issues(analysis)

        except Exception as e:
            analysis['error'] = f"Failed to analyze TensorBoard logs: {e}"

        return analysis

    def _diagnose_issues(self, analysis: Dict) -> List[str]:
        """Diagnose potential issues from analysis."""
        issues = []

        # Check action distribution
        action_dist = analysis.get('action_distribution', {})
        if action_dist:
            hold_pct = action_dist.get('HOLD', {}).get('latest', 0)
            if hold_pct > 90:
                issues.append(f"Agent is stuck on HOLD ({hold_pct:.1f}%) - not taking meaningful actions")
            elif hold_pct < 10:
                issues.append(f"Agent rarely holds ({hold_pct:.1f}%) - may be overtrading")

            # Check if entry actions are balanced
            if 'BUY' in action_dist and 'SELL' in action_dist:
                buy_pct = action_dist['BUY'].get('latest', 0)
                sell_pct = action_dist['SELL'].get('latest', 0)
                if abs(buy_pct - sell_pct) > 30:
                    issues.append(f"Unbalanced entry actions - BUY: {buy_pct:.1f}%, SELL: {sell_pct:.1f}%")

        # Check invalid action rate
        invalid_rate = analysis.get('invalid_action_rate', {})
        if invalid_rate:
            latest_invalid = invalid_rate.get('latest', 0)
            if latest_invalid > 10:
                issues.append(f"High invalid action rate ({latest_invalid:.1f}%) - action masking may not be working")

        # Check rollout metrics
        rollout = analysis.get('rollout_metrics', {})
        if 'rollout/ep_rew_mean' in rollout:
            ep_rew = rollout['rollout/ep_rew_mean'].get('latest', 0)
            if ep_rew < -1.0:
                issues.append(f"Strongly negative rewards ({ep_rew:.2f}) - agent is performing very poorly")
            elif ep_rew < 0:
                issues.append(f"Negative rewards ({ep_rew:.2f}) - agent is losing money")

        # Check checkpoint metrics
        checkpoint = analysis.get('checkpoint_metrics', {})
        if all(metric in checkpoint and checkpoint[metric].get('latest', 0) == 0.0
               for metric in ['checkpoint/val_reward', 'checkpoint/sharpe_ratio']):
            issues.append("All checkpoint metrics are 0 - evaluation metrics may not be logging correctly")

        return issues

    def print_report(self, analysis: Dict, title: str = "Action Distribution Analysis"):
        """Print a formatted analysis report."""
        print("\n" + "=" * 80)
        print(f"{title}")
        print("=" * 80)

        if 'error' in analysis:
            print(f"\n[ERROR] {analysis['error']}")
            return

        # Print action distribution
        if 'action_distribution' in analysis and analysis['action_distribution']:
            print("\n[ACTION DISTRIBUTION]")
            for action, stats in analysis['action_distribution'].items():
                print(f"  {action:20} Mean: {stats['mean']:6.2f}%  Latest: {stats['latest']:6.2f}%")

        # Print invalid action rate
        if 'invalid_action_rate' in analysis and analysis['invalid_action_rate']:
            inv = analysis['invalid_action_rate']
            print(f"\n[INVALID ACTION RATE]")
            print(f"  Mean: {inv['mean']:6.2f}%  Latest: {inv['latest']:6.2f}%  Max: {inv['max']:6.2f}%")

        # Print rollout metrics
        if 'rollout_metrics' in analysis and analysis['rollout_metrics']:
            print(f"\n[ROLLOUT METRICS]")
            for metric, stats in analysis['rollout_metrics'].items():
                metric_name = metric.split('/')[-1]
                print(f"  {metric_name:20} Mean: {stats['mean']:8.4f}  Latest: {stats['latest']:8.4f}")

        # Print checkpoint metrics
        if 'checkpoint_metrics' in analysis and analysis['checkpoint_metrics']:
            print(f"\n[CHECKPOINT METRICS (Validation)]")
            for metric, stats in analysis['checkpoint_metrics'].items():
                metric_name = metric.split('/')[-1]
                print(f"  {metric_name:20} Mean: {stats['mean']:8.4f}  Latest: {stats['latest']:8.4f}")

        # Print evaluation analysis
        if 'mean_rewards' in analysis:
            print(f"\n[EVALUATION SUMMARY]")
            print(f"  Number of evaluations: {analysis['num_evaluations']}")
            if analysis['mean_rewards']:
                print(f"  Mean reward range: {min(analysis['mean_rewards']):.4f} to {max(analysis['mean_rewards']):.4f}")
                if 'reward_improvement' in analysis:
                    print(f"  Reward improvement: {analysis['reward_improvement']:+.4f}")

        # Print issues
        if 'issues' in analysis and analysis['issues']:
            print(f"\n[DETECTED ISSUES]")
            for i, issue in enumerate(analysis['issues'], 1):
                print(f"  {i}. {issue}")

        print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze action distributions from training logs')
    parser.add_argument('--phase', type=int, default=2, choices=[1, 2, 3],
                       help='Training phase (1, 2, or 3)')
    parser.add_argument('--tensorboard', type=str, default=None,
                       help='Path to TensorBoard log directory')
    parser.add_argument('--logdir', type=str, default=None,
                       help='Path to log directory containing evaluations.npz')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for analysis results')
    args = parser.parse_args()

    analyzer = ActionDistributionAnalyzer(phase=args.phase)

    results = {}

    # Analyze TensorBoard logs
    if args.tensorboard:
        print(f"\n[INFO] Analyzing TensorBoard logs: {args.tensorboard}")
        tb_analysis = analyzer.analyze_tensorboard_logs(args.tensorboard)
        results['tensorboard'] = tb_analysis
        analyzer.print_report(tb_analysis, f"TensorBoard Analysis - Phase {args.phase}")

    # Analyze evaluations.npz
    if args.logdir:
        npz_path = os.path.join(args.logdir, 'evaluations.npz')
        print(f"\n[INFO] Analyzing evaluation logs: {npz_path}")
        eval_analysis = analyzer.analyze_evaluations_npz(npz_path)
        results['evaluations'] = eval_analysis
        analyzer.print_report(eval_analysis, f"Evaluation Analysis - Phase {args.phase}")

    # Auto-detect if no paths specified
    if not args.tensorboard and not args.logdir:
        print("\n[INFO] No paths specified. Searching for logs...")

        # Try to find logs
        phase_name = f'phase{args.phase}'
        tensorboard_path = f'tensorboard_logs/{phase_name}'
        log_path = f'logs/{phase_name}'

        if os.path.exists(tensorboard_path):
            print(f"[INFO] Found TensorBoard logs: {tensorboard_path}")
            tb_analysis = analyzer.analyze_tensorboard_logs(tensorboard_path)
            results['tensorboard'] = tb_analysis
            analyzer.print_report(tb_analysis, f"TensorBoard Analysis - Phase {args.phase}")

        if os.path.exists(os.path.join(log_path, 'evaluations.npz')):
            npz_path = os.path.join(log_path, 'evaluations.npz')
            print(f"[INFO] Found evaluation logs: {npz_path}")
            eval_analysis = analyzer.analyze_evaluations_npz(npz_path)
            results['evaluations'] = eval_analysis
            analyzer.print_report(eval_analysis, f"Evaluation Analysis - Phase {args.phase}")

    # Save results to JSON if requested
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Analysis results saved to: {args.output}")

    # Print summary recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    all_issues = []
    for analysis_type, analysis in results.items():
        if 'issues' in analysis:
            all_issues.extend(analysis['issues'])

    if all_issues:
        print("\nIdentified Issues:")
        for i, issue in enumerate(set(all_issues), 1):
            print(f"  {i}. {issue}")
    else:
        print("\nNo major issues detected. Training appears healthy!")

    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
