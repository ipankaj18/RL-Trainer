#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 Hybrid Agent Evaluation Script

Evaluates the hybrid RL + LLM trading agent on a holdout slice of market data.
Optionally compares performance against the latest Phase 2 (RL-only) baseline
and generates both a human-readable report and machine-readable JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

# Ensure src/ is on sys.path so intra-project imports resolve
sys.path.insert(0, str(Path(__file__).parent))

try:
    from environment_phase3_llm import TradingEnvironmentPhase3LLM
    from llm_reasoning import LLMReasoningModule
    from hybrid_agent import HybridTradingAgent
    from feature_engineering import add_market_regime_features
    from market_specs import get_market_spec
    from model_utils import detect_models_in_folder
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor
    from action_mask_utils import get_action_masks
    PHASE3_AVAILABLE = True
except ImportError as exc:
    print(f"Warning: Phase 3 modules not available: {exc}")
    PHASE3_AVAILABLE = False


def safe_print(message: str = "") -> None:
    """Print text while gracefully degrading on terminals without UTF-8."""
    try:
        print(message)
    except UnicodeEncodeError:
        replacements = {
            '✓': '[OK]', '✅': '[OK]', '✗': '[X]', '❌': '[X]', '→': '->',
            '⚠': '[WARN]', '⚠️': '[WARN]', '—': '-', '–': '-', '’': "'",
            '“': '"', '”': '"',
        }
        ascii_message = message
        for src, target in replacements.items():
            ascii_message = ascii_message.replace(src, target)
        ascii_message = ascii_message.encode('ascii', errors='ignore').decode('ascii')
        print(ascii_message)


def resolve_baseline_model_path(arg: str) -> Optional[str]:
    """Resolve the RL baseline model path based on CLI input."""
    if not arg or arg.lower() == "none":
        return None

    if arg.lower() == "auto":
        phase2_models = detect_models_in_folder("models", phase="phase2")
        if phase2_models:
            path = phase2_models[0]['path']
            safe_print(f"[INFO] Auto-detected Phase 2 baseline: {path}")
            return path
        safe_print("[WARN] No Phase 2 models found for baseline; skipping RL-only evaluation.")
        return None

    if os.path.exists(arg):
        return arg

    safe_print(f"[WARN] Baseline model path not found: {arg}. Skipping RL-only evaluation.")
    return None


def load_evaluation_data(market_name: str, holdout_fraction: float = 0.2) -> pd.DataFrame:
    """
    Load the most recent segment of market data for evaluation.

    Args:
        market_name: Market symbol (e.g., 'NQ', 'ES')
        holdout_fraction: Fraction (0 < x < 1) of the tail data reserved for evaluation

    Returns:
        DataFrame containing only the holdout segment
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    processed_dir = data_dir / "processed"

    data_files = sorted(glob.glob(str(processed_dir / f"{market_name}_*.csv")))
    all_data: List[pd.DataFrame] = []

    if data_files:
        safe_print(f"[DATA] Using processed evaluation files from {processed_dir}")
        for file in data_files:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
            all_data.append(df)
    else:
        fallback_path = data_dir / f"{market_name}_D1M.csv"
        if not fallback_path.exists():
            fallback_path = data_dir / "D1M.csv"
        if not fallback_path.exists():
            raise ValueError(
                f"No evaluation data found for {market_name}. "
                f"Expected {processed_dir}/{market_name}_*.csv or {fallback_path.name}."
            )
        safe_print(f"[DATA] Using fallback minute data: {fallback_path.name}")
        df = pd.read_csv(fallback_path, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
        all_data.append(df)

    data = pd.concat(all_data, axis=0)
    data.sort_index(inplace=True)
    data = add_market_regime_features(data)

    if not 0 < holdout_fraction < 1:
        holdout_fraction = 0.2

    split_idx = int(len(data) * (1 - holdout_fraction))
    if split_idx <= 0 or split_idx >= len(data):
        raise ValueError("Holdout fraction leaves no data for evaluation.")

    holdout_data = data.iloc[split_idx:]
    safe_print(f"[DATA] Loaded {len(holdout_data)} holdout rows (last {holdout_fraction:.0%} of dataset)")
    safe_print(f"[DATA] Holdout range: {holdout_data.index[0]} to {holdout_data.index[-1]}")
    return holdout_data


def create_evaluation_env(data: pd.DataFrame, market_name: Optional[str], use_llm_features: bool) -> Monitor:
    """Instantiate the Phase 3 environment with or without LLM features."""
    market_spec = get_market_spec(market_name) if market_name else None

    env = TradingEnvironmentPhase3LLM(
        data=data,
        use_llm_features=use_llm_features,
        initial_balance=50000,
        window_size=20,
        second_data=None,
        market_spec=market_spec,
        commission_override=None,
        initial_sl_multiplier=1.5,
        initial_tp_ratio=3.0,
        position_size_contracts=1.0,
        trailing_drawdown_limit=2500,
        tighten_sl_step=0.5,
        extend_tp_step=1.0,
        trailing_activation_profit=1.0
    )
    env = Monitor(env)
    return ActionMasker(env, lambda env: get_action_masks(env))


def evaluate_agent(agent, env, num_episodes: int = 10) -> Dict[str, Any]:
    """Roll out an agent against the provided environment and collect metrics."""
    from collections import defaultdict

    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    episode_trades: List[int] = []
    episode_win_rates: List[float] = []
    aggregate_trades: List[float] = []

    fusion_stats = defaultdict(list)

    safe_print(f"[EVAL] Running {num_episodes} episodes...")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_return = 0.0
        episode_length = 0
        episode_trades_list: List[float] = []

        while not (done or truncated):
            if hasattr(agent, 'predict'):  # Hybrid agent
                position_state = {
                    'position': env.position,
                    'balance': env.balance,
                    'win_rate': env._calculate_win_rate(),
                    'consecutive_losses': env.consecutive_losses,
                    'dd_buffer_ratio': (
                        (env.balance - (env.peak_balance - 2500)) / env.balance if env.peak_balance > 0 else 1.0
                    )
                }
                market_context = env.get_llm_context()
                action_mask = get_action_masks(env)
                action, meta = agent.predict(obs, action_mask, position_state, market_context)

                if meta:
                    if meta.get('fusion_method'):
                        fusion_stats[meta['fusion_method']].append(1)
                    if meta.get('risk_veto'):
                        fusion_stats['risk_veto'].append(1)
                    if meta.get('llm_queried'):
                        fusion_stats['llm_queried'].append(1)
            else:  # Plain RL model
                action, _ = agent.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)
            episode_return += reward
            episode_length += 1

            pnl = info.get('trade_pnl')
            if pnl:
                episode_trades_list.append(pnl)
                aggregate_trades.append(pnl)

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        episode_trades.append(len(episode_trades_list))
        win_rate = (
            sum(1 for pnl in episode_trades_list if pnl > 0) / len(episode_trades_list)
            if episode_trades_list else 0.0
        )
        episode_win_rates.append(win_rate)

        if (episode + 1) % max(1, num_episodes // 5) == 0:
            safe_print(f"[EVAL] Completed {episode + 1}/{num_episodes} episodes...")

    returns_array = np.array(episode_returns)
    lengths_array = np.array(episode_lengths)
    trades_array = np.array(episode_trades)
    win_rates_array = np.array(episode_win_rates)

    metrics = {
        'mean_return': float(np.mean(returns_array)),
        'std_return': float(np.std(returns_array)),
        'sharpe_ratio': float(np.mean(returns_array) / (np.std(returns_array) + 1e-8)),
        'mean_length': float(np.mean(lengths_array)),
        'mean_trades': float(np.mean(trades_array)),
        'win_rate': float(np.mean(win_rates_array)),
        'total_return': float(np.sum(returns_array)),
        'max_drawdown': float(np.max(np.maximum.accumulate(returns_array) - returns_array)),
        'profit_factor': float(calculate_profit_factor(aggregate_trades)) if aggregate_trades else 0.0
    }

    if fusion_stats:
        total_counts = sum(len(v) for v in fusion_stats.values())
        metrics['fusion_stats'] = {
            stat: len(entries) / total_counts * 100 if total_counts else 0
            for stat, entries in fusion_stats.items()
        }

    return metrics


def calculate_profit_factor(trades_list: List[float]) -> float:
    """Compute profit factor for a list of trade P&Ls."""
    if not trades_list:
        return 0.0
    gross_profit = sum(pnl for pnl in trades_list if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in trades_list if pnl < 0))
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def run_component_ablation_study(
    data: pd.DataFrame,
    hybrid_model_path: str,
    market_name: str,
    config: Dict[str, Any],
    baseline_model_path: Optional[str]
) -> Dict[str, Any]:
    """Evaluate RL baseline (optional) and hybrid agent on the provided dataset."""
    results: Dict[str, Any] = {}

    if baseline_model_path:
        safe_print(f"[LOAD] Loading RL baseline: {baseline_model_path}")
        try:
            baseline_model = MaskablePPO.load(baseline_model_path)
            safe_print("[OK] RL baseline loaded")
            safe_print("\n[EVAL] Evaluating RL-only baseline...")
            rl_env = create_evaluation_env(data, market_name, use_llm_features=False)
            results['rl_only'] = evaluate_agent(baseline_model, rl_env, num_episodes=config.get('episodes', 10))
            print_metrics("RL-Only", results['rl_only'])
            rl_env.close()
        except Exception as exc:
            safe_print(f"[WARN] Baseline evaluation skipped (load failed): {exc}")
    else:
        safe_print("[INFO] No RL baseline model provided. Skipping RL-only evaluation.")

    safe_print("\n[LOAD] Loading hybrid RL model...")
    custom_objects = {}
    try:
        from hybrid_policy_with_adapter import HybridAgentPolicyWithAdapter
        custom_objects["policy_class"] = HybridAgentPolicyWithAdapter
    except ImportError:
        safe_print("[WARN] Could not import HybridAgentPolicyWithAdapter. Attempting default policy load.")

    try:
        hybrid_model = MaskablePPO.load(hybrid_model_path, custom_objects=custom_objects)
        safe_print("[OK] Hybrid RL weights loaded")
    except Exception as exc:
        safe_print(f"[ERROR] Failed to load hybrid model: {exc}")
        return results

    try:
        llm_model = LLMReasoningModule(
            config_path=config['llm_config_path']
        )
        safe_print("[OK] LLM advisor initialized")
    except Exception as exc:
        safe_print(f"[ERROR] Failed to initialize LLM module: {exc}")
        safe_print("[INFO] Ensure Phi-3-mini-4k-instruct model is downloaded to project folder")
        return results

    hybrid_agent = HybridTradingAgent(rl_model=hybrid_model, llm_model=llm_model, config=config)
    safe_print("\n[EVAL] Evaluating hybrid agent...")
    hybrid_env = create_evaluation_env(data, market_name, use_llm_features=True)
    results['hybrid'] = evaluate_agent(hybrid_agent, hybrid_env, num_episodes=config.get('episodes', 10))
    print_metrics("Hybrid", results['hybrid'])
    hybrid_env.close()

    return results


def print_metrics(agent_name: str, metrics: Dict[str, Any]) -> None:
    """Pretty-print metrics for a given agent."""
    safe_print(f"\n{agent_name} Results:")
    safe_print(f"  Mean Return: ${metrics['mean_return']:,.2f}")
    safe_print(f"  Std Return: ${metrics['std_return']:,.2f}")
    safe_print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    safe_print(f"  Win Rate: {metrics['win_rate']:.1%}")
    safe_print(f"  Mean Trades/Episode: {metrics['mean_trades']:.1f}")
    safe_print(f"  Max Drawdown: ${metrics['max_drawdown']:,.2f}")
    safe_print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    if 'fusion_stats' in metrics:
        safe_print("  Fusion Stats:")
        for key, value in metrics['fusion_stats'].items():
            safe_print(f"    {key}: {value:.1f}%")


def generate_evaluation_report(results: Dict[str, Any], output_path: Optional[Path]) -> None:
    """Write a text report summarizing evaluation results."""
    safe_print("\n" + "=" * 70)
    safe_print("EVALUATION REPORT")
    safe_print("=" * 70)

    if not results:
        safe_print("No results to report")
        return

    hybrid = results.get('hybrid')
    rl_only = results.get('rl_only')

    if hybrid and rl_only:
        safe_print("\nComponent Comparison:")
        safe_print("-" * 70)
        return_improvement = hybrid['mean_return'] - rl_only['mean_return']
        sharpe_improvement = hybrid['sharpe_ratio'] - rl_only['sharpe_ratio']
        winrate_improvement = hybrid['win_rate'] - rl_only['win_rate']

        safe_print(f"Mean Return:     ${hybrid['mean_return']:,.2f} vs ${rl_only['mean_return']:,.2f} "
                   f"({return_improvement:+.2f})")
        safe_print(f"Sharpe Ratio:    {hybrid['sharpe_ratio']:.2f} vs {rl_only['sharpe_ratio']:.2f} "
                   f"({sharpe_improvement:+.2f})")
        safe_print(f"Win Rate:        {hybrid['win_rate']:.1%} vs {rl_only['win_rate']:.1%} "
                   f"({winrate_improvement:+.1%})")

        if hybrid['std_return'] > 0 and rl_only['std_return'] > 0:
            hybrid_sem = hybrid['std_return'] / np.sqrt(10)
            rl_sem = rl_only['std_return'] / np.sqrt(10)
            hybrid_ci = 1.96 * hybrid_sem
            rl_ci = 1.96 * rl_sem

            safe_print("\nStatistical Significance (95% CI):")
            safe_print(f"  Hybrid:  ${hybrid['mean_return']:,.2f} ± ${hybrid_ci:,.2f}")
            safe_print(f"  RL-Only: ${rl_only['mean_return']:,.2f} ± ${rl_ci:,.2f}")

            if abs(return_improvement) > (hybrid_ci + rl_ci):
                safe_print("  → Improvement is statistically significant")
            else:
                safe_print("  → Improvement is not statistically significant")

    if hybrid and 'fusion_stats' in hybrid:
        safe_print("\nLLM Contribution Analysis:")
        safe_print("-" * 70)
        fusion_stats = hybrid['fusion_stats']
        safe_print(f"LLM Query Rate: {fusion_stats.get('llm_queried', 0):.1f}%")
        safe_print(f"Agreement Rate: {fusion_stats.get('agreement', 0):.1f}%")
        safe_print(f"Risk Veto Rate: {fusion_stats.get('risk_veto', 0):.1f}%")
        override_rate = fusion_stats.get('llm_weighted', 0) + fusion_stats.get('llm_confident', 0)
        safe_print(f"LLM Override Rate: {override_rate:.1f}%")

    if output_path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as handle:
                handle.write("Phase 3 Hybrid Agent Evaluation Report\n")
                handle.write("=" * 70 + "\n\n")
                handle.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                if hybrid:
                    handle.write("Hybrid Metrics:\n")
                    for key, value in hybrid.items():
                        if isinstance(value, dict):
                            handle.write(f"  {key}:\n")
                            for sub_key, sub_value in value.items():
                                handle.write(f"    {sub_key}: {sub_value}\n")
                        else:
                            handle.write(f"  {key}: {value}\n")
                    handle.write("\n")

                if hybrid and rl_only:
                    handle.write("Component Comparison:\n")
                    handle.write("-" * 70 + "\n")
                    handle.write(f"Mean Return Improvement: ${return_improvement:,.2f}\n")
                    handle.write(f"Sharpe Ratio Improvement: {sharpe_improvement:+.2f}\n")
                    handle.write(f"Win Rate Improvement: {winrate_improvement:+.1%}\n")

            safe_print(f"\n[SAVE] Report saved to: {output_path}")
        except Exception as exc:
            safe_print(f"[WARNING] Could not save report: {exc}")


def save_results_json(results: Dict[str, Any], output_path: Path) -> None:
    """Persist evaluation results as JSON for downstream tooling."""
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.integer, np.int32, np.int64)):
            return int(value)
        return value

    serializable = {
        agent: {k: convert(v) for k, v in metrics.items()}
        for agent, metrics in results.items()
    }

    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(serializable, handle, indent=2)
    safe_print(f"[SAVE] JSON summary saved to: {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Phase 3 Hybrid Agent Evaluation")
    parser.add_argument('--model', required=True, help='Path to trained Phase 3 hybrid model')
    parser.add_argument('--market', required=True, help='Market symbol to evaluate')
    parser.add_argument('--config', default="./config/llm_config.yaml", help='LLM config file')
    parser.add_argument('--episodes', type=int, default=20, help='Episodes per evaluation agent')
    parser.add_argument('--output', help='Optional report output path (defaults to results/phase3_evaluation_TIMESTAMP.txt)')
    parser.add_argument('--holdout-fraction', type=float, default=0.2, help='Fraction of data used as holdout set')
    parser.add_argument('--baseline-model', type=str, default="auto",
                        help="Phase 2 baseline model path. Use 'auto' to detect, or 'none' to skip.")

    args = parser.parse_args()

    if not PHASE3_AVAILABLE:
        safe_print("[ERROR] Phase 3 modules not available")
        sys.exit(1)

    if not os.path.exists(args.model):
        safe_print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"phase3_evaluation_{timestamp}.txt"
    summary_json_path = output_path.with_suffix('.json')

    safe_print("=" * 70)
    safe_print("Phase 3 Hybrid Agent Evaluation")
    safe_print("=" * 70)
    safe_print(f"Model: {args.model}")
    safe_print(f"Market: {args.market}")
    safe_print(f"Episodes: {args.episodes}")
    safe_print(f"Holdout Fraction: {args.holdout_fraction:.0%}")
    safe_print("=" * 70)

    config = {
        'llm_config_path': args.config,
        'episodes': args.episodes
    }

    baseline_path = resolve_baseline_model_path(args.baseline_model)

    try:
        data = load_evaluation_data(args.market, holdout_fraction=args.holdout_fraction)
        results = run_component_ablation_study(
            data=data,
            hybrid_model_path=args.model,
            market_name=args.market,
            config=config,
            baseline_model_path=baseline_path
        )

        generate_evaluation_report(results, output_path)
        if results:
            save_results_json(results, summary_json_path)

        safe_print("\n✅ Evaluation completed successfully")
    except Exception as exc:
        safe_print(f"\n[ERROR] Evaluation failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
