#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Episode Termination Diagnostic Tool

Purpose: Analyze why Phase 3 episodes terminate early (currently ~19.9 bars)
Target: Identify root causes and recommend fixes

Usage:
    python src/diagnostics/episode_termination_analysis.py --market NQ --episodes 100

Author: RL Trading System Team
Date: November 2025
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import argparse
from typing import Dict, List
from collections import defaultdict

from environment_phase3_llm import TradingEnvironmentPhase3LLM
from market_specs import get_market_spec
from feature_engineering import add_market_regime_features


class TerminationAnalyzer:
    """Analyze episode termination patterns."""

    def __init__(self, market: str = 'NQ', verbose: bool = True):
        """
        Initialize analyzer.

        Args:
            market: Market symbol (e.g., 'NQ', 'ES')
            verbose: Print progress messages
        """
        self.market = market
        self.verbose = verbose

        # Statistics containers
        self.termination_reasons = defaultdict(int)
        self.episode_lengths = []
        self.episode_rewards = []
        self.episode_pnls = []
        self.trades_per_episode = []
        self.termination_details = []

    def load_data(self) -> pd.DataFrame:
        """Load and prepare market data."""
        if self.verbose:
            print(f"\n[DATA] Loading {self.market} data...")

        data_path = Path(f'./data/{self.market}_D1M.csv')
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n"
                f"Run data processing first: python src/update_training_data.py --market {self.market}"
            )

        # Load data
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)

        # Ensure timezone-aware datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index, utc=True)

        # Handle timezone conversion
        if hasattr(data.index, 'tz') and data.index.tz is None:
            # No timezone - add UTC then convert to ET
            data.index = data.index.tz_localize('UTC').tz_convert("America/New_York")
        elif hasattr(data.index, 'tz') and data.index.tz is not None:
            # Already has timezone - just convert to ET if needed
            if str(data.index.tz) != 'America/New_York':
                data.index = data.index.tz_convert("America/New_York")

        # Add market regime features (including LLM features for Phase 3)
        if self.verbose:
            print(f"[DATA] Adding market regime features...")
        data = add_market_regime_features(data)

        if self.verbose:
            print(f"[DATA] Loaded {len(data)} rows from {data.index[0]} to {data.index[-1]}")
            print(f"[DATA] Features: {len(data.columns)} columns")

        return data

    def create_environment(self, data: pd.DataFrame) -> TradingEnvironmentPhase3LLM:
        """Create Phase 3 environment for testing."""
        market_spec = get_market_spec(self.market)

        env = TradingEnvironmentPhase3LLM(
            data=data,
            use_llm_features=True,  # Use 261D observations
            initial_balance=50000,
            window_size=20,
            second_data=None,
            market_spec=market_spec,
            commission_override=None,
            initial_sl_multiplier=2.5,  # Widened from 1.5 to 2.5 ATR for learning phase
            initial_tp_ratio=3.0,
            position_size_contracts=1.0,
            trailing_drawdown_limit=15000,  # TRAINING: $15K matches Phase 1/2 (Apex uses $2,500)
            tighten_sl_step=0.5,
            extend_tp_step=1.0,
            trailing_activation_profit=1.0,
            hybrid_agent=None,
            start_index=None,
            randomize_start_offsets=True,
            min_episode_bars=300  # Reduced from 1500 to match training config
        )

        return env

    def classify_termination(self, env: TradingEnvironmentPhase3LLM,
                            info: Dict, terminated: bool, truncated: bool) -> str:
        """
        Classify why the episode terminated.

        Args:
            env: Environment instance
            info: Info dictionary from step
            terminated: Terminated flag
            truncated: Truncated flag

        Returns:
            Termination reason string
        """
        # Check for explicit termination reason in info
        if 'done_reason' in info and info['done_reason']:
            return info['done_reason']
        if 'termination_reason' in info:
            return info['termination_reason']

        # Infer from environment state
        current_balance = env.balance
        initial_balance = env.initial_balance
        drawdown = initial_balance - current_balance
        dd_limit = getattr(env, 'trailing_dd_limit', getattr(env, 'trailing_drawdown_limit', 2500))

        # Check drawdown limit
        if drawdown >= dd_limit:
            return 'drawdown_limit_hit'

        # Check if we ran out of data
        if env.current_step >= len(env.data) - env.window_size - 10:
            return 'data_exhausted'

        # Check if episode was artificially truncated (max steps)
        if truncated and not terminated:
            return 'max_steps_reached'

        # Check if position hit SL/TP
        if 'trade_pnl' in info and info['trade_pnl'] != 0:
            if info['trade_pnl'] > 0:
                return 'take_profit_hit'
            else:
                return 'stop_loss_hit'

        # Unknown reason
        return 'unknown'

    def run_episodes(self, num_episodes: int = 100) -> None:
        """
        Run multiple episodes and collect termination statistics.

        Args:
            num_episodes: Number of episodes to run
        """
        if self.verbose:
            print(f"\n[DIAGNOSTIC] Running {num_episodes} episodes...")

        # Load data and create environment
        data = self.load_data()
        env = self.create_environment(data)

        # Run episodes
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            steps = 0
            episode_reward = 0
            trades_count = 0

            while not done:
                # Get action mask
                try:
                    action_mask = env.action_masks()
                except AttributeError:
                    # Fallback if action_masks not directly available
                    action_mask = np.ones(env.action_space.n, dtype=bool)

                # Random valid action
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) == 0:
                    valid_actions = [0]  # Default to HOLD
                action = np.random.choice(valid_actions)

                # Step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                episode_reward += reward

                # Track trades
                if 'trade_pnl' in info and info['trade_pnl'] != 0:
                    trades_count += 1

                # Check for emergency exit (infinite loop protection)
                if steps > 10000:
                    if self.verbose:
                        print(f"[WARNING] Episode {ep+1} exceeded 10K steps, forcing termination")
                    done = True
                    reason = 'emergency_exit'
                    break

            # Classify termination reason
            if steps <= 10000:
                reason = self.classify_termination(env, info, terminated, truncated)

            # Store statistics
            self.termination_reasons[reason] += 1
            self.episode_lengths.append(steps)
            self.episode_rewards.append(episode_reward)
            self.trades_per_episode.append(trades_count)

            final_pnl = env.balance - env.initial_balance
            self.episode_pnls.append(final_pnl)

            self.termination_details.append({
                'episode': ep + 1,
                'length': steps,
                'reason': reason,
                'reward': episode_reward,
                'final_pnl': final_pnl,
                'trades': trades_count
            })

            # Progress update
            if self.verbose and (ep + 1) % 10 == 0:
                print(f"[PROGRESS] Completed {ep+1}/{num_episodes} episodes " +
                      f"(avg length: {np.mean(self.episode_lengths):.1f} bars)")

        if self.verbose:
            print(f"[DIAGNOSTIC] Completed {num_episodes} episodes\n")

    def print_report(self) -> None:
        """Print comprehensive analysis report."""
        print("=" * 80)
        print(" EPISODE TERMINATION ANALYSIS - DIAGNOSTIC REPORT")
        print("=" * 80)

        # Basic statistics
        print(f"\n{'EPISODE STATISTICS':^80}")
        print("-" * 80)
        print(f"  Total Episodes:        {len(self.episode_lengths)}")
        print(f"  Average Length:        {np.mean(self.episode_lengths):.1f} bars")
        print(f"  Median Length:         {np.median(self.episode_lengths):.1f} bars")
        print(f"  Min Length:            {np.min(self.episode_lengths)} bars")
        print(f"  Max Length:            {np.max(self.episode_lengths)} bars")
        print(f"  Std Deviation:         {np.std(self.episode_lengths):.1f} bars")

        # Termination reasons
        print(f"\n{'TERMINATION REASONS':^80}")
        print("-" * 80)
        total = len(self.episode_lengths)
        sorted_reasons = sorted(self.termination_reasons.items(),
                               key=lambda x: -x[1])

        for reason, count in sorted_reasons:
            pct = 100 * count / total
            bar = "█" * int(pct / 2)  # Visual bar (50 chars = 100%)
            print(f"  {reason:25s}: {count:4d} ({pct:5.1f}%) {bar}")

        # Rewards and P&L
        print(f"\n{'PERFORMANCE METRICS':^80}")
        print("-" * 80)
        print(f"  Average Reward:        {np.mean(self.episode_rewards):.2f}")
        print(f"  Average P&L:           ${np.mean(self.episode_pnls):.2f}")
        print(f"  Average Trades/Ep:     {np.mean(self.trades_per_episode):.1f}")

        # Analysis and recommendations
        print(f"\n{'ROOT CAUSE ANALYSIS':^80}")
        print("-" * 80)

        primary_reason = sorted_reasons[0][0] if sorted_reasons else "unknown"
        primary_pct = sorted_reasons[0][1] / total * 100 if sorted_reasons else 0

        print(f"\n  PRIMARY CAUSE: {primary_reason.upper()} ({primary_pct:.1f}%)")
        print()

        # Recommendations based on primary cause
        if primary_reason == 'stop_loss_hit':
            print("  💡 RECOMMENDATION:")
            print("     Stop-loss is too tight. Episodes end before strategy can work.")
            print()
            print("  🔧 FIX:")
            print("     In train_phase3_llm.py, increase initial_sl_multiplier:")
            print("     initial_sl_multiplier=2.0  # From 1.5 to 2.0 ATR")
            print("     OR")
            print("     initial_sl_multiplier=2.5  # Even wider for learning")

        elif primary_reason in ['drawdown_limit_hit', 'minute_trailing_drawdown', 'second_level_trailing_drawdown']:
            print("  💡 RECOMMENDATION:")
            print("     Agent hitting $2,500 trailing drawdown limit in ~23 bars.")
            print()
            avg_trades = np.mean(self.trades_per_episode)
            if avg_trades < 0.1:
                print("  ⚠️  CRITICAL BUG DETECTED:")
                print(f"     Average trades: {avg_trades:.1f} (NO TRADES!)")
                print(f"     Average loss: ${np.mean(self.episode_pnls):.2f}")
                print("     → Environment is losing money WITHOUT executing trades!")
                print()
                print("  🔧 URGENT FIX NEEDED:")
                print("     1. Check environment_phase2.py for commission on HOLD actions")
                print("     2. Verify balance calculation doesn't deduct without trades")
                print("     3. Check action masking - are BUY/SELL actions blocked?")
                print("     4. Run single episode with verbose logging to trace bug")
                print()
            print("  🔧 TEMPORARY WORKAROUND (TRAINING ONLY):")
            print("     Relax drawdown limit to observe full episode behavior:")
            print("     trailing_drawdown_limit=10000  # From $2,500 to $10,000")
            print("     This will help identify if trades execute in longer episodes.")
            print()
            print("     ⚠️  IMPORTANT: Keep $2,500 limit for Apex evaluation!")

        elif primary_reason == 'data_exhausted':
            print("  💡 RECOMMENDATION:")
            print("     Running out of data - need more historical bars.")
            print()
            print("  🔧 FIX:")
            print("     1. Download 6-12 months of data (target: 150K+ rows)")
            print("     2. Current setting is 300 bars - already reduced from 1500")
            print("     3. If still insufficient, reduce further to 200-250:")

        elif primary_reason == 'max_steps_reached':
            print("  💡 RECOMMENDATION:")
            print("     Episodes hitting artificial step limit.")
            print()
            print("  🔧 FIX:")
            print("     Check if max_episode_steps is set too low in config.")

        else:
            print("  💡 RECOMMENDATION:")
            print("     Unknown termination pattern - requires deeper investigation.")
            print()
            print("  🔧 NEXT STEPS:")
            print("     1. Check environment termination logic in environment_phase3_llm.py")
            print("     2. Add logging to step() method to track state changes")
            print("     3. Run with --verbose flag for detailed per-step output")

        # Data volume warning
        print(f"\n{'DATA VOLUME CHECK':^80}")
        print("-" * 80)

        # This would require loading data again, so estimate from episode stats
        avg_length = np.mean(self.episode_lengths)
        min_bars = 300  # From config (reduced from 1500)

        if avg_length < 50:
            print("  ⚠️  WARNING: Episodes are VERY SHORT (<50 bars)")
            print("     This prevents learning proper position management.")
            print("     Agent only learns entry patterns, not trade lifecycle.")
        elif avg_length < 80:
            print("  ⚠️  WARNING: Episodes are SHORT (<80 bars)")
            print("     Target is 100-300 bars for full trade lifecycle learning.")
        else:
            print("  ✅ GOOD: Episode length is adequate for learning.")

        print("\n" + "=" * 80)
        print(" END OF DIAGNOSTIC REPORT")
        print("=" * 80 + "\n")

    def save_details_csv(self, output_path: str = './logs/episode_termination_details.csv') -> None:
        """Save detailed episode data to CSV for further analysis."""
        df = pd.DataFrame(self.termination_details)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[SAVE] Detailed episode data saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze Phase 3 episode termination patterns')
    parser.add_argument('--market', type=str, default='NQ',
                       help='Market symbol (default: NQ)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes to run (default: 100)')
    parser.add_argument('--save-csv', action='store_true',
                       help='Save detailed results to CSV')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')

    args = parser.parse_args()

    # Run analysis
    analyzer = TerminationAnalyzer(
        market=args.market,
        verbose=not args.quiet
    )

    analyzer.run_episodes(num_episodes=args.episodes)
    analyzer.print_report()

    if args.save_csv:
        analyzer.save_details_csv()

    print("\n[COMPLETE] Diagnostic analysis finished.")
    print("[NEXT STEP] Apply recommended fixes and rerun training.\n")


if __name__ == "__main__":
    main()
