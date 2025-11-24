#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment Diagnostic Tool

Tests the trading environment with processed data to ensure:
1. Data files load correctly
2. Environment initializes properly
3. Basic trading actions work (buy, sell, hold)
4. No critical errors occur

Usage:
    python diagnose_environment.py --phase 1 --steps 100
    python diagnose_environment.py --phase 2 --steps 100

Returns:
    Exit code 0: All tests passed
    Exit code 1: Tests failed
    Exit code 2: Data files not found
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def check_data_files(data_dir, market="ES"):
    """
    Check if required data files exist.

    Args:
        data_dir: Directory containing data files
        market: Market prefix (ES, NQ, etc.)

    Returns:
        Tuple of (minute_file, second_file) or (None, None) if not found
    """
    minute_file = os.path.join(data_dir, f"{market}_D1M.csv")
    second_file = os.path.join(data_dir, f"{market}_D1S.csv")

    if not os.path.exists(minute_file):
        print(f"[ERROR] Minute data file not found: {minute_file}")
        return None, None

    if not os.path.exists(second_file):
        print(f"[WARN] Second data file not found: {second_file}")
        # Second data is optional for Phase 1
        second_file = None

    return minute_file, second_file


def load_data(minute_file, second_file=None):
    """
    Load and validate data files.

    Args:
        minute_file: Path to minute-level data
        second_file: Path to second-level data (optional)

    Returns:
        Tuple of (minute_df, second_df) or None on error
    """
    try:
        # Load minute data
        print(f"Loading minute data: {minute_file}")
        minute_df = pd.read_csv(minute_file, index_col=0, parse_dates=True)
        print(f"  Loaded {len(minute_df):,} rows with {len(minute_df.columns)} columns")

        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in minute_df.columns]

        if missing_cols:
            print(f"[ERROR] Missing required columns: {missing_cols}")
            return None, None

        # Load second data if available
        second_df = None
        if second_file and os.path.exists(second_file):
            print(f"Loading second data: {second_file}")
            second_df = pd.read_csv(second_file, index_col=0, parse_dates=True)
            print(f"  Loaded {len(second_df):,} rows")

        return minute_df, second_df

    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_phase1_environment(minute_df, second_df=None, num_steps=100):
    """
    Test Phase 1 trading environment.

    Args:
        minute_df: Minute-level data
        second_df: Second-level data (optional)
        num_steps: Number of steps to simulate

    Returns:
        True if test passed, False otherwise
    """
    try:
        from environment_phase1 import TradingEnvironmentPhase1

        print("\n[TEST] Phase 1 Environment")
        print("-" * 60)

        # Create environment
        env = TradingEnvironmentPhase1(
            data=minute_df,
            second_data=second_df,
            initial_balance=50000,
            window_size=20
        )

        print("  Environment created successfully")

        # Reset environment
        obs, info = env.reset()
        print(f"  Reset successful - observation shape: {obs.shape}")

        # Run simulation
        print(f"  Running {num_steps} step simulation...")

        actions_taken = {'hold': 0, 'buy': 0, 'sell': 0}

        for step in range(num_steps):
            # Random action selection
            if env.position == 0:
                action = np.random.choice([0, 1, 2], p=[0.7, 0.15, 0.15])
            else:
                action = 0  # Hold when in position

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)

            # Track actions
            if action == 0:
                actions_taken['hold'] += 1
            elif action == 1:
                actions_taken['buy'] += 1
            elif action == 2:
                actions_taken['sell'] += 1

            # Check for termination
            if terminated or truncated:
                print(f"  Episode ended at step {step + 1}")
                break

        # Verify actions occurred
        print(f"\n  Actions taken:")
        print(f"    Hold: {actions_taken['hold']}")
        print(f"    Buy: {actions_taken['buy']}")
        print(f"    Sell: {actions_taken['sell']}")

        # Ensure at least some actions were taken
        if actions_taken['buy'] == 0 and actions_taken['sell'] == 0:
            print("  [WARN] No buy/sell actions executed")
        else:
            print("  Buy action: OK" if actions_taken['buy'] > 0 else "  Buy action: Not executed")
            print("  Sell action: OK" if actions_taken['sell'] > 0 else "  Sell action: Not executed")

        print(f"\n  Final portfolio value: ${info.get('portfolio_value', 0):,.2f}")
        print(f"  Total trades: {env.num_trades}")

        print("\n[PASS] Phase 1 Environment Test")
        return True

    except ImportError as e:
        print(f"[ERROR] Failed to import Phase 1 environment: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Phase 1 environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase2_environment(minute_df, second_df=None, num_steps=100):
    """
    Test Phase 2 trading environment.

    Args:
        minute_df: Minute-level data
        second_df: Second-level data (optional)
        num_steps: Number of steps to simulate

    Returns:
        True if test passed, False otherwise
    """
    try:
        from environment_phase2 import TradingEnvironmentPhase2

        print("\n[TEST] Phase 2 Environment")
        print("-" * 60)

        # Create environment
        env = TradingEnvironmentPhase2(
            data=minute_df,
            second_data=second_df,
            initial_balance=50000,
            window_size=20
        )

        print("  Environment created successfully")

        # Reset environment
        obs, info = env.reset()
        print(f"  Reset successful - observation shape: {obs.shape}")
        print(f"  Action space size: {env.action_space.n}")

        # Run simulation
        print(f"  Running {num_steps} step simulation...")

        for step in range(num_steps):
            # Simple action selection
            if env.position == 0:
                action = np.random.choice([0, 1, 2], p=[0.7, 0.15, 0.15])
            else:
                action = 0  # Hold when in position

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)

            # Check for termination
            if terminated or truncated:
                print(f"  Episode ended at step {step + 1}")
                break

        print(f"\n  Final portfolio value: ${info.get('portfolio_value', 0):,.2f}")
        print(f"  Total trades: {env.num_trades}")

        print("\n[PASS] Phase 2 Environment Test")
        return True

    except ImportError as e:
        print(f"[ERROR] Failed to import Phase 2 environment: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Phase 2 environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main diagnostic function."""
    parser = argparse.ArgumentParser(
        description='Diagnose trading environment setup',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--phase', type=int, choices=[1, 2], default=1,
                       help='Environment phase to test (1 or 2)')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of simulation steps')
    parser.add_argument('--market', type=str, default='ES',
                       help='Market to test (ES, NQ, etc.)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')

    args = parser.parse_args()

    print("=" * 60)
    print("ENVIRONMENT DIAGNOSTIC TOOL")
    print("=" * 60)
    print(f"Phase: {args.phase}")
    print(f"Market: {args.market}")
    print(f"Steps: {args.steps}")
    print(f"Data directory: {args.data_dir}")

    # Change to project root (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")

    # Check data files
    print("\n[1/3] CHECKING DATA FILES")
    print("-" * 60)
    minute_file, second_file = check_data_files(args.data_dir, args.market)

    if minute_file is None:
        print("\n[FAIL] Required data files not found")
        return 2

    # Load data
    print("\n[2/3] LOADING DATA")
    print("-" * 60)
    minute_df, second_df = load_data(minute_file, second_file)

    if minute_df is None:
        print("\n[FAIL] Failed to load data")
        return 1

    # Test environment
    print("\n[3/3] TESTING ENVIRONMENT")
    print("-" * 60)

    if args.phase == 1:
        success = test_phase1_environment(minute_df, second_df, args.steps)
    else:
        success = test_phase2_environment(minute_df, second_df, args.steps)

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("DIAGNOSTIC COMPLETE: ALL TESTS PASSED")
        print("=" * 60)
        return 0
    else:
        print("DIAGNOSTIC COMPLETE: TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Diagnostic interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
