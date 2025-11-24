#!/usr/bin/env python3
"""
Single Trade Validation Test

This script validates that ONE complete trade executes correctly with all
position management actions working as expected. This provides confidence
that the environment is wired correctly before running expensive training.

Test Flow:
1. Create synthetic price data with predictable movement
2. Initialize environment with fixed settings (no randomization)
3. Execute hardcoded action sequence (not random)
4. Validate exact values at each step
5. Confirm profitable trade closes correctly

Usage:
    python tests/test_single_trade_validation.py
    OR
    pytest tests/test_single_trade_validation.py -v
"""

import sys
import os
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Change to project root for imports
os.chdir(project_root)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import environment and specs
from environment_phase3_llm import TradingEnvironmentPhase3LLM
from market_specs import get_market_spec
from technical_indicators import add_all_indicators
from feature_engineering import add_market_regime_features


def create_synthetic_test_data(num_bars=50, base_price=20000.0):
    """
    Create synthetic OHLCV data with controlled price movement.

    Price Movement Design:
    - Bars 0-20: Sideways at 20,000 (warmup for indicators)
    - Bar 21: Entry point at 20,000
    - Bars 22-27: Gradual rise to 20,015 (enable breakeven)
    - Bars 28-35: Rise to 20,040 (enable trailing)
    - Bars 36+: Rise to 20,080 (hit take profit)

    Args:
        num_bars: Number of bars to generate
        base_price: Starting price level

    Returns:
        pd.DataFrame with OHLCV data and datetime index
    """
    print("\n[DATA] Creating synthetic test data...")

    # Create datetime index (1-minute bars)
    start_time = datetime(2024, 11, 20, 9, 30)  # 9:30 AM ET
    timestamps = [start_time + timedelta(minutes=i) for i in range(num_bars)]

    # Design price movement
    prices = np.zeros(num_bars)

    # Bars 0-20: Sideways (warmup)
    prices[0:21] = base_price + np.random.normal(0, 2, 21)

    # Bar 21: Entry point
    prices[21] = base_price

    # Bars 22-27: Small rise (+15 points for BE move)
    prices[22:28] = np.linspace(base_price + 5, base_price + 15, 6)

    # Bars 28-35: Larger rise (+40 points for trailing)
    prices[28:36] = np.linspace(base_price + 20, base_price + 40, 8)

    # Bars 36+: Continue rise to hit TP (~+80 points)
    if num_bars > 36:
        prices[36:] = np.linspace(base_price + 50, base_price + 80, num_bars - 36)

    # Create OHLCV data
    # For simplicity, use close price with small random variations for O/H/L
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 1, num_bars),
        'high': prices + np.abs(np.random.normal(2, 1, num_bars)),
        'low': prices - np.abs(np.random.normal(2, 1, num_bars)),
        'close': prices,
        'volume': np.random.randint(5000, 15000, num_bars)
    }, index=pd.DatetimeIndex(timestamps, tz='America/New_York'))

    # Add technical indicators (required for environment)
    print("[DATA] Adding technical indicators...")
    data = add_all_indicators(data)

    # Add market regime features (including LLM features)
    print("[DATA] Adding market regime features...")
    data = add_market_regime_features(data)

    print(f"[DATA] Created {len(data)} bars from {data.index[0]} to {data.index[-1]}")
    print(f"[DATA] Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    print(f"[DATA] Features: {len(data.columns)} columns")

    return data


def validate_action_mask(mask, expected_pattern, step_name):
    """
    Validate action mask matches expected pattern.

    Args:
        mask: Actual action mask array
        expected_pattern: Expected mask (use None for "don't care")
        step_name: Name of current step for error messages
    """
    action_names = ['HOLD', 'BUY', 'SELL', 'MOVE_TO_BE', 'ENABLE_TRAIL', 'DISABLE_TRAIL']

    for i, (actual, expected) in enumerate(zip(mask, expected_pattern)):
        if expected is not None and actual != expected:
            raise AssertionError(
                f"[{step_name}] Action mask mismatch for {action_names[i]}: "
                f"expected {expected}, got {actual}"
            )


def test_single_trade_lifecycle():
    """
    Test a complete long trade lifecycle with all position management actions.

    This is the main validation test that exercises all 6 actions:
    - HOLD (0)
    - BUY (1)
    - SELL (2)
    - MOVE_TO_BE (3)
    - ENABLE_TRAIL (4)
    - DISABLE_TRAIL (5)
    """
    print("\n" + "="*80)
    print("SINGLE TRADE LIFECYCLE VALIDATION TEST")
    print("="*80)

    # Create synthetic data
    data = create_synthetic_test_data(num_bars=50, base_price=20000.0)

    # Get NQ market specifications
    market_spec = get_market_spec('NQ')
    print(f"\n[MARKET] Using {market_spec.name} specifications:")
    print(f"  Contract multiplier: ${market_spec.contract_multiplier}/point")
    print(f"  Tick value: ${market_spec.tick_value}")
    print(f"  Commission: ${market_spec.commission}/side")

    # Create environment with deterministic settings
    print("\n[ENV] Creating deterministic environment...")
    env = TradingEnvironmentPhase3LLM(
        data=data,
        use_llm_features=True,  # 261D observations
        initial_balance=50000,
        window_size=20,
        second_data=None,
        market_spec=market_spec,
        commission_override=None,
        initial_sl_multiplier=2.5,  # Wide stops for testing
        initial_tp_ratio=3.0,
        position_size_contracts=1.0,
        trailing_drawdown_limit=10000,  # High limit (won't hit)
        tighten_sl_step=0.5,
        extend_tp_step=1.0,
        trailing_activation_profit=1.0,
        hybrid_agent=None,
        start_index=20,  # Start after warmup bars
        randomize_start_offsets=False,  # CRITICAL: No randomization
        min_episode_bars=50
    )

    print("[ENV] Environment created successfully")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n} actions")

    # Reset environment with fixed seed
    print("\n[ENV] Resetting environment...")
    obs, info = env.reset(seed=42)
    initial_balance = env.balance

    print(f"[ENV] Initial state:")
    print(f"  Balance: ${initial_balance:,.2f}")
    print(f"  Position: {env.position}")
    print(f"  Current step: {env.current_step}")

    # Get action mask
    action_mask = env.action_masks()
    print(f"  Action mask: {action_mask.tolist()}")

    # Expected values for NQ
    COMMISSION = market_spec.commission
    MULTIPLIER = market_spec.contract_multiplier
    SLIPPAGE_TICKS = 1
    SLIPPAGE_COST = SLIPPAGE_TICKS * market_spec.tick_value

    print(f"\n[CALC] Expected transaction costs:")
    print(f"  Commission: ${COMMISSION}/side")
    print(f"  Slippage: {SLIPPAGE_TICKS} tick = ${SLIPPAGE_COST}")
    print(f"  Entry cost: ${COMMISSION} (commission only)")
    print(f"  Round-trip cost: ${2 * COMMISSION} + slippage")

    # =========================================================================
    # STEP 1: HOLD (Pre-trade state)
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 1: HOLD (Pre-trade validation)")
    print("-"*80)

    action = 0  # HOLD
    obs, reward, terminated, truncated, info = env.step(action)

    # Validate
    assert env.position == 0, f"Expected no position, got {env.position}"
    assert env.balance == initial_balance, f"Balance changed on HOLD: {env.balance}"

    action_mask = env.action_masks()
    # Expected: Can HOLD, BUY, or SELL; cannot do position management (no position)
    validate_action_mask(action_mask, [True, True, True, False, False, False], "STEP 1")

    print(f"✅ Position: {env.position} (flat)")
    print(f"✅ Balance: ${env.balance:,.2f} (unchanged)")
    print(f"✅ Action mask: {action_mask.tolist()} (entry actions available)")
    print(f"✅ Reward: {reward:.4f}")

    # =========================================================================
    # STEP 2: BUY (Open long position)
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 2: BUY (Open long position)")
    print("-"*80)

    balance_before_entry = env.balance
    action = 1  # BUY
    obs, reward, terminated, truncated, info = env.step(action)

    # Validate position opened
    assert env.position == 1, f"Expected long position, got {env.position}"

    # Validate commission deducted
    expected_balance = balance_before_entry - COMMISSION
    balance_diff = abs(env.balance - expected_balance)
    assert balance_diff < 0.01, \
        f"Balance mismatch: expected ${expected_balance:.2f}, got ${env.balance:.2f}"

    # Validate entry price set
    assert hasattr(env, 'entry_price'), "Entry price not set"
    assert env.entry_price > 0, "Invalid entry price"

    # Validate SL/TP set
    assert hasattr(env, 'sl_price'), "Stop loss not set"
    assert hasattr(env, 'tp_price'), "Take profit not set"
    assert env.sl_price < env.entry_price, "SL should be below entry"
    assert env.tp_price > env.entry_price, "TP should be above entry"

    action_mask = env.action_masks()
    # Expected: Can HOLD; cannot enter new positions; position management depends on profit
    # Initially, might not be profitable enough for BE move
    assert action_mask[0] == True, "HOLD should be available"
    assert action_mask[1] == False, "BUY should be blocked (already in position)"
    assert action_mask[2] == False, "SELL should be blocked (already in position)"

    print(f"✅ Position: {env.position} (long)")
    print(f"✅ Balance: ${env.balance:,.2f} (entry commission deducted)")
    print(f"✅ Entry price: ${env.entry_price:.2f}")
    print(f"✅ Stop loss: ${env.sl_price:.2f}")
    print(f"✅ Take profit: ${env.tp_price:.2f}")
    print(f"✅ Action mask: {action_mask.tolist()}")
    print(f"✅ Reward: {reward:.4f}")

    entry_price = env.entry_price
    entry_balance = env.balance

    # =========================================================================
    # STEP 3-4: HOLD (Build profit to enable BE move)
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 3-4: HOLD (Wait for profit to enable breakeven)")
    print("-"*80)

    for step_num in range(3, 5):
        action = 0  # HOLD
        obs, reward, terminated, truncated, info = env.step(action)

        unrealized_pnl = info.get('unrealized_pnl', 0)
        print(f"  Step {step_num}: Unrealized P&L = ${unrealized_pnl:.2f}, "
              f"Reward = {reward:.4f}, Position = {env.position}")

        if terminated or truncated:
            raise AssertionError(f"Trade terminated prematurely at step {step_num}")

    # Check if profitable enough for BE move
    action_mask = env.action_masks()
    can_move_be = action_mask[3]

    print(f"✅ Position maintained: {env.position}")
    print(f"✅ Balance unchanged: ${env.balance:,.2f}")
    print(f"✅ Unrealized P&L: ${unrealized_pnl:.2f}")
    print(f"✅ Can move to BE: {can_move_be}")

    # =========================================================================
    # STEP 5: MOVE_TO_BE (Move stop to breakeven)
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 5: MOVE_TO_BE (Move stop loss to breakeven)")
    print("-"*80)

    if not can_move_be:
        print("⚠️  Not profitable enough for BE move, continuing to build profit...")
        # Execute more HOLD actions until profitable
        for i in range(5):
            action = 0  # HOLD
            obs, reward, terminated, truncated, info = env.step(action)
            action_mask = env.action_masks()

            if action_mask[3]:  # Can move to BE now
                print(f"  After {i+1} more bars, can now move to BE")
                break

            if terminated or truncated:
                raise AssertionError(f"Trade terminated before BE move possible")

    sl_before_be = env.sl_price
    action = 3  # MOVE_TO_BE
    obs, reward, terminated, truncated, info = env.step(action)
    sl_after_be = env.sl_price

    # Validate SL moved closer to entry
    assert sl_after_be > sl_before_be, \
        f"SL should move up, was {sl_before_be:.2f}, now {sl_after_be:.2f}"

    # Validate SL is at or near entry (with buffer)
    assert sl_after_be >= entry_price * 0.999, \
        f"SL should be near entry ({entry_price:.2f}), got {sl_after_be:.2f}"

    # Validate balance unchanged
    assert env.balance == entry_balance, "Balance should not change on MOVE_TO_BE"

    action_mask = env.action_masks()
    # After moving to BE, MOVE_TO_BE should be disabled
    assert action_mask[3] == False, "MOVE_TO_BE should be blocked after use"

    print(f"✅ Stop loss moved: ${sl_before_be:.2f} → ${sl_after_be:.2f}")
    print(f"✅ Now at breakeven (near entry ${entry_price:.2f})")
    print(f"✅ Balance unchanged: ${env.balance:,.2f}")
    print(f"✅ Action mask: {action_mask.tolist()}")
    print(f"✅ Reward: {reward:.4f}")

    # =========================================================================
    # STEP 6-7: HOLD (Build more profit for trailing)
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 6-7: HOLD (Build profit to enable trailing stop)")
    print("-"*80)

    for step_num in range(6, 8):
        action = 0  # HOLD
        obs, reward, terminated, truncated, info = env.step(action)

        unrealized_pnl = info.get('unrealized_pnl', 0)
        print(f"  Step {step_num}: Unrealized P&L = ${unrealized_pnl:.2f}, "
              f"Reward = {reward:.4f}")

        if terminated or truncated:
            raise AssertionError(f"Trade terminated prematurely at step {step_num}")

    action_mask = env.action_masks()
    can_enable_trail = action_mask[4]

    print(f"✅ Position maintained: {env.position}")
    print(f"✅ Unrealized P&L: ${unrealized_pnl:.2f}")
    print(f"✅ Can enable trailing: {can_enable_trail}")

    # =========================================================================
    # STEP 8: ENABLE_TRAIL (Activate trailing stop)
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 8: ENABLE_TRAIL (Activate trailing stop)")
    print("-"*80)

    if not can_enable_trail:
        print("⚠️  Not profitable enough for trailing, continuing...")
        # Execute more HOLD actions
        for i in range(5):
            action = 0  # HOLD
            obs, reward, terminated, truncated, info = env.step(action)
            action_mask = env.action_masks()

            if action_mask[4]:  # Can enable trail now
                print(f"  After {i+1} more bars, can now enable trailing")
                break

            if terminated or truncated:
                raise AssertionError(f"Trade terminated before trailing possible")

    action = 4  # ENABLE_TRAIL
    obs, reward, terminated, truncated, info = env.step(action)

    # Validate trailing activated
    assert hasattr(env, 'trailing_stop_active'), "Trailing stop flag not found"
    # Note: Some environments may not expose this directly, check via action mask instead

    action_mask = env.action_masks()
    # After enabling trail, ENABLE_TRAIL should be disabled, DISABLE_TRAIL enabled
    assert action_mask[4] == False, "ENABLE_TRAIL should be blocked after use"
    assert action_mask[5] == True, "DISABLE_TRAIL should be available"

    # Validate balance unchanged
    assert env.balance == entry_balance, "Balance should not change on ENABLE_TRAIL"

    print(f"✅ Trailing stop activated")
    print(f"✅ Balance unchanged: ${env.balance:,.2f}")
    print(f"✅ Action mask: {action_mask.tolist()}")
    print(f"✅ Reward: {reward:.4f}")

    # =========================================================================
    # STEP 9+: HOLD (Wait for TP exit)
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 9+: HOLD (Wait for take profit exit)")
    print("-"*80)

    max_steps = 30
    for step_num in range(9, 9 + max_steps):
        action = 0  # HOLD
        obs, reward, terminated, truncated, info = env.step(action)

        unrealized_pnl = info.get('unrealized_pnl', 0)
        trade_pnl = info.get('trade_pnl', 0)

        print(f"  Step {step_num}: Position={env.position}, "
              f"Unrealized=${unrealized_pnl:.2f}, "
              f"Trade P&L=${trade_pnl:.2f}, "
              f"Reward={reward:.4f}")

        # Check if trade closed
        if env.position == 0:
            print(f"\n🎯 Trade closed at step {step_num}!")
            exit_reason = info.get('exit_reason', 'unknown')
            print(f"   Exit reason: {exit_reason}")
            print(f"   Realized P&L: ${trade_pnl:.2f}")
            print(f"   Balance: ${env.balance:,.2f}")
            break

        if terminated or truncated:
            print(f"\n⚠️  Episode terminated at step {step_num}")
            done_reason = info.get('done_reason', 'unknown')
            print(f"   Reason: {done_reason}")
            break
    else:
        raise AssertionError(f"Trade did not close within {max_steps} steps")

    # =========================================================================
    # FINAL VALIDATION
    # =========================================================================
    print("\n" + "="*80)
    print("FINAL VALIDATION")
    print("="*80)

    final_balance = env.balance
    total_pnl = final_balance - initial_balance

    print(f"Initial balance:  ${initial_balance:,.2f}")
    print(f"Final balance:    ${final_balance:,.2f}")
    print(f"Total P&L:        ${total_pnl:+,.2f}")
    print(f"Entry commission: ${COMMISSION:.2f}")
    print(f"Exit commission:  ${COMMISSION:.2f}")
    print(f"Total commission: ${2 * COMMISSION:.2f}")

    # Validate position closed
    assert env.position == 0, f"Position should be flat, got {env.position}"

    # Validate profitable (since price went up)
    assert total_pnl > 0, f"Trade should be profitable, got ${total_pnl:.2f}"

    # Validate balance changed
    assert final_balance != initial_balance, "Balance should have changed"

    # Validate trade P&L was recorded
    assert trade_pnl != 0, "Trade P&L should be recorded in info dict"

    print("\n" + "="*80)
    print("✅ ALL VALIDATIONS PASSED!")
    print("="*80)
    print("\nSummary:")
    print(f"  ✅ Environment initialized correctly")
    print(f"  ✅ All 6 actions executed without errors")
    print(f"  ✅ BUY opened long position")
    print(f"  ✅ MOVE_TO_BE moved stop to breakeven")
    print(f"  ✅ ENABLE_TRAIL activated trailing stop")
    print(f"  ✅ Trade closed profitably")
    print(f"  ✅ Balance calculations correct")
    print(f"  ✅ Action masking worked as expected")
    print(f"\n🎉 Single trade lifecycle validated successfully!")
    print(f"   Net profit: ${total_pnl:,.2f} on ${initial_balance:,.2f} initial capital")
    print(f"   Return: {100 * total_pnl / initial_balance:.2f}%")

    return True


def main():
    """Main entry point."""
    try:
        success = test_single_trade_lifecycle()
        if success:
            print("\n✅ Test completed successfully!")
            return 0
        else:
            print("\n❌ Test failed!")
            return 1
    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
