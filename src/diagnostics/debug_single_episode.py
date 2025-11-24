#!/usr/bin/env python3
"""
Debug Single Episode - Trace Why Balance Drops Without Trades

This script runs a single episode with detailed logging to identify:
1. Why balance is decreasing without trades
2. What actions are being taken
3. Whether action masking is blocking entries
4. Balance changes at each step
"""

import sys
import os
from pathlib import Path

# Add both project root and src to path
project_root = Path(__file__).resolve().parent.parent.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Change to project root for data paths
os.chdir(project_root)

import pandas as pd
import numpy as np

# Import without prefix
from environment_phase3_llm import TradingEnvironmentPhase3LLM
from market_specs import get_market_spec
from feature_engineering import add_market_regime_features

# Load data
print("Loading NQ data...")
data = pd.read_csv('./data/NQ_D1M.csv', index_col=0, parse_dates=True)

# Ensure timezone
if not isinstance(data.index, pd.DatetimeIndex):
    data.index = pd.to_datetime(data.index, utc=True)
if hasattr(data.index, 'tz') and data.index.tz is None:
    data.index = data.index.tz_localize('UTC').tz_convert("America/New_York")
elif hasattr(data.index, 'tz') and str(data.index.tz) != 'America/New_York':
    data.index = data.index.tz_convert("America/New_York")

# Add features
print("Adding features...")
data = add_market_regime_features(data)

# Create environment
print(f"Creating environment (data: {len(data)} rows)...")
market_spec = get_market_spec('NQ')
env = TradingEnvironmentPhase3LLM(
    data=data,
    use_llm_features=True,
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
    min_episode_bars=300
)

# Reset
print("\nResetting environment...")
obs, info = env.reset()
initial_balance = env.balance
print(f"Initial balance: ${initial_balance:,.2f}")
dd_limit = getattr(env, 'trailing_dd_limit', getattr(env, 'trailing_drawdown_limit', 2500))
print(f"Trailing DD limit: ${dd_limit:,.2f}")

# Run episode with detailed logging
print("\n" + "="*80)
print("RUNNING SINGLE EPISODE WITH DETAILED LOGGING")
print("="*80)

done = False
step = 0
max_steps = 100  # Limit for debugging
trades_executed = 0
balance_history = [initial_balance]

while not done and step < max_steps:
    # Get action mask
    try:
        action_mask = env.action_masks()
    except AttributeError:
        action_mask = np.ones(env.action_space.n, dtype=bool)

    # Random valid action
    valid_actions = np.where(action_mask)[0]
    if len(valid_actions) == 0:
        valid_actions = [0]  # Default to HOLD
    action = np.random.choice(valid_actions)

    # Action names
    action_names = ['HOLD', 'BUY', 'SELL', 'MOVE_TO_BE', 'ENABLE_TRAIL', 'DISABLE_TRAIL']
    action_name = action_names[action] if action < len(action_names) else f"ACTION_{action}"

    # Step
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Track balance
    new_balance = env.balance
    balance_change = new_balance - balance_history[-1]
    balance_history.append(new_balance)

    # Track trades
    if 'num_trades' in info:
        if info['num_trades'] > trades_executed:
            trades_executed = info['num_trades']
            print(f"\n  🔔 TRADE EXECUTED! Total trades: {trades_executed}")

    # Log every step
    print(f"\nStep {step+1}:")
    print(f"  Action: {action_name} (mask: {action_mask.tolist()})")
    print(f"  Position: {info.get('position', 'unknown')}")
    print(f"  Balance: ${new_balance:,.2f} (change: ${balance_change:+,.2f})")
    print(f"  Reward: {reward:.4f}")
    print(f"  Trades: {info.get('num_trades', 0)}")
    print(f"  Unrealized P&L: ${info.get('unrealized_pnl', 0):,.2f}")
    if info.get('trade_pnl') not in (None, 0):
        print(f"  Realized trade P&L: ${info['trade_pnl']:,.2f} (exit_reason={info.get('exit_reason')})")

    if balance_change != 0 and info.get('trade_pnl') in (None, 0):
        print(f"  ⚠️  Balance changed WITHOUT new trade!")

    if done:
        print(f"\n  🛑 Episode terminated!")
        print(f"  Reason: {info.get('done_reason', 'unknown')}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")

    step += 1

# Summary
print("\n" + "="*80)
print("EPISODE SUMMARY")
print("="*80)
print(f"Steps: {step}")
print(f"Initial balance: ${initial_balance:,.2f}")
print(f"Final balance: ${env.balance:,.2f}")
print(f"Total P&L: ${env.balance - initial_balance:+,.2f}")
print(f"Trades executed: {trades_executed}")
print(f"Termination reason: {info.get('done_reason', 'unknown')}")

if trades_executed == 0 and env.balance < initial_balance:
    print("\n⚠️  CRITICAL BUG CONFIRMED:")
    print(f"Lost ${initial_balance - env.balance:,.2f} WITHOUT executing ANY trades!")
    print("This should be IMPOSSIBLE - balance should only change on trades + commission.")

print("\n" + "="*80)
print("Balance history (first 20 steps):")
for i, bal in enumerate(balance_history[:20]):
    change = bal - balance_history[i-1] if i > 0 else 0
    print(f"  Step {i}: ${bal:,.2f} ({change:+,.2f})")
