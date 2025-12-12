"""
Debug Evaluation Script - Phase 2 JAX
Runs a single episode step-by-step with detailed logging to diagnose early termination.

Usage:
    python src/jax_migration/debug_evaluation.py --data-path data/NQ_D1M_test.csv --model-path models/phase2_jax_nq/phase2_jax_final_3051

This script will print detailed state after each step to identify:
1. Why episodes terminate early
2. What triggers dd_violation vs end_of_data vs time_violation
3. The exact state of the environment at termination
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.jax_migration.data_loader import MarketData, load_market_data
from src.jax_migration.env_phase2_jax import (
    EnvStatePhase2, EnvParamsPhase2,
    reset_phase2, step_phase2, action_masks_phase2,
    ACTION_HOLD, ACTION_BUY, ACTION_SELL, ACTION_MOVE_SL_TO_BE, ACTION_ENABLE_TRAIL, ACTION_DISABLE_TRAIL
)
from src.jax_migration.train_phase2_jax import create_train_state
from src.jax_migration.train_ppo_jax_fixed import PPOConfig, normalize_obs, create_normalizer, masked_softmax
from src.jax_migration.evaluate_phase2_jax import load_checkpoint, load_normalizer
from src.market_specs import get_market_spec


ACTION_NAMES = ["HOLD", "BUY", "SELL", "SL→BE", "TRAIL+", "TRAIL-"]

class Colors:
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"


def run_debug_episode(
    model_path: str,
    data_path: str,
    data_path_1s: str = None,
    max_steps: int = 50,
    seed: int = 42,
    market: str = "NQ"
):
    """Run a single episode with detailed step-by-step debugging."""
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}DEBUG EVALUATION - Step-by-Step Analysis{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*80}{Colors.RESET}\n")
    
    # 1. Load Data
    print(f"{Colors.YELLOW}[1] Loading data...{Colors.RESET}")
    data = load_market_data(data_path, second_data_path=data_path_1s)
    print(f"    Total bars: {len(data.features)}")
    print(f"    RTH indices: {len(data.rth_indices)}")
    print(f"    Min RTH index: {int(data.rth_indices.min())}")
    print(f"    Max RTH index: {int(data.rth_indices.max())}")
    print(f"    Data length: {data.features.shape[0]}")
    print(f"    Max start allowed: {data.features.shape[0] - 400} (with 400-bar buffer)")
    
    # 2. Load Model
    print(f"\n{Colors.YELLOW}[2] Loading model...{Colors.RESET}")
    obs_shape = (233,)
    config = PPOConfig(num_envs=1, num_steps=10000, total_timesteps=10000)
    
    try:
        train_state = load_checkpoint(model_path, obs_shape, config, seed=0)
        normalizer = load_normalizer(model_path, obs_shape)
        print(f"    Model loaded successfully")
    except Exception as e:
        print(f"{Colors.RED}    Failed to load model: {e}{Colors.RESET}")
        return
    
    # 3. Setup Environment with MARKET-SPECIFIC contract values
    print(f"\n{Colors.YELLOW}[3] Setting up environment for {market}...{Colors.RESET}")
    
    # Get market specs
    market_spec = get_market_spec(market)
    if market_spec:
        contract_size = market_spec.contract_multiplier
        tick_size = market_spec.tick_size
        commission = market_spec.commission
        slippage_ticks = market_spec.slippage_ticks
        print(f"    {Colors.GREEN}Using {market} contract specs: ${contract_size}/point{Colors.RESET}")
    else:
        contract_size = 50.0
        tick_size = 0.25
        commission = 2.50
        slippage_ticks = 1
        print(f"    {Colors.YELLOW}Warning: Unknown market {market}, using ES defaults{Colors.RESET}")
    
    env_params = EnvParamsPhase2(
        initial_balance=50000.0,
        min_episode_bars=5,
        window_size=20,
        contract_size=contract_size,
        contract_value=contract_size,
        tick_size=tick_size,
        commission=commission,
        slippage_ticks=slippage_ticks,
    )
    print(f"    Initial balance: ${env_params.initial_balance:,.2f}")
    print(f"    Contract size: ${contract_size}/point")
    print(f"    Trailing DD limit: ${env_params.trailing_dd_limit:,.2f}")
    print(f"    Expected trailing_dd_level: ${env_params.initial_balance - env_params.trailing_dd_limit:,.2f}")
    
    # 4. Reset Episode
    print(f"\n{Colors.YELLOW}[4] Resetting episode (seed={seed})...{Colors.RESET}")
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    
    obs, state = reset_phase2(reset_key, env_params, data)
    
    print(f"    Episode start index: {int(state.episode_start_idx)}")
    print(f"    Initial step_idx: {int(state.step_idx)}")
    print(f"    Bars until end of data: {data.features.shape[0] - int(state.step_idx)}")
    
    # Get the timestamp for this index
    start_hour = float(data.timestamps_hour[int(state.episode_start_idx)])
    print(f"    Start time (decimal hour): {start_hour:.2f}")
    print(f"    Start time (HH:MM): {int(start_hour)}:{int((start_hour % 1) * 60):02d}")
    
    # 5. Run Episode Step-by-Step
    print(f"\n{Colors.YELLOW}[5] Running episode step-by-step (max {max_steps} steps)...{Colors.RESET}")
    print(f"\n{Colors.BOLD}{'='*80}{Colors.RESET}")
    
    step = 0
    done = False
    total_reward = 0.0
    
    while not done and step < max_steps:
        key, action_key, step_key = jax.random.split(key, 3)
        
        # Get observation and mask
        mask = action_masks_phase2(state, data, env_params)
        obs_norm = normalize_obs(obs.reshape(1, -1), normalizer)
        
        # Get action from model
        logits, _ = train_state.apply_fn(train_state.params, obs_norm)
        probs = masked_softmax(logits, mask.reshape(1, -1), exploration_floor=0.03)
        action = jax.random.categorical(action_key, jnp.log(probs + 1e-10))[0]
        
        # Take step
        next_obs, next_state, reward, done, info = step_phase2(step_key, state, action, env_params, data)
        total_reward += float(reward)
        
        # Print step info
        print(f"\n{Colors.BOLD}Step {step}:{Colors.RESET}")
        print(f"  {Colors.CYAN}Action:{Colors.RESET} {ACTION_NAMES[int(action)]}")
        print(f"  {Colors.CYAN}Step Index:{Colors.RESET} {int(info['step_idx'])}")
        
        # Position info
        position = int(info.get('position', 0))
        pos_str = {0: "FLAT", 1: "LONG", -1: "SHORT"}.get(position, str(position))
        print(f"  {Colors.CYAN}Position:{Colors.RESET} {pos_str}")
        
        if position != 0:
            print(f"    Entry Price: ${float(info.get('entry_price', 0)):.2f}")
            print(f"    Current Price: ${float(info.get('current_price', 0)):.2f}")
        
        # Financial state
        print(f"  {Colors.CYAN}Financials:{Colors.RESET}")
        print(f"    Balance: ${float(info.get('final_balance', 0)):,.2f}")
        print(f"    Portfolio Value: ${float(info.get('portfolio_value', 0)):,.2f}")
        print(f"    Trailing DD Level: ${float(info.get('trailing_dd_level', 47500)):,.2f}")
        print(f"    Worst Case Equity: ${float(info.get('worst_case_equity', 0)):,.2f}")
        
        # Reward
        print(f"  {Colors.CYAN}Reward:{Colors.RESET} {float(reward):.4f}")
        
        # Termination flags
        dd_viol = bool(info.get('dd_violation', False))
        intra_dd = bool(info.get('intra_bar_dd_violation', False))
        eod = bool(info.get('end_of_data', False))
        time_viol = bool(info.get('time_violation', False))
        
        print(f"  {Colors.CYAN}Termination Flags:{Colors.RESET}")
        print(f"    dd_violation: {Colors.RED if dd_viol else Colors.GREEN}{dd_viol}{Colors.RESET}")
        print(f"    intra_bar_dd_violation: {Colors.RED if intra_dd else Colors.GREEN}{intra_dd}{Colors.RESET}")
        print(f"    end_of_data: {Colors.RED if eod else Colors.GREEN}{eod}{Colors.RESET}")
        print(f"    time_violation: {Colors.RED if time_viol else Colors.GREEN}{time_viol}{Colors.RESET}")
        print(f"    done: {Colors.RED if done else Colors.GREEN}{done}{Colors.RESET}")
        
        # If done, explain why
        if done:
            print(f"\n{Colors.BOLD}{Colors.RED}{'='*40}{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.RED}EPISODE TERMINATED!{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.RED}{'='*40}{Colors.RESET}")
            
            reasons = []
            if dd_viol:
                reasons.append("DD_VIOLATION")
                if intra_dd:
                    print(f"{Colors.YELLOW}  Reason: INTRA-BAR DRAWDOWN VIOLATION{Colors.RESET}")
                    print(f"  Explanation: The worst-case equity (${float(info.get('worst_case_equity', 0)):,.2f})")
                    print(f"               dropped below trailing DD level (${float(info.get('trailing_dd_level', 47500)):,.2f})")
                else:
                    print(f"{Colors.YELLOW}  Reason: END-OF-BAR DRAWDOWN VIOLATION{Colors.RESET}")
                    print(f"  Explanation: Portfolio value (${float(info.get('portfolio_value', 0)):,.2f})")
                    print(f"               dropped below trailing DD level (${float(info.get('trailing_dd_level', 47500)):,.2f})")
            if eod:
                reasons.append("END_OF_DATA")
                print(f"{Colors.YELLOW}  Reason: END OF DATA{Colors.RESET}")
                print(f"  Explanation: step_idx ({int(info['step_idx'])}) >= data_length - 2 ({data.features.shape[0] - 2})")
            if time_viol:
                reasons.append("TIME_VIOLATION")
                print(f"{Colors.YELLOW}  Reason: RTH CLOSE (TIME VIOLATION){Colors.RESET}")
                print(f"  Explanation: Position was open when RTH closed (past 4:59 PM ET)")
            
            if not reasons:
                print(f"{Colors.YELLOW}  Reason: UNKNOWN{Colors.RESET}")
                print(f"  This shouldn't happen - done=True but no termination flag is set!")
        
        # Update state
        obs = next_obs
        state = next_state
        step += 1
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}EPISODE SUMMARY{Colors.RESET}")
    print(f"{'='*80}")
    print(f"  Total Steps: {step}")
    print(f"  Total Reward: {total_reward:.4f}")
    print(f"  Final Balance: ${float(state.balance):,.2f}")
    print(f"  Terminated: {'Yes' if done else 'No (max steps reached)'}")
    
    return


def main():
    parser = argparse.ArgumentParser(description="Debug Phase 2 JAX Evaluation")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-path", type=str, required=True, help="Path to 1-minute data CSV")
    parser.add_argument("--data-path-1s", type=str, default=None, help="Path to 1-second data CSV")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--market", type=str, default="NQ", help="Market symbol for contract specs")
    
    args = parser.parse_args()
    
    run_debug_episode(
        model_path=args.model_path,
        data_path=args.data_path,
        data_path_1s=args.data_path_1s,
        max_steps=args.max_steps,
        seed=args.seed,
        market=args.market
    )


if __name__ == "__main__":
    main()
