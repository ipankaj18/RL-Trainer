"""
Phase 2 JAX Evaluator
Evaluates a trained JAX model on fresh data (1-min + 1-sec) with real-time logic synchronization.
"""

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from flax.training import checkpoints
import numpy as np
import os
import json
import pandas as pd
import pickle
import argparse
import sys
import time
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.jax_migration.data_loader import MarketData, load_market_data, precompute_time_features
from src.jax_migration.env_phase2_jax import (
    EnvStatePhase2, EnvParamsPhase2,
    batch_reset_phase2, batch_step_phase2, reset_phase2,
    get_observation_phase2, batch_action_masks_phase2
)
from src.jax_migration.train_phase2_jax import (
    TrainState, create_train_state, get_batched_observations_phase2
)
from src.jax_migration.train_ppo_jax_fixed import (
    PPOConfig, normalize_obs, create_normalizer, masked_softmax
)
from src.market_specs import get_market_spec, MARKET_SPECS

class Colors:
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

def _is_windows_unc_path(path: str, *, platform_name: str | None = None) -> bool:
    platform_name = platform_name or os.name
    return platform_name == "nt" and (path.startswith("\\\\") or path.startswith("//"))

def load_checkpoint(checkpoint_path: str, obs_shape: Tuple[int, ...], config: PPOConfig, seed: int = 0):
    """Load JAX model checkpoint using Orbax/Flax.
    
    CRITICAL FIX (2025-12-08): Use target=train_state to restore with proper structure,
    matching how train_phase2_jax.py saves checkpoints.
    
    FIX (2025-12-10): Convert relative paths to absolute to avoid Orbax errors.
    """
    # FIX (2025-12-10): Convert relative paths to absolute
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
        print(f"Converted to absolute path: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    if _is_windows_unc_path(checkpoint_path):
        raise ValueError("Cannot load from UNC path. Please use a local path.")

    # Initialize model structure (this is the template for restoration)
    key = jax.random.key(seed)
    train_state = create_train_state(key, obs_shape, config, num_actions=6)

    # Determine checkpoint directory and prefix
    if os.path.isdir(checkpoint_path):
        ckpt_dir = checkpoint_path
        # Try to infer the prefix from the directory name
        dirname = os.path.basename(checkpoint_path)
        if dirname.startswith("phase2_jax_final_"):
            prefix = "phase2_jax_final_"
        elif dirname.startswith("phase2_jax_"):
            prefix = "phase2_jax_"
        else:
            prefix = ""  # Let Orbax find it
    else:
        ckpt_dir = os.path.dirname(checkpoint_path)
        prefix = os.path.basename(checkpoint_path)

    # CRITICAL FIX: Use target=train_state for templated restoration
    # This ensures params are properly loaded into the TrainState structure
    try:
        restored = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=train_state,  # Use template for proper structure restoration
            prefix=prefix
        )
    except Exception as e:
        print(f"{Colors.YELLOW}[WARN] Templated restore failed: {e}{Colors.RESET}")
        print(f"{Colors.CYAN}Trying dict-based restore as fallback...{Colors.RESET}")
        
        # Fallback: Load as dict and manually reconstruct
        restored = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=None,
            prefix=prefix
        )
        
        if restored is None:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        # Try to extract and restructure params
        if isinstance(restored, dict):
            # Navigate nested structure to find params
            params = restored.get('params', restored)
            
            # Debug: Show structure
            print(f"{Colors.CYAN}[DEBUG] Restored dict keys: {list(restored.keys())}{Colors.RESET}")
            if isinstance(params, dict):
                print(f"{Colors.CYAN}[DEBUG] Params keys: {list(params.keys())}{Colors.RESET}")
            
            # Handle double-nesting: {'params': {'params': {...}}}
            if isinstance(params, dict) and 'params' in params and isinstance(params['params'], dict):
                inner = params['params']
                # Check if inner looks like layer params
                if any(k.startswith('Dense') or k.startswith('LayerNorm') for k in inner.keys()):
                    params = params  # Keep the outer {'params': {...}} structure
                else:
                    params = inner  # Unwrap one level
            
            try:
                restored = train_state.replace(params=params)
                if 'step' in restored:
                    restored = restored.replace(step=restored['step'])
            except Exception as replace_err:
                raise ValueError(f"Failed to reconstruct TrainState from dict: {replace_err}")
        else:
            raise ValueError(f"Unexpected restored type: {type(restored)}")

    if restored is None:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    return restored

def load_normalizer(checkpoint_path: str, obs_shape: Tuple[int, ...]):
    """Load the normalizer pickle associated with the checkpoint.
    
    FIX (2025-12-10): Search both checkpoint dir AND parent dir,
    since normalizers are saved at model root (e.g., models/phase2_jax_nq/)
    not inside checkpoint subfolders (e.g., .../phase2_jax_final_3051/)
    """
    # FIX: Convert relative paths to absolute first
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
    
    ckpt_dir = os.path.dirname(checkpoint_path) if not os.path.isdir(checkpoint_path) else checkpoint_path
    parent_dir = os.path.dirname(ckpt_dir)
    
    # FIX: Search both checkpoint dir and parent dir
    search_dirs = [ckpt_dir, parent_dir]
    
    candidates = []
    for search_dir in search_dirs:
        candidates.extend([
            os.path.join(search_dir, "normalizer_final.pkl"),
            os.path.join(search_dir, "normalizer.pkl")
        ])
        
        # Also check for normalizer_{step}.pkl
        import glob
        pkls = sorted(glob.glob(os.path.join(search_dir, "normalizer_*.pkl")), key=os.path.getmtime, reverse=True)
        candidates.extend(pkls)

    for path in candidates:
        if os.path.exists(path):
            print(f"Loading normalizer from {path}")
            with open(path, "rb") as f:
                normalizer = pickle.load(f)
            
            # FIX (2025-12-10): Validate normalizer shape matches expected obs_shape
            # If shape mismatch (e.g., 231 vs 233 after Apex features added), use identity normalizer
            if hasattr(normalizer, 'mean') and normalizer.mean.shape != obs_shape:
                print(f"{Colors.RED}[ERROR] Normalizer shape mismatch!{Colors.RESET}")
                print(f"  Expected: {obs_shape}, Got: {normalizer.mean.shape}")
                print(f"  This can happen if model was trained with different observation features.")
                print(f"{Colors.YELLOW}  Using identity normalizer instead (no normalization).{Colors.RESET}")
                return create_normalizer(obs_shape)
            
            return normalizer
    
    print(f"{Colors.YELLOW}Warning: No normalizer found in {ckpt_dir} or {parent_dir}. Using identity normalizer.{Colors.RESET}")
    return create_normalizer(obs_shape)

def run_evaluation(
    model_path: str,
    data_path_1m: str,
    data_path_1s: Optional[str] = None,
    num_episodes: int = 1,
    num_envs: int | None = None,
    render: bool = False,
    output_dir: str = "results/evaluation",
    market: str = "NQ",
    mode: str = "deterministic",  # deterministic, stochastic, or argmax
    walkthrough: bool = False,  # NEW: Sequential walkthrough mode
    max_steps: int = 0  # NEW: Max steps (0 = unlimited for walkthrough, auto for episodes)
):
    """Run the evaluation loop.
    
    Args:
        walkthrough: If True, evaluates entire dataset sequentially (ignores num_episodes).
                     Episodes reset at RTH close but continue to next trading day.
        num_envs: Parallel envs for episodes mode. Defaults to min(num_episodes, 50).
        max_steps: Maximum steps to evaluate. 0 = unlimited.
    """
    
    # 1. Load Data
    print(f"{Colors.CYAN}Loading fresh data...{Colors.RESET}")
    try:
        data = load_market_data(data_path_1m, second_data_path=data_path_1s)
        print(f"Loaded {len(data.features)} bars of data.")
    except Exception as e:
        print(f"{Colors.RED}Failed to load data: {e}{Colors.RESET}")
        return

    # 2. Setup Environment with MARKET-SPECIFIC contract values
    # FIX (2025-12-10): Use market_specs to get correct contract size
    # Default was ES (50.0) but NQ is 20.0 - this caused 2.5x larger losses!
    market_spec = get_market_spec(market)
    if market_spec:
        contract_size = market_spec.contract_multiplier
        tick_size = market_spec.tick_size
        commission = market_spec.commission
        slippage_ticks = market_spec.slippage_ticks
        print(f"{Colors.GREEN}Using {market} contract specs: ${contract_size}/point, tick={tick_size}{Colors.RESET}")
    else:
        # Fallback to ES defaults
        contract_size = 50.0
        tick_size = 0.25
        commission = 2.50
        slippage_ticks = 1
        print(f"{Colors.YELLOW}Warning: Unknown market {market}, using ES defaults{Colors.RESET}")
    
    env_params = EnvParamsPhase2(
        initial_balance=50000.0,
        min_episode_bars=5, 
        window_size=20,
        # Market-specific values
        contract_size=contract_size,
        contract_value=contract_size,
        tick_size=tick_size,
        commission=commission,
        slippage_ticks=slippage_ticks,
        # NEW (2025-12-12): Market-specific position sizing
        max_position_size=float(market_spec.max_position_size) if market_spec else 3.0,
    )
    # Determine parallelism vs. stopping criteria
    # - walkthrough: single env, stop by max_steps/data length
    # - episodes: stop after `num_episodes` episode terminations, run across `num_envs` envs in parallel
    if walkthrough:
        parallel_envs = 1
        target_episodes = 0
    else:
        target_episodes = int(num_episodes)
        parallel_envs = int(num_envs) if num_envs is not None else min(target_episodes, 50)
        if parallel_envs <= 0:
            parallel_envs = min(target_episodes, 50)

    # Ensure config matches training assumptions
    config = PPOConfig(
        num_envs=parallel_envs,
        num_steps=100000, # Large enough to cover full dataset if needed
        total_timesteps=100000
    )
    
    # CRITICAL FIX (2025-12-10): obs_shape updated to 233 for drawdown features
    # Observation breakdown (from env_phase2_jax.py get_observation_phase2):
    #   - market_obs: (window=20, 8) = 160
    #   - time_obs: (window=20, 3) = 60
    #   - combined flattened: 220
    #   - position_features: 5 dims
    #   - phase2_features: 5 dims (was 3, added drawdown_ratio + drawdown_room)
    #   - validity_features: 3 dims (can_enter, can_manage, has_position)
    # TOTAL: 220 + 5 + 5 + 3 = 233
    obs_shape = (233,)  # Updated from 231 for Apex drawdown awareness

    # 3. Load Model
    try:
        train_state = load_checkpoint(model_path, obs_shape, config)
        normalizer = load_normalizer(model_path, obs_shape)
    except Exception as e:
        print(f"{Colors.RED}Failed to load model: {e}{Colors.RESET}")
        return

    # 4. Execution Loop
    
    if walkthrough:
        print(f"{Colors.CYAN}Starting WALKTHROUGH evaluation (full dataset)...{Colors.RESET}")
    else:
        print(
            f"{Colors.CYAN}Starting evaluation: {target_episodes} episodes across {parallel_envs} envs...{Colors.RESET}"
        )

    key = jax.random.key(42)
    key, reset_key = jax.random.split(key)
    
    # Initialize environments
    obs, env_states = batch_reset_phase2(
        jax.random.split(reset_key, parallel_envs),
        env_params,
        parallel_envs,
        data
    )

    # FIX (2025-12-11): Improved progress bar and step limits
    import tqdm
    import math
    total_data_bars = len(data.features)
    if walkthrough:
        # Walkthrough: aim to cover all data
        effective_max_steps = max_steps if max_steps > 0 else total_data_bars
        pbar = tqdm.tqdm(total=effective_max_steps, desc="Evaluating bars")
    else:
        # Episodes mode: progress by completed episodes; keep a conservative step cap to avoid infinite loops
        # Estimate based on avg episode length (~400 1-minute bars per RTH session)
        est_episode_len = 500
        effective_max_steps = (
            max_steps if max_steps > 0 else int(math.ceil(target_episodes / parallel_envs) * est_episode_len)
        )
        pbar = tqdm.tqdm(total=target_episodes, desc="Evaluating episodes")
    
    total_steps = 0
    start_time = time.time()
    episodes_completed = 0  # NEW: Track completed episodes
    
    # Accumulate trade info
    all_trade_pnl = []
    
    # FIX (2025-12-08): Action distribution tracking
    ACTION_NAMES = ["HOLD", "BUY", "SELL", "SL→BE", "TRAIL+", "TRAIL-"]
    action_counts = np.zeros(6, dtype=np.int64)
    
    # FIX (2025-12-08): Apex compliance tracking
    max_equity = env_params.initial_balance
    current_drawdown = 0.0
    max_drawdown = 0.0
    min_apex_margin = float('inf')  # NEW: Track minimum distance to trailing floor
    trailing_drawdown_limit = 2500.0  # Apex limit
    apex_violations = 0
    
    # NEW (2025-12-11): Coverage tracking
    unique_step_indices = set()  # Track unique bar indices visited
    
    termination_counts = {
        "dd_intra": 0,
        "dd_eob": 0,
        "time_violation": 0,
        "end_of_data": 0,
        "unknown": 0,
    }

    while True:
        if walkthrough:
            if total_steps >= effective_max_steps:
                break
        else:
            if episodes_completed >= target_episodes:
                break
            if total_steps >= effective_max_steps:
                break
        key, action_key, step_key = jax.random.split(key, 3)
        
        # Observe
        obs_batch = get_batched_observations_phase2(env_states, data, env_params)
        obs_norm = normalize_obs(obs_batch, normalizer)
        masks = batch_action_masks_phase2(env_states, data, env_params)
        
        # Act
        logits, _ = train_state.apply_fn(train_state.params, obs_norm)
        
        # DEBUG: Log mask statistics on first step
        if total_steps == 0:
            mask_sums = masks.sum(axis=0)
            print(
                f"\n{Colors.CYAN}[DEBUG] Initial action mask validity (across {parallel_envs} envs):{Colors.RESET}"
            )
            for i, name in enumerate(ACTION_NAMES):
                print(f"  {name}: {int(mask_sums[i])}/{parallel_envs} valid")
        
        # FIX: Apply masked_softmax with exploration floor to prevent 100% HOLD
        # Mode: deterministic (default), stochastic, or argmax
        if mode == "argmax":
            # Pure argmax for debugging - may produce 100% HOLD
            actions = jnp.argmax(jnp.where(masks, logits, -1e9), axis=-1)
        else:
            # Apply masked_softmax with 3% floor on BUY/SELL actions
            eval_floor = 0.03  # 3% minimum probability for BUY/SELL
            probs = masked_softmax(logits, masks, exploration_floor=eval_floor)
            
            if mode == "stochastic":
                # Sample from probability distribution (matches training)
                actions = jax.random.categorical(action_key, jnp.log(probs + 1e-10))
            else:
                # Deterministic: weighted random selection that respects the 3% floor
                # FIX (2025-12-10): Pure argmax defeats the floor (HOLD always wins at ~85%)
                # Instead, use categorical sampling which respects the floor probabilities
                # This gives BUY/SELL ~3% chance each instead of 0%
                actions = jax.random.categorical(action_key, jnp.log(probs + 1e-10))
        
        # Track action distribution
        actions_np = np.array(actions)
        for a in actions_np:
            if 0 <= a < 6:
                action_counts[a] += 1
        
        # Step
        key_steps = jax.random.split(step_key, parallel_envs)
        next_obs, next_states, rewards, dones, infos = batch_step_phase2(
            key_steps, env_states, actions, env_params, data
        )
        
        # FIX (2025-12-08): Safe infos access with defaults
        trade_pnls = np.array(infos.get('trade_pnl', np.zeros(parallel_envs)))
        pos_closed = np.array(infos.get('position_closed', np.zeros(parallel_envs, dtype=bool)))
        # FIX (2025-12-10): Include forced EOD closes in trade count
        forced_closed = np.array(infos.get('forced_close', np.zeros(parallel_envs, dtype=bool)))
        any_close = pos_closed | forced_closed
        
        for i in range(parallel_envs):
            if any_close[i]:
                pnl = float(trade_pnls[i])
                all_trade_pnl.append(pnl)
                
                # OLD (Incorrect): Aggregated usage across all envs -> huge DD
                # We track max_drawdown correctly at the batch level now
                pass
        
        env_states = next_states
        
        # FIX (2025-12-12): Track actual max drawdown from environment state (BATCH LEVEL)
        if 'trailing_dd' in infos:
             # JAX often returns DeviceArray, convert to numpy
             step_dds = np.array(infos['trailing_dd'])
             if step_dds.size > 0:
                 step_max_dd = float(np.max(step_dds))
                 max_drawdown = max(max_drawdown, step_max_dd)
             
             # Track violations (sum of terminated episodes due to DD)
             if 'dd_violation' in infos:
                 step_violations = np.sum(infos['dd_violation'])
                 apex_violations += int(step_violations)
                 
             # NEW: Track Minimum Margin (Distance to Floor)
             if 'apex_margin' in infos:
                 step_margins = np.array(infos['apex_margin'])
                 if step_margins.size > 0:
                     step_min_margin = float(np.min(step_margins))
                     min_apex_margin = min(min_apex_margin, step_min_margin)
        
        # FIX (2025-12-12): Auto-reset terminated episodes for all modes
        dones_np = np.array(dones)
        if np.any(dones_np):
            dd_viol_arr = np.array(infos.get('dd_violation', np.zeros(parallel_envs, dtype=bool)))
            intra_dd_arr = np.array(infos.get('intra_bar_dd_violation', np.zeros(parallel_envs, dtype=bool)))
            eod_arr = np.array(infos.get('end_of_data', np.zeros(parallel_envs, dtype=bool)))
            time_viol_arr = np.array(infos.get('time_violation', np.zeros(parallel_envs, dtype=bool)))
            step_idx_arr = np.array(infos.get('step_idx', np.zeros(parallel_envs, dtype=np.int32)))

            # Reset terminated envs to continue evaluation
            key, reset_key = jax.random.split(key)
            reset_keys = jax.random.split(reset_key, parallel_envs)
            done_indices = np.where(dones_np)[0]
            for i in done_indices:
                dd_viol = bool(dd_viol_arr[i])
                intra_dd = bool(intra_dd_arr[i])
                eod = bool(eod_arr[i])
                time_viol = bool(time_viol_arr[i])
                step_idx = int(step_idx_arr[i])

                if dd_viol:
                    if intra_dd:
                        termination_counts["dd_intra"] += 1
                        reason = "INTRA-BAR DD VIOLATION (worst_case < trailing_dd)"
                    else:
                        termination_counts["dd_eob"] += 1
                        reason = "END-OF-BAR DD VIOLATION (portfolio < trailing_dd)"
                elif eod:
                    termination_counts["end_of_data"] += 1
                    reason = f"END OF DATA (step_idx={step_idx} >= data_len-2)"
                elif time_viol:
                    termination_counts["time_violation"] += 1
                    reason = "RTH CLOSE (time_violation)"
                else:
                    termination_counts["unknown"] += 1
                    reason = f"UNKNOWN (all flags False, step_idx={step_idx})"

                if total_steps < 100:
                    print(f"[DEBUG] Env {i} terminated at step {total_steps}: {reason}")

                if walkthrough:
                    episodes_completed += 1
                else:
                    if episodes_completed >= target_episodes:
                        continue
                    episodes_completed += 1
                    pbar.update(1)

                # Reset if we still need more episodes (or if we're in walkthrough mode)
                if walkthrough or episodes_completed < target_episodes:
                    single_obs, single_state = reset_phase2(reset_keys[i], env_params, data)
                    env_states = EnvStatePhase2(
                        step_idx=env_states.step_idx.at[i].set(single_state.step_idx),
                        position=env_states.position.at[i].set(single_state.position),
                        entry_price=env_states.entry_price.at[i].set(single_state.entry_price),
                        sl_price=env_states.sl_price.at[i].set(single_state.sl_price),
                        tp_price=env_states.tp_price.at[i].set(single_state.tp_price),
                        position_entry_step=env_states.position_entry_step.at[i].set(single_state.position_entry_step),
                        balance=env_states.balance.at[i].set(single_state.balance),
                        highest_balance=env_states.highest_balance.at[i].set(single_state.highest_balance),
                        trailing_dd_level=env_states.trailing_dd_level.at[i].set(single_state.trailing_dd_level),
                        num_trades=env_states.num_trades.at[i].set(single_state.num_trades),
                        winning_trades=env_states.winning_trades.at[i].set(single_state.winning_trades),
                        losing_trades=env_states.losing_trades.at[i].set(single_state.losing_trades),
                        total_pnl=env_states.total_pnl.at[i].set(single_state.total_pnl),
                        episode_start_idx=env_states.episode_start_idx.at[i].set(single_state.episode_start_idx),
                        trailing_stop_active=env_states.trailing_stop_active.at[i].set(single_state.trailing_stop_active),
                        highest_profit_point=env_states.highest_profit_point.at[i].set(single_state.highest_profit_point),
                        be_move_count=env_states.be_move_count.at[i].set(single_state.be_move_count),
                        original_sl_price=env_states.original_sl_price.at[i].set(single_state.original_sl_price),
                        trail_activation_price=env_states.trail_activation_price.at[i].set(single_state.trail_activation_price),
                        position_size=env_states.position_size.at[i].set(single_state.position_size),
                    )

        # Track unique bar indices for coverage
        for i in range(parallel_envs):
            unique_step_indices.add(int(env_states.step_idx[i]))
        
        total_steps += 1
        if walkthrough:
            pbar.update(1)
        
    pbar.close()
    
    # Calculate statistics
    all_trade_pnl = np.array(all_trade_pnl)
    total_trades = len(all_trade_pnl)
    total_pnl = np.sum(all_trade_pnl) if total_trades > 0 else 0.0
    wins = all_trade_pnl[all_trade_pnl > 0]
    losses = all_trade_pnl[all_trade_pnl <= 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
    
    # Calculate action percentages
    total_actions = action_counts.sum()
    action_pcts = (action_counts / total_actions * 100) if total_actions > 0 else np.zeros(6)
    
    # NEW (2025-12-11): Calculate coverage metrics
    coverage_pct = (len(unique_step_indices) / total_data_bars) * 100 if total_data_bars > 0 else 0
    elapsed_time = time.time() - start_time
    steps_per_sec = total_steps / elapsed_time if elapsed_time > 0 else 0
    env_steps = total_steps * parallel_envs
    env_steps_per_sec = env_steps / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n{Colors.BOLD}═══ Evaluation Results ═══{Colors.RESET}")
    print(f"\n{Colors.BOLD}Coverage:{Colors.RESET}")
    print(f"  Vector Steps: {total_steps:,}")
    print(f"  Env Steps: {env_steps:,} ({parallel_envs} envs)")
    print(f"  Unique Bars: {len(unique_step_indices):,} / {total_data_bars:,} ({coverage_pct:.1f}%)")
    print(f"  Episodes Completed: {episodes_completed if walkthrough else min(episodes_completed, target_episodes)}")
    print(f"  Speed: {steps_per_sec:.1f} vec steps/sec ({env_steps_per_sec:.1f} env steps/sec)")
    print(f"  Elapsed: {elapsed_time:.1f}s")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate*100:.1f}%")
    print(f"Total PnL: ${total_pnl:.2f}")
    if total_trades > 0:
        print(f"Avg Win: ${avg_win:.2f}")
        print(f"Avg Loss: ${avg_loss:.2f}")
    
    # FIX (2025-12-08): Action distribution display
    print(f"\n{Colors.BOLD}Action Distribution:{Colors.RESET}")
    for i, name in enumerate(ACTION_NAMES):
        print(f"  {name}: {action_pcts[i]:.1f}% ({action_counts[i]:,})")
    
    # FIX (2025-12-08): Apex compliance display
    print(f"\n{Colors.BOLD}Apex Compliance:{Colors.RESET}")
    print(f"  Max Drawdown (Peak-to-Valley): ${max_drawdown:.2f} (Limit: ${trailing_drawdown_limit:.2f})")
    print(f"  Min Margin (Dist to Floor): ${min_apex_margin:.2f}")
    
    # FIX (2025-12-12): Pass based strictly on violations (Env knows Capped Rules)
    apex_pass = apex_violations == 0
    apex_status = f"{Colors.GREEN}PASS{Colors.RESET}" if apex_pass else f"{Colors.RED}FAIL{Colors.RESET}"
    print(f"  Status: {apex_status}")
    if apex_violations > 0:
        print(f"  {Colors.YELLOW}Violations: {apex_violations} trades exceeded drawdown limit{Colors.RESET}")

    if 'termination_counts' in locals():
        print(f"\n{Colors.BOLD}Termination Reasons:{Colors.RESET}")
        print(f"  INTRA-BAR DD: {termination_counts['dd_intra']}")
        print(f"  END-OF-BAR DD: {termination_counts['dd_eob']}")
        print(f"  RTH CLOSE: {termination_counts['time_violation']}")
        print(f"  END OF DATA: {termination_counts['end_of_data']}")
        if termination_counts['unknown'] > 0:
            print(f"  UNKNOWN: {termination_counts['unknown']}")
    
    # Save results with all new metrics
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "model": model_path,
        "market": market,
        "total_trades": int(total_trades),
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        # New: Action distribution
        "action_distribution": {name: float(pct) for name, pct in zip(ACTION_NAMES, action_pcts)},
        "action_counts": {name: int(cnt) for name, cnt in zip(ACTION_NAMES, action_counts)},
        # New: Apex compliance
        "apex_compliance": {
            "max_drawdown": float(max_drawdown),
            "trailing_dd_limit": float(trailing_drawdown_limit),
            "passed": bool(apex_pass),
            "violations": int(apex_violations)
        },
        # New: Equity curve (for plotting)
        # FIX (2025-12-12): Use total PnL since equity_curve list was removed
        "final_equity": float(env_params.initial_balance + total_pnl),
        # NEW (2025-12-11): Coverage metrics
        "coverage": {
            # Backwards-compatible alias (older dashboards/scripts may read total_steps)
            "total_steps": int(total_steps),
            "vector_steps": int(total_steps),
            "env_steps": int(env_steps),
            "unique_bars": len(unique_step_indices),
            "total_bars": int(total_data_bars),
            "coverage_pct": float(coverage_pct),
            "episodes_completed": int(episodes_completed if walkthrough else min(episodes_completed, target_episodes)),
            "steps_per_sec": float(steps_per_sec),
            "env_steps_per_sec": float(env_steps_per_sec),
            "elapsed_seconds": float(elapsed_time)
        },
        "termination_reasons": termination_counts if 'termination_counts' in locals() else {},
        "evaluation_mode": "walkthrough" if walkthrough else f"{num_episodes}_episodes"
    }
    
    with open(os.path.join(output_dir, "eval_summary.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n{Colors.GREEN}Results saved to {output_dir}{Colors.RESET}")


if __name__ == "__main__":
    print(f"{Colors.CYAN}JAX Devices: {jax.devices()}{Colors.RESET}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data-path-1m", type=str, default=None, help="Path to 1-minute data CSV (auto-detects test data if not provided)")
    parser.add_argument("--data-path-1s", type=str, help="Path to 1-second data CSV")
    parser.add_argument("--episodes", type=int, default=50,
                       help="Number of episodes (default: 50 for better coverage)")
    parser.add_argument("--num-envs", type=int, default=None,
                       help="Parallel envs for episodes mode (default: min(episodes, 50))")
    parser.add_argument("--market", type=str, default="NQ")
    parser.add_argument("--mode", type=str, default="deterministic",
                       choices=["deterministic", "stochastic", "argmax"],
                       help="Action selection mode (default: deterministic with 3%% floor)")
    parser.add_argument("--walkthrough", action="store_true",
                       help="Sequential walkthrough mode - evaluates entire dataset")
    parser.add_argument("--max-steps", type=int, default=0,
                       help="Maximum steps to evaluate (0 = auto based on data/episodes)")
    
    args = parser.parse_args()
    
    # TRAIN/TEST SPLIT: Auto-detect test data if not provided
    if args.data_path_1m is None:
        import glob
        
        # Prefer test-specific file (unseen evaluation data)
        test_file = f"data/{args.market}_D1M_test.csv"
        if os.path.exists(test_file):
            args.data_path_1m = test_file
            print(f"{Colors.GREEN}[AUTO-DETECT] Using TEST data (20% unseen): {test_file}{Colors.RESET}")
        else:
            # Fall back to full market data
            market_file = f"data/{args.market}_D1M.csv"
            if os.path.exists(market_file):
                args.data_path_1m = market_file
                print(f"{Colors.YELLOW}[AUTO-DETECT] Using full data (may include training data): {market_file}{Colors.RESET}")
            else:
                # Search for any test or full data
                test_candidates = sorted(glob.glob("data/*_D1M_test.csv"))
                if test_candidates:
                    args.data_path_1m = test_candidates[0]
                    print(f"{Colors.GREEN}[AUTO-DETECT] Using found test data: {args.data_path_1m}{Colors.RESET}")
                else:
                    full_candidates = sorted(glob.glob("data/*_D1M.csv"))
                    if full_candidates:
                        args.data_path_1m = full_candidates[0]
                        print(f"{Colors.YELLOW}[AUTO-DETECT] Using found data: {args.data_path_1m}{Colors.RESET}")
                    else:
                        raise FileNotFoundError(f"No data found. Expected: {test_file} or {market_file}")
        
        # Also auto-detect second data if minute test data was found
        if args.data_path_1s is None and "_test.csv" in args.data_path_1m:
            second_test = args.data_path_1m.replace("_D1M_test.csv", "_D1S_test.csv")
            if os.path.exists(second_test):
                args.data_path_1s = second_test
                print(f"{Colors.CYAN}[AUTO-DETECT] Also using second test data: {second_test}{Colors.RESET}")
    
    print(f"\n{Colors.CYAN}Evaluation mode: {args.mode}{Colors.RESET}")
    if args.mode == "deterministic":
        print(f"{Colors.GREEN}Using 3% probability floor on BUY/SELL to prevent HOLD trap{Colors.RESET}")
    elif args.mode == "stochastic":
        print(f"{Colors.YELLOW}Sampling from probability distribution (results will vary){Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}Pure argmax mode - may produce 100% HOLD{Colors.RESET}")
    
    # Show evaluation mode
    if args.walkthrough:
        print(f"{Colors.CYAN}Mode: WALKTHROUGH (full dataset sequential evaluation){Colors.RESET}")
    else:
        print(f"{Colors.CYAN}Mode: {args.episodes} EPISODES (random RTH starts){Colors.RESET}")
    
    run_evaluation(
        model_path=args.model_path,
        data_path_1m=args.data_path_1m,
        data_path_1s=args.data_path_1s,
        num_episodes=args.episodes,
        num_envs=args.num_envs,
        market=args.market,
        mode=args.mode,
        walkthrough=args.walkthrough,
        max_steps=args.max_steps
    )

