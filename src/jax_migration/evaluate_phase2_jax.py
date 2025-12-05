"""
Phase 2 JAX Evaluator
Loads a trained model and evaluates it on the environment.
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
import tempfile
from typing import Dict, Any, Tuple

# Relative imports
from .data_loader import MarketData
from .env_phase2_jax import (
    EnvStatePhase2, EnvParamsPhase2,
    batch_reset_phase2, batch_step_phase2,
    get_observation_phase2, batch_action_masks_phase2
)
from .train_phase2_jax import (
    TrainState, create_train_state, get_batched_observations_phase2
)
from .train_ppo_jax_fixed import (
    PPOConfig, normalize_obs, sample_action, create_normalizer
)


def _is_windows_unc_path(path: str, *, platform_name: str | None = None) -> bool:
    """
    Return True if the given path is a UNC-style path while running on Windows.
    """
    platform_name = platform_name or os.name
    return platform_name == "nt" and (path.startswith("\\\\") or path.startswith("//"))


def _assert_checkpoint_path_safe_for_loading(checkpoint_path: str, *, platform_name: str | None = None) -> None:
    """
    Raise a descriptive error when attempting to load from a Windows UNC path (Orbax limitation).
    """
    if _is_windows_unc_path(checkpoint_path, platform_name=platform_name):
        raise ValueError(
            "Checkpoint path resolves to a UNC share. Orbax/TensorStore cannot read UNC paths on Windows. "
            "Run this script inside WSL (bash) so paths look like /home/... or provide a Windows-native "
            "checkpoint directory (e.g., C:\\Users\\<user>\\AppData\\Local\\Temp)."
        )


def _resolve_checkpoint_dir(preferred_dir: str | None = None, *, platform_name: str | None = None) -> Tuple[str, bool]:
    """
    Choose a safe checkpoint directory for saving temporary artifacts.

    Returns (path, cleanup_flag). When a temp dir is created (or a UNC path is avoided),
    cleanup_flag is True so callers can delete it after use.
    """
    platform_name = platform_name or os.name
    cleanup_needed = False

    if preferred_dir:
        abs_dir = os.path.abspath(preferred_dir)
        if not _is_windows_unc_path(abs_dir, platform_name=platform_name):
            return abs_dir, cleanup_needed
        print(f"[WARN] Requested checkpoint dir '{abs_dir}' is a UNC path; using a local temp directory instead.")
        cleanup_needed = True

    temp_dir = tempfile.mkdtemp(prefix="jax_ckpt_eval_")
    return temp_dir, cleanup_needed or True


def evaluate_model(
    checkpoint_path: str,
    data: MarketData,
    num_episodes: int = 100,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate a trained model.
    """
    print(f"Evaluating model from {checkpoint_path}...")

    # Guard against UNC paths when running from Windows/PowerShell
    _assert_checkpoint_path_safe_for_loading(checkpoint_path)
    
    # Setup
    key = jax.random.key(seed)
    key, init_key, reset_key = jax.random.split(key, 3)
    
    env_params = EnvParamsPhase2()
    config = PPOConfig(num_envs=num_episodes, num_steps=1000, total_timesteps=1000) # num_envs = num_episodes for parallel eval
    
    obs_shape = (env_params.window_size * env_params.num_features + 8,)
    
    # Load checkpoint
    # Try loading with target=None to see what we get, then reconstruct if needed
    if os.path.isdir(checkpoint_path):
        restored = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_path, 
            target=None,
            prefix="phase2_jax_"
        )
    else:
        restored = checkpoints.restore_checkpoint(
            ckpt_dir=os.path.dirname(checkpoint_path),
            target=None,
            prefix=os.path.basename(checkpoint_path)
        )
    
    if restored is None:
        raise ValueError(f"No checkpoint found at {checkpoint_path}")
        
    # If restored is a dict (raw state dict), we need to put it into TrainState
    if isinstance(restored, dict):
        print("Restored raw dictionary. Reconstructing TrainState...")
        # Assuming restored has 'params', 'opt_state', 'step'
        # We can use train_state.replace
        # But we need to be careful about structure.
        # If we saved TrainState, the dict should match TrainState fields.
        try:
            train_state = train_state.replace(
                params=restored['params'],
                opt_state=restored['opt_state'],
                step=restored['step']
            )
        except Exception as e:
            print(f"Failed to reconstruct TrainState: {e}")
            print(f"Restored keys: {restored.keys()}")
            raise
    else:
        # It might be a TrainState object if Orbax restored it as such (unlikely with target=None)
        train_state = restored
    
    print("Model loaded successfully.")
    
    # Load normalizer from pickle
    # We assume it's in the same directory with name normalizer_{step}.pkl or normalizer_final.pkl
    # We need to find the matching step.
    # restore_checkpoint usually returns the step number if we ask? No.
    # But we can look for the file.
    
    # Heuristic: Look for normalizer_final.pkl first, then try to match step if possible.
    # Since we don't know the exact step easily from restored state (unless we check step),
    # let's try to find the latest normalizer pickle in the dir.
    
    ckpt_dir = os.path.dirname(checkpoint_path) if not os.path.isdir(checkpoint_path) else checkpoint_path
    
    normalizer_path = os.path.join(ckpt_dir, "normalizer_final.pkl")
    if not os.path.exists(normalizer_path):
        # Try finding any normalizer pickle
        import glob
        pkls = glob.glob(os.path.join(ckpt_dir, "normalizer_*.pkl"))
        if pkls:
            # Sort by modification time or name? Name might be normalizer_100.pkl
            # Let's pick the one with highest number
            try:
                pkls.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else -1)
                normalizer_path = pkls[-1]
            except:
                normalizer_path = pkls[-1] # Fallback
    
    if os.path.exists(normalizer_path):
        print(f"Loading normalizer from {normalizer_path}...")
        with open(normalizer_path, "rb") as f:
            normalizer = pickle.load(f)
    else:
        print("[WARN] Normalizer not found. Using identity normalization.")
        normalizer = create_normalizer(obs_shape)
    
    # Run Evaluation
    # We run 'num_episodes' environments in parallel until they are all done.
    
    obs, env_states = batch_reset_phase2(reset_key, env_params, num_episodes, data)
    
    # Track stats
    active_mask = jnp.ones(num_episodes, dtype=jnp.bool_)
    episode_returns = jnp.zeros(num_episodes)
    episode_lengths = jnp.zeros(num_episodes)
    
    # Trade stats tracking
    total_trades = jnp.zeros(num_episodes)
    winning_trades = jnp.zeros(num_episodes)
    gross_profit = jnp.zeros(num_episodes)
    gross_loss = jnp.zeros(num_episodes)
    max_drawdown = jnp.zeros(num_episodes)
    
    # Scan loop
    def eval_step(carry, _):
        env_states, active_mask, episode_returns, episode_lengths, total_trades, winning_trades, gross_profit, gross_loss, max_drawdown, key = carry
        key, key_action, key_step = jax.random.split(key, 3)
        
        obs_batch = get_batched_observations_phase2(env_states, data, env_params)
        obs_norm = normalize_obs(obs_batch, normalizer)
        masks = batch_action_masks_phase2(env_states, data, env_params)
        
        logits, _ = train_state.apply_fn(train_state.params, obs_norm)
        
        # Deterministic action for evaluation (argmax)
        actions = jnp.argmax(jnp.where(masks, logits, -1e9), axis=-1)
        
        key_steps = jax.random.split(key_step, num_episodes)
        next_obs, next_states, rewards, dones, infos = batch_step_phase2(
            key_steps, env_states, actions, env_params, data
        )
        
        # Update stats only for active episodes
        episode_returns += rewards * active_mask
        episode_lengths += 1 * active_mask
        
        # Extract trade info from info dict
        # info['trade_pnl'] and info['position_closed'] are arrays of shape (num_episodes,)
        trade_pnl = infos['trade_pnl']
        position_closed = infos['position_closed']
        
        # Update trade stats
        new_trades = position_closed & active_mask
        total_trades += new_trades
        winning_trades += (trade_pnl > 0) & new_trades
        
        gross_profit += jnp.where((trade_pnl > 0) & new_trades, trade_pnl, 0.0)
        gross_loss += jnp.where((trade_pnl <= 0) & new_trades, trade_pnl, 0.0)
        
        # Calculate Drawdown for this step
        # DD = (Highest Balance - Current Balance) / Highest Balance
        # We can get this from state
        current_dd = (next_states.highest_balance - next_states.balance) / next_states.highest_balance
        max_drawdown = jnp.maximum(max_drawdown, current_dd * active_mask)
        
        new_active_mask = active_mask & (~dones)
        
        return (next_states, new_active_mask, episode_returns, episode_lengths, total_trades, winning_trades, gross_profit, gross_loss, max_drawdown, key), dones

    # Run for max steps (e.g. 1000)
    max_steps = 1000
    final_carry, dones_history = lax.scan(
        eval_step,
        (env_states, active_mask, episode_returns, episode_lengths, total_trades, winning_trades, gross_profit, gross_loss, max_drawdown, key),
        None,
        max_steps
    )
    
    # Unpack results
    final_returns = final_carry[2]
    final_total_trades = final_carry[4]
    final_winning_trades = final_carry[5]
    final_gross_profit = final_carry[6]
    final_gross_loss = final_carry[7]
    final_max_dd = final_carry[8]
    
    # Aggregate metrics
    mean_return = float(jnp.mean(final_returns))
    total_trades_sum = float(jnp.sum(final_total_trades))
    total_wins_sum = float(jnp.sum(final_winning_trades))
    total_gross_profit = float(jnp.sum(final_gross_profit))
    total_gross_loss = float(jnp.sum(final_gross_loss))
    
    win_rate = total_wins_sum / total_trades_sum if total_trades_sum > 0 else 0.0
    profit_factor = total_gross_profit / abs(total_gross_loss) if abs(total_gross_loss) > 0 else 0.0
    mean_max_dd = float(jnp.mean(final_max_dd))
    
    # Sharpe Ratio (approximate from episode returns)
    # Ideally we want daily returns, but episode returns give a proxy for consistency
    sharpe = float(jnp.mean(final_returns) / (jnp.std(final_returns) + 1e-6))
    
    metrics = {
        "mean_return": mean_return,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "max_drawdown": mean_max_dd,
        "total_trades": total_trades_sum,
        "min_return": float(jnp.min(final_returns)),
        "max_return": float(jnp.max(final_returns)),
    }
    
    print("Evaluation Metrics:", json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description="Phase 2 JAX evaluator self-test (saves a temporary checkpoint).")
    parser.add_argument(
        "--checkpoint-dir",
        help="Optional directory to write the temporary checkpoint. "
             "On Windows provide a local path like C:\\Users\\<user>\\AppData\\Local\\Temp "
             "to avoid UNC path issues."
    )
    args = parser.parse_args()

    # Test run
    print("Running Evaluator Test...")
    num_timesteps = 2000
    key = jax.random.key(0)
    dummy_data = MarketData(
        features=jax.random.normal(key, (num_timesteps, 8)),
        prices=jnp.abs(jax.random.normal(key, (num_timesteps, 4))) * 100 + 5000,
        atr=jnp.abs(jax.random.normal(key, (num_timesteps,))) * 10 + 5,
        time_features=jax.random.uniform(key, (num_timesteps, 3)),
        trading_mask=jnp.ones(num_timesteps),
        timestamps_hour=jnp.linspace(9.5, 16.9, num_timesteps),
        rth_indices=jnp.arange(60, num_timesteps - 100),  # Valid RTH start indices
        low_s=jnp.abs(jax.random.normal(key, (num_timesteps,))) * 100 + 4990,
        high_s=jnp.abs(jax.random.normal(key, (num_timesteps,))) * 100 + 5010,
    )
    
    # Create a dummy checkpoint
    config = PPOConfig(num_envs=1, num_steps=1, total_timesteps=1)
    env_params = EnvParamsPhase2()
    obs_shape = (env_params.window_size * env_params.num_features + 8,)
    train_state = create_train_state(key, obs_shape, config, num_actions=6)
    normalizer = create_normalizer(obs_shape)
    
    ckpt_dir, cleanup_needed = _resolve_checkpoint_dir(args.checkpoint_dir)
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print(f"Saving checkpoint to {ckpt_dir}...")
    try:
        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir, 
            target=train_state, 
            step=1, 
            prefix="phase2_jax_", 
            keep=1,
            overwrite=True
        )
        with open(os.path.join(ckpt_dir, "normalizer_1.pkl"), "wb") as f:
            pickle.dump(normalizer, f)
        print("Checkpoint saved.")
        evaluate_model(ckpt_dir, dummy_data, num_episodes=10)
    except Exception as e:
        print(f"Save failed: {e}")
    finally:
        if cleanup_needed:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
