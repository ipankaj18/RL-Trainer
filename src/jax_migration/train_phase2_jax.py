"""
Phase 2 JAX Training Script
Implements PPO with Transfer Learning from Phase 1.
"""

import jax
import jax.numpy as jnp
from jax import lax
import optax
import flax.linen as nn
from flax.training import train_state, checkpoints
import numpy as np
import os
import time
import pickle
from typing import Tuple, NamedTuple, Any, Dict
from functools import partial
import yaml

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Absolute imports
from src.jax_migration.data_loader import MarketData, load_market_data
from src.jax_migration.env_phase2_jax import (
    EnvStatePhase2, EnvParamsPhase2,
    reset_phase2, step_phase2, action_masks_phase2, get_observation_phase2,
    batch_reset_phase2, batch_step_phase2, batch_action_masks_phase2
)
from src.jax_migration.train_ppo_jax_fixed import (
    ActorCritic, NormalizerState, RunnerState, Transition, PPOConfig,
    create_normalizer, update_normalizer, normalize_obs,
    compute_gae, ppo_loss, sample_action, log_prob_action,
    masked_softmax
)
from src.jax_migration.training_metrics_tracker import TrainingMetricsTracker

class Colors:
    """Simple color constants for terminal output."""
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    RESET = "\033[0m"

class TrainState(train_state.TrainState):
    """Custom TrainState to include batch stats if needed."""
    pass

def create_train_state(
    key: jax.random.PRNGKey,
    obs_shape: Tuple[int, ...],
    config: PPOConfig,
    num_actions: int = 6
) -> TrainState:
    """Initialize network and optimizer for Phase 2."""
    network = ActorCritic(num_actions=num_actions)
    
    dummy_obs = jnp.zeros((1,) + obs_shape)
    params = network.init(key, dummy_obs)
    
    # Learning rate schedule
    num_updates = config.total_timesteps // (config.num_envs * config.num_steps)
    
    if config.anneal_lr:
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=config.learning_rate,
            transition_steps=config.lr_warmup_steps
        )
        decay_fn = optax.linear_schedule(
            init_value=config.learning_rate,
            end_value=0.0,
            transition_steps=num_updates - config.lr_warmup_steps
        )
        schedule = optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[config.lr_warmup_steps]
        )
    else:
        schedule = config.learning_rate
    
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(schedule, eps=1e-5)
    )
    
    return TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx
    )

def load_phase1_and_transfer(
    phase2_params,
    phase1_checkpoint_path: str,
    config: PPOConfig
):
    """
    Transfer weights from Phase 1 to Phase 2.
    
    Returns: Modified params (FrozenDict) to be used to create new TrainState.
    
    Strategy:
    1. Load Phase 1 params (3 actions, 225-dim observations).
    2. Handle Dense_0 shape mismatch (225 -> 228) by padding.
    3. Copy shared layers (Dense_1, LayerNorm_0, LayerNorm_1, Dense_2, Dense_4, Dense_5).
    4. Expand action head from 3 to 6 actions.
    
    Note: Returns params only. Caller must create new TrainState with these params
    to reinitialize optimizer state properly.
    """
    print(f"[TRANSFER] Loading Phase 1 checkpoint from {phase1_checkpoint_path}")
    
    try:
        # Restore as a dict first to inspect
        phase1_restored = checkpoints.restore_checkpoint(
            ckpt_dir=os.path.dirname(phase1_checkpoint_path),
            target=None,
            prefix=os.path.basename(phase1_checkpoint_path)
        )
        
        if phase1_restored is None:
             # Try restoring from directory if path is a directory
            phase1_restored = checkpoints.restore_checkpoint(
                ckpt_dir=phase1_checkpoint_path,
                target=None
            )
            
        if phase1_restored is None:
            raise FileNotFoundError(f"Could not restore checkpoint from {phase1_checkpoint_path}")
            
        phase1_params = phase1_restored['params']
        print("[TRANSFER] Phase 1 params loaded.")
        
        # Handle nesting if present (common in Flax checkpoints)
        # Both Phase 1 and Phase 2 might have nested 'params' structure
        if 'params' in phase1_params and isinstance(phase1_params['params'], (dict, flax.core.FrozenDict)):
            print("[TRANSFER] Detected nested 'params' in Phase 1, drilling down.")
            phase1_params = phase1_params['params']
            
        # Create a mutable copy of Phase 2 params
        new_params = flax.core.unfreeze(phase2_params)
        
        # Drill down Phase 2 if it also has nested params
        if 'params' in new_params and isinstance(new_params['params'], (dict, flax.core.FrozenDict)):
            print("[TRANSFER] Detected nested 'params' in Phase 2, drilling down.")
            phase2_inner = flax.core.unfreeze(new_params['params'])
        else:
            phase2_inner = new_params
            
        print(f"[DEBUG] Phase 1 layer keys: {list(phase1_params.keys())}")
        print(f"[DEBUG] Phase 2 layer keys: {list(phase2_inner.keys())}")
        
        # Transfer Dense_0 with shape handling (observation input layer)
        if 'Dense_0' in phase1_params and 'Dense_0' in phase2_inner:
            p1_kernel = phase1_params['Dense_0']['kernel']
            p1_bias = phase1_params['Dense_0']['bias']
            p2_kernel = phase2_inner['Dense_0']['kernel']
            p2_bias = phase2_inner['Dense_0']['bias']
            
            p1_shape = p1_kernel.shape
            p2_shape = p2_kernel.shape
            
            print(f"[DEBUG] Dense_0 shapes: P1={p1_shape}, P2={p2_shape}")
            
            if p1_shape != p2_shape:
                p1_in, p1_out = p1_shape
                p2_in, p2_out = p2_shape
                
                if p1_in < p2_in and p1_out == p2_out:
                    # Pad Phase 1 kernel to match Phase 2 input size
                    padding = p2_in - p1_in
                    print(f"  [INFO] Padding Dense_0: {p1_in} -> {p2_in} inputs (+{padding} dims)")
                    
                    # Initialize padding with small random values (Xavier-like)
                    key = jax.random.key(42)
                    stddev = jnp.sqrt(2.0 / (p2_in + p1_out))
                    pad_weights = jax.random.normal(key, (padding, p1_out)) * stddev * 0.1
                    
                    # Transfer P1 weights and add padding for new input dims
                    p2_kernel = p2_kernel.at[:p1_in, :].set(p1_kernel)
                    p2_kernel = p2_kernel.at[p1_in:, :].set(pad_weights)
                    
                    # Bias stays the same (output dims unchanged)
                    p2_bias = p1_bias
                    
                    phase2_inner['Dense_0']['kernel'] = p2_kernel
                    phase2_inner['Dense_0']['bias'] = p2_bias
                    print(f"  [OK] Transferred Dense_0 with padding ({padding} new input dims)")
                elif p1_in > p2_in:
                    print(f"  [ERROR] Cannot transfer Dense_0: P1 has more inputs than P2")
                    print(f"  [WARN] Using random initialization for Dense_0")
                else:
                    print(f"  [ERROR] Dense_0 output dims mismatch: P1={p1_out}, P2={p2_out}")
                    print(f"  [WARN] Using random initialization for Dense_0")
            else:
                # Shapes match perfectly, direct transfer
                phase2_inner['Dense_0'] = phase1_params['Dense_0']
                print(f"  [OK] Transferred Dense_0 (exact match)")
        else:
            print(f"  [WARN] Dense_0 missing in Phase 1 or Phase 2")
        
        # Transfer other shared layers (excluding Dense_0, Dense_3)
        shared_layers = ['Dense_1', 'LayerNorm_0', 'LayerNorm_1', 'Dense_2', 'Dense_4', 'Dense_5']
        
        for layer in shared_layers:
            if layer in phase1_params and layer in phase2_inner:
                phase2_inner[layer] = phase1_params[layer]
                print(f"  [OK] Transferred {layer}")
            else:
                print(f"  [WARN] Layer {layer} missing in P1 or P2")
                
        # Transfer Action Head (Dense_3) with expansion
        # P1: (128, 3) kernel, (3,) bias
        # P2: (128, 6) kernel, (6,) bias
        if 'Dense_3' in phase1_params and 'Dense_3' in phase2_inner:
            try:
                p1_kernel = phase1_params['Dense_3']['kernel']
                p1_bias = phase1_params['Dense_3']['bias']
                
                p2_kernel = phase2_inner['Dense_3']['kernel']
                p2_bias = phase2_inner['Dense_3']['bias']
                
                # Copy actions 0-2 (Hold, Buy, Sell)
                p2_kernel = p2_kernel.at[:, :3].set(p1_kernel)
                p2_bias = p2_bias.at[:3].set(p1_bias)
                
                # Initialize new actions 3-5 with negative bias (discourage initially)
                p2_bias = p2_bias.at[3:].set(-5.0)
                
                phase2_inner['Dense_3']['kernel'] = p2_kernel
                phase2_inner['Dense_3']['bias'] = p2_bias
                print(f"  [OK] Transferred Action Head (Dense_3) with expansion 3->6")
            except Exception as e:
                print(f"  [ERROR] Failed to transfer Dense_3: {e}")
        else:
            print("  [WARN] Dense_3 missing in Phase 1 or Phase 2 checkpoint")
            
        # Reconstruct proper param structure
        if 'params' in new_params:
            # Phase 2 has nested structure, need to wrap back
            new_params['params'] = phase2_inner
            final_params = new_params
        else:
            # Phase 2 was flat, use the inner params directly
            final_params = phase2_inner
            
        # Return frozen params only (not TrainState)
        return flax.core.freeze(final_params)
        
    except Exception as e:
        print(f"[ERROR] Transfer failed: {e}")
        import traceback
        traceback.print_exc()
        print("[WARN] Starting from scratch")
        return phase2_params  # Return original params if transfer fails

import flax

@partial(jax.jit, static_argnums=(1, 3, 4))
def collect_rollouts_phase2(
    runner_state: RunnerState,
    env_params: EnvParamsPhase2,
    data: MarketData,
    num_steps: int,
    num_envs: int
) -> Tuple[Transition, RunnerState, jnp.ndarray]:
    """
    Collect rollouts for Phase 2.
    Identical to Phase 1 but uses Phase 2 env functions.
    """
    train_state, env_states, normalizer, key, update_step = runner_state
    
    def step_fn(carry, _):
        env_states, key, normalizer = carry
        key, key_step, key_action, key_reset = jax.random.split(key, 4)
        
        # Phase 2 Observation
        obs_batch = get_batched_observations_phase2(env_states, data, env_params)
        obs_normalized = normalize_obs(obs_batch, normalizer)
        
        # Phase 2 Masks
        masks = batch_action_masks_phase2(env_states, data, env_params)
        
        logits, values = train_state.apply_fn(train_state.params, obs_normalized)
        
        key_actions = jax.random.split(key_action, num_envs)
        actions = jax.vmap(sample_action)(key_actions, logits, masks)
        log_probs = jax.vmap(log_prob_action)(logits, actions, masks)
        
        key_steps = jax.random.split(key_step, num_envs)
        next_obs, next_states, rewards, dones, infos = batch_step_phase2(
            key_steps, env_states, actions, env_params, data
        )
        
        key_resets = jax.random.split(key_reset, num_envs)
        reset_obs_batch, reset_states = batch_reset_phase2(key_reset, env_params, num_envs, data)
        
        final_states = jax.tree.map(
            lambda reset_val, next_val: jnp.where(
                dones[:, None] if reset_val.ndim > 1 else dones,
                reset_val,
                next_val
            ),
            reset_states,
            next_states
        )
        
        transition = Transition(
            obs=obs_normalized,
            action=actions,
            reward=rewards,
            done=dones.astype(jnp.float32),
            value=values,
            log_prob=log_probs,
            mask=masks
        )
        
        return (final_states, key, normalizer), transition
    
    (final_states, key, normalizer), transitions = lax.scan(
        step_fn,
        (env_states, key, normalizer),
        None,
        num_steps
    )
    
    all_obs = transitions.obs.reshape(-1, transitions.obs.shape[-1])
    new_normalizer = update_normalizer(normalizer, all_obs)
    
    final_obs = get_batched_observations_phase2(final_states, data, env_params)
    final_obs_norm = normalize_obs(final_obs, new_normalizer)
    _, last_values = train_state.apply_fn(train_state.params, final_obs_norm)
    
    advantages, returns = compute_gae(
        transitions.reward,
        transitions.value,
        transitions.done,
        last_values,
        0.99,
        0.95
    )
    
    transitions = transitions._replace(reward=returns, value=advantages)
    episode_returns = (transitions.reward * (1 - transitions.done)).sum(axis=0)
    
    new_runner_state = RunnerState(
        train_state=train_state,
        env_states=final_states,
        normalizer=new_normalizer,
        key=key,
        update_step=update_step
    )
    
    return transitions, new_runner_state, episode_returns

def get_batched_observations_phase2(env_states, data, params):
    return jax.vmap(
        lambda state: get_observation_phase2(state, data, params)
    )(env_states)

def train_phase2(
    config: PPOConfig,
    env_params: EnvParamsPhase2,
    data: MarketData,
    phase1_checkpoint: str = None,
    checkpoint_dir: str = "models/phase2_jax",
    market: str = "NQ",
    seed: int = 0
) -> RunnerState:
    """Main training loop for Phase 2."""
    
    # Ensure checkpoint path is absolute (required by orbax)
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    # Validate training configuration and checkpoint compatibility
    print(f"\n{Colors.CYAN}[VALIDATION] Checking training configuration...{Colors.RESET}")
    
    from src.jax_migration.validation_utils import validate_training_config, compute_observation_shape
    
    validation_result = validate_training_config(
        env_params,
        config,
        phase1_checkpoint=phase1_checkpoint
    )
    
    # Display warnings
    if validation_result['warnings']:
        print(f"{Colors.YELLOW}[VALIDATION WARNINGS]{Colors.RESET}")
        for warning in validation_result['warnings']:
            print(f"  ⚠ {warning}")
    
    # Check for errors
    if not validation_result['valid']:
        print(f"\n{Colors.RED}[VALIDATION FAILED]{Colors.RESET}")
        for error in validation_result['errors']:
            print(f"  ✗ {error}")
        raise ValueError("Training configuration validation failed")
    
    expected_obs_dim = validation_result['expected_obs_dim']
    print(f"{Colors.GREEN}✓ Validation passed{Colors.RESET}")
    print(f"  Expected observation shape: ({expected_obs_dim},)")
    
    if phase1_checkpoint:
        ckpt_val = validation_result['checkpoint_validation']
        print(f"  Phase 1 checkpoint observation shape: {ckpt_val['obs_shape']}")
        print(f"  Phase 1 checkpoint actions: {ckpt_val['num_actions']}")
    print()
    
    key = jax.random.key(seed)
    key, network_key = jax.random.split(key)
    
    obs_shape = (expected_obs_dim,)
    
    # Create initial train_state
    train_state = create_train_state(network_key, obs_shape, config, num_actions=6)
    
    # If transfer learning, get transferred params and recreate train_state
    if phase1_checkpoint:
        print(f"\n{Colors.CYAN}[TRANSFER] Applying transfer learning from Phase 1{Colors.RESET}")
        transferred_params = load_phase1_and_transfer(train_state.params, phase1_checkpoint, config)
        
        # Recreate train_state with transferred params to reinitialize optimizer state
        print(f"{Colors.CYAN}[TRANSFER] Recreating TrainState with transferred params{Colors.RESET}")
        train_state = TrainState.create(
            apply_fn=train_state.apply_fn,
            params=transferred_params,
            tx=train_state.tx
        )
        print(f"{Colors.GREEN}✓ Transfer learning complete{Colors.RESET}\n")
        
    normalizer = create_normalizer(obs_shape)
    
    key, reset_key = jax.random.split(key)
    obs, env_states = batch_reset_phase2(reset_key, env_params, config.num_envs, data)
    
    runner_state = RunnerState(
        train_state=train_state,
        env_states=env_states,
        normalizer=normalizer,
        key=key,
        update_step=jnp.array(0)
    )
    
    num_updates = config.total_timesteps // (config.num_envs * config.num_steps)
    if num_updates == 0:
        print(f"[WARN] Total timesteps ({config.total_timesteps}) < Batch size ({config.num_envs * config.num_steps}). Running 1 update.")
        num_updates = 1
    
    print(f"Starting Phase 2 Training:")
    print(f"  Total timesteps: {config.total_timesteps:,}")
    print(f"  Num envs: {config.num_envs}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Initialize metrics tracker
    metrics_tracker = TrainingMetricsTracker(
        market=market,
        phase=2,
        initial_balance=env_params.initial_balance,
        drawdown_limit=env_params.trailing_dd_limit,
        enable_tensorboard=False  # Can enable later
    )
        
    start_time = time.time()
    
    # Import train_step from fixed PPO (it's generic)
    from src.jax_migration.train_ppo_jax_fixed import train_step
    
    for update in range(num_updates):
        key, rollout_key, train_key = jax.random.split(runner_state.key, 3)
        runner_state = runner_state._replace(key=key, update_step=jnp.array(update))
        
        transitions, runner_state, episode_returns = collect_rollouts_phase2(
            runner_state, env_params, data, config.num_steps, config.num_envs
        )
        
        advantages = transitions.value
        
        new_train_state, train_metrics = train_step(
            runner_state.train_state,
            transitions,
            advantages,
            config,
            train_key
        )
        
        runner_state = runner_state._replace(train_state=new_train_state)
        jax.block_until_ready(runner_state.train_state.params)
        
        # Update metrics tracker with episode completions
        # Note: infos contains aggregated data - we track cumulative metrics
        # For proper episode tracking, we'd need to detect individual episode dones
        # For now, we log aggregated metrics periodically
        
        if (update + 1) % 1 == 0:
            elapsed = time.time() - start_time
            timesteps = (update + 1) * config.num_envs * config.num_steps
            sps = timesteps / elapsed
            print(f"Update {update + 1}/{num_updates} | SPS: {sps:,.0f} | Return: {episode_returns.mean():.2f} | Loss: {train_metrics['policy_loss']:.4f}")
        
        # Log metrics tracker every 10 updates
        if (update + 1) % 10 == 0:
            metrics_tracker.log_to_console(update + 1, num_updates)
            # Save metrics to JSON
            metrics_json_path = f"results/{market}_jax_phase2_realtime_metrics.json"
            metrics_tracker.save_to_json(metrics_json_path, include_history=True)
            
        if (update + 1) % 50 == 0:
            # Save checkpoint
            checkpoints.save_checkpoint(
                ckpt_dir=checkpoint_dir,
                target=runner_state.train_state,
                step=update + 1,
                prefix="phase2_jax_",
                keep=3
            )
            # Save normalizer
            with open(os.path.join(checkpoint_dir, f"normalizer_{update + 1}.pkl"), "wb") as f:
                pickle.dump(runner_state.normalizer, f)
            
    print("Training complete.")
    print("Training complete.")
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=runner_state.train_state,
        step=num_updates,
        prefix="phase2_jax_final_",
        keep=1,
        overwrite=True
    )
    # Save final normalizer
    with open(os.path.join(checkpoint_dir, "normalizer_final.pkl"), "wb") as f:
        pickle.dump(runner_state.normalizer, f)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2 JAX Training")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total timesteps")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
    parser.add_argument("--num_steps", type=int, default=128, help="Steps per environment per update")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--phase1_checkpoint", type=str, default=None, help="Path to Phase 1 checkpoint for transfer")
    parser.add_argument("--checkpoint_dir", type=str, default="models/phase2_jax", help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--test", action="store_true", help="Run in test mode with dummy data")
    parser.add_argument("--data_path", type=str, default=None, help="Path to market data CSV (auto-detects if not provided)")
    parser.add_argument("--market", type=str, default="NQ", help="Market symbol (NQ, ES, etc.)")
    parser.add_argument("--hardware-profile", type=str, default=None, help="Path to hardware profile YAML")
    
    args = parser.parse_args()
    
    # Auto-detect data path if not provided
    if args.data_path is None and not args.test:
        import glob
        
        # Try market-specific file first
        market_file = f"data/{args.market}_D1M.csv"
        if os.path.exists(market_file):
            args.data_path = market_file
            print(f"[AUTO-DETECT] Using market data: {market_file}")
        else:
            # Try generic D1M.csv
            generic_file = "data/D1M.csv"
            if os.path.exists(generic_file):
                args.data_path = generic_file
                print(f"[AUTO-DETECT] Using generic data: {generic_file}")
            else:
                # Search for any *_D1M.csv file
                pattern = "data/*_D1M.csv"
                candidates = sorted(glob.glob(pattern))
                if candidates:
                    args.data_path = candidates[0]
                    print(f"[AUTO-DETECT] Using found data: {args.data_path}")
                else:
                    raise FileNotFoundError(
                        f"No market data found. Expected: {market_file}, {generic_file}, or any data/*_D1M.csv file. "
                        f"Please specify --data_path explicitly or ensure data files are in the data/ directory."
                    )

    # Apply hardware profile if provided
    if args.hardware_profile:
        try:
            with open(args.hardware_profile, 'r') as f:
                profile = yaml.safe_load(f)
                print(f"Loading hardware profile from {args.hardware_profile}")
                
                if 'num_envs' in profile:
                    args.num_envs = int(profile['num_envs'])
                    print(f"  - Overriding num_envs: {args.num_envs}")
                
                if 'num_steps' in profile:
                    args.num_steps = int(profile['num_steps'])
                    print(f"  - Overriding num_steps: {args.num_steps}")
                    
                if 'batch_size' in profile:
                    # JAX PPO usually calculates batch_size = num_envs * num_steps
                    # We can check consistency or ignore
                    pass
        except Exception as e:
            print(f"[WARN] Failed to load hardware profile: {e}")
    
    config = PPOConfig(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate
    )
    
    env_params = EnvParamsPhase2()
    
    if args.test:
        print("Running Phase 2 JAX Test Mode...")
        key = jax.random.key(args.seed)
        num_timesteps = args.total_timesteps
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
        data = dummy_data
    else:
        # Load real data
        try:
            print(f"Loading market data from {args.data_path}")
            # Infer second-level data path
            second_data_path = args.data_path.replace('_D1M.csv', '_D1S.csv')
            if not os.path.exists(second_data_path):
                second_data_path = None
                
            data = load_market_data(args.data_path, second_data_path=second_data_path)
            print("Loaded real market data.")
        except Exception as e:
            print(f"[ERROR] Could not load market data from {args.data_path}: {e}")
            if not args.test:
                raise e
            print("[WARN] Using dummy data for safety.")
            key = jax.random.key(args.seed)
            num_timesteps = args.total_timesteps
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
            data = dummy_data

    train_phase2(
        config,
        env_params,
        data,
        phase1_checkpoint=args.phase1_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        market=args.market,
        seed=args.seed
    )
