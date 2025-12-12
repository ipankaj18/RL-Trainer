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
from typing import Tuple, NamedTuple, Any, Dict, Optional
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
from src.market_specs import get_market_spec, MARKET_SPECS

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

# MEMORY FIX (2025-12-08): Removed env_params (arg 1) from static_argnums
# env_params changes every update due to curriculum values (training_progress, 
# exploration bonuses, forced_position_ratio). Keeping it static caused JAX to
# recompile on every iteration, leaking ~10-50 MB per update → server crash.
@partial(jax.jit, static_argnums=(3, 4))  # Only num_steps and num_envs are truly static
def collect_rollouts_phase2(
    runner_state: RunnerState,
    env_params: EnvParamsPhase2,
    data: MarketData,
    num_steps: int,
    num_envs: int,
    exploration_floor: float = 0.0  # CRITICAL FIX: Add PM action floor
) -> Tuple[Transition, RunnerState, jnp.ndarray]:
    """
    Collect rollouts for Phase 2.
    Identical to Phase 1 but uses Phase 2 env functions.
    HOLD TRAP FIX: Now supports exploration_floor for PM action sampling.
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
        # CRITICAL FIX: Pass exploration_floor to ensure PM action floor is applied
        actions = jax.vmap(lambda k, l, m: sample_action(k, l, m, exploration_floor))(
            key_actions, logits, masks
        )
        log_probs = jax.vmap(lambda l, a, m: log_prob_action(l, a, m, exploration_floor))(
            logits, actions, masks
        )
        
        key_steps = jax.random.split(key_step, num_envs)
        next_obs, next_states, rewards, dones, infos = batch_step_phase2(
            key_steps, env_states, actions, env_params, data
        )
        
        key_resets = jax.random.split(key_reset, num_envs)
        reset_obs_batch, reset_states = batch_reset_phase2(
            jax.random.split(key_reset, num_envs),
            env_params,
            num_envs,
            data
        )
        
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

def find_latest_checkpoint(checkpoint_dir: str) -> tuple[Optional[str], Optional[int]]:
    """Find the most recent checkpoint in the directory.

    Returns:
        (checkpoint_path, step_number) or (None, None) if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None, None

    checkpoint_dirs = sorted([
        d for d in Path(checkpoint_dir).iterdir()
        if d.is_dir() and d.name.startswith("phase2_jax_") and not d.name.endswith("_final")
    ], key=lambda x: int(x.name.split("_")[-1]))

    if not checkpoint_dirs:
        return None, None

    latest_checkpoint = checkpoint_dirs[-1]
    step_number = int(latest_checkpoint.name.split("_")[-1])
    return str(latest_checkpoint), step_number

def train_phase2(
    config: PPOConfig,
    env_params: EnvParamsPhase2,
    data: MarketData,
    phase1_checkpoint: str = None,
    checkpoint_dir: str = "models/phase2_jax",
    market: str = "NQ",
    seed: int = 0,
    resume: bool = False,
    resume_from: Optional[str] = None
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

    # Calculate num_updates early (needed for resume messages)
    num_updates = config.total_timesteps // (config.num_envs * config.num_steps)
    if num_updates == 0:
        print(f"[WARN] Total timesteps ({config.total_timesteps}) < Batch size ({config.num_envs * config.num_steps}). Running 1 update.")
        num_updates = 1

    # Create initial train_state
    train_state = create_train_state(network_key, obs_shape, config, num_actions=6)

    # Checkpoint resumption
    initial_update = 0
    resume_checkpoint_path = None
    resume_step = None

    if resume or resume_from:
        if resume_from:
            # User specified exact checkpoint
            resume_checkpoint_path = os.path.join(checkpoint_dir, resume_from)
            if not os.path.exists(resume_checkpoint_path):
                print(f"{Colors.RED}[ERROR] Specified checkpoint not found: {resume_checkpoint_path}{Colors.RESET}")
                sys.exit(1)
            resume_step = int(resume_from.split("_")[-1])
        else:
            # Auto-detect latest checkpoint
            resume_checkpoint_path, resume_step = find_latest_checkpoint(checkpoint_dir)

        if resume_checkpoint_path and resume_step:
            print(f"\n{Colors.YELLOW}[RESUME] Found checkpoint: {os.path.basename(resume_checkpoint_path)} (update {resume_step}){Colors.RESET}")

            if not resume_from:  # Only prompt if auto-detected
                user_input = input(f"Resume from update {resume_step}? (y/n): ").strip().lower()
                if user_input != 'y':
                    print(f"{Colors.CYAN}Starting fresh training...{Colors.RESET}")
                    resume_checkpoint_path = None
                    resume_step = None

            if resume_checkpoint_path:
                print(f"{Colors.CYAN}[RESUME] Restoring checkpoint...{Colors.RESET}")

                try:
                    # Restore train_state using Orbax
                    restored_state = checkpoints.restore_checkpoint(
                        ckpt_dir=resume_checkpoint_path,
                        target=train_state,
                        step=None  # Use step from directory
                    )
                    train_state = restored_state
                    print(f"{Colors.GREEN}✓ Restored model checkpoint{Colors.RESET}")

                    # Restore normalizer
                    normalizer_path = os.path.join(checkpoint_dir, f"normalizer_{resume_step}.pkl")
                    if os.path.exists(normalizer_path):
                        with open(normalizer_path, "rb") as f:
                            normalizer = pickle.load(f)
                        print(f"{Colors.GREEN}✓ Restored normalizer from step {resume_step}{Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}[WARN] Normalizer not found for step {resume_step}, using fresh normalizer{Colors.RESET}")
                        normalizer = create_normalizer(obs_shape)

                    initial_update = resume_step
                    print(f"{Colors.GREEN}✓ Resuming from update {initial_update}/{num_updates}{Colors.RESET}\n")

                except Exception as e:
                    print(f"{Colors.RED}[ERROR] Failed to restore checkpoint: {e}{Colors.RESET}")
                    print(f"{Colors.YELLOW}Starting fresh training...{Colors.RESET}")
                    initial_update = 0
                    normalizer = create_normalizer(obs_shape)
        else:
            print(f"{Colors.YELLOW}[RESUME] No checkpoints found in {checkpoint_dir}{Colors.RESET}")
            print(f"{Colors.CYAN}Starting fresh training...{Colors.RESET}")
            normalizer = create_normalizer(obs_shape)
    else:
        # No resume - standard initialization
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
    obs, env_states = batch_reset_phase2(
        jax.random.split(reset_key, config.num_envs),
        env_params,
        config.num_envs,
        data
    )
    
    runner_state = RunnerState(
        train_state=train_state,
        env_states=env_states,
        normalizer=normalizer,
        key=key,
        update_step=jnp.array(initial_update)
    )

    print(f"Starting Phase 2 Training:")
    print(f"  Total timesteps: {config.total_timesteps:,}")
    print(f"  Num envs: {config.num_envs}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    if initial_update > 0:
        print(f"  Resuming from update: {initial_update}/{num_updates}")
    
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
    
    # Delta tracking variables for metrics (Phase 1 pattern)
    prev_total_trades = 0      # Track cumulative trades for delta calculation
    prev_total_winning = 0     # Track cumulative wins for delta calculation
    prev_avg_balance = env_params.initial_balance  # Track avg balance for delta calculation
    
    # Import train_step from fixed PPO (it's generic)
    from src.jax_migration.train_ppo_jax_fixed import train_step

    for update in range(initial_update, num_updates):
        key, rollout_key, train_key = jax.random.split(runner_state.key, 3)
        runner_state = runner_state._replace(key=key, update_step=jnp.array(update))
        
        # ===== 3-SUB-PHASE CURRICULUM (Critical for solving HOLD trap) =====
        # Phase 2A (0-20%): Position Boot Camp - 50% forced positions, high PM bonus
        # Phase 2B (20-80%): Integrated Trading - forced ratio decays 50%→10%
        # Phase 2C (80-100%): Production Hardening - 0% forced, realistic conditions
        
        current_global_timestep = update * config.num_envs * config.num_steps
        training_progress = update / num_updates
        
        # Calculate curriculum parameters based on progress
        # HOLD TRAP FIX (2025-12-07): Increased forced ratios and PM bonus retention
        if training_progress < 0.20:  # Phase 2A: Boot Camp
            forced_position_ratio = 0.7  # 70% start with position (was 50%)
            pm_bonus_mult = 1.0  # Full PM bonus
            entry_bonus_mult = 1.0  # Full entry bonus
            commission_mult = 0.1  # Very low commission
            phase_name = "2A-Boot"
        elif training_progress < 0.80:  # Phase 2B: Integrated
            phase_progress = (training_progress - 0.20) / 0.60  # 0→1 within Phase 2B
            forced_position_ratio = 0.7 - 0.5 * phase_progress  # 70%→20% (was 50%→10%)
            pm_bonus_mult = 1.0 - 0.5 * phase_progress  # 100%→50% (was 100%→25%)
            entry_bonus_mult = 1.0 - 0.5 * phase_progress  # 100%→50%
            commission_mult = 0.1 + 0.7 * phase_progress  # 10%→80%
            phase_name = "2B-Int"
        else:  # Phase 2C: Production Hardening
            forced_position_ratio = 0.1  # 10% forced (was 0%) - maintains PM practice
            pm_bonus_mult = 0.0  # No exploration bonus
            entry_bonus_mult = 0.0  # No exploration bonus
            commission_mult = 1.0  # Full commission
            phase_name = "2C-Prod"
        
        # Calculate actual bonus values (use updated base values from env_params)
        # CRITICAL FIX: Use $200 base (not $400) - updated in env_params default
        pm_bonus = 200.0 * pm_bonus_mult
        entry_bonus = 100.0 * entry_bonus_mult
        curriculum_commission = 2.5 * commission_mult
        
        # Update env_params with current progress (chex dataclass requires recreate)
        env_params = EnvParamsPhase2(
            contract_size=env_params.contract_size, tick_size=env_params.tick_size,
            tick_value=env_params.tick_value, commission=env_params.commission,
            slippage_ticks=env_params.slippage_ticks, initial_balance=env_params.initial_balance,
            sl_atr_mult=env_params.sl_atr_mult, tp_sl_ratio=env_params.tp_sl_ratio,
            position_size=env_params.position_size, trailing_dd_limit=env_params.trailing_dd_limit,
            risk_per_trade=env_params.risk_per_trade, max_position_size=env_params.max_position_size,
            contract_value=env_params.contract_value, window_size=env_params.window_size,
            num_features=env_params.num_features, rth_open=env_params.rth_open,
            rth_close=env_params.rth_close, min_episode_bars=env_params.min_episode_bars,
            trail_activation_mult=env_params.trail_activation_mult,
            trail_distance_atr=env_params.trail_distance_atr,
            be_min_profit_atr=env_params.be_min_profit_atr,
            initial_commission=curriculum_commission,  # Use curriculum commission
            final_commission=env_params.final_commission,
            commission_curriculum=True,
            training_progress=training_progress,
            current_global_timestep=float(current_global_timestep),
            total_training_timesteps=float(config.total_timesteps),
            # NEW: Forced position curriculum
            forced_position_ratio=forced_position_ratio,
            forced_position_profit_range=1.0,
            # NEW: Action-specific exploration bonuses
            pm_action_exploration_bonus=pm_bonus,
            entry_action_exploration_bonus=entry_bonus
        )
        
        # CRITICAL FIX: Calculate exploration floor for PM actions (same as Phase 1)
        # Permanent 5% floor prevents HOLD trap convergence
        exploration_floor_horizon = config.total_timesteps * 1.0
        floor_progress = min(1.0, current_global_timestep / exploration_floor_horizon)
        base_floor = 0.08    # Start at 8%
        min_floor = 0.05     # Never go below 5%
        current_floor = min_floor + (base_floor - min_floor) * max(0.0, 1.0 - floor_progress)
        
        transitions, runner_state, episode_returns = collect_rollouts_phase2(
            runner_state, env_params, data, config.num_steps, config.num_envs,
            exploration_floor=current_floor  # CRITICAL FIX: Pass floor to rollouts
        )
        
        advantages = transitions.value
        
        # CRITICAL FIX: Pass exploration_floor to train_step for proper loss calculation
        new_train_state, train_metrics = train_step(
            runner_state.train_state,
            transitions,
            advantages,
            config,
            train_key,
            exploration_floor=current_floor  # CRITICAL FIX: Ensure floor used in training
        )
        
        runner_state = runner_state._replace(train_state=new_train_state)
        jax.block_until_ready(runner_state.train_state.params)
        
        # ===== UPDATE METRICS TRACKER WITH DELTA STATS (Phase 1 pattern) =====
        # Extract current statistics from environment states
        final_env_states = runner_state.env_states  # Shape: (num_envs,)
        
        current_total_trades = int(final_env_states.num_trades.sum())
        current_total_winning = int(final_env_states.winning_trades.sum())
        avg_balance = float(final_env_states.balance.mean())  # CRITICAL: Use .mean() not .sum()!
        
        # Calculate DELTAS (changes since last update)
        if update == 0:
            # First update - all current values are deltas
            new_trades_this_update = current_total_trades
            new_wins_this_update = current_total_winning
            avg_pnl_delta = 0.0  # No PnL delta on first update
        else:
            # Calculate deltas from previous update
            new_trades_this_update = current_total_trades - prev_total_trades
            new_wins_this_update = current_total_winning - prev_total_winning
            avg_pnl_delta = avg_balance - prev_avg_balance  # Average PnL change per env
        
        # Calculate win rate for NEW trades this update only
        win_rate_this_update = new_wins_this_update / max(new_trades_this_update, 1) if new_trades_this_update > 0 else 0.0
        
        # Update tracker with DELTA stats (per-env averages, not num_envs× sums)
        metrics_tracker.record_episode(
            final_balance=avg_balance,
            num_trades=new_trades_this_update,      # DELTA, not cumulative
            win_rate=win_rate_this_update,          # Win rate of new trades only
            total_pnl=avg_pnl_delta                 # DELTA, not cumulative
        )
        
        # Store current values for next delta calculation
        prev_total_trades = current_total_trades
        prev_total_winning = current_total_winning
        prev_avg_balance = avg_balance
        
        if (update + 1) % 1 == 0:
            elapsed = time.time() - start_time
            timesteps = (update + 1) * config.num_envs * config.num_steps
            sps = timesteps / elapsed
            
            # Use curriculum values calculated above (phase_name, pm_bonus, entry_bonus, forced_position_ratio)
            bonus_str = f"Entry: ${entry_bonus:.0f}, PM: ${pm_bonus:.0f}" if pm_bonus > 0 or entry_bonus > 0 else "Disabled"
            forced_str = f"Forced: {forced_position_ratio*100:.0f}%" if forced_position_ratio > 0 else ""
            floor_str = f"Floor: {current_floor*100:.1f}%"  # ADDED: Show action floor value
            
            # Enhanced logging with phase name and curriculum info
            print(f"[{phase_name}] Update {update + 1}/{num_updates} | SPS: {sps:,.0f} | Return: {episode_returns.mean():.2f} | Loss: {train_metrics['policy_loss']:.4f} | {forced_str} 🎯 {bonus_str} | Comm: ${curriculum_commission:.2f} | {floor_str}")
        
        # ===== ACTION DISTRIBUTION MONITORING (Critical for detecting HOLD trap) =====
        # Track distribution for all 6 actions every 10 updates
        if (update + 1) % 10 == 0:
            # Get action distribution from rollout
            actions_flat = np.array(transitions.action).flatten()
            action_counts = np.bincount(actions_flat, minlength=6)
            action_pcts = action_counts / max(action_counts.sum(), 1)
            
            # Action names for Phase 2
            action_names = ['HOLD', 'BUY', 'SELL', 'SL→BE', 'TRAIL+', 'TRAIL-']
            
            # Build action distribution string
            dist_parts = [f"{action_names[i]}: {action_pcts[i]*100:.1f}%" for i in range(6)]
            action_dist_str = " | ".join(dist_parts)
            print(f"     📊 Actions: {action_dist_str}")
            
            # Warn if PM actions are underutilized (below 2% each after Phase 2A)
            if training_progress > 0.25:  # After Boot Camp
                pm_actions_pct = action_pcts[3:6].sum()
                if pm_actions_pct < 0.03:  # Less than 3% total PM actions
                    print(f"     ⚠️ WARNING: PM actions underutilized ({pm_actions_pct*100:.1f}% total)")
            
            # Get current metrics from tracker
            status = metrics_tracker.get_metrics()
            total_trades = status.get('total_trades', 0)
            win_rate = status.get('win_rate', 0.0)
            total_pnl = status.get('total_pnl', 0.0)
            current_balance = status.get('current_balance', 50000.0)
            
            # Simple one-line format like Phase 1
            print(f"     Trades: {total_trades} | Win Rate: {win_rate:.1%} | P&L: ${total_pnl:.2f} | Balance: ${current_balance:,.2f}")
            
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
                keep=3,
                overwrite=True
            )
            # Save normalizer
            with open(os.path.join(checkpoint_dir, f"normalizer_{update + 1}.pkl"), "wb") as f:
                pickle.dump(runner_state.normalizer, f)
        
        # MEMORY SAFETY: Clear JAX caches every 500 updates as safety net
        if (update + 1) % 500 == 0:
            jax.clear_caches()
            print(f"  [MEMORY] Cleared JAX caches at update {update + 1}")
            
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
    parser.add_argument("--resume", action="store_true", help="Resume from most recent checkpoint")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from specific checkpoint (e.g., 'phase2_jax_2400')")

    args = parser.parse_args()
    
    # Auto-detect data path if not provided
    if args.data_path is None and not args.test:
        import glob
        
        # TRAIN/TEST SPLIT: Prefer train-specific file first
        train_file = f"data/{args.market}_D1M_train.csv"
        if os.path.exists(train_file):
            args.data_path = train_file
            print(f"[AUTO-DETECT] Using TRAIN data (80%): {train_file}")
        else:
            # Fall back to full market-specific file
            market_file = f"data/{args.market}_D1M.csv"
            if os.path.exists(market_file):
                args.data_path = market_file
                print(f"[AUTO-DETECT] Using full market data: {market_file}")
                print(f"  ⚠️  Note: Consider running Data Processing to create train/test split")
            else:
                # Try generic D1M.csv
                generic_file = "data/D1M.csv"
                if os.path.exists(generic_file):
                    args.data_path = generic_file
                    print(f"[AUTO-DETECT] Using generic data: {generic_file}")
                else:
                    # Search for any *_D1M_train.csv or *_D1M.csv file
                    train_candidates = sorted(glob.glob("data/*_D1M_train.csv"))
                    if train_candidates:
                        args.data_path = train_candidates[0]
                        print(f"[AUTO-DETECT] Using found train data: {args.data_path}")
                    else:
                        full_candidates = sorted(glob.glob("data/*_D1M.csv"))
                        if full_candidates:
                            args.data_path = full_candidates[0]
                            print(f"[AUTO-DETECT] Using found data: {args.data_path}")
                        else:
                            raise FileNotFoundError(
                                f"No market data found. Expected: {train_file}, {market_file}, or any data/*_D1M.csv file. "
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

    # FIX: Ensure checkpoint directory matches main.py expectation (models/phase2_jax_{market})
    # If the user didn't specify a custom dir (it's the default), append the market
    if args.checkpoint_dir == "models/phase2_jax":
        args.checkpoint_dir = f"models/phase2_jax_{args.market.lower()}"
        print(f"[CONFIG] Auto-updated checkpoint dir to: {args.checkpoint_dir}")
    else:
        print(f"[CONFIG] Using specified checkpoint dir: {args.checkpoint_dir}")
        
    config = PPOConfig(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate
    )
    
    # FIX (2025-12-10): Use market-specific contract values instead of ES defaults
    # Previously used EnvParamsPhase2() which defaults to ES (contract_size=50.0)
    # NQ should use $20/point, not $50/point - losses were calculated 2.5x too large!
    market_spec = get_market_spec(args.market)
    if market_spec:
        print(f"{Colors.GREEN}Using {args.market} contract specs: ${market_spec.contract_multiplier}/point, max {market_spec.max_position_size} contracts{Colors.RESET}")
        env_params = EnvParamsPhase2(
            contract_size=market_spec.contract_multiplier,
            contract_value=market_spec.contract_multiplier,
            tick_size=market_spec.tick_size,
            tick_value=market_spec.tick_value,
            commission=market_spec.commission,
            slippage_ticks=market_spec.slippage_ticks,
            # NEW (2025-12-12): Market-specific position sizing
            max_position_size=float(market_spec.max_position_size),
        )
    else:
        print(f"{Colors.YELLOW}Warning: Unknown market {args.market}, using ES defaults{Colors.RESET}")
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
        seed=args.seed,
        resume=args.resume,
        resume_from=args.resume_from
    )
