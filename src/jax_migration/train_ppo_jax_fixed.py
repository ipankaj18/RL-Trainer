"""
PureJAX PPO Training Loop - FIXED VERSION

Critical fixes applied:
1. Fixed collect_rollouts() observation construction (was placeholder zeros)
2. Added observation normalization
3. Proper auto-reset mechanism
4. Episode boundary handling
5. Learning rate warmup

Based on PureJaxRL patterns with custom modifications for trading.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, NamedTuple, Callable, Any, Optional
from functools import partial
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import time
import chex

# Relative imports
from .data_loader import MarketData
from .env_phase1_jax import (
    EnvState, EnvParams,
    reset, step, action_masks, get_observation,
    batch_reset, batch_step, batch_action_masks
)
from .training_metrics_tracker import TrainingMetricsTracker  # Phase 2 integration
from .training_quality_monitor import TrainingQualityMonitor  # NEW: Phase C integration
from .hyperparameter_auto_adjuster import HyperparameterAutoAdjuster  # NEW: Phase C integration

# Market specifications for correct contract values
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.market_specs import get_market_spec, MARKET_SPECS


# =============================================================================
# Observation Normalization
# =============================================================================

class NormalizerState(NamedTuple):
    """Running statistics for observation normalization."""
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def create_normalizer(obs_shape: Tuple[int, ...]) -> NormalizerState:
    """Initialize normalizer with zeros."""
    return NormalizerState(
        mean=jnp.zeros(obs_shape, dtype=jnp.float32),
        var=jnp.ones(obs_shape, dtype=jnp.float32),
        count=jnp.array(1e-4, dtype=jnp.float32)
    )


def update_normalizer(
    state: NormalizerState, 
    batch: jnp.ndarray
) -> NormalizerState:
    """Update running mean/var with new batch (Welford's algorithm)."""
    batch_mean = batch.mean(axis=0)
    batch_var = batch.var(axis=0)
    batch_count = jnp.array(batch.shape[0], dtype=jnp.float32)
    
    delta = batch_mean - state.mean
    total_count = state.count + batch_count
    
    new_mean = state.mean + delta * batch_count / total_count
    
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta**2 * state.count * batch_count / total_count
    new_var = M2 / total_count
    
    # Clamp variance to avoid division by zero
    new_var = jnp.maximum(new_var, 1e-6)
    
    return NormalizerState(new_mean, new_var, total_count)


def normalize_obs(obs: jnp.ndarray, normalizer: NormalizerState) -> jnp.ndarray:
    """Normalize observation using running statistics."""
    return (obs - normalizer.mean) / jnp.sqrt(normalizer.var + 1e-8)


# =============================================================================
# Network Architecture
# =============================================================================

class ActorCritic(nn.Module):
    """Actor-Critic network with action masking support."""
    num_actions: int = 3
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Shared feature extractor
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)  # Added layer norm for stability
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        
        # Actor head (policy logits)
        actor = nn.Dense(self.hidden_dim // 2)(x)
        actor = nn.relu(actor)
        logits = nn.Dense(self.num_actions)(actor)
        
        # Critic head (value function)
        critic = nn.Dense(self.hidden_dim // 2)(x)
        critic = nn.relu(critic)
        value = nn.Dense(1)(critic)
        
        return logits, value.squeeze(-1)


def masked_softmax(
    logits: jnp.ndarray,
    mask: jnp.ndarray,
    exploration_floor: float = 0.0,  # NEW parameter
    floor_actions: tuple = (1, 2)     # BUY=1, SELL=2 (not used when traced)
) -> jnp.ndarray:
    """
    Apply mask to logits and compute softmax with optional exploration floor.

    PHASE B: Minimum action probability floor prevents HOLD trap by guaranteeing
    minimum probability for BUY/SELL actions during exploration phase.
    
    HOLD TRAP FIX (2025-12-07): Extended to support PM actions (3,4,5) in Phase 2.
    Auto-detects Phase 2 via logits.shape[-1] == 6.
    
    CRITICAL FIX: Uses jnp.where() instead of Python if/for to avoid
    TracerBoolConversionError when exploration_floor is traced inside JIT.
    """
    masked_logits = jnp.where(mask, logits, -1e10)
    probs = jax.nn.softmax(masked_logits, axis=-1)

    # PHASE B: Apply minimum probability floor for BUY (1) and SELL (2) actions
    # CRITICAL: Use jnp.where() instead of Python 'if' to avoid TracerBoolConversionError
    # when exploration_floor becomes a traced value inside lax.scan/vmap
    
    # Apply floor to BUY action (index 1)
    floored_probs = probs.at[..., 1].set(
        jnp.maximum(probs[..., 1], exploration_floor)
    )
    # Apply floor to SELL action (index 2)
    floored_probs = floored_probs.at[..., 2].set(
        jnp.maximum(floored_probs[..., 2], exploration_floor)
    )
    
    # HOLD TRAP FIX (2025-12-07): Apply floor to PM actions for Phase 2 (6 actions)
    # Phase 2 has actions: 0=HOLD, 1=BUY, 2=SELL, 3=SL→BE, 4=TRAIL+, 5=TRAIL-
    # Auto-detect Phase 2 by checking if logits have 6 dimensions
    is_phase2 = logits.shape[-1] == 6
    
    # Apply PM floor only in Phase 2 (use jnp.where for JAX compatibility)
    pm_floor = jnp.where(is_phase2, exploration_floor, 0.0)
    
    # Apply floor to PM actions (indices 3, 4, 5)
    floored_probs = floored_probs.at[..., 3].set(
        jnp.where(is_phase2, jnp.maximum(floored_probs[..., 3], pm_floor), floored_probs[..., 3])
    )
    floored_probs = floored_probs.at[..., 4].set(
        jnp.where(is_phase2, jnp.maximum(floored_probs[..., 4], pm_floor), floored_probs[..., 4])
    )
    floored_probs = floored_probs.at[..., 5].set(
        jnp.where(is_phase2, jnp.maximum(floored_probs[..., 5], pm_floor), floored_probs[..., 5])
    )
    
    # Renormalize to ensure probabilities sum to 1.0
    floored_probs = floored_probs / floored_probs.sum(axis=-1, keepdims=True)
    
    # Use jnp.where to conditionally apply floored probs (JAX-compatible)
    # When exploration_floor > 0, use floored_probs; otherwise use original probs
    probs = jnp.where(exploration_floor > 0.0, floored_probs, probs)

    return probs




def sample_action(
    key: jax.random.PRNGKey,
    logits: jnp.ndarray,
    mask: jnp.ndarray,
    exploration_floor: float = 0.0
) -> jnp.ndarray:
    """Sample action from masked categorical distribution."""
    probs = masked_softmax(logits, mask, exploration_floor=exploration_floor)
    return jax.random.categorical(key, jnp.log(probs + 1e-8))


def log_prob_action(
    logits: jnp.ndarray,
    action: jnp.ndarray,
    mask: jnp.ndarray,
    exploration_floor: float = 0.0
) -> jnp.ndarray:
    """Compute log probability of action under masked distribution."""
    probs = masked_softmax(logits, mask, exploration_floor=exploration_floor)
    return jnp.log(probs[action] + 1e-8)


# =============================================================================
# PPO Components
# =============================================================================

class Transition(NamedTuple):
    """Single transition for PPO training."""
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    mask: jnp.ndarray


class PPOConfig(NamedTuple):
    """PPO hyperparameters."""
    num_envs: int = 2048
    num_steps: int = 128
    num_minibatches: int = 4
    num_epochs: int = 4
    gamma: float = 0.95  # Reduced from 0.99 for trading's shorter horizon
    gae_lambda: float = 0.95
    clip_eps: float = 0.15
    ent_coef: float = 0.05  # Increased from 0.01 to prevent entropy collapse/HOLD trap
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    total_timesteps: int = 2_000_000
    # New additions
    normalize_obs: bool = True
    lr_warmup_steps: int = 1000


class RunnerState(NamedTuple):
    """Complete training state for pure JAX training loop."""
    train_state: TrainState
    env_states: EnvState
    normalizer: NormalizerState
    key: jax.random.PRNGKey
    update_step: jnp.ndarray


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: (num_steps, num_envs)
        values: (num_steps, num_envs)
        dones: (num_steps, num_envs)
        last_value: (num_envs,)
        gamma: discount factor
        gae_lambda: GAE lambda
        
    Returns:
        advantages: (num_steps, num_envs)
        returns: (num_steps, num_envs)
    """
    def _gae_step(carry, transition):
        gae, next_value = carry
        reward, value, done = transition
        
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        
        return (gae, value), gae
    
    # Scan backwards through transitions
    _, advantages = lax.scan(
        _gae_step,
        (jnp.zeros_like(last_value), last_value),
        (rewards[::-1], values[::-1], dones[::-1]),
    )
    advantages = advantages[::-1]
    returns = advantages + values
    
    return advantages, returns


def ppo_loss(
    params,
    apply_fn: Callable,
    batch: Transition,
    advantages: jnp.ndarray,
    clip_eps: float,
    ent_coef: float,
    vf_coef: float,
    exploration_floor: float = 0.0  # NEW: Phase B parameter
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute PPO loss with action masking.

    Returns:
        loss: scalar loss value
        metrics: dictionary of logged metrics
    """
    logits, values = apply_fn(params, batch.obs)

    # Compute log probabilities under current policy (with exploration floor)
    probs = masked_softmax(logits, batch.mask, exploration_floor=exploration_floor)
    log_probs = jnp.log(probs + 1e-8)
    new_log_prob = log_probs[jnp.arange(batch.action.shape[0]), batch.action]
    
    # Entropy bonus (masked)
    entropy = -jnp.sum(probs * log_probs * batch.mask, axis=-1)
    
    # Policy ratio
    ratio = jnp.exp(new_log_prob - batch.log_prob)
    
    # Normalize advantages
    adv_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Clipped surrogate objective
    pg_loss1 = -adv_normalized * ratio
    pg_loss2 = -adv_normalized * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
    
    # Value loss (clipped)
    value_pred_clipped = batch.value + jnp.clip(
        values - batch.value, -clip_eps, clip_eps
    )
    value_losses = (values - batch.reward) ** 2
    value_losses_clipped = (value_pred_clipped - batch.reward) ** 2
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    
    # Entropy loss
    entropy_loss = -entropy.mean()
    
    # Total loss
    total_loss = pg_loss + vf_coef * value_loss + ent_coef * entropy_loss
    
    metrics = {
        'policy_loss': pg_loss,
        'value_loss': value_loss,
        'entropy': entropy.mean(),
        'approx_kl': ((ratio - 1) - jnp.log(ratio)).mean(),
        'clip_fraction': jnp.mean(jnp.abs(ratio - 1) > clip_eps),
    }
    
    return total_loss, metrics


# =============================================================================
# Training Loop - FIXED VERSION
# =============================================================================

def create_train_state(
    key: jax.random.PRNGKey,
    obs_shape: Tuple[int, ...],
    config: PPOConfig,
    num_actions: int = 3
) -> TrainState:
    """Initialize network and optimizer."""
    network = ActorCritic(num_actions=num_actions)
    
    dummy_obs = jnp.zeros((1,) + obs_shape)
    params = network.init(key, dummy_obs)
    
    # Learning rate schedule with warmup
    num_updates = config.total_timesteps // (config.num_envs * config.num_steps)
    
    if config.anneal_lr:
        # Warmup then decay
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


def get_batched_observations(
    env_states: EnvState,
    data: MarketData,
    params: EnvParams
) -> jnp.ndarray:
    """
    FIXED: Properly compute observations for all environments.
    Uses vmap over get_observation function.
    """
    return jax.vmap(
        lambda state: get_observation(state, data, params)
    )(env_states)


# MEMORY FIX (2025-12-08): Removed env_params (arg 1) from static_argnums
# env_params changes every update due to curriculum values (training_progress,
# exploration bonuses). Keeping it static caused JAX to recompile every iteration.
@partial(jax.jit, static_argnums=(3, 4))  # Only num_steps and num_envs are truly static
def collect_rollouts(
    runner_state: RunnerState,
    env_params: EnvParams,
    data: MarketData,
    num_steps: int,
    num_envs: int,
    exploration_floor: float = 0.0  # NEW: Phase B parameter
) -> Tuple[Transition, RunnerState, jnp.ndarray]:
    """
    FIXED: Properly collect rollouts with correct observation computation.

    Returns:
        transitions: collected experience
        new_runner_state: updated state
        episode_returns: sum of rewards per episode
    """
    train_state, env_states, normalizer, key, update_step = runner_state

    def step_fn(carry, _):
        env_states, key, normalizer = carry
        key, key_step, key_action, key_reset = jax.random.split(key, 4)

        # FIXED: Proper observation computation
        obs_batch = get_batched_observations(env_states, data, env_params)

        # Normalize observations
        obs_normalized = normalize_obs(obs_batch, normalizer)

        # Get action masks
        masks = batch_action_masks(env_states)

        # Get policy outputs
        logits, values = train_state.apply_fn(train_state.params, obs_normalized)

        # Sample actions with masking (Phase B: apply exploration floor)
        key_actions = jax.random.split(key_action, num_envs)
        actions = jax.vmap(lambda k, l, m: sample_action(k, l, m, exploration_floor))(
            key_actions, logits, masks
        )

        # Compute log probs (Phase B: apply exploration floor)
        log_probs = jax.vmap(lambda l, a, m: log_prob_action(l, a, m, exploration_floor))(
            logits, actions, masks
        )
        
        # Step environments
        key_steps = jax.random.split(key_step, num_envs)
        next_obs, next_states, rewards, dones, infos = batch_step(
            key_steps, env_states, actions, env_params, data
        )
        
        # FIXED: Auto-reset done environments
        key_resets = jax.random.split(key_reset, num_envs)
        reset_obs_batch, reset_states = batch_reset(key_resets, env_params, data)
        
        # Select reset or continued state based on done
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
    
    # Run scan over steps
    (final_states, key, normalizer), transitions = lax.scan(
        step_fn,
        (env_states, key, normalizer),
        None,
        num_steps
    )
    
    # Update normalizer with collected observations
    all_obs = transitions.obs.reshape(-1, transitions.obs.shape[-1])
    new_normalizer = update_normalizer(normalizer, all_obs)
    
    # Compute last value for GAE
    final_obs = get_batched_observations(final_states, data, env_params)
    final_obs_norm = normalize_obs(final_obs, new_normalizer)
    _, last_values = train_state.apply_fn(train_state.params, final_obs_norm)
    
    # Compute advantages and returns
    advantages, returns = compute_gae(
        transitions.reward,
        transitions.value,
        transitions.done,
        last_values,
        0.99,  # gamma from config
        0.95   # gae_lambda from config
    )
    
    # FIXED: Calculate episode returns using ACTUAL rewards, not GAE returns
    # Must be done BEFORE replacing transitions.reward with GAE returns
    episode_returns = (transitions.reward * (1 - transitions.done)).sum(axis=0)
    
    # Replace value with advantages, reward with returns for loss
    transitions = transitions._replace(
        reward=returns,
        value=advantages
    )
    
    # NEW: Phase B2 - Calculate action distribution for monitoring
    action_counts = jnp.bincount(transitions.action.flatten(), length=3)
    action_dist = action_counts / transitions.action.size
    
    new_runner_state = RunnerState(
        train_state=train_state,
        env_states=final_states,
        normalizer=new_normalizer,
        key=key,
        update_step=update_step
    )
    
    return transitions, new_runner_state, episode_returns, action_dist  # Added action_dist


@partial(jax.jit, static_argnums=(3,))  # config is static for batch size computation
def train_step(
    train_state: TrainState,
    transitions: Transition,
    advantages: jnp.ndarray,
    config: PPOConfig,
    key: jax.random.PRNGKey,
    exploration_floor: float = 0.0  # NEW: Phase B parameter
) -> Tuple[TrainState, dict]:
    """
    Single PPO training step with multiple epochs and minibatches.
    """
    batch_size = config.num_envs * config.num_steps
    minibatch_size = batch_size // config.num_minibatches
    
    # Flatten transitions
    flat_transitions = jax.tree.map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]),
        transitions
    )
    flat_advantages = advantages.reshape(batch_size)
    
    def epoch_step(carry, _):
        train_state, key = carry
        key, shuffle_key = jax.random.split(key)
        
        # Shuffle
        permutation = jax.random.permutation(shuffle_key, batch_size)
        shuffled_trans = jax.tree.map(lambda x: x[permutation], flat_transitions)
        shuffled_adv = flat_advantages[permutation]
        
        # Split into minibatches
        def reshape_batch(x):
            return x.reshape((config.num_minibatches, minibatch_size) + x.shape[1:])
        
        minibatched_trans = jax.tree.map(reshape_batch, shuffled_trans)
        minibatched_adv = shuffled_adv.reshape(config.num_minibatches, minibatch_size)
        
        def minibatch_step(train_state, batch_and_adv):
            batch, adv = batch_and_adv

            loss_fn = partial(
                ppo_loss,
                apply_fn=train_state.apply_fn,
                batch=batch,
                advantages=adv,
                clip_eps=config.clip_eps,
                ent_coef=config.ent_coef,
                vf_coef=config.vf_coef,
                exploration_floor=exploration_floor  # NEW: Phase B
            )
            
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                train_state.params
            )
            train_state = train_state.apply_gradients(grads=grads)
            
            return train_state, metrics
        
        # Process minibatches
        train_state, metrics = lax.scan(
            minibatch_step,
            train_state,
            (minibatched_trans, minibatched_adv)
        )
        
        return (train_state, key), metrics
    
    # Multiple epochs
    (train_state, _), all_metrics = lax.scan(
        epoch_step,
        (train_state, key),
        None,
        config.num_epochs
    )
    
    # Average metrics
    avg_metrics = jax.tree.map(lambda x: x.mean(), all_metrics)
    
    return train_state, avg_metrics


def train(
    config: PPOConfig,
    env_params: EnvParams,
    data: MarketData,
    seed: int = 0,
    market: str = "UNKNOWN",  # NEW: For metrics tracker
    checkpoint_dir: str = "models/phase1_jax"  # NEW: For metrics saving
) -> Tuple[TrainState, NormalizerState, list]:
    """
    Main training loop - FIXED VERSION with Metrics Tracker (Phase 2).

    Returns:
        trained_state: final trained parameters
        normalizer: observation normalization stats
        metrics_history: training metrics over time
    """
    key = jax.random.key(seed)
    key, init_key, reset_key = jax.random.split(key, 3)
    
    # Derive observation shape from params
    obs_shape = (env_params.window_size * env_params.num_features + 5,)
    
    # Initialize
    train_state = create_train_state(init_key, obs_shape, config)
    normalizer = create_normalizer(obs_shape)
    
    # Initial environment states
    obs, env_states = batch_reset(
        jax.random.split(reset_key, config.num_envs),
        env_params,
        data
    )
    
    runner_state = RunnerState(
        train_state=train_state,
        env_states=env_states,
        normalizer=normalizer,
        key=key,
        update_step=jnp.array(0)
    )
    
    num_updates = config.total_timesteps // (config.num_envs * config.num_steps)

    # NEW: Initialize metrics tracker (Phase 2 integration)
    tracker = TrainingMetricsTracker(
        market=market,
        checkpoint_dir=checkpoint_dir,
        phase=1
    )
    print(f"Metrics tracker initialized: {checkpoint_dir}/training_metrics_{market}.json")
    
    # NEW: Phase C - Initialize adaptive training system
    quality_monitor = TrainingQualityMonitor(
        window_size=10,
        min_trades_per_update=config.num_envs // 100,  # 1 trade per 100 envs
        min_entropy=0.10,
        max_hold_ratio=0.95
    )
    hyperparameter_adjuster = HyperparameterAutoAdjuster(log_dir=checkpoint_dir)
    curriculum_slowdown_active = False
    print(f"Adaptive training system initialized (auto-apply enabled)")

    print(f"Starting training:")
    print(f"  Total timesteps: {config.total_timesteps:,}")
    print(f"  Num envs: {config.num_envs}")
    print(f"  Steps per rollout: {config.num_steps}")
    print(f"  Num updates: {num_updates}")
    print(f"  Observation shape: {obs_shape}")

    metrics_history = []
    start_time = time.time()
    prev_total_trades = 0  # Track trades between updates for delta calculation
    prev_total_winning = 0  # Track winning trades for delta calculation
    prev_avg_balance = 10000.0  # Track average balance for delta calculation (starts at initial_balance)
    
    
    for update in range(num_updates):
        key, rollout_key, train_key = jax.random.split(runner_state.key, 3)
        
        # NEW: Phase A1 - Calculate training progress for commission curriculum
        base_progress = float(update) / num_updates
        
        # NEW: Phase C - Apply curriculum slowdown if quality is poor
        if curriculum_slowdown_active:
            progress = base_progress * 0.5  # Slow down curriculum by 50%
        else:
            progress = base_progress
        
        # Update environment params with current training progress
        # NEW: Phase 1B - Also update global timestep for exploration bonus
        current_global_timestep = float(update * config.num_envs * config.num_steps)
        env_params = env_params._replace(
            training_progress=progress,
            current_global_timestep=current_global_timestep,
            total_training_timesteps=float(config.total_timesteps)
        )
        
        # Update runner state key
        runner_state = runner_state._replace(
            key=key,
            update_step=jnp.array(update)
        )

        # PHASE B: Calculate exploration floor (decays over training)
        # Guarantees minimum 8% BUY and 8% SELL probabilities
        # FIXED: Decays over 100% of training (was 40%) and stops at 5% (was 0%)
        exploration_floor_horizon = config.total_timesteps * 1.0
        floor_progress = min(1.0, current_global_timestep / exploration_floor_horizon)
        base_floor = 0.08    # Start at 8%
        min_floor = 0.05     # Never go below 5%
        current_floor = min_floor + (base_floor - min_floor) * max(0.0, 1.0 - floor_progress)

        # Collect rollouts with updated params (includes curriculum commission + exploration floor)
        transitions, runner_state, episode_returns, action_dist = collect_rollouts(
            runner_state, env_params, data, config.num_steps, config.num_envs,
            exploration_floor=current_floor  # NEW: Phase B
        )

        # Extract advantages for training
        advantages = transitions.value  # GAE stored in value field
        
        # NEW: Update metrics tracker with DELTA stats from environment states
        # CRITICAL FIX v2: Use AVERAGE balance delta per env, not total PnL sum
        final_env_states = runner_state.env_states  # Shape: (num_envs,)
        
        # Extract current statistics
        current_total_trades = int(final_env_states.num_trades.sum())
        current_total_winning = int(final_env_states.winning_trades.sum())
        avg_balance = float(final_env_states.balance.mean())  # Average balance per env
        
        # Calculate DELTAS (changes since last update)
        if update == 0:
            # First update - all current values are deltas
            new_trades_this_update = current_total_trades
            new_wins_this_update = current_total_winning
            avg_pnl_delta = 0.0  # No PnL delta on first update
            prev_total_trades = 0
            prev_total_winning = 0
            prev_avg_balance = avg_balance
        else:
            # Calculate deltas from previous update
            new_trades_this_update = current_total_trades - prev_total_trades
            new_wins_this_update = current_total_winning - prev_total_winning
            # FIXED: Use average balance change per env (realistic $50-$500 range)
            avg_pnl_delta = avg_balance - prev_avg_balance
        
        # Calculate win rate for NEW trades this update only
        win_rate_this_update = new_wins_this_update / max(new_trades_this_update, 1) if new_trades_this_update > 0 else 0.0
        
        # Update tracker with DELTA stats (per-env averages, not 384x sums)
        tracker.record_episode(
            final_balance=avg_balance,
            num_trades=new_trades_this_update,
            win_rate=win_rate_this_update,
            total_pnl=avg_pnl_delta  # Average PnL change per env this update
        )
        
        # Store current values for next delta calculation
        prev_total_trades = current_total_trades
        prev_total_winning = current_total_winning
        prev_avg_balance = avg_balance

        new_train_state, train_metrics = train_step(
            runner_state.train_state,
            transitions,
            advantages,
            config,
            train_key,
            exploration_floor=current_floor  # NEW: Phase B
        )
        
        runner_state = runner_state._replace(train_state=new_train_state)
        
        # Block for accurate timing
        jax.block_until_ready(runner_state.train_state.params)
        
        # Logging + telemetry snapshot
        elapsed = time.time() - start_time
        timesteps = (update + 1) * config.num_envs * config.num_steps
        # Calculate SPS with zero-check for elapsed time
        # When training fails immediately, elapsed can be 0 or very small
        sps = timesteps / elapsed if elapsed > 0 else 0.0
        current_entropy = float(train_metrics['entropy'])
        current_kl = float(train_metrics['approx_kl'])

        tracker.record_update(
            timesteps=timesteps,
            sps=sps,
            entropy=current_entropy,
            approx_kl=current_kl,
            action_dist=action_dist,
            policy_loss=float(train_metrics['policy_loss']),
            value_loss=float(train_metrics['value_loss']),
            win_rate=win_rate_this_update,
            mean_return=float(episode_returns.mean()),
            learning_rate=float(config.learning_rate),
            timesteps_target=config.total_timesteps
        )

        # Logging (every 10 updates)
        if (update + 1) % 10 == 0:
            # Calculate current commission for curriculum tracking
            current_commission = env_params.initial_commission + \
                (env_params.final_commission - env_params.initial_commission) * min(1.0, progress * 2.0)
            
            # NEW: Phase 1B - Calculate exploration bonus for monitoring
            # Updated to match env_phase1_jax.py ACTUAL values
            exploration_horizon = config.total_timesteps * 1.0  # Decays over full run
            exploration_progress = current_global_timestep / exploration_horizon
            current_exploration_bonus = 100.0 * max(0.0, 1.0 - exploration_progress)  # Match env base_bonus=100
            
            print(f"Update {update + 1}/{num_updates}")
            print(f"  Timesteps: {timesteps:,}")
            print(f"  SPS: {sps:,.0f}")
            print(f"  Progress: {progress:.1%} | Commission: ${current_commission:.2f}")
            if current_exploration_bonus > 0.0:
                print(f"  🎯 Exploration Bonus: ${current_exploration_bonus:.2f} (decays to $0 at {exploration_horizon:,.0f} steps)")
            if current_floor > 0.0:
                print(f"  🎯 Action Floor: {current_floor:.1%} min probability for BUY/SELL (guarantees ~{current_floor*2:.0%} trading)")
            print(f"  Mean episode return: {episode_returns.mean():.2f}")
            print(f"  Policy loss: {train_metrics['policy_loss']:.4f}")
            print(f"  Value loss: {train_metrics['value_loss']:.4f}")
            print(f"  Entropy: {train_metrics['entropy']:.4f}")

            # PHASE A: Earlier entropy intervention with higher max coefficient
            # Trigger threshold: 0.15 → 0.25 (earlier detection)
            # Max ent_coef: 0.30 → 0.50 (stronger intervention)
            if current_entropy < 0.25:
                # Entropy collapsing - increase coefficient more aggressively
                new_ent_coef = min(config.ent_coef * 1.5, 0.50)
                config = config._replace(ent_coef=new_ent_coef)
                print(f"  ⚠️  Entropy low ({current_entropy:.3f}) - increasing ent_coef to {new_ent_coef:.3f}")
            elif current_entropy > 0.40:
                # Entropy too high (unfocused policy) - decrease
                new_ent_coef = max(config.ent_coef * 0.95, 0.05)
                config = config._replace(ent_coef=new_ent_coef)
                print(f"  ℹ️  Entropy high ({current_entropy:.3f}) - decreasing ent_coef to {new_ent_coef:.3f}")

            print(f"  Approx KL: {train_metrics['approx_kl']:.4f}")
            
            # NEW: Phase B2 - Action distribution monitoring
            print(f"  Action dist: HOLD {action_dist[0]:.1%} | BUY {action_dist[1]:.1%} | SELL {action_dist[2]:.1%}")
            if action_dist[0] > 0.95:
                print(f"  ⚠️  HOLD-dominated policy detected! Consider increasing exploration.")

            # NEW: Log metrics tracker summary (Phase 2 integration)
            tracker.log_summary()
            
            # NEW: Phase C - Quality monitoring and adaptive adjustments (every 20 updates)
            if (update + 1) % 20 == 0:
                # Update quality monitor with current metrics
                quality_metrics = {
                    'entropy': current_entropy,
                    'trades': tracker.trades_last_update if hasattr(tracker, 'trades_last_update') else 0,
                    'win_rate': tracker.win_rate if hasattr(tracker, 'win_rate') else 0.0,
                    'mean_return': float(episode_returns.mean()),
                    'action_dist': action_dist
                }
                quality_score = quality_monitor.update(quality_metrics)
                
                print(f"\n🔍 Quality Check (Update {update + 1}):")
                print(f"  Quality Score: {quality_score:.2f}/1.00")
                
                # Check if adaptive intervention needed
                if quality_monitor.should_adjust():
                    recommendations = quality_monitor.get_recommendations()
                    print(f"  ⚙️  Auto-adjustments triggered:")
                    for rec in recommendations:
                        print(f"     • {rec.replace('_', ' ').title()}")
                    
                    # Apply adjustments (AUTO-APPLY enabled per user request)
                    result = hyperparameter_adjuster.apply(
                        config, env_params, recommendations, update + 1
                    )
                    config = result['config']
                    env_params = result['env_params']
                    curriculum_slowdown_active = result['curriculum_slowdown']
                    
                    # Log applied adjustments
                    for adjustment in result['adjustments']:
                        print(f"     ✓ {adjustment}")
                    
                    if curriculum_slowdown_active:
                        print(f"     ✓ Commission curriculum slowed (0.5x progress multiplier)")
                else:
                    status = quality_monitor.get_status_summary()
                    print(f"  ✅ Training healthy - no adjustments needed")
                    print(f"     Avg entropy: {status['avg_entropy']:.3f} | Avg trades: {status['avg_trades']:.1f}")
                print()  # Blank line for readability

        # Persist metrics frequently for dashboard (every 5 updates)
        if (update + 1) % 5 == 0 or (update + 1) == num_updates:
            tracker.save_metrics()

        metrics_history.append({
            'update': update + 1,
            'timesteps': timesteps,
            'sps': sps,
            'mean_return': float(episode_returns.mean()),
            **{k: float(v) for k, v in train_metrics.items()}
        })
        
        # MEMORY SAFETY: Limit metrics history to prevent unbounded growth
        if len(metrics_history) > 200:
            metrics_history = metrics_history[-200:]
        
        # MEMORY SAFETY: Clear JAX caches every 500 updates as safety net
        if (update + 1) % 500 == 0:
            jax.clear_caches()
            print(f"  [MEMORY] Cleared JAX caches at update {update + 1}")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Final SPS: {config.total_timesteps / total_time:,.0f}")
    
    return runner_state.train_state, runner_state.normalizer, metrics_history


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import pickle
    from flax.training import checkpoints
    
    parser = argparse.ArgumentParser(description="JAX PPO Training - Phase 1")
    parser.add_argument("--market", type=str, required=True, help="Market symbol (e.g., NQ, ES)")
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel environments")
    parser.add_argument("--total_timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--data_path", type=str, required=True, help="Path to market data CSV file")
    parser.add_argument("--checkpoint_dir", type=str, default="models/phase1_jax", help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # NEW: Hyperparameter arguments (Phase 1B improvements)
    parser.add_argument('--ent_coef', type=float, default=0.15,
                       help='Entropy coefficient (default: 0.15, was 0.05)')
    parser.add_argument('--initial_lr', type=float, default=3e-4,
                       help='Initial learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                       help='Final learning rate (for annealing)')
    parser.add_argument('--lr_annealing', action='store_true',
                       help='Enable learning rate annealing')
    parser.add_argument('--data_filter', type=str, default=None,
                       choices=['high_volatility', 'trending', 'ranging'],
                       help='Filter training data by market conditions')
    parser.add_argument('--hardware_profile', type=str, default=None,
                       help='Path to hardware profile YAML file')

    args = parser.parse_args()

    # FIX: Ensure checkpoint directory matches convention (models/phase1_jax_{market})
    if args.checkpoint_dir == "models/phase1_jax":
        args.checkpoint_dir = f"models/phase1_jax_{args.market.lower()}"
        print(f"[CONFIG] Auto-updated checkpoint dir to: {args.checkpoint_dir}")
    else:
        print(f"[CONFIG] Using specified checkpoint dir: {args.checkpoint_dir}")

    # Load hardware profile if provided
    if args.hardware_profile:
        import yaml
        try:
            with open(args.hardware_profile, 'r') as f:
                profile = yaml.safe_load(f)
                print(f"Loading hardware profile from {args.hardware_profile}")
        except FileNotFoundError:
            print(f"Error: Hardware profile file not found at {args.hardware_profile}")
            exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing hardware profile {args.hardware_profile}: {e}")
            exit(1)
    
    # Validate arguments
    if args.num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {args.num_envs}")
    if args.total_timesteps <= 0:
        raise ValueError(f"total_timesteps must be positive, got {args.total_timesteps}")
    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    print("JAX PPO Training Script - FIXED VERSION")
    print(f"JAX devices: {jax.devices()}")
    print(f"Market: {args.market}")
    print(f"Data path: {args.data_path}")
    print()
    
    # Load real market data
    from .data_loader import load_market_data

    # Infer second-level data path from minute data path
    data_path = Path(args.data_path)
    second_data_path = data_path.parent / data_path.name.replace('_D1M.csv', '_D1S.csv')

    # TODO (Phase 3): Add data filtering support via data_filter.py
    # For now, --data_filter argument is accepted but not yet implemented
    # Future: Filter by high_volatility, trending, or ranging markets
    if args.data_filter:
        print(f"Note: Data filtering ({args.data_filter}) is not yet implemented in JAX pipeline")
        print(f"  This will be added in a future update")

    data = load_market_data(
        args.data_path,
        second_data_path=str(second_data_path) if second_data_path.exists() else None
    )
    
    # Configuration with command-line arguments (with Phase 1B improvements)
    config = PPOConfig(
        num_envs=args.num_envs,
        num_steps=128,
        total_timesteps=args.total_timesteps,
        normalize_obs=True,
        ent_coef=args.ent_coef,  # NEW: configurable entropy (default 0.15)
        learning_rate=args.initial_lr,  # NEW: use initial_lr
        anneal_lr=args.lr_annealing,  # NEW: enable LR annealing if flag set
    )
    
    # FIX (2025-12-10): Use market-specific contract values instead of ES defaults
    # Previously used EnvParams() which defaults to ES (contract_size=50.0)
    # NQ should use $20/point, not $50/point - losses were calculated 2.5x too large!
    market_spec = get_market_spec(args.market)
    if market_spec:
        print(f"Using {args.market} contract specs: ${market_spec.contract_multiplier}/point")
        env_params = EnvParams(
            contract_size=market_spec.contract_multiplier,
            tick_size=market_spec.tick_size,
            tick_value=market_spec.tick_value,
            commission=market_spec.commission,
            slippage_ticks=market_spec.slippage_ticks,
            rth_start_count=int(data.rth_indices.shape[0])
        )
    else:
        print(f"Warning: Unknown market {args.market}, using ES defaults")
        env_params = EnvParams(
            rth_start_count=int(data.rth_indices.shape[0])
        )
    
    # Train with real data (with Phase 2 metrics tracker integration)
    trained_state, normalizer, metrics = train(
        config, env_params, data,
        seed=args.seed,
        market=args.market,
        checkpoint_dir=str(checkpoint_dir)
    )
    
    print(f"\nFinal metrics:")
    if metrics:
        final = metrics[-1]
        print(f"  Mean return: {final['mean_return']:.2f}")
        print(f"  SPS: {final['sps']:,.0f}")
    
    # ========================================================================
    # SAVE CHECKPOINTS
    # ========================================================================
    print(f"\nSaving checkpoints to {checkpoint_dir}...")
    
    # Save final checkpoint using Flax checkpoints module
    checkpoints.save_checkpoint(
        ckpt_dir=str(checkpoint_dir),
        target=trained_state,
        step=config.total_timesteps,
        prefix="phase1_jax_final_",
        keep=1,
        overwrite=True
    )
    print(f"  ✓ Saved final model checkpoint")
    
    # Save normalizer separately
    normalizer_path = checkpoint_dir / "normalizer_final.pkl"
    with open(normalizer_path, "wb") as f:
        pickle.dump(normalizer, f)
    print(f"  ✓ Saved normalizer to {normalizer_path.name}")
    
    # Save training metrics
    metrics_path = checkpoint_dir / f"training_metrics_{args.market}.json"
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Saved training metrics to {metrics_path.name}")
    
    # Save metadata
    metadata_path = checkpoint_dir / "metadata.json"
    metadata = {
        "market": args.market,
        "total_timesteps": args.total_timesteps,
        "num_envs": args.num_envs,
        "final_mean_return": float(metrics[-1]['mean_return']) if metrics else 0.0,
        "phase": 1,
        "observation_shape": (env_params.window_size * env_params.num_features + 5,),
        "num_actions": 3
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata to {metadata_path.name}")
    
    print(f"\n✅ Phase 1 training complete! Models saved in {checkpoint_dir}/")
