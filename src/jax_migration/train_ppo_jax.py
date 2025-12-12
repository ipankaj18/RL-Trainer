"""
⚠️ DEPRECATED - 2025-12-08

This file has been superseded by train_ppo_jax_fixed.py.
DO NOT USE THIS FILE FOR TRAINING.

This file is kept for historical reference only.
Use train_ppo_jax_fixed.py for Phase 1 JAX training.

Issues with this file:
- Missing observation normalization
- Missing curriculum learning
- Missing exploration bonus
- Missing proper metrics tracking

PureJAX PPO Training Loop (LEGACY)

End-to-end GPU training using JAX with no CPU-GPU data movement.
Implements PPO with action masking for the trading environment.

Based on PureJaxRL patterns with custom modifications for trading.
"""

import warnings
warnings.warn(
    "train_ppo_jax.py is DEPRECATED. Use train_ppo_jax_fixed.py instead.",
    DeprecationWarning,
    stacklevel=2
)

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, NamedTuple, Callable, Any
from functools import partial
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import time

from env_phase1_jax import (
    EnvState, EnvParams, MarketData,
    reset, step, action_masks, batch_reset, batch_step
)


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
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Actor head (policy logits)
        actor = nn.Dense(self.hidden_dim // 2)(x)
        actor = nn.relu(actor)
        logits = nn.Dense(self.num_actions)(actor)
        
        # Critic head (value function)
        critic = nn.Dense(self.hidden_dim // 2)(x)
        critic = nn.relu(critic)
        value = nn.Dense(1)(critic)
        
        return logits, value.squeeze(-1)


def masked_softmax(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Apply mask to logits and compute softmax."""
    # Set invalid actions to large negative number
    masked_logits = jnp.where(mask, logits, -1e10)
    return jax.nn.softmax(masked_logits, axis=-1)


def sample_action(key: jax.random.PRNGKey, logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Sample action from masked categorical distribution."""
    probs = masked_softmax(logits, mask)
    return jax.random.categorical(key, jnp.log(probs + 1e-8))


def log_prob(logits: jnp.ndarray, action: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Compute log probability of action under masked distribution."""
    probs = masked_softmax(logits, mask)
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
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    total_timesteps: int = 2_000_000


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
    clip_eps: float,
    ent_coef: float,
    vf_coef: float
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute PPO loss with action masking.
    
    Returns:
        loss: scalar loss value
        metrics: dictionary of logged metrics
    """
    logits, values = apply_fn(params, batch.obs)
    
    # Compute log probabilities under current policy
    probs = masked_softmax(logits, batch.mask)
    log_probs = jnp.log(probs + 1e-8)
    new_log_prob = log_probs[jnp.arange(batch.action.shape[0]), batch.action]
    
    # Entropy bonus
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    
    # Policy ratio
    ratio = jnp.exp(new_log_prob - batch.log_prob)
    
    # Normalize advantages
    advantages = batch.reward  # Actually returns from compute_gae
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Clipped surrogate objective
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
    
    # Value loss
    value_loss = 0.5 * ((values - batch.value) ** 2).mean()
    
    # Entropy loss
    entropy_loss = -entropy.mean()
    
    # Total loss
    total_loss = pg_loss + vf_coef * value_loss + ent_coef * entropy_loss
    
    metrics = {
        'policy_loss': pg_loss,
        'value_loss': value_loss,
        'entropy': entropy.mean(),
        'approx_kl': ((ratio - 1) - jnp.log(ratio)).mean(),
    }
    
    return total_loss, metrics


# =============================================================================
# Training Loop
# =============================================================================

def create_train_state(
    key: jax.random.PRNGKey,
    obs_shape: Tuple[int, ...],
    config: PPOConfig
) -> TrainState:
    """Initialize network and optimizer."""
    network = ActorCritic()
    
    dummy_obs = jnp.zeros((1,) + obs_shape)
    params = network.init(key, dummy_obs)
    
    # Learning rate schedule
    if config.anneal_lr:
        num_updates = config.total_timesteps // (config.num_envs * config.num_steps)
        schedule = optax.linear_schedule(
            init_value=config.learning_rate,
            end_value=0.0,
            transition_steps=num_updates
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


def train_step(
    train_state: TrainState,
    transitions: Transition,
    config: PPOConfig,
    key: jax.random.PRNGKey
) -> Tuple[TrainState, dict]:
    """
    Single PPO training step.
    
    Processes collected transitions through multiple epochs and minibatches.
    """
    # Flatten batch
    batch_size = config.num_envs * config.num_steps
    minibatch_size = batch_size // config.num_minibatches
    
    # Flatten transitions
    flat_transitions = jax.tree.map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]),
        transitions
    )
    
    def epoch_step(carry, _):
        train_state, key = carry
        key, shuffle_key = jax.random.split(key)
        
        # Shuffle and split into minibatches
        permutation = jax.random.permutation(shuffle_key, batch_size)
        shuffled = jax.tree.map(lambda x: x[permutation], flat_transitions)
        
        # Process minibatches
        minibatches = jax.tree.map(
            lambda x: x.reshape((config.num_minibatches, minibatch_size) + x.shape[1:]),
            shuffled
        )
        
        def minibatch_step(train_state, batch):
            loss_fn = partial(
                ppo_loss,
                apply_fn=train_state.apply_fn,
                batch=batch,
                clip_eps=config.clip_eps,
                ent_coef=config.ent_coef,
                vf_coef=config.vf_coef
            )
            
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                train_state.params
            )
            train_state = train_state.apply_gradients(grads=grads)
            
            return train_state, metrics
        
        train_state, metrics = lax.scan(minibatch_step, train_state, minibatches)
        
        return (train_state, key), metrics
    
    # Multiple epochs
    (train_state, _), all_metrics = lax.scan(
        epoch_step,
        (train_state, key),
        None,
        config.num_epochs
    )
    
    # Average metrics across epochs and minibatches
    avg_metrics = jax.tree.map(lambda x: x.mean(), all_metrics)
    
    return train_state, avg_metrics


def collect_rollouts(
    key: jax.random.PRNGKey,
    train_state: TrainState,
    env_states: EnvState,
    env_params: EnvParams,
    data: MarketData,
    config: PPOConfig
) -> Tuple[Transition, EnvState, jnp.ndarray]:
    """
    Collect rollouts from vectorized environments.
    
    Returns:
        transitions: collected experience
        final_states: environment states after collection
        total_rewards: sum of rewards per environment
    """
    
    def step_fn(carry, _):
        env_states, key = carry
        key, key_step, key_action = jax.random.split(key, 3)
        
        # Get observations and masks
        obs_batch = jax.vmap(
            lambda s: jnp.concatenate([
                data.features[s.step_idx - env_params.window_size:s.step_idx].flatten(),
                data.time_features[s.step_idx - env_params.window_size:s.step_idx].flatten(),
                jnp.array([
                    s.position.astype(jnp.float32),
                    jnp.where(s.position != 0, s.entry_price / data.prices[s.step_idx, 3], 1.0),
                    jnp.where(s.position != 0, jnp.abs(s.sl_price - data.prices[s.step_idx, 3]) / data.atr[s.step_idx], 0.0),
                    jnp.where(s.position != 0, jnp.abs(s.tp_price - data.prices[s.step_idx, 3]) / data.atr[s.step_idx], 0.0),
                    jnp.where(s.position != 0, (s.step_idx - s.position_entry_step).astype(jnp.float32) / 390.0, 0.0),
                ])
            ])
        )(env_states)
        
        # Simple observation construction for vectorized case
        # Using a simplified version - in production, use proper get_observation
        masks = jax.vmap(action_masks)(env_states)
        
        # Get policy outputs
        logits, values = train_state.apply_fn(train_state.params, obs_batch)
        
        # Sample actions
        key_actions = jax.random.split(key_action, config.num_envs)
        actions = jax.vmap(sample_action)(key_actions, logits, masks)
        
        # Compute log probs
        log_probs = jax.vmap(log_prob)(logits, actions, masks)
        
        # Step environments
        key_steps = jax.random.split(key_step, config.num_envs)
        next_obs, next_states, rewards, dones, infos = batch_step(
            key_steps, env_states, actions, env_params, data
        )
        
        # Auto-reset done environments
        key_resets = jax.random.split(key, config.num_envs)
        reset_obs, reset_states = batch_reset(
            jax.random.split(key, config.num_envs),
            env_params,
            data
        )
        
        final_states = jax.tree.map(
            lambda x, y: jnp.where(dones[:, None] if x.ndim > 1 else dones, x, y),
            reset_states, next_states
        )
        
        transition = Transition(
            obs=obs_batch,
            action=actions,
            reward=rewards,
            done=dones,
            value=values,
            log_prob=log_probs,
            mask=masks
        )
        
        return (final_states, key), transition
    
    (final_states, _), transitions = lax.scan(
        step_fn,
        (env_states, key),
        None,
        config.num_steps
    )
    
    # Compute last value for GAE
    # (simplified - in production use proper observation)
    final_obs = jnp.zeros((config.num_envs, 225))  # Placeholder
    _, last_values = train_state.apply_fn(train_state.params, final_obs)
    
    # Compute advantages and returns
    advantages, returns = compute_gae(
        transitions.reward,
        transitions.value,
        transitions.done,
        last_values,
        config.gamma,
        config.gae_lambda
    )
    
    # Replace reward with returns for loss computation
    transitions = transitions._replace(
        reward=returns,
        value=advantages  # Store advantages in value field for loss
    )
    
    total_rewards = transitions.reward.sum(axis=0)
    
    return transitions, final_states, total_rewards


def train(
    config: PPOConfig,
    env_params: EnvParams,
    data: MarketData,
    seed: int = 0
) -> Tuple[TrainState, dict]:
    """
    Main training loop.
    
    Returns:
        trained_state: final trained parameters
        metrics_history: training metrics over time
    """
    key = jax.random.key(seed)
    key, init_key, reset_key = jax.random.split(key, 3)
    
    # Initialize
    obs_shape = (225,)  # Phase 1 observation shape
    train_state = create_train_state(init_key, obs_shape, config)
    
    # Initial environment states
    obs, env_states = batch_reset(
        jax.random.split(reset_key, config.num_envs),
        env_params,
        data
    )
    
    num_updates = config.total_timesteps // (config.num_envs * config.num_steps)
    
    print(f"Starting training:")
    print(f"  Total timesteps: {config.total_timesteps:,}")
    print(f"  Num envs: {config.num_envs}")
    print(f"  Steps per rollout: {config.num_steps}")
    print(f"  Num updates: {num_updates}")
    
    metrics_history = []
    start_time = time.time()
    
    for update in range(num_updates):
        key, rollout_key, train_key = jax.random.split(key, 3)
        
        # Collect rollouts
        transitions, env_states, episode_rewards = collect_rollouts(
            rollout_key, train_state, env_states, env_params, data, config
        )
        
        # Train on collected data
        train_state, train_metrics = train_step(
            train_state, transitions, config, train_key
        )
        
        # Block for accurate timing
        jax.block_until_ready(train_state.params)
        
        # Logging
        if (update + 1) % 10 == 0:
            elapsed = time.time() - start_time
            timesteps = (update + 1) * config.num_envs * config.num_steps
            sps = timesteps / elapsed
            
            print(f"Update {update + 1}/{num_updates}")
            print(f"  Timesteps: {timesteps:,}")
            print(f"  SPS: {sps:,.0f}")
            print(f"  Mean reward: {episode_rewards.mean():.2f}")
            print(f"  Policy loss: {train_metrics['policy_loss']:.4f}")
            print(f"  Value loss: {train_metrics['value_loss']:.4f}")
            print(f"  Entropy: {train_metrics['entropy']:.4f}")
            
            metrics_history.append({
                'update': update + 1,
                'timesteps': timesteps,
                'sps': sps,
                'mean_reward': float(episode_rewards.mean()),
                **{k: float(v) for k, v in train_metrics.items()}
            })
    
    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Final SPS: {config.total_timesteps / total_time:,.0f}")
    
    return train_state, metrics_history


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("JAX PPO Training Script")
    print(f"JAX devices: {jax.devices()}")
    
    # Create dummy data for testing
    num_timesteps = 50000
    num_features = 8
    
    key = jax.random.key(0)
    dummy_data = MarketData(
        features=jax.random.normal(key, (num_timesteps, num_features)),
        prices=jnp.abs(jax.random.normal(key, (num_timesteps, 4))) * 6000 + 100,
        atr=jnp.abs(jax.random.normal(key, (num_timesteps,))) * 10 + 1,
        time_features=jax.random.uniform(key, (num_timesteps, 3)),
        trading_mask=jnp.ones(num_timesteps),
        timestamps_hour=jnp.linspace(9.5, 16.9, num_timesteps),
        rth_indices=jnp.arange(60, num_timesteps - 100),  # Valid RTH start indices
        low_s=jnp.abs(jax.random.normal(key, (num_timesteps,))) * 6000 + 90,
        high_s=jnp.abs(jax.random.normal(key, (num_timesteps,))) * 6000 + 110,
    )
    
    # Configuration
    config = PPOConfig(
        num_envs=1024,
        num_steps=128,
        total_timesteps=500_000,  # Reduced for testing
    )
    
    env_params = EnvParams(
        rth_start_count=int(data.rth_indices.shape[0])
    )
    
    # Train
    trained_state, metrics = train(config, env_params, dummy_data, seed=42)
    
    print(f"\nFinal metrics:")
    if metrics:
        final = metrics[-1]
        print(f"  Mean reward: {final['mean_reward']:.2f}")
        print(f"  SPS: {final['sps']:,.0f}")
