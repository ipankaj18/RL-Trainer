"""
JAX Migration Module for RL Trader System

This module provides pure JAX implementations of the trading environment
for GPU-accelerated training with 10k+ parallel environments.

Main Components:
- data_loader: Convert Pandas DataFrames to GPU-resident JAX arrays
- env_phase1_jax: Phase 1 environment (entry timing)
- env_phase2_jax: Phase 2 environment (position management)
- train_ppo_jax_fixed: PPO training loop with fixes
- test_validation: Validation and benchmark suite

Usage:
    from jax_migration import (
        MarketData, load_market_data,
        EnvParams, EnvState, reset, step, action_masks,
        batch_reset, batch_step, batch_action_masks,
        PPOConfig, train
    )
    
    # Load data
    data = load_market_data("data/ES_D1M.csv")
    
    # Create environment
    params = EnvParams()
    obs, state = reset(key, params, data)
    
    # Train
    config = PPOConfig(num_envs=2048, total_timesteps=2_000_000)
    trained_state, normalizer, metrics = train(config, params, data)
"""

# Data loading
from .data_loader import (
    MarketData,
    load_market_data,
    load_all_markets,
    create_batched_data,
    precompute_time_features,
    compute_trading_mask,
)

# Phase 1 Environment
from .env_phase1_jax import (
    EnvState,
    EnvParams,
    reset,
    step,
    action_masks,
    get_observation,
    batch_reset,
    batch_step,
    batch_action_masks,
    calculate_sl_tp,
    calculate_pnl,
    calculate_reward,
    ACTION_HOLD,
    ACTION_BUY,
    ACTION_SELL,
)

# Phase 2 Environment (when available)
try:
    from .env_phase2_jax import (
        EnvStatePhase2,
        EnvParamsPhase2,
        reset_phase2,
        step_phase2,
        action_masks_phase2,
        get_observation_phase2,
        batch_reset_phase2,
        batch_step_phase2,
        batch_action_masks_phase2,
        ACTION_MOVE_SL_TO_BE,
        ACTION_ENABLE_TRAIL,
        ACTION_DISABLE_TRAIL,
    )
    _PHASE2_AVAILABLE = True
except ImportError:
    _PHASE2_AVAILABLE = False

# Training (fixed version)
try:
    from .train_ppo_jax_fixed import (
        PPOConfig,
        ActorCritic,
        Transition,
        RunnerState,
        NormalizerState,
        create_train_state,
        create_normalizer,
        update_normalizer,
        normalize_obs,
        collect_rollouts,
        train_step,
        train,
        compute_gae,
        ppo_loss,
        masked_softmax,
        sample_action,
        log_prob_action,
    )
    _TRAINING_AVAILABLE = True
except ImportError:
    _TRAINING_AVAILABLE = False

# Version info
__version__ = "0.2.0"
__author__ = "JVLora"

# Feature flags
PHASE1_READY = True
PHASE2_READY = _PHASE2_AVAILABLE
TRAINING_READY = _TRAINING_AVAILABLE

def check_status():
    """Print module status and available features."""
    import jax
    
    print(f"JAX Migration Module v{__version__}")
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()
    print("Feature Status:")
    print(f"  Phase 1 Environment: {'✓ Ready' if PHASE1_READY else '✗ Not available'}")
    print(f"  Phase 2 Environment: {'✓ Ready' if PHASE2_READY else '✗ Not available'}")
    print(f"  Training Pipeline:   {'✓ Ready' if TRAINING_READY else '✗ Not available'}")
    
    if TRAINING_READY:
        print()
        print("Quick Start:")
        print("  from jax_migration import load_market_data, EnvParams, PPOConfig, train")
        print("  data = load_market_data('data/ES_D1M.csv')")
        print("  config = PPOConfig(num_envs=1024, total_timesteps=500_000)")
        print("  trained_state, normalizer, metrics = train(config, EnvParams(), data)")


__all__ = [
    # Data
    "MarketData",
    "load_market_data",
    "load_all_markets",
    "create_batched_data",
    # Phase 1
    "EnvState",
    "EnvParams",
    "reset",
    "step",
    "action_masks",
    "get_observation",
    "batch_reset",
    "batch_step",
    "batch_action_masks",
    # Training
    "PPOConfig",
    "train",
    "ActorCritic",
    # Utilities
    "check_status",
    "__version__",
]
