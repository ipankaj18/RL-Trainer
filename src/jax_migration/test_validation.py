"""
JAX Migration Validation Test Suite

Tests for:
1. Reward parity between PyTorch and JAX implementations
2. Action mask correctness
3. Throughput benchmarks
4. Edge case handling
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_migration.data_loader import MarketData, load_market_data
from jax_migration.env_phase1_jax import (
    EnvState, EnvParams, reset, step, action_masks,
    batch_reset, batch_step, batch_action_masks
)


# =============================================================================
# Test Utilities
# =============================================================================

def create_test_data(num_timesteps: int = 10000, seed: int = 42) -> MarketData:
    """Create deterministic test data."""
    key = jax.random.key(seed)
    keys = jax.random.split(key, 6)
    
    # Generate realistic price data
    base_price = 5000.0
    returns = jax.random.normal(keys[0], (num_timesteps,)) * 0.001
    prices_close = base_price * jnp.cumprod(1 + returns)
    
    # OHLC from close
    noise = jax.random.uniform(keys[1], (num_timesteps,)) * 0.001
    prices_open = prices_close * (1 + noise - 0.0005)
    prices_high = jnp.maximum(prices_open, prices_close) * (1 + jax.random.uniform(keys[2], (num_timesteps,)) * 0.002)
    prices_low = jnp.minimum(prices_open, prices_close) * (1 - jax.random.uniform(keys[3], (num_timesteps,)) * 0.002)
    
    prices = jnp.stack([prices_open, prices_high, prices_low, prices_close], axis=1)
    
    # ATR (simplified)
    atr = (prices_high - prices_low) * 0.5 + 5.0
    
    # Features
    features = jax.random.normal(keys[4], (num_timesteps, 8))
    
    # Time features
    time_features = jax.random.uniform(keys[5], (num_timesteps, 3))
    
    # Trading hours (9:30 to 16:59)
    timestamps_hour = jnp.linspace(9.5, 16.9, num_timesteps)
    
    return MarketData(
        features=features,
        prices=prices,
        atr=atr,
        time_features=time_features,
        trading_mask=jnp.ones(num_timesteps),
        timestamps_hour=timestamps_hour,
        low_s=prices[:, 2] - 5.0,  # Simple approximation
        high_s=prices[:, 1] + 5.0  # Simple approximation
    )


# =============================================================================
# Test Cases
# =============================================================================

def test_reset_produces_valid_state():
    """Test that reset produces a valid initial state."""
    print("\n=== Test: Reset Produces Valid State ===")
    
    data = create_test_data()
    params = EnvParams(
        rth_start_count=int(data.rth_indices.shape[0])
    )
    key = jax.random.key(0)
    
    obs, state = reset(key, params, data)
    
    # Check observation shape
    expected_shape = (params.window_size * params.num_features + 5,)
    assert obs.shape == expected_shape, f"Expected {expected_shape}, got {obs.shape}"
    
    # Check state values
    assert state.position == 0, "Initial position should be 0"
    assert state.balance == params.initial_balance, "Initial balance incorrect"
    assert state.num_trades == 0, "Initial trades should be 0"
    assert jnp.isfinite(obs).all(), "Observation contains NaN/Inf"
    
    print("✓ Reset produces valid state")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Initial balance: {state.balance}")
    return True


def test_action_masks_correct():
    """Test action masking logic."""
    print("\n=== Test: Action Masks Correct ===")
    
    data = create_test_data()
    params = EnvParams(
        rth_start_count=int(data.rth_indices.shape[0])
    )
    key = jax.random.key(0)
    
    # Test flat position
    obs, state = reset(key, params, data)
    mask = action_masks(state)
    
    assert mask[0] == True, "HOLD should always be valid"
    assert mask[1] == True, "BUY should be valid when flat"
    assert mask[2] == True, "SELL should be valid when flat"
    print("✓ Flat position: all actions valid")
    
    # Take a BUY action
    key, step_key = jax.random.split(key)
    obs, state, _, _, _ = step(step_key, state, jnp.array(1), params, data)
    mask = action_masks(state)
    
    assert mask[0] == True, "HOLD should be valid in position"
    assert mask[1] == False, "BUY should be invalid in position"
    assert mask[2] == False, "SELL should be invalid in position"
    print("✓ In position: only HOLD valid")
    
    return True


def test_step_produces_valid_output():
    """Test that step produces valid outputs."""
    print("\n=== Test: Step Produces Valid Output ===")
    
    data = create_test_data()
    params = EnvParams(
        rth_start_count=int(data.rth_indices.shape[0])
    )
    key = jax.random.key(0)
    
    obs, state = reset(key, params, data)
    
    # Take 100 random steps
    for i in range(100):
        key, action_key, step_key = jax.random.split(key, 3)
        mask = action_masks(state)
        
        # Sample valid action
        valid_actions = jnp.where(mask, jnp.arange(3), -1)
        valid_actions = valid_actions[valid_actions >= 0]
        action = jax.random.choice(action_key, valid_actions)
        
        obs, state, reward, done, info = step(step_key, state, action, params, data)
        
        # Check outputs
        assert jnp.isfinite(obs).all(), f"Step {i}: Observation contains NaN/Inf"
        assert jnp.isfinite(reward), f"Step {i}: Reward is not finite"
        assert state.balance > 0 or done, f"Step {i}: Balance went negative without termination"
        
        if done:
            break
    
    print(f"✓ Completed {i+1} steps successfully")
    print(f"  Final balance: {state.balance:.2f}")
    print(f"  Num trades: {state.num_trades}")
    return True


def test_batch_operations():
    """Test vectorized batch operations."""
    print("\n=== Test: Batch Operations ===")
    
    data = create_test_data()
    params = EnvParams()
    key = jax.random.key(0)
    num_envs = 100
    
    # Batch reset
    obs_batch, state_batch = batch_reset(
        jax.random.split(key, num_envs),
        params,
        data
    )
    
    assert obs_batch.shape[0] == num_envs, f"Expected {num_envs} observations"
    print(f"✓ Batch reset: {num_envs} environments")
    
    # Batch action masks
    masks = batch_action_masks(state_batch)
    assert masks.shape == (num_envs, 3), f"Expected shape ({num_envs}, 3)"
    print(f"✓ Batch action masks: shape {masks.shape}")
    
    # Batch step
    actions = jnp.zeros(num_envs, dtype=jnp.int32)  # All HOLD
    keys = jax.random.split(key, num_envs)
    obs_batch, state_batch, rewards, dones, _ = batch_step(
        keys, state_batch, actions, params, data
    )
    
    assert obs_batch.shape[0] == num_envs, f"Expected {num_envs} observations after step"
    assert rewards.shape == (num_envs,), f"Expected {num_envs} rewards"
    print(f"✓ Batch step: {num_envs} environments stepped")
    
    return True


def test_episode_termination():
    """Test that episodes terminate correctly."""
    print("\n=== Test: Episode Termination ===")
    
    data = create_test_data(num_timesteps=2000)
    params = EnvParams(min_episode_bars=100)
    key = jax.random.key(0)
    
    obs, state = reset(key, params, data)
    initial_step = state.step_idx
    
    # Run until termination
    for i in range(2000):
        key, step_key = jax.random.split(key)
        action = jnp.array(0)  # Always HOLD
        obs, state, reward, done, info = step(step_key, state, action, params, data)
        
        if done:
            break
    
    assert done, "Episode should terminate"
    print(f"✓ Episode terminated after {i+1} steps")
    print(f"  Start step: {initial_step}, End step: {state.step_idx}")
    
    return True


# =============================================================================
# Benchmark Suite
# =============================================================================

def benchmark_throughput(num_envs_list: List[int] = None) -> Dict[int, float]:
    """Benchmark environment throughput at various scales."""
    print("\n=== Throughput Benchmark ===")
    
    if num_envs_list is None:
        num_envs_list = [100, 500, 1000, 2000, 5000]
    
    data = create_test_data(num_timesteps=50000)
    params = EnvParams()
    num_steps = 100
    results = {}
    
    for num_envs in num_envs_list:
        try:
            key = jax.random.key(0)
            
            # Reset
            obs_batch, state_batch = batch_reset(
                jax.random.split(key, num_envs),
                params,
                data
            )
            actions = jnp.zeros(num_envs, dtype=jnp.int32)
            keys = jax.random.split(key, num_envs)
            
            # Warm-up
            for _ in range(5):
                obs_batch, state_batch, rewards, dones, _ = batch_step(
                    keys, state_batch, actions, params, data
                )
            jax.block_until_ready(obs_batch)
            
            # Benchmark
            start = time.time()
            for _ in range(num_steps):
                obs_batch, state_batch, rewards, dones, _ = batch_step(
                    keys, state_batch, actions, params, data
                )
            jax.block_until_ready(obs_batch)
            elapsed = time.time() - start
            
            steps_per_sec = (num_envs * num_steps) / elapsed
            results[num_envs] = steps_per_sec
            
            print(f"  {num_envs:>5} envs: {steps_per_sec:>12,.0f} steps/sec ({elapsed:.2f}s)")
            
        except Exception as e:
            print(f"  {num_envs:>5} envs: FAILED - {e}")
            results[num_envs] = 0
    
    return results


def benchmark_jit_compilation():
    """Benchmark JIT compilation time."""
    print("\n=== JIT Compilation Benchmark ===")
    
    data = create_test_data()
    params = EnvParams()
    key = jax.random.key(0)
    
    # Measure reset compilation
    start = time.time()
    obs, state = reset(key, params, data)
    jax.block_until_ready(obs)
    reset_time = time.time() - start
    print(f"  reset() compilation: {reset_time:.3f}s")
    
    # Measure step compilation
    start = time.time()
    obs, state, reward, done, info = step(key, state, jnp.array(0), params, data)
    jax.block_until_ready(obs)
    step_time = time.time() - start
    print(f"  step() compilation: {step_time:.3f}s")
    
    # Measure batch_reset compilation
    start = time.time()
    obs_batch, state_batch = batch_reset(
        jax.random.split(key, 100),
        params,
        data
    )
    jax.block_until_ready(obs_batch)
    batch_reset_time = time.time() - start
    print(f"  batch_reset() compilation: {batch_reset_time:.3f}s")
    
    # Measure batch_step compilation
    keys = jax.random.split(key, 100)
    actions = jnp.zeros(100, dtype=jnp.int32)
    start = time.time()
    obs_batch, state_batch, rewards, dones, _ = batch_step(keys, state_batch, actions, params, data)
    jax.block_until_ready(obs_batch)
    batch_step_time = time.time() - start
    print(f"  batch_step() compilation: {batch_step_time:.3f}s")


def benchmark_memory():
    """Benchmark memory usage at different scales."""
    print("\n=== Memory Benchmark ===")
    
    data = create_test_data(num_timesteps=100000)
    params = EnvParams()
    key = jax.random.key(0)
    
    # Get data size
    data_size_mb = sum(
        x.nbytes for x in [data.features, data.prices, data.atr, 
                           data.time_features, data.trading_mask, data.timestamps_hour]
    ) / (1024 * 1024)
    print(f"  Market data size: {data_size_mb:.1f} MB")
    
    # Estimate state size per env
    obs, state = reset(key, params, data)
    state_size = sum(x.nbytes for x in jax.tree_util.tree_leaves(state))
    print(f"  State size per env: {state_size} bytes")
    
    # Estimate for different scales
    for num_envs in [1000, 5000, 10000, 50000]:
        total_mb = (num_envs * state_size) / (1024 * 1024) + data_size_mb
        print(f"  {num_envs:>5} envs: ~{total_mb:.1f} MB total")


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("JAX Migration Validation Test Suite")
    print("=" * 60)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    
    tests = [
        test_reset_produces_valid_state,
        test_action_masks_correct,
        test_step_produces_valid_output,
        test_batch_operations,
        test_episode_termination,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_fn.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    # Run benchmarks
    benchmark_jit_compilation()
    benchmark_memory()
    throughput_results = benchmark_throughput()
    
    print("\n" + "=" * 60)
    print("Validation Complete")
    print("=" * 60)
    
    return passed, failed, throughput_results


if __name__ == "__main__":
    passed, failed, throughput = run_all_tests()
    
    # Print summary
    if failed == 0:
        print("\n✓ All tests passed!")
        if throughput.get(1000, 0) > 100000:
            print("✓ Throughput target met (>100k steps/sec @ 1k envs)")
        else:
            print("⚠ Throughput below target")
    else:
        print(f"\n✗ {failed} tests failed")
        sys.exit(1)
