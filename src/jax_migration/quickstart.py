#!/usr/bin/env python3
"""
JAX Migration Quick Start Script

Verifies installation, runs validation tests, and performs a minimal training run.

Usage:
    python quickstart.py [--full]
    
Options:
    --full  Run full validation suite including throughput benchmarks
"""

import sys
import time
from pathlib import Path

# Add src to path so we can import jax_migration package
# This allows running the script from anywhere (e.g. project root or src folder)
src_path = str(Path(__file__).resolve().parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def main():
    print("=" * 70)
    print("JAX Migration Quick Start")
    print("=" * 70)
    
    # Check JAX installation
    print("\n[1/5] Checking JAX installation...")
    try:
        import jax
        import jax.numpy as jnp
        print(f"  ✓ JAX version: {jax.__version__}")
        print(f"  ✓ Devices: {jax.devices()}")
        
        # Check GPU
        if any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in jax.devices()):
            print("  ✓ GPU detected!")
        else:
            print("  ⚠ No GPU detected - running on CPU (will be slower)")
    except ImportError as e:
        print(f"  ✗ JAX not installed: {e}")
        print("\n  Install with: pip install jax[cuda12]")
        return 1
    
    # Check dependencies
    print("\n[2/5] Checking dependencies...")
    deps = ['flax', 'optax', 'chex']
    missing = []
    for dep in deps:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} not installed")
            missing.append(dep)
    
    if missing:
        print(f"\n  Install missing: pip install {' '.join(missing)}")
        return 1
    
    # Check module imports
    print("\n[3/5] Checking JAX migration module...")
    try:
        from jax_migration import (
            MarketData, EnvParams, EnvState,
            reset, step, action_masks,
            batch_reset, batch_step,
            check_status
        )
        check_status()
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        print("\n  Make sure you're running from the src directory")
        return 1
    
    # Run basic tests
    print("\n[4/5] Running basic tests...")
    try:
        # Create dummy data
        key = jax.random.key(42)
        num_timesteps = 5000
        
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
        
        params = EnvParams(
            rth_start_count=int(dummy_data.rth_indices.shape[0])
        )
        
        # Test single env
        obs, state = reset(key, params, dummy_data)
        assert obs.shape == (225,), f"Wrong obs shape: {obs.shape}"
        print(f"  ✓ Single env reset: obs shape {obs.shape}")
        
        # Test step
        key, step_key = jax.random.split(key)
        obs, state, reward, done, _ = step(step_key, state, jnp.array(1), params, dummy_data)
        print(f"  ✓ Single env step: position={state.position}")
        
        # Test batch
        num_envs = 100
        obs_batch, state_batch = batch_reset(
            jax.random.split(key, num_envs),
            params,
            dummy_data
        )
        assert obs_batch.shape == (num_envs, 225), f"Wrong batch shape: {obs_batch.shape}"
        print(f"  ✓ Batch reset: {num_envs} envs")
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Quick throughput test
    print("\n[5/5] Quick throughput benchmark...")
    try:
        num_envs = 1000
        num_steps = 100
        
        obs_batch, state_batch = batch_reset(
            jax.random.split(key, num_envs),
            params,
            dummy_data
        )
        actions = jnp.zeros(num_envs, dtype=jnp.int32)
        keys = jax.random.split(key, num_envs)
        
        # Warm-up
        for _ in range(5):
            obs_batch, state_batch, _, _, _ = batch_step(keys, state_batch, actions, params, dummy_data)
        jax.block_until_ready(obs_batch)
        
        # Benchmark
        start = time.time()
        for _ in range(num_steps):
            obs_batch, state_batch, rewards, dones, _ = batch_step(
                keys, state_batch, actions, params, dummy_data
            )
        jax.block_until_ready(obs_batch)
        elapsed = time.time() - start
        
        steps_per_sec = (num_envs * num_steps) / elapsed
        print(f"  ✓ {num_envs} envs × {num_steps} steps: {steps_per_sec:,.0f} steps/sec")
        
        if steps_per_sec > 100000:
            print("  ✓ Throughput target met! (>100k steps/sec)")
        elif steps_per_sec > 10000:
            print("  ⚠ Throughput moderate - consider GPU if not already using")
        else:
            print("  ⚠ Throughput low - check GPU availability")
        
    except Exception as e:
        print(f"  ✗ Benchmark failed: {e}")
        return 1
    
    # Summary
    print("\n" + "=" * 70)
    print("Quick Start Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Load your market data:")
    print("     from jax_migration import load_market_data")
    print("     data = load_market_data('data/ES_D1M.csv')")
    print()
    print("  2. Run validation tests:")
    print("     python -m jax_migration.test_validation")
    print()
    print("  3. Start training:")
    print("     from jax_migration import PPOConfig, train")
    print("     config = PPOConfig(num_envs=2048, total_timesteps=1_000_000)")
    print("     trained_state, normalizer, metrics = train(config, EnvParams(rth_start_count=<len>), data)")
    
    # Full validation if requested
    if '--full' in sys.argv:
        print("\n\nRunning full validation suite...")
        from jax_migration.test_validation import run_all_tests
        passed, failed, throughput = run_all_tests()
        return 0 if failed == 0 else 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
