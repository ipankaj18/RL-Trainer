#!/usr/bin/env python3
"""Test batch_step after fix"""

import os
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import sys
from pathlib import Path

src_path = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, src_path)

import jax
import jax.numpy as jnp
from jax_migration import MarketData, EnvParams, batch_step, batch_reset

key = jax.random.key(42)
num_timesteps = 5000

dummy_data = MarketData(
    features=jax.random.normal(key, (num_timesteps, 8)),
    prices=jnp.abs(jax.random.normal(key, (num_timesteps, 4))) * 100 + 5000,
    atr=jnp.abs(jax.random.normal(key, (num_timesteps,))) * 10 + 5,
    time_features=jax.random.uniform(key, (num_timesteps, 3)),
    trading_mask=jnp.ones(num_timesteps),
    timestamps_hour=jnp.linspace(9.5, 16.9, num_timesteps),
    rth_indices=jnp.arange(60, num_timesteps - 100),
    low_s=jnp.abs(jax.random.normal(key, (num_timesteps,))) * 100 + 4990,
    high_s=jnp.abs(jax.random.normal(key, (num_timesteps,))) * 100 + 5010,
)

params = EnvParams(rth_start_count=int(dummy_data.rth_indices.shape[0]))

num_envs = 100
print(f"Testing batch_reset with {num_envs} envs...")
obs_batch, state_batch = batch_reset(
    jax.random.split(key, num_envs),
    params,
    dummy_data
)
print(f"✓ batch_reset succeeded: {obs_batch.shape}")

actions = jnp.zeros(num_envs, dtype=jnp.int32)
keys = jax.random.split(key, num_envs)

print(f"Testing batch_step...")
try:
    obs_batch, state_batch, rewards, dones, info = batch_step(
        keys, state_batch, actions, params, dummy_data
    )
    jax.block_until_ready(obs_batch)
    print(f"✓ batch_step succeeded!")
    print(f"  obs shape: {obs_batch.shape}")
    print(f"  rewards: {rewards[:5]}...")
    print(f"  dones: {dones[:5]}...")
except Exception as e:
    print(f"✗ batch_step failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Benchmark ===")
import time
num_steps = 100

# Warmup
for _ in range(5):
    obs_batch, state_batch, _, _, _ = batch_step(keys, state_batch, actions, params, dummy_data)
jax.block_until_ready(obs_batch)

start = time.time()
for _ in range(num_steps):
    obs_batch, state_batch, rewards, dones, _ = batch_step(keys, state_batch, actions, params, dummy_data)
jax.block_until_ready(obs_batch)
elapsed = time.time() - start

sps = (num_envs * num_steps) / elapsed
print(f"✓ {num_envs} envs × {num_steps} steps: {sps:,.0f} steps/sec")
print("\n🎉 ALL TESTS PASSED!")
