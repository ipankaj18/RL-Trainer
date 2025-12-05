# Comprehensive Prompt: Rebuild JAX Stress Test from Scratch

## Context
We have a JAX RL trading environment that repeatedly fails with `TracerIntegerConversionError` during stress testing. The core issue is that the current implementation mixes Python control flow with JAX traced values in ways that break JIT compilation.

## Current Architecture (What's Broken)
```
stress_hardware_jax.py → train_ppo_jax_fixed.py → env_phase1_jax.py
                                   ↓
                         collect_rollouts() [JIT]
                                   ↓
                              step_fn() [lax.scan]
                                   ↓
                            batch_reset() [vmap]
                                   ↓
                              reset() [vmapped]
                                   ↓
                    data.rth_indices[traced_idx] ← BREAKS HERE!
```

## Root Causes of Failures
1. **Python `[]` indexing with traced arrays** - `data.rth_indices[rth_idx]` fails when `rth_idx` is traced
2. **Python `int()` calls on traced values** - `int(params.value)` fails inside JIT
3. **Python `if` statements with traced booleans** - `if traced_condition:` fails inside JIT
4. **Dynamic shapes inside JIT** - `jnp.zeros(traced_shape)` fails
5. **`random.split(key, traced_n)`** - Can't use traced value for split count

## Design Requirements for New Implementation

### 1. Stress Test Script (`scripts/stress_hardware_jax_v2.py`)
- **Purpose**: Test different num_envs/batch_size combinations to find optimal GPU utilization
- **Data handling**: Load real market data, subset if needed, ensure all arrays are JAX arrays
- **Minimal training loop**: Only needs 50-100 updates to measure GPU utilization
- **No complex environment needed**: Can use a simplified dummy environment for stress testing

### 2. Simplified Stress-Only Environment
Create a NEW simplified environment ONLY for stress testing that:
- Has NO complex trading logic (no SL/TP, no RTH checks)
- Uses FIXED episode starts (not random)
- Has pure JAX operations with ZERO Python control flow
- Returns dummy rewards (we only care about GPU speed, not training quality)

### 3. Key JAX Patterns to Follow

#### ✅ CORRECT: Use `jnp.take()` for traced indexing
```python
# Instead of: arr[traced_idx]
value = jnp.take(arr, traced_idx)
```

#### ✅ CORRECT: Use `jnp.where()` for conditionals
```python
# Instead of: if condition: result = a else: result = b
result = jnp.where(condition, a, b)
```

#### ✅ CORRECT: Use static values for random splits
```python
# Inside JIT, num must be static (Python int, not traced)
# Split BEFORE entering JIT, then pass pre-split keys
keys = jax.random.split(master_key, num_envs)  # Outside JIT
jax.vmap(func)(keys)  # Each func gets one key
```

#### ✅ CORRECT: Use `lax.dynamic_slice` for safe slicing
```python
# Instead of: arr[start:start+window]
result = lax.dynamic_slice_in_dim(arr, start, window, axis=0)
```

#### ✅ CORRECT: Avoid Python ints from traced values
```python
# Don't do this inside JIT:
# n = int(params.count)

# Instead, pass count as a static argument or use it directly as JAX array
```

## Proposed New Architecture

### Option A: Ultra-Simple Stress Test (Recommended)
```python
# stress_test_simple.py - Completely self-contained, no external dependencies

import jax
import jax.numpy as jnp
from jax import lax, vmap
import time

# Simple observation: just random features
# Simple action: 0, 1, 2 (hold, buy, sell)
# Simple reward: random small value
# No trading logic, no market data, no RTH

@jax.jit
def dummy_step(key, state, action):
    """Pure JAX step with no Python control flow."""
    next_state = state + 1  # Simple counter
    reward = jax.random.normal(key) * 0.01
    done = next_state >= 1000  # Fixed episode length
    obs = jax.random.normal(key, shape=(128,))
    return obs, next_state, reward, done

def stress_test(num_envs, num_steps, num_updates):
    """Measure training speed."""
    key = jax.random.PRNGKey(0)
    
    # Pre-split all keys
    keys = jax.random.split(key, num_envs)
    states = jnp.zeros(num_envs)
    
    start = time.time()
    for _ in range(num_updates):
        # Vectorized step
        keys = jax.random.split(keys[0], num_envs)
        actions = jax.random.randint(keys[0], (num_envs,), 0, 3)
        obs, states, rewards, dones = vmap(dummy_step)(keys, states, actions)
        
        # Reset done envs
        states = jnp.where(dones, 0, states)
    
    jax.block_until_ready(states)
    elapsed = time.time() - start
    
    sps = (num_envs * num_steps * num_updates) / elapsed
    return sps
```

### Option B: Minimal Trading Env (If Market Data Needed)
- Load market data ONCE, convert ALL arrays to JAX arrays
- Use STATIC episode starts (jnp.linspace) instead of random sampling
- Remove all SL/TP/RTH logic from stress test
- Keep environment logic simple: step forward, compute observation

## Files to DELETE Before Fresh Start
```bash
# On server:
rm /workspace/scripts/stress_hardware_jax.py
rm -rf /workspace/src/jax_migration/__pycache__
rm -rf /workspace/scripts/__pycache__
```

## Files to KEEP (Core Training, Don't Touch)
- `src/jax_migration/data_loader.py` - Data loading utilities
- `src/jax_migration/train_ppo_jax_fixed.py` - Main training (needs fixes but keep)
- `src/jax_migration/env_phase1_jax.py` - Trading env (needs fixes but keep)

## Deliverables
1. `scripts/stress_test_simple.py` - Self-contained stress test with dummy env
2. If stress test passes, THEN fix `env_phase1_jax.py` incrementally
3. Document ALL places where `[]` indexing is used with traced values
4. Replace ALL with `jnp.take()` or `lax.dynamic_slice`

## Success Criteria
- Stress test completes 3 runs without errors
- GPU utilization > 70%
- Hardware profiles generated
- No `TracerIntegerConversionError` or `TracerBoolConversionError`
