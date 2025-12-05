# JAX Migration Improvement Plan - Comprehensive Update

**Generated:** 2025-01-27
**Status:** Critical fixes identified, implementation plan ready

## Executive Summary

Analysis reveals the existing JAX migration is ~60% complete but has **critical bugs blocking training**. This plan prioritizes fixes and establishes a path to 1M+ steps/sec throughput.

---

## Current State Assessment

### ✅ Completed Components
| Component | Status | Notes |
|-----------|--------|-------|
| EnvState/EnvParams | ✅ Complete | Proper NamedTuples |
| Phase 1 Environment Core | ✅ 90% | step/reset pure functions |
| Data Loader | ✅ Complete | Pandas → JAX conversion |
| Action Masking | ✅ Complete | Phase 1 3-action mask |
| PPO Loss Function | ✅ Complete | Clipped surrogate + masking |
| GAE Computation | ✅ Complete | lax.scan implementation |

### ❌ Critical Issues Found
| Issue | Severity | Impact |
|-------|----------|--------|
| `collect_rollouts()` uses placeholder obs | 🔴 CRITICAL | Training produces garbage |
| No observation normalization | 🔴 HIGH | Unstable training |
| Import path errors | 🟡 MEDIUM | Can't run as package |
| No auto-reset in batch_step | 🟡 MEDIUM | Episode boundary issues |
| Hardcoded observation shape | 🟡 MEDIUM | Breaks Phase 2 |

---

## Phase 1: Critical Fixes (Week 1)

### 1.1 Fix collect_rollouts() - PRIORITY 1

**Problem:** Line ~200 in `train_ppo_jax.py`:
```python
# BROKEN - This is a placeholder!
final_obs = jnp.zeros((config.num_envs, 225))
```

**Solution:** Implemented in `train_ppo_jax_fixed.py`

### 1.2 Add Observation Normalization - PRIORITY 2

**Implementation:**
```python
class NormalizerState(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

def update_normalizer(state: NormalizerState, batch: jnp.ndarray) -> NormalizerState:
    batch_mean = batch.mean(axis=0)
    batch_var = batch.var(axis=0)
    batch_count = batch.shape[0]
    
    delta = batch_mean - state.mean
    total_count = state.count + batch_count
    
    new_mean = state.mean + delta * batch_count / total_count
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta**2 * state.count * batch_count / total_count
    new_var = M2 / total_count
    
    return NormalizerState(new_mean, new_var, total_count)
```

### 1.3 Fix Import Structure - PRIORITY 3

Create proper `__init__.py`:
```python
from .data_loader import MarketData, load_market_data
from .env_phase1_jax import EnvState, EnvParams, reset, step, action_masks
from .train_ppo_jax import PPOConfig, train
```

---

## Phase 2: Validation & Testing (Week 1-2)

### 2.1 Reward Parity Test Suite

```python
def test_reward_parity():
    """Ensure JAX and PyTorch produce identical results."""
    seed = 42
    
    # Run PyTorch episode
    pytorch_env = TradingEnvironmentPhase1(data, seed=seed)
    pytorch_rewards = run_episode(pytorch_env)
    
    # Run JAX episode
    jax_data = load_market_data(data_path)
    jax_rewards = run_jax_episode(jax_data, seed=seed)
    
    assert np.allclose(pytorch_rewards, jax_rewards, rtol=0.05)
```

### 2.2 Throughput Benchmarks

Target metrics:
| Envs | Target Steps/Sec | Memory |
|------|-----------------|--------|
| 100 | 50,000 | < 1 GB |
| 1,000 | 200,000 | < 2 GB |
| 5,000 | 500,000 | < 4 GB |
| 10,000 | 800,000 | < 6 GB |

---

## Phase 3: Vectorization Optimization (Week 2)

### 3.1 Optimize Memory Layout

```python
# Use scan-friendly state layout
@jax.jit
def batched_rollout(key, states, params, data, num_steps):
    def scan_body(carry, _):
        states, key = carry
        key, *subkeys = jax.random.split(key, 4)
        
        # Vectorized observation
        obs = vmap(get_observation)(states, data, params)
        
        # Vectorized action selection
        logits, values = network.apply(params, obs)
        actions = vmap(sample_masked_action)(subkeys[0], logits, masks)
        
        # Vectorized step
        next_obs, next_states, rewards, dones, _ = batch_step(
            subkeys[1], states, actions, params, data
        )
        
        # Auto-reset
        reset_states = vmap(partial(reset, params=params, data=data))(subkeys[2])
        final_states = tree_map(
            lambda r, n: jnp.where(dones[:, None], r, n),
            reset_states, next_states
        )
        
        return (final_states, key), (obs, actions, rewards, dones, values)
    
    return lax.scan(scan_body, (states, key), None, num_steps)
```

### 3.2 Memory Optimization

- Use `jnp.float16` for observations during training
- Gradient checkpointing for batches > 5000
- Lazy evaluation with `jax.checkpoint`

---

## Phase 4: Training Pipeline (Week 2-3)

### 4.1 PureJaxRL-Style Training

Key patterns from Context7 Gymnax docs:
```python
@partial(jax.jit, static_argnums=(4, 5))
def train_epoch(runner_state, data, params, num_steps, num_epochs):
    """Single training epoch in pure JAX."""
    
    def _update_step(runner_state, _):
        train_state, env_states, key = runner_state
        
        # Collect rollouts (all in JAX)
        key, rollout_key = jax.random.split(key)
        transitions, new_env_states = collect_rollouts(...)
        
        # Update policy (PPO)
        train_state, metrics = ppo_update(train_state, transitions)
        
        return (train_state, new_env_states, key), metrics
    
    return lax.scan(_update_step, runner_state, None, num_epochs)
```

### 4.2 Logging & Checkpointing

```python
# Tensorboard logging
from flax.metrics import tensorboard

summary_writer = tensorboard.SummaryWriter(log_dir)

# Orbax checkpointing
from orbax.checkpoint import PyTreeCheckpointer

checkpointer = PyTreeCheckpointer()
checkpointer.save(ckpt_dir, train_state)
```

---

## Phase 5: Phase 2 Environment (Week 3-4)

### 5.1 Extended EnvState

```python
class EnvStatePhase2(NamedTuple):
    # Inherit all Phase 1 fields
    step_idx: jnp.ndarray
    position: jnp.ndarray
    entry_price: jnp.ndarray
    sl_price: jnp.ndarray
    tp_price: jnp.ndarray
    # ... other Phase 1 fields ...
    
    # Phase 2 additions
    trailing_stop_active: jnp.ndarray   # bool
    highest_profit_point: jnp.ndarray   # float32
    be_move_count: jnp.ndarray          # int32
    original_sl_price: jnp.ndarray      # float32 (for BE calculation)
```

### 5.2 Phase 2 Action Masking

```python
def action_masks_phase2(state: EnvStatePhase2, params: EnvParams, data: MarketData) -> jnp.ndarray:
    is_flat = state.position == 0
    has_position = state.position != 0
    
    # Check if within RTH
    current_hour = data.timestamps_hour[state.step_idx]
    within_rth = (current_hour >= params.rth_open) & (current_hour < params.rth_close)
    
    # Check profitability for BE move
    current_price = data.prices[state.step_idx, 3]
    unrealized_pnl = jnp.where(
        state.position == 1,
        (current_price - state.entry_price) * params.contract_size,
        (state.entry_price - current_price) * params.contract_size
    )
    is_profitable = unrealized_pnl > 0
    
    return jnp.array([
        True,                              # 0: HOLD
        is_flat & within_rth,              # 1: BUY
        is_flat & within_rth,              # 2: SELL
        has_position & is_profitable,      # 3: MOVE_SL_TO_BE
        has_position,                      # 4: ENABLE_TRAIL
        has_position,                      # 5: DISABLE_TRAIL
    ], dtype=jnp.bool_)
```

---

## Phase 6: Multi-Market & Curriculum (Week 4+)

### 6.1 Multi-Market Training

```python
# Randomize market selection during training
def sample_market_batch(key, market_data, num_envs):
    market_ids = jax.random.randint(key, (num_envs,), 0, len(market_data))
    # Index into batched market data
    return tree_map(lambda x: x[market_ids], batched_data)
```

### 6.2 Curriculum Learning

```python
# Curriculum schedule
def get_curriculum_params(progress: float) -> EnvParams:
    """Progress: 0.0 (start) to 1.0 (end)"""
    return EnvParams(
        trailing_dd_limit=jnp.interp(progress, [0, 0.5, 1.0], [15000, 5000, 2500]),
        sl_atr_mult=jnp.interp(progress, [0, 1.0], [2.0, 1.5]),
        # ... other params
    )
```

---

## Expected Outcomes

| Metric | Current (PyTorch) | Target (JAX) | Improvement |
|--------|-------------------|--------------|-------------|
| Steps/sec | ~5,000 | >1,000,000 | 200x |
| Training 2M steps | 6-8 hours | 15-30 min | 15-20x |
| GPU utilization | ~20% | >80% | 4x |
| Memory per env | ~50 MB | ~0.1 MB | 500x |
| Parallel envs | 8 (SubprocVec) | 10,000+ | 1000x |

---

## Implementation Order

```
Week 1:
├── Day 1-2: Fix collect_rollouts() + obs normalization
├── Day 3: Fix imports + add tests
└── Day 4-5: Validation tests + initial benchmarks

Week 2:
├── Day 1-2: Optimize batch_step
├── Day 3-4: Memory optimization
└── Day 5: Throughput benchmarks

Week 3:
├── Day 1-3: PureJaxRL training loop
└── Day 4-5: Logging + checkpointing

Week 4:
├── Day 1-3: Phase 2 environment
└── Day 4-5: Transfer learning + validation

Week 5+:
├── Multi-market training
├── Curriculum learning
└── Production optimization
```

---

## Files to Modify/Create

1. `train_ppo_jax.py` → Fix collect_rollouts (CRITICAL)
2. `normalizer.py` → NEW: Observation normalization
3. `env_phase1_jax.py` → Minor fixes
4. `env_phase2_jax.py` → NEW: Phase 2 environment
5. `test_parity.py` → NEW: Validation tests
6. `benchmark.py` → NEW: Throughput benchmarks
7. `__init__.py` → Fix imports
