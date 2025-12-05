# JAX Migration Plan for RL Trader AI Trading System

## Executive Summary

This document outlines the comprehensive plan to migrate the RL Trader system from PyTorch/Stable-Baselines3 to a JAX-first architecture. The migration will enable:

- **10,000+ parallel environments** running on GPU
- **1-5 million environment steps per second** throughput
- **100-1000x speedup** over current SubprocVecEnv implementation
- **End-to-end GPU training** with no CPU-GPU data movement

## Current Architecture Analysis

### Bottlenecks Identified

| Component | Current Implementation | JAX Solution |
|-----------|----------------------|--------------|
| Data Access | `df.iloc[start:end]` (Pandas) | Pre-computed JAX arrays |
| State Management | Mutable Python attributes | Immutable `EnvState` dataclass |
| Control Flow | `if/else` branches | `jnp.where()` chains |
| String Comparisons | `exit_reason == "stop_loss"` | Integer exit codes |
| Timezone Conversion | `pd.Timestamp.tz_convert()` | Pre-computed hour arrays |
| Trade History | `list.append()` | Fixed counters in state |

### Performance Comparison (Estimated)

| Metric | Current (PyTorch) | JAX Target | Improvement |
|--------|-------------------|------------|-------------|
| Env Steps/sec | ~5,000 | ~2,000,000 | 400x |
| GPU Utilization | ~20% | ~85% | 4x |
| Training Time (2M steps) | 6-8 hours | 15-30 min | 12-24x |
| Memory per Env | ~50 MB | ~0.1 MB | 500x |

## Migration Phases

### Phase A: Data Pipeline (Week 1-2)

**Goal**: Eliminate Pandas from training hot path

**Deliverables**:
1. `data_loader.py` - Convert CSVs to GPU-resident JAX arrays
2. `MarketData` namedtuple with pre-computed features
3. Time features as numeric arrays (hour decimals)
4. Trading mask arrays for RTH filtering

**Validation**:
- [ ] All 8 markets loadable as JAX arrays
- [ ] Memory footprint < 500MB per market
- [ ] GPU residency confirmed via `jax.devices()`

### Phase B: Core Environment (Week 2-3)

**Goal**: Implement Phase 1 env as pure JAX functions

**Deliverables**:
1. `env_phase1_jax.py` - Pure JAX environment
2. `EnvState` dataclass with all position/portfolio state
3. `EnvParams` dataclass for configuration
4. `step()` and `reset()` as pure functions
5. Action masking via `action_masks()`

**Key Transformations**:
```python
# BEFORE (Python control flow)
if self.position == 1:
    pnl = (exit_price - self.entry_price) * contract_size
else:
    pnl = (self.entry_price - exit_price) * contract_size

# AFTER (JAX tensor ops)
pnl = jnp.where(
    state.position == 1,
    (exit_price - state.entry_price) * contract_size,
    (state.entry_price - exit_price) * contract_size
)
```

**Validation**:
- [ ] Single env step completes without errors
- [ ] Reward parity with PyTorch version (±5%)
- [ ] Action masking produces correct masks

### Phase C: Vectorization (Week 3-4)

**Goal**: Scale to 10k+ parallel environments

**Deliverables**:
1. `batch_reset()` - vmap over reset
2. `batch_step()` - vmap over step  
3. `batch_action_masks()` - vmap over masks
4. Benchmark script for throughput testing

**Target Metrics**:
- [ ] 1k envs: >100k steps/sec
- [ ] 10k envs: >500k steps/sec
- [ ] No Python fallbacks in traced code

### Phase D: PPO Integration (Week 4-5)

**Goal**: Complete training pipeline

**Deliverables**:
1. `train_ppo_jax.py` - Full training loop
2. `ActorCritic` network in Flax
3. GAE computation via `lax.scan`
4. Masked action sampling
5. Observation normalization (running mean/std)

**Architecture**:
```
┌─────────────────────────────────────────────────────┐
│                   Training Loop                      │
├─────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌─────────┐    ┌─────────────────┐ │
│  │  Data   │ -> │  Envs   │ -> │ Policy Network  │ │
│  │ (GPU)   │    │ (vmap)  │    │    (Flax)       │ │
│  └─────────┘    └─────────┘    └─────────────────┘ │
│       │              │                   │          │
│       └──────────────┴───────────────────┘          │
│                  All GPU-resident                   │
└─────────────────────────────────────────────────────┘
```

**Validation**:
- [ ] Training completes 100k steps without errors
- [ ] Loss curves show learning
- [ ] Throughput matches targets

### Phase E: Phase 2 Environment (Week 5-6)

**Goal**: Extend to position management

**Deliverables**:
1. `env_phase2_jax.py` - 6-action environment
2. Transfer learning from Phase 1 weights
3. Additional state for trailing stops
4. Extended action masking logic

**New Actions**:
- 0: Hold
- 1: Buy
- 2: Sell
- 3: Move SL to Break-Even
- 4: Enable Trailing Stop
- 5: Disable Trailing Stop

## Technical Specifications

### EnvState Structure

```python
class EnvState(NamedTuple):
    # Position tracking
    step_idx: jnp.int32          # Current timestep
    position: jnp.int32          # -1/0/1
    entry_price: jnp.float32
    sl_price: jnp.float32
    tp_price: jnp.float32
    position_entry_step: jnp.int32
    
    # Portfolio
    balance: jnp.float32
    highest_balance: jnp.float32
    trailing_dd_level: jnp.float32
    
    # Statistics
    num_trades: jnp.int32
    winning_trades: jnp.int32
    losing_trades: jnp.int32
    total_pnl: jnp.float32
```

### Observation Space

- **Shape**: (225,) = 20 window × 11 features + 5 position features
- **Market features**: close, volume, sma_5, sma_20, rsi, macd, momentum, atr
- **Time features**: hour_decimal, min_from_open, min_to_close
- **Position features**: position, entry_ratio, sl_dist, tp_dist, time_in_position

### Action Masking

```python
def action_masks(state: EnvState) -> jnp.ndarray:
    is_flat = state.position == 0
    return jnp.array([True, is_flat, is_flat])  # [HOLD, BUY, SELL]
```

## Validation & Testing

### Reward Parity Test

```python
def test_reward_parity():
    # Run same episode in both PyTorch and JAX
    pytorch_rewards = run_pytorch_episode(seed=42)
    jax_rewards = run_jax_episode(seed=42)
    
    assert np.allclose(pytorch_rewards, jax_rewards, rtol=0.05)
```

### Throughput Benchmark

```python
def benchmark_throughput(num_envs: int, num_steps: int = 1000):
    # Warm-up
    for _ in range(10):
        batch_step(...)
    
    # Benchmark
    start = time.time()
    for _ in range(num_steps):
        batch_step(...)
        jax.block_until_ready(...)
    elapsed = time.time() - start
    
    return (num_envs * num_steps) / elapsed
```

## Risk Mitigation

### Challenge: Variable Episode Lengths
**Solution**: Use done masking in `lax.scan`, auto-reset within scan loop

### Challenge: Trade History Logging
**Solution**: Keep only aggregate statistics in state (num_trades, wins, losses, total_pnl)

### Challenge: Second-Level Drawdown Checks
**Solution**: Skip in JAX version (minute-level sufficient for training); add for evaluation only

### Challenge: LLM Integration (Phase 3)
**Solution**: Keep LLM inference as separate pass outside JAX env, decision fusion at action level

### Challenge: Model Deployment
**Solution**: Export final JAX weights to PyTorch format if needed for NinjaTrader integration

## Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2 | A | Data pipeline, MarketData loader |
| 2-3 | B | Phase 1 JAX environment |
| 3-4 | C | Vectorization, benchmarks |
| 4-5 | D | PPO training loop |
| 5-6 | E | Phase 2 environment, transfer learning |

## Next Steps

1. **Install JAX with GPU support**:
   ```bash
   pip install jax[cuda12]
   ```

2. **Test data loader**:
   ```bash
   python src/jax_migration/data_loader.py
   ```

3. **Run environment tests**:
   ```bash
   python src/jax_migration/env_phase1_jax.py
   ```

4. **Benchmark vectorization**:
   ```bash
   python src/jax_migration/train_ppo_jax.py
   ```

## Files Created

```
src/jax_migration/
├── __init__.py
├── data_loader.py        # Pandas → JAX array conversion
├── env_phase1_jax.py     # Phase 1 trading environment
├── train_ppo_jax.py      # PPO training loop
├── requirements_jax.txt  # JAX dependencies
└── MIGRATION_PLAN.md     # This document
```

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [Gymnax](https://github.com/RobertTLange/gymnax)
- [PureJaxRL](https://github.com/luchris429/purejaxrl)
- [RLax](https://github.com/google-deepmind/rlax)
- [Flax](https://github.com/google/flax)
