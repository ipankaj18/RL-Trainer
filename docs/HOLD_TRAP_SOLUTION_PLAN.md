# HOLD Trap Solution Plan

## Problem Statement

**Observed Behavior (from stress test output):**
- Action Distribution: 93-94% HOLD consistently throughout training
- Entropy collapse: 0.09 → 0.05 (agent becoming deterministic)
- Exploration Bonus ($75 → $1.50) was ignored by the agent
- Auto entropy adjustment (0.015 → 0.30) not solving the problem
- Despite 65% win rate and $255K PnL, agent refuses to trade more

**This is the "HOLD Trap"** - a classic risk-averse policy collapse in RL trading systems.

---

## Root Cause Analysis

### Why the Agent Prefers HOLD

The agent has learned that HOLD is a "safe harbor" that:

| Scenario | Reward/Penalty | Risk |
|----------|---------------|------|
| HOLD when flat | +0.001/step | Zero |
| BUY/SELL entry | +0.75 (bonus) - 0.005 (commission) | High (uncertain PnL, drawdown risk) |
| Bad trade | -$100 to -$1000 scaled | -10.0 on drawdown violation |

**The Math:**
- 100 HOLD steps = +0.10 reward (guaranteed)
- 1 BUY with bonus = +0.745 reward BUT:
  - If trade loses $200: +0.745 - 2.00 = **-1.255** net reward
  - If drawdown violation: **-10.0** penalty

**Agent's Rational Conclusion:** Trading is risky, HOLD is safe. The $75 exploration bonus doesn't compensate for the perceived risk.

### Why Current Solutions Failed

1. **Exploration Bonus ($75)**: Too small relative to worst-case trade loss
2. **Entropy Auto-Adjustment**: Reactive, not proactive - entropy already collapsed
3. **No Entropy Floor**: Entropy plummets from 0.09 to 0.05 unchecked
4. **Patience Reward (+0.001)**: Creates "safe harbor" incentive for HOLD

---

## Research-Based Solutions

Based on recent RL research (2024-2025):

### EPO (Entropy-regularized Policy Optimization)
- **Entropy smoothing regularizer**: Bounds policy entropy within historical averages
- **Adaptive phase-based weighting**: Balances exploration/exploitation across training
- Results: Up to 152% performance improvement
- Source: [arxiv.org/html/2509.22576](https://arxiv.org/html/2509.22576)

### axPPO (Adaptive Exploration PPO)
- **Performance-dependent entropy scaling**: `ent_coef * (1 - normalized_recent_return)`
- When performance poor → higher entropy → more exploration
- Source: [arxiv.org/html/2405.04664v1](https://arxiv.org/html/2405.04664v1)

### CE-GPPO (Coordinating Entropy)
- PPO's clipping discards valuable gradient signals from low-probability actions
- These clipped gradients are crucial for **entropy evolution**
- Source: [arxiv.org/html/2509.20712](https://arxiv.org/html/2509.20712)

---

## Proposed Solution Architecture

### Three-Layer Approach

```
LAYER 1: Make Trading MORE Attractive
├─ Increase exploration bonus 4-6x ($300-500)
├─ Extend exploration period (25% → 40%)
└─ Add entry signal reward bonus

LAYER 2: Make HOLD LESS Attractive
├─ Remove patience reward during exploration (0.001 → 0.0)
├─ Add HOLD penalty: -0.005/step when flat
└─ Create asymmetric incentive favoring action

LAYER 3: Structural Guarantee
├─ Minimum action probability floor (5% BUY, 5% SELL)
├─ Temperature scaling curriculum (3.0 → 1.0)
└─ Entropy smoothing to prevent collapse
```

---

## Implementation Plan

### Phase A: Quick Wins (1-2 hours)

**Priority: HIGHEST - Implement First**

#### A1. Increase Exploration Bonus
**File:** `src/jax_migration/env_phase1_jax.py:366`

```python
# BEFORE
base_bonus = 75.0

# AFTER
base_bonus = 400.0  # 5.3x increase
```

**Rationale:** Bonus must exceed perceived risk of worst-case trade.

#### A2. Extend Exploration Period
**File:** `src/jax_migration/env_phase1_jax.py:362`

```python
# BEFORE
exploration_horizon = total_timesteps * 0.25

# AFTER
exploration_horizon = total_timesteps * 0.40  # 40% of training
```

#### A3. Remove Patience Reward During Exploration
**File:** `src/jax_migration/env_phase1_jax.py:349-351`

```python
# BEFORE
reward = jnp.where(is_flat & no_exit, 0.001, reward)

# AFTER - Conditional patience reward
if current_timestep is not None:
    exploration_progress = current_timestep / (total_timesteps * 0.40)
    # No patience reward during exploration phase
    in_exploration = exploration_progress < 1.0
    patience_reward = jnp.where(in_exploration, 0.0, 0.001)
    reward = jnp.where(is_flat & no_exit, patience_reward, reward)
else:
    reward = jnp.where(is_flat & no_exit, 0.001, reward)
```

#### A4. Earlier Entropy Intervention
**File:** `src/jax_migration/train_ppo_jax_fixed.py:772`

```python
# BEFORE
if current_entropy < 0.15:
    new_ent_coef = min(config.ent_coef * 1.5, 0.30)

# AFTER
if current_entropy < 0.25:  # Earlier trigger
    new_ent_coef = min(config.ent_coef * 1.5, 0.50)  # Higher max
```

**Expected Result from Phase A:**
- HOLD% drops from 93-94% to 80-85%
- Entropy stays above 0.15

---

### Phase B: Minimum Action Probability Floor (2-4 hours)

**Priority: HIGH - Structural Guarantee**

#### B1. Add Exploration Probability Floor
**File:** `src/jax_migration/train_ppo_jax_fixed.py:119-122`

```python
def masked_softmax(
    logits: jnp.ndarray,
    mask: jnp.ndarray,
    exploration_floor: float = 0.0,  # NEW parameter
    floor_actions: tuple = (1, 2)     # BUY=1, SELL=2
) -> jnp.ndarray:
    """Apply mask to logits and compute softmax with optional exploration floor."""
    masked_logits = jnp.where(mask, logits, -1e10)
    probs = jax.nn.softmax(masked_logits, axis=-1)

    # Apply minimum probability floor for specified actions
    if exploration_floor > 0.0:
        for action_idx in floor_actions:
            probs = probs.at[..., action_idx].set(
                jnp.maximum(probs[..., action_idx], exploration_floor)
            )
        # Renormalize
        probs = probs / probs.sum(axis=-1, keepdims=True)

    return probs
```

#### B2. Curriculum-Based Floor Decay
**File:** `src/jax_migration/train_ppo_jax_fixed.py` (training loop)

```python
# Calculate exploration floor (decays over training)
exploration_floor_horizon = config.total_timesteps * 0.40
floor_progress = current_global_timestep / exploration_floor_horizon
base_floor = 0.08  # 8% minimum for BUY and SELL
current_floor = base_floor * max(0.0, 1.0 - floor_progress)

# Pass to action sampling
probs = masked_softmax(logits, mask, exploration_floor=current_floor)
```

**Expected Result from Phase B:**
- HOLD% capped at ~84% maximum (100% - 8% BUY - 8% SELL)
- Guaranteed 8% BUY and 8% SELL actions during exploration
- ~65% more trading actions than current

---

### Phase C: Temperature Curriculum (4-8 hours)

**Priority: MEDIUM - Polish**

#### C1. Temperature-Scaled Softmax
**File:** `src/jax_migration/train_ppo_jax_fixed.py:119-122`

```python
def masked_softmax(
    logits: jnp.ndarray,
    mask: jnp.ndarray,
    temperature: float = 1.0  # NEW parameter
) -> jnp.ndarray:
    """Apply mask to logits and compute temperature-scaled softmax."""
    masked_logits = jnp.where(mask, logits, -1e10)
    # Temperature scaling: higher temp = more uniform distribution
    scaled_logits = masked_logits / temperature
    return jax.nn.softmax(scaled_logits, axis=-1)
```

#### C2. Temperature Decay Schedule
```python
# Start with high temperature (random-ish), decay to 1.0 (normal)
temperature_decay_horizon = config.total_timesteps * 0.50
temp_progress = current_global_timestep / temperature_decay_horizon
temperature = 1.0 + 2.0 * max(0.0, 1.0 - temp_progress)  # 3.0 → 1.0
```

#### C3. Entropy Smoothing (EPO-style)
```python
# Track entropy history
entropy_history = []
entropy_window = 20

def get_smoothed_entropy_target(current_entropy):
    entropy_history.append(current_entropy)
    if len(entropy_history) > entropy_window:
        entropy_history.pop(0)

    avg_entropy = sum(entropy_history) / len(entropy_history)
    std_entropy = np.std(entropy_history)

    # Bound entropy within historical average ± 1 std
    min_target = avg_entropy - std_entropy
    max_target = avg_entropy + std_entropy

    return max(min_target, 0.15)  # Never go below 0.15
```

---

## Success Metrics

| Metric | Current | Phase A Target | Phase B Target | Final Target |
|--------|---------|----------------|----------------|--------------|
| HOLD % | 93-94% | 80-85% | 70-80% | 60-70% |
| BUY % | 3-4% | 8-10% | 12-15% | 15-20% |
| SELL % | 2-3% | 5-8% | 8-12% | 15-20% |
| Entropy | 0.05-0.09 | 0.15+ | 0.20+ | 0.20-0.30 |
| Trades per 1M steps | ~50 | ~100 | ~200 | ~300 |

---

## Testing Protocol

### Quick Validation Test
```bash
# Run 50 updates (~1M timesteps) with Phase A changes
python scripts/stress_hardware_jax.py --phase 1 --market NQ --max-runs 1 \
    --test-updates 50

# Monitor these metrics:
# - Action distribution (should see HOLD% drop)
# - Entropy (should stay above 0.15)
# - Exploration bonus logs
```

### Success Criteria for Each Phase
- **Phase A Success:** HOLD% < 85% by update 50
- **Phase B Success:** HOLD% < 75% by update 50, BUY+SELL > 20%
- **Phase C Success:** Entropy stable 0.20-0.30, smooth action distribution

---

## Risk Mitigation

### Potential Issues

1. **Too Much Trading (Over-Exploration)**
   - Monitor: If BUY+SELL > 60%, reduce exploration bonus
   - Safety: Commission curriculum will naturally penalize excessive trading

2. **Random Trading (No Learning)**
   - Monitor: Win rate should stay > 45%
   - Safety: If win rate < 40%, reduce temperature/floor

3. **Training Instability**
   - Monitor: Policy loss spikes, KL divergence > 0.1
   - Safety: Reduce ent_coef if unstable

### Rollback Plan
Keep original parameter values documented. If any phase causes degradation:
```python
# Original values for rollback
ORIGINAL_BONUS = 75.0
ORIGINAL_EXPLORATION_HORIZON = 0.25
ORIGINAL_PATIENCE_REWARD = 0.001
ORIGINAL_ENTROPY_THRESHOLD = 0.15
ORIGINAL_MAX_ENT_COEF = 0.30
```

---

## Implementation Priority

1. **IMMEDIATE (Today):** Phase A - Quick wins
2. **NEXT (If Phase A insufficient):** Phase B - Probability floor
3. **LATER (Optimization):** Phase C - Temperature curriculum

---

## References

- [EPO: Entropy-regularized Policy Optimization](https://arxiv.org/html/2509.22576)
- [axPPO: Adaptive Exploration PPO](https://arxiv.org/html/2405.04664v1)
- [CE-GPPO: Coordinating Entropy](https://arxiv.org/html/2509.20712)
- [EEPO: Exploration-Enhanced Policy Optimization](https://arxiv.org/html/2510.05837)
- [Exploration-Exploitation Dilemma Revisited](https://arxiv.org/html/2408.09974)

---

**Document Version:** 1.0
**Created:** 2025-12-03
**Author:** Claude Code (AI Assistant)
