# Phase 3 Training Diagnostic Report
**Date**: 2025-11-23
**Training Run**: phase3_hybrid test (81,920 timesteps)
**Status**: ❌ FAILED - Multiple Critical Issues

---

## Executive Summary

The Phase 3 hybrid training run **failed to learn** due to a configuration error that froze the RL policy for the entire training duration. Additionally, the LLM integration is non-functional, contributing zero decision-making value while adding significant latency overhead.

**Key Findings**:
1. RL policy frozen for 100% of training (warmup misconfiguration)
2. LLM queried only 0.5% of the time (should be ~20%)
3. LLM provides near-zero confidence (0.023%) - essentially abstaining
4. Massive train/val overfitting gap (1.55) despite no learning
5. Episodes terminate after only 19.9 bars (abnormally short)

---

## Critical Issues

### Issue #1: RL Policy Frozen (CRITICAL)

**Symptoms**:
```
approx_kl: 1.24e-10          ← Policy unchanged
clip_fraction: 0             ← No gradient updates
policy_gradient_loss: -6.62e-10  ← Zero gradients
explained_variance: 0.192    ← Poor value estimation
```

**Root Cause**:
```python
# src/train_phase3_llm.py:315
'adapter_warmup_steps': 100_000  # Unfreeze Phase 2 after 100K steps

# Actual training:
total_timesteps: 81,920          # Never reached threshold!
```

**Impact**:
- Phase 2 weights remained **frozen** for entire 82K timesteps
- Only 261D→228D adapter layer trained
- Main policy network (90%+ of parameters) never updated
- RL agent incapable of adapting to Phase 3 observation space

**Fix**:
```python
'adapter_warmup_steps': 10_000,  # 10K steps (12% of test run)
# OR disable freezing entirely for test runs:
'freeze_phase2_initially': False,
```

---

### Issue #2: LLM Query Rate 0.5% (CRITICAL)

**Symptoms**:
```
LLM queries: 372 / 81,920 = 0.45%
Expected: ~20% (every 5 bars)
Actual: ~1 query per 220 steps
```

**Root Causes**:
1. **Query interval too high**:
   ```yaml
   # config/llm_config.yaml:20
   query_interval: 20  # Should be 5
   ```

2. **Cooldown too aggressive**:
   ```yaml
   # config/llm_config.yaml:21
   query_cooldown: 10  # Combined with interval=20 kills queries
   ```

3. **Selective querying logic**:
   - RL confidence threshold (0.75) never met (RL avg = 0.076)
   - Position state change detection too strict
   - Interval logic overrides uncertainty triggers

**Impact**:
- LLM contributes to <0.5% of decisions
- 99.9% agreement because LLM rarely disagrees (rarely queried)
- 14 minutes of latency overhead (372 × 2.3s) for zero benefit

**Fix**:
```yaml
fusion:
  query_interval: 5      # Query every 5 bars = 20%
  query_cooldown: 2      # Reduce cooldown
  confidence_threshold: 0.60  # Lower threshold (RL confidence is low)
```

---

### Issue #3: LLM Near-Zero Confidence (CRITICAL)

**Symptoms**:
```
Avg LLM confidence: 0.000232 = 0.023%
LLM outputs: "HOLD | 0.0 | ..." for 99%+ of responses
Agreement rate: 99.9% (both default to HOLD)
```

**Root Causes**:
1. **Model mismatch**: Phi-3-mini is a general-purpose model, NOT trained for trading
   - No financial domain knowledge
   - Doesn't understand trading terminology
   - Prompts are 100+ lines of specialized trading rules

2. **Prompt complexity**:
   - Template is ~150 lines ([llm_config.yaml:43-114](config/llm_config.yaml#L43))
   - Phi-3-4k context struggles with dense financial jargon
   - Instructions conflict with model's training distribution

3. **Temperature too low**:
   ```yaml
   temperature: 0.3  # Makes model conservative → defaults to HOLD
   do_sample: false  # Greedy = always picks safest option
   ```

**Impact**:
- LLM essentially abstains from all decisions
- No value added to decision fusion
- Pure latency overhead (2.3s per query)

**Fix Options**:

**Option A: Use FinGPT (Recommended)**
```yaml
llm_model:
  name: "FinGPT/fingpt-mt_llama3-8b_lora"
  # Trained on financial data, understands trading
```

**Option B: Simplify Phi-3 Prompts**
```yaml
llm_model:
  temperature: 0.7        # Increase for decisive outputs
  do_sample: true         # Enable sampling
  max_new_tokens: 50      # Reduce from 96 (faster)

prompts:
  # Reduce from 150 lines to ~30 lines
  # Focus on 3-5 key rules only
```

**Option C: Disable LLM (Fastest Fix)**
```yaml
fusion:
  use_selective_querying: false  # Disable entirely
  llm_weight: 0.0                # Pure RL mode
```

---

### Issue #4: Overfitting Gap 1.55 (HIGH)

**Symptoms**:
```
Train reward: 1.79
Val reward:   0.244
Gap:          1.55 (7.3× worse on validation!)
```

**Root Causes**:
1. **Data distribution mismatch**:
   - Training data: Different market conditions
   - Validation data: Different regime (volatility, trend)
   - No data shuffling or stratification

2. **Short episodes prevent generalization**:
   - 19.9 bars average = model overfits to entry patterns
   - Doesn't learn full trade lifecycle

3. **Frozen policy can't adapt**:
   - Phase 2 learned on 228D observations
   - Phase 3 uses 261D observations
   - Adapter too weak to bridge gap while policy frozen

**Impact**:
- Model won't generalize to live trading
- Performance metrics misleading (train Sharpe 0.12 unusable)

**Fix**:
```python
# 1. Ensure train/val from same time period
val_split = 0.2  # Last 20% of data for validation

# 2. Enable Phase 2 unfreezing earlier
'adapter_warmup_steps': 10_000,  # Not 100K

# 3. Longer episodes via termination logic
'max_episode_steps': 300,        # Not auto-terminating at 20 bars
```

---

### Issue #5: Episodes Too Short (19.9 bars) (HIGH)

**Symptoms**:
```
Episode length: 19.9 bars  (should be 100-300+)
Episode reward: 1.41       (very low)
Win rate: 66.7%            (decent but tiny profits)
```

**Root Causes**:
1. **Early stop-loss hits**: SL too tight (1.5x ATR)
2. **Drawdown limits**: Hitting trailing DD limit early
3. **Done conditions**: Environment terminating prematurely

**Impact**:
- Model doesn't learn position management
- Only learns entry signals
- Overfits to short-term patterns

**Fix** (requires environment investigation):
```python
# Check environment_phase3_llm.py termination logic
# Likely candidates:
# - Apex DD check too aggressive
# - SL distance too small
# - Time-based termination active
```

---

### Issue #6: Training Crashes on Exit (MEDIUM)

**Symptoms**:
```
terminate called without an active exception
Aborted
```

**Root Cause**:
- Async LLM cleanup not joining threads
- Model save during shutdown race condition
- TensorBoard writer not closing properly

**Impact**:
- Training completes but exits dirty
- May corrupt final checkpoint
- Log flush incomplete

**Fix**:
```python
# Add proper cleanup in train_phase3_llm.py
try:
    model.learn(...)
finally:
    # Shutdown async LLM
    if hasattr(hybrid_agent, 'async_llm'):
        hybrid_agent.async_llm.shutdown()

    # Close TensorBoard
    if hasattr(model, 'logger'):
        model.logger.close()
```

---

### Issue #7: High LLM Latency (2.3s/query) (MEDIUM)

**Symptoms**:
```
Avg latency: 2,346 ms per query
Total overhead: 372 queries × 2.3s = 14.7 minutes
```

**Root Causes**:
1. Long prompts (~150 lines)
2. `max_new_tokens: 96` (generates too much text)
3. No batching (each query isolated)
4. Possible CPU inference despite `device: auto`

**Impact**:
- Training takes 17% longer
- Live trading would be unusable (2.3s lag)

**Fix**:
```yaml
llm_model:
  max_new_tokens: 30      # Down from 96

# Simplify prompt to <50 lines
# OR switch to FinGPT (faster inference on trading queries)
```

---

## Recommended Action Plan

### Phase A: Emergency Fixes (Required for ANY useful training)

1. **Fix frozen policy** ✅ PRIORITY 1
   ```python
   # src/train_phase3_llm.py:315
   'adapter_warmup_steps': 10_000,  # Was: 100_000
   ```

2. **Fix LLM query rate** ✅ PRIORITY 2
   ```yaml
   # config/llm_config.yaml
   query_interval: 5       # Was: 20
   query_cooldown: 2       # Was: 10
   confidence_threshold: 0.60  # Was: 0.75
   ```

3. **Disable non-functional LLM** ✅ PRIORITY 3 (temporary)
   ```yaml
   fusion:
     llm_weight: 0.0  # Disable until Phi-3 prompts fixed
   ```

### Phase B: Run Pure RL Baseline (Validate fixes)

```bash
# Test with 50K timesteps
python src/train_phase3_llm.py --test --timesteps 50000

# Expected results:
# - approx_kl: 0.01-0.05 (policy learning)
# - clip_fraction: 0.05-0.15 (gradients flowing)
# - explained_variance: >0.5 (value improving)
```

### Phase C: Fix LLM Integration (Once RL working)

**Option 1: Switch to FinGPT** (Best for production)
- Download FinGPT LoRA model
- Update config to use financial-trained model
- Re-enable LLM weight

**Option 2: Fix Phi-3 Prompts** (Faster iteration)
- Simplify prompt to 30-50 lines
- Focus on 3-5 core rules
- Increase temperature to 0.7
- Enable sampling

**Option 3: Disable Hybrid** (Fallback)
- Train pure RL Phase 2 extended
- Skip LLM integration entirely

### Phase D: Fix Environment Issues

1. **Episode length**: Investigate termination logic
2. **Overfitting**: Use same time period for train/val
3. **Exit crash**: Add proper cleanup handlers

---

## Testing Protocol

### Test 1: Pure RL (No LLM)
```bash
# Disable LLM, test RL learning
python src/train_phase3_llm.py --test --timesteps 50000

# Success criteria:
✅ approx_kl > 0.01
✅ clip_fraction > 0.05
✅ explained_variance > 0.5
✅ Sharpe ratio improving over time
```

### Test 2: LLM Query Rate
```bash
# Enable LLM, verify query rate
# With query_interval=5, expect ~20% query rate

# Success criteria:
✅ LLM query rate: 15-25%
✅ LLM queries: >10,000 in 50K steps
```

### Test 3: LLM Confidence
```bash
# Monitor LLM outputs in logs
# Add: save_llm_responses: true

# Success criteria:
✅ LLM confidence > 0.3 avg
✅ Disagreement rate > 5%
✅ LLM override rate > 2%
```

### Test 4: Full Hybrid (50K)
```bash
# All fixes applied, full integration test

# Success criteria:
✅ RL learning (KL > 0.01)
✅ LLM contributing (confidence > 0.3)
✅ Fusion working (disagreements handled)
✅ No crashes
✅ Episodes > 50 bars avg
✅ Overfitting gap < 0.5
```

---

## Configuration Diff (Quick Fix)

```diff
# src/train_phase3_llm.py
- 'adapter_warmup_steps': 100_000,
+ 'adapter_warmup_steps': 10_000,

# config/llm_config.yaml
fusion:
- query_interval: 20
+ query_interval: 5
- query_cooldown: 10
+ query_cooldown: 2
- confidence_threshold: 0.75
+ confidence_threshold: 0.60

llm_model:
- max_new_tokens: 96
+ max_new_tokens: 40
- temperature: 0.3
+ temperature: 0.7
- do_sample: false
+ do_sample: true

logging:
- save_llm_responses: false
+ save_llm_responses: true  # For debugging
```

---

## Expected Results After Fixes

| Metric | Before | After (Target) |
|--------|--------|----------------|
| approx_kl | 1.2e-10 | 0.01-0.05 |
| clip_fraction | 0 | 0.05-0.15 |
| explained_variance | 0.192 | >0.6 |
| LLM query rate | 0.5% | 15-20% |
| LLM confidence | 0.023% | >30% |
| Episode length | 19.9 | >80 |
| Overfitting gap | 1.55 | <0.5 |
| Training crash | Yes | No |

---

## Long-Term Recommendations

1. **Model Selection**: Switch to FinGPT for production (Phi-3 inadequate for trading)
2. **Prompt Engineering**: Simplify to <50 lines, focus on decision trees
3. **Observation Space**: Consider reducing from 261D (may be overkill)
4. **Episode Design**: Longer episodes for better position management learning
5. **Data Strategy**: Stratified train/val split by market regime
6. **Checkpointing**: Save every 10K steps during development

---

## Files to Modify

1. `src/train_phase3_llm.py` - Fix warmup steps
2. `config/llm_config.yaml` - Fix query rate, temperature
3. `src/environment_phase3_llm.py` - Investigate termination logic (future)
4. `src/llm_reasoning.py` - Simplify prompts (optional)

---

**Next Steps**: Apply Phase A fixes → Run Test 1 (Pure RL) → Validate learning before re-enabling LLM
