# Phase 3 Critical Fixes Applied

## Summary
Fixed 3 critical issues preventing LLM from generating useful trading advice:
1. **Llama-3 chat template** - No longer using wrong prompt format
2. **Observation de-normalization** - LLM now sees real indicator values
3. **Action mask debugging** - Better visibility into why trading is blocked

---

## Fix #1: Llama-3 Chat Template

### Problem
- Config had `use_instruction_template: true` (line 57)
- This used FinGPT's "Instruction/Input/Answer" format
- Llama-3 expects `<|start_header_id|>assistant` format
- Result: Immediate `<|end_of_text|>` token, 0 output, 100% parsing errors

### Solution
**File: config/llm_config.yaml**
```yaml
# Line 57 (changed)
prompts:
  use_instruction_template: false  # Use Llama-3's native chat template
```

**File: src/llm_reasoning.py (_generate_raw method)**
- Reordered template priority: Native chat → Instruction → Plain
- Llama-3's `apply_chat_template()` now used first
- Adds proper `<|start_header_id|>assistant` markers
- Generation should now produce actual text

### Verification
```bash
python test_phase3_fixes.py
```
Should see: `✓ LLM generates text: ...`

---

## Fix #2: Observation De-normalization

### Problem
- `_build_prompt()` read raw obs slots: `obs[228]`, `obs[229]`, `obs[240]`
- These are **standardized tensors** (mean=0, std=1) from RL preprocessing
- LLM saw nonsense like:
  - `ADX: -0.5` (should be 0-100)
  - `RSI: -0.1` (should be 0-100)
  - `Price: $0.00` (should be ~$5000)
  - `Time: 00:00` (normalized timestamp)
- Result: LLM always recommended HOLD (couldn't interpret data)

### Solution
**New File: data/normalization_stats.json**
- Stores mean/std for each indicator
- Default stats for ES futures (ADX, RSI, VWAP, price, momentum)

**File: src/llm_reasoning.py (new class)**
```python
class ObservationDenormalizer:
    """Convert normalized obs back to physical scales"""
    
    def denorm(self, obs: np.ndarray, feature_indices: dict) -> dict:
        # Formula: real_value = normalized_value * std + mean
        # Example: obs[240]=-0.1 → RSI = -0.1*20 + 50 = 48
```

**File: src/llm_reasoning.py (_build_prompt updated)**
```python
# Before: adx = float(obs[228])  # Normalized value
# After:  denorm_obs = self.denormalizer.denorm(obs, self.feature_indices)
#         adx = float(denorm_obs['adx'])  # Real ADX value (0-100)
```

Now LLM sees:
- `ADX: 32.5` (strong trend)
- `RSI: 48.0` (neutral)
- `Price: $5127.50` (real ES price)
- `Time: 10:32:15` (readable time)

### Verification
```bash
python test_phase3_fixes.py
```
Should see: `✓ ADX in valid range`, `✓ RSI in valid range`, etc.

---

## Fix #3: Action Mask Debug Logging

### Problem
- LLM always had `available_actions = ['HOLD']`
- Could not recommend BUY/SELL even for FLAT positions
- Root cause: action_mask coming from env/compliance with only [1,0,0,0,0,0]
- No visibility into WHY mask was restricted

### Solution
**File: src/hybrid_agent.py (_get_available_actions)**
```python
# Added debug logging
self.logger.debug(f"[MASK] Raw action_mask: {action_mask}")
self.logger.debug(f"[MASK] Enabled indices: {np.where(action_mask)[0]}")

# Added warning for restricted masks
if len(available) <= 1:
    self.logger.warning(
        f"[MASK] Only {len(available)} action(s) available: {available}. "
        f"Mask: {action_mask}. This prevents trading!"
    )
```

**File: src/hybrid_agent.py (predict method)**
```python
# Added state logging before mask usage
self.logger.debug(
    f"[HYBRID] Env {env_id}: position={position}, balance=${balance:.0f}, "
    f"losses={consecutive_losses}, mask={action_mask}"
)
```

Now you'll see in logs:
```
[MASK] Raw action_mask: [1 0 0 0 0 0]
[MASK] Enabled indices: [0]
[MASK] Only 1 action(s) available: ['HOLD']. Mask: [1 0 0 0 0 0]. This prevents trading!
[HYBRID] Env 0: position=0, balance=$50000, losses=0, mask=[1 0 0 0 0 0]
```

This helps debug:
1. Is mask coming from environment correctly?
2. Is risk manager blocking actions inappropriately?
3. Is compliance layer too restrictive?

### Next Steps to Debug Mask
If you see restricted masks:
1. Check `environment.get_action_mask()` - should return [1,1,1,0,0,0] for FLAT
2. Check `risk_manager.get_state()` - may be blocking entries
3. Check async LLM - may be exercised before env populates tradable actions

---

## Testing

### Quick Test
```bash
python test_phase3_fixes.py
```

### Full Pipeline Test
```bash
python src/testing_framework.py --mode pipeline --market NQ
```

Look for in logs:
- ✓ `[LLM] Using Llama-3 native chat template` (not Instruction mode)
- ✓ `ADX: 32.5` (not -0.5)
- ✓ `available_actions=['BUY', 'SELL', 'HOLD']` (not just HOLD)
- ✓ No `<|end_of_text|>` in responses
- ✓ Parsing success rate > 0%

### Expected Improvements
- **LLM output**: Text generation works, no immediate termination
- **Prompt quality**: Real indicator values, readable timestamps
- **Action variety**: LLM can recommend BUY/SELL/HOLD (if mask allows)
- **Parse rate**: Should go from 0% → >50%

---

## Files Modified

### Core Fixes
1. `config/llm_config.yaml` - Disable instruction template (line 57)
2. `src/llm_reasoning.py` - Chat template priority, denormalization
3. `src/hybrid_agent.py` - Action mask debug logging

### New Files
4. `data/normalization_stats.json` - Indicator mean/std values
5. `test_phase3_fixes.py` - Validation script

---

## Rollback Instructions

If fixes cause issues:

```bash
# Revert config
git checkout config/llm_config.yaml

# Revert llm_reasoning.py  
git checkout src/llm_reasoning.py

# Revert hybrid_agent.py
git checkout src/hybrid_agent.py

# Remove new files
rm data/normalization_stats.json
rm test_phase3_fixes.py
```

---

## Next Debugging Steps

After applying fixes, if problems persist:

1. **Still no LLM output?**
   - Check `[LLM] Using Llama-3 native chat template` in logs
   - Try increasing `temperature` in config (currently 0.55)
   - Verify model loaded: Look for `[LLM] LoRA adapter attached`

2. **Still seeing weird indicator values?**
   - Check `data/normalization_stats.json` values match your data
   - Update means/stds based on your actual ES data range
   - Verify feature_indices match observation array structure

3. **Still restricted to HOLD only?**
   - Check environment's `get_action_mask()` implementation
   - Check risk_manager restrictions
   - Verify position state is passed correctly
   - Look for compliance layer blocking trades

---

## Contact

If issues persist after these fixes:
- Check logs for ERROR messages
- Run test script and share output
- Provide sample of normalized observation values
