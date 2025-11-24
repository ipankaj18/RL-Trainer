# Single Trade Validation Test

## Overview

The single trade validation test (`tests/test_single_trade_validation.py`) validates that ONE complete trade executes correctly with all position management actions working as expected. This provides **high confidence** that the Phase 3 environment is wired correctly before running expensive 5M-step training.

## Purpose

Before investing 12-16 hours in full training, this test confirms:

✅ Environment initializes without errors
✅ All 6 actions work correctly (HOLD, BUY, SELL, MOVE_TO_BE, ENABLE_TRAIL, DISABLE_TRAIL)
✅ Trade opens with correct commission and slippage
✅ Position management actions execute properly
✅ Balance and P&L calculations are accurate
✅ Action masking prevents invalid actions
✅ Trade closes profitably at take profit

**If this test passes**, you have confidence that the environment will provide correct learning signals during training.

## How It Works

### 1. Synthetic Data Generation

Creates 50 bars of deterministic price movement:

```
Bars 0-20:  Sideways at $20,000 (warmup for indicators)
Bar 21:     Entry point at $20,000
Bars 22-27: Gradual rise to $20,015 (enables breakeven move)
Bars 28-35: Rise to $20,040 (enables trailing stop)
Bars 36+:   Rise to $20,080 (hits take profit)
```

### 2. Hardcoded Action Sequence

Instead of random actions, the test executes a specific sequence:

```
Step 1: HOLD          (verify pre-trade state)
Step 2: BUY           (open long position at $20,000)
Step 3-4: HOLD        (build profit)
Step 5: MOVE_TO_BE    (move stop loss to breakeven)
Step 6-7: HOLD        (build more profit)
Step 8: ENABLE_TRAIL  (activate trailing stop)
Step 9+: HOLD         (wait for take profit exit)
```

### 3. Validation at Each Step

At each step, the test validates:

- **Balance changes** match expected values (±$0.01 tolerance)
- **Position state** is correct (0=flat, 1=long, -1=short)
- **Action masking** only allows valid actions
- **Price levels** (SL/TP) set correctly based on ATR
- **Trade P&L** (unrealized and realized) calculated correctly
- **Info dict** contains expected keys and values
- **Exit reason** is correct when trade closes

## Running the Test

### Method 1: Direct Python Execution

```bash
cd "/home/javlo/Code Projects/RL Trainner & Executor System/AI Trainer"
python tests/test_single_trade_validation.py
```

### Method 2: Via pytest

```bash
cd "/home/javlo/Code Projects/RL Trainner & Executor System/AI Trainer"
pytest tests/test_single_trade_validation.py -v
```

### Expected Runtime

**30-60 seconds** (most time spent generating features)

## Expected Output

### Success Output

```
================================================================================
SINGLE TRADE LIFECYCLE VALIDATION TEST
================================================================================

[DATA] Creating synthetic test data...
[DATA] Adding technical indicators...
[INDICATORS] Adding technical indicators...
  [OK] Moving averages (SMA-5, SMA-20)
  [OK] RSI
  [OK] MACD
  ...
[DATA] Created 50 bars from 2024-11-20 09:30:00-05:00 to 2024-11-20 10:19:00-05:00

[MARKET] Using E-mini Nasdaq-100 specifications:
  Contract multiplier: $20.0/point
  Tick value: $5.0
  Commission: $2.5/side

[ENV] Creating deterministic environment...
[ENV] Phase 3 LLM environment initialized with 261D observations
[ENV] Environment created successfully

--------------------------------------------------------------------------------
STEP 2: BUY (Open long position)
--------------------------------------------------------------------------------
✅ Position: 1 (long)
✅ Balance: $49,997.50 (entry commission deducted)
✅ Entry price: $20000.25
✅ Stop loss: $19987.05
✅ Take profit: $20039.86

--------------------------------------------------------------------------------
STEP 5: MOVE_TO_BE (Move stop loss to breakeven)
--------------------------------------------------------------------------------
✅ Stop loss moved: $19987.05 → $20000.50
✅ Now at breakeven (near entry $20000.25)

--------------------------------------------------------------------------------
STEP 8: ENABLE_TRAIL (Activate trailing stop)
--------------------------------------------------------------------------------
✅ Trailing stop activated

🎯 Trade closed at step 15!
   Exit reason: take_profit
   Realized P&L: $787.21
   Balance: $50,784.71

================================================================================
✅ ALL VALIDATIONS PASSED!
================================================================================

🎉 Single trade lifecycle validated successfully!
   Net profit: $784.71 on $50,000.00 initial capital
   Return: 1.57%

✅ Test completed successfully!
```

### Failure Output

If a validation fails, you'll see:

```
❌ VALIDATION FAILED: Balance mismatch: expected $49,997.50, got $50,000.00

AssertionError: Balance should decrease by commission on entry
```

## What Gets Validated

### Step 1: HOLD (Pre-trade)
- ✅ Position = 0 (flat)
- ✅ Balance = $50,000 (unchanged)
- ✅ Action mask = `[T, T, T, F, F, F]` (can enter, cannot manage)

### Step 2: BUY (Entry)
- ✅ Position = 1 (long)
- ✅ Balance = $49,997.50 (commission $2.50 deducted)
- ✅ Entry price set with slippage
- ✅ SL/TP levels set based on 2.5 ATR
- ✅ Action mask = `[T, F, F, ?, ?, ?]` (cannot re-enter)

### Step 3-4: HOLD (Build profit)
- ✅ Position maintained
- ✅ Balance unchanged
- ✅ Unrealized P&L increases
- ✅ Action mask enables breakeven move when profitable

### Step 5: MOVE_TO_BE
- ✅ Stop loss moves from ~$19,987 → ~$20,000 (near entry)
- ✅ Balance unchanged (no transaction cost)
- ✅ Action mask disables MOVE_TO_BE (can't move twice)
- ✅ Positive reward for protective action

### Step 6-7: HOLD (Build more profit)
- ✅ Unrealized P&L continues growing
- ✅ Action mask enables trailing stop

### Step 8: ENABLE_TRAIL
- ✅ Trailing stop activated
- ✅ Action mask disables ENABLE_TRAIL, enables DISABLE_TRAIL
- ✅ Balance unchanged

### Step 9+: HOLD (Wait for exit)
- ✅ Position closes when TP hit
- ✅ Realized P&L = ~$787 (price move - commissions)
- ✅ Balance = $50,784.71
- ✅ Exit reason = 'take_profit'

## Expected Values (NQ Contract)

Based on the synthetic data and NQ specifications:

| Metric | Expected Value | Calculation |
|--------|---------------|-------------|
| **Entry Price** | $20,000.25 | Base price + 1 tick slippage |
| **Initial SL** | $19,987.05 | Entry - (2.5 ATR × 15) |
| **Initial TP** | $20,039.86 | Entry + (2.5 ATR × 15 × 3) |
| **BE Stop** | $20,000.50 | Entry + buffer |
| **Exit Price** | ~$20,039 | TP level |
| **Price Move** | ~39 points | Exit - Entry |
| **Gross Profit** | ~$780 | 39 points × $20/point |
| **Commissions** | $5.00 | $2.50 entry + $2.50 exit |
| **Net Profit** | ~$785 | Gross - commissions |
| **Return** | ~1.57% | $785 / $50,000 |

## Interpreting Results

### ✅ Test Passed

**Meaning**: Your Phase 3 environment is working correctly!

**Next Steps**:
1. **Run 10K test training** (15 min) to validate learning loop
2. **Launch full 5M training** (12-16 hours) with confidence

### ❌ Test Failed

**Common Failures**:

1. **Balance Mismatch**
   - **Symptom**: Balance doesn't match expected after entry/exit
   - **Cause**: Commission or slippage calculation error
   - **Fix**: Check `environment_phase1.py` step() method

2. **Action Masking Error**
   - **Symptom**: Invalid actions allowed or valid actions blocked
   - **Cause**: Action mask logic incorrect
   - **Fix**: Check `environment_phase2.py` action_masks() method

3. **Position Not Opening**
   - **Symptom**: Position = 0 after BUY action
   - **Cause**: Action not processed correctly
   - **Fix**: Check step() method in environment

4. **SL/TP Not Set**
   - **Symptom**: AttributeError for sl_price or tp_price
   - **Cause**: Entry logic not setting price levels
   - **Fix**: Check entry logic in environment

5. **Trade Not Closing**
   - **Symptom**: Position stays open for 30+ steps
   - **Cause**: TP level never reached or exit logic broken
   - **Fix**: Check synthetic data prices or exit logic

## Customization

### Test Different Scenarios

You can modify the test to validate other scenarios:

#### Test a Short Trade

```python
# Change line 240 from:
action = 1  # BUY

# To:
action = 2  # SELL
```

#### Test Stop Loss Exit

Modify `create_synthetic_test_data()` to make prices drop instead of rise:

```python
# Change bars 22+ from rising to falling
prices[22:28] = np.linspace(base_price - 5, base_price - 15, 6)
prices[28:36] = np.linspace(base_price - 20, base_price - 50, 8)
```

#### Test Higher Volatility

```python
# In create_synthetic_test_data(), increase price variations
data['high'] = prices + np.abs(np.random.normal(5, 2, num_bars))  # More spread
data['low'] = prices - np.abs(np.random.normal(5, 2, num_bars))
```

## Integration with Training Pipeline

### Before Phase 3 Training

**Always run this test before starting expensive training:**

```bash
# 1. Run validation test
python tests/test_single_trade_validation.py

# 2. If test passes, run 10K test
python src/train_phase3_llm.py --test --vec-env dummy --n-envs 4

# 3. If test succeeds, launch full training
python src/train_phase3_llm.py --vec-env dummy --n-envs 8
```

### After Environment Changes

If you modify the environment code, **rerun this test**:

```bash
# After changes to:
# - environment_phase1.py (step logic)
# - environment_phase2.py (action masking)
# - environment_phase3_llm.py (observation space)
# - market_specs.py (contract specs)

python tests/test_single_trade_validation.py
```

### Continuous Integration (CI/CD)

Add to your CI pipeline:

```yaml
# .github/workflows/test.yml
- name: Run Single Trade Validation
  run: |
    python tests/test_single_trade_validation.py
```

## Technical Details

### Deterministic Settings

The test uses these settings to ensure reproducibility:

```python
env = TradingEnvironmentPhase3LLM(
    randomize_start_offsets=False,  # Always start at same point
    start_index=20,                  # Fixed start after warmup
    trailing_drawdown_limit=10000,   # High limit (won't hit)
    initial_sl_multiplier=2.5,       # Wide stops (won't hit early)
)

obs, info = env.reset(seed=42)       # Fixed random seed
```

### Synthetic Data Features

The generated data includes all 46 required features:

**Base OHLCV**: open, high, low, close, volume
**Technical Indicators** (15): SMA, RSI, MACD, ATR, Bollinger Bands, etc.
**Market Regime** (10): Volatility regime, ADX, VWAP, microstructure
**LLM Features** (21): Multi-timeframe, patterns, support/resistance

**Total**: 42 features → 261D observations (with 20-bar window)

### Why This Test Matters

**Problem**: Training 5M timesteps takes 12-16 hours. If the environment has a bug (incorrect P&L, broken action masking, etc.), you waste that time and get a broken model.

**Solution**: This 60-second test validates the environment works correctly for a single profitable trade, giving you confidence that:

1. Actions execute correctly
2. Balance calculations are accurate
3. Position management works
4. Rewards are meaningful
5. Training will learn from correct signals

**ROI**: 60 seconds of testing saves 12-16 hours of wasted training.

## Troubleshooting

### Test Hangs at Data Generation

**Symptom**: Stuck at "Adding technical indicators..."

**Solution**:
```bash
# Check if indicators are calculating correctly
python -c "from technical_indicators import add_all_indicators; print('OK')"
```

### Environment Initialization Fails

**Symptom**: Error during environment creation

**Solution**:
```bash
# Test environment import
python -c "from environment_phase3_llm import TradingEnvironmentPhase3LLM; print('OK')"
```

### Feature Count Mismatch

**Symptom**: "Expected 46 features, got 42"

**Solution**: Ensure `add_market_regime_features()` is called after `add_all_indicators()`

## Summary

✅ **Purpose**: Validate one complete trade before expensive training
✅ **Runtime**: 30-60 seconds
✅ **Location**: `tests/test_single_trade_validation.py`
✅ **Usage**: `python tests/test_single_trade_validation.py`
✅ **Success**: All 6 actions work, trade closes profitably
✅ **Confidence**: Environment ready for 5M-step training

**Next Steps After Test Passes**:
1. Run 10K test training (15 min)
2. Launch full 5M training (12-16 hours)
3. Profit! 🚀
