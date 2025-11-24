# Phase Training Guide - RL Trading System
**Complete Guide to Training Phases 1, 2, and 3**

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Phase 1: Entry Signal Learning](#phase-1-entry-signal-learning)
3. [Phase 2: Position Management](#phase-2-position-management)
4. [Phase 3: Extended RL (261D Observations)](#phase-3-extended-rl-261d-observations)
5. [Training Expectations](#training-expectations)
6. [Troubleshooting](#troubleshooting)
7. [Performance Targets](#performance-targets)

---

## Overview

The RL Trading System uses a **three-phase curriculum learning approach** to train profitable trading agents for futures markets while maintaining strict Apex Trader Funding compliance.

### Training Philosophy

Each phase builds on the previous one:
- **Phase 1**: Teaches the agent WHEN to enter trades (entry signals)
- **Phase 2**: Teaches the agent HOW to manage positions (stop-loss, take-profit, trailing stops)
- **Phase 3**: Enhances decision-making with extended market context (261D vs 228D observations)

### Key Metrics by Phase

| Phase | Observations | Actions | Primary Goal | Duration |
|-------|--------------|---------|--------------|----------|
| Phase 1 | 225D | 3 (Hold/Buy/Sell) | Entry Quality | 6-8 hours (2M steps) |
| Phase 2 | 228D | 6 (+ SL/TP mgmt) | Risk Management | 8-10 hours (5M steps) |
| Phase 3 | 261D | 6 (same as P2) | Pattern Recognition | 12-16 hours (5M steps) |

---

## Phase 1: Entry Signal Learning

### Purpose
Teach the agent to identify **high-quality entry points** for long and short trades based on market conditions, technical indicators, and price action.

### Configuration
- **Total Timesteps**: 2,000,000 (2M)
- **Parallel Environments**: 80 (CPU-dependent)
- **Learning Rate**: 3e-4
- **Batch Size**: 512
- **Action Space**: 3 actions
  - 0: Hold (do nothing)
  - 1: Buy (enter long position)
  - 2: Sell (enter short position)

### Observation Space (225 Dimensions)
The agent observes 11 technical indicators across a 20-bar window plus current position state:

**Base Indicators** (11 × 20 bars = 220D):
- Price Action: Open, High, Low, Close
- Moving Averages: SMA (5, 20, 50, 200)
- Momentum: RSI (14), MACD (12, 26, 9)
- Volatility: ATR (14), Bollinger Bands (20, 2)
- Oscillators: Stochastic, Williams %R, CCI
- Strength: ADX, ROC, MFI

**Position State** (5D):
- Current position (-1 long, 0 flat, +1 short)
- Entry price
- Unrealized P&L
- Stop-loss distance
- Take-profit distance

### Reward Function
```python
# Simplified - actual implementation in environment_phase1.py
reward = (
    trade_pnl * 0.01           # Primary: Profit/loss
    - holding_penalty * 0.001  # Small penalty for inaction
    + directional_bonus        # Bonus for correct market direction
)
```

### What to Expect During Training

**Early Training (0-500K steps)**:
- Random exploration, high entropy
- approx_kl: 0.02-0.05 (high policy changes)
- clip_fraction: 0.10-0.20 (many gradient updates)
- explained_variance: 0.2-0.4 (value network learning)
- Reward: Highly volatile, often negative

**Mid Training (500K-1.5M steps)**:
- Agent begins recognizing patterns
- approx_kl: 0.01-0.03 (policy stabilizing)
- clip_fraction: 0.05-0.15 (moderate updates)
- explained_variance: 0.4-0.7 (value improving)
- Reward: Starts trending positive

**Late Training (1.5M-2M steps)**:
- Policy convergence, exploitation focus
- approx_kl: 0.005-0.02 (small policy changes)
- clip_fraction: 0.02-0.10 (fine-tuning)
- explained_variance: 0.7-0.9 (good value estimation)
- Reward: Stable, positive (if successful)

### Expected Outcomes
- **Win Rate**: 50-55% (break-even or slightly profitable)
- **Sharpe Ratio**: 1.0-1.5 (moderate risk-adjusted returns)
- **Max Drawdown**: ~10-15% (unoptimized SL/TP)
- **Entry Quality**: Agent learns to avoid bad entries

### TensorBoard Metrics to Monitor
```
rollout/ep_rew_mean          # Average episode reward (should trend up)
train/approx_kl              # Policy change magnitude (should decrease)
train/clip_fraction          # Gradient clipping frequency (should stabilize)
train/explained_variance     # Value network quality (should approach 1.0)
train/entropy_loss           # Exploration level (should gradually decrease)
train/policy_gradient_loss   # Policy optimization progress
train/value_loss            # Value network training loss
```

### Success Criteria
✅ **Phase 1 Complete When**:
- explained_variance > 0.7
- approx_kl < 0.02 (stable policy)
- Mean reward positive for last 100K steps
- Model saved to `models/phase1_foundational_final.zip`

---

## Phase 2: Position Management

### Purpose
Teach the agent **advanced risk management** including dynamic stop-loss adjustment, trailing stops, and position sizing while maintaining strict Apex compliance.

### Configuration
- **Total Timesteps**: 5,000,000 (5M)
- **Parallel Environments**: 80 (CPU-dependent)
- **Learning Rate**: 3e-4
- **Batch Size**: 512
- **Action Space**: 6 actions
  - 0: Hold
  - 1: Buy (long entry)
  - 2: Sell (short entry)
  - 3: Move SL to Break-Even
  - 4: Enable Trailing Stop
  - 5: Disable Trailing Stop

### Observation Space (228 Dimensions)
Extends Phase 1 (225D) with action validity features:

**Base Observations** (225D):
- Same as Phase 1 (11 indicators × 20 bars + 5 position state)

**Action Masking Features** (3D):
- Can enter new position? (bool)
- Can adjust SL/TP? (bool)
- Can enable trailing? (bool)

### Transfer Learning from Phase 1
Phase 2 **automatically loads the newest Phase 1 model** to inherit learned entry signals:

```python
# From train_phase2.py
phase1_models = detect_models_in_folder(phase='phase1')
if phase1_models:
    phase1_path = phase1_models[0]['path']  # Newest model
    model = MaskablePPO.load(phase1_path, env=env)
    print(f"[TRANSFER] Loaded Phase 1 model: {phase1_path}")
```

**Benefits**:
- Faster convergence (20-30% faster than training from scratch)
- Better final performance
- Agent starts with knowledge of good entry points

### Apex Compliance Enforcement

Phase 2 enforces Apex rules through a **three-layer safety system**:

**Layer 1: Environment Rewards**
```python
# Penalty for violating trailing drawdown
if current_drawdown > 2500:
    reward -= 100  # Severe penalty
    done = True
```

**Layer 2: Action Masking**
```python
# Prevent new entries if near DD limit
def action_masks():
    can_enter = (drawdown < 2000)  # $500 buffer
    can_exit = (position != 0)
    return [True, can_enter, can_enter, can_exit, ...]
```

**Layer 3: Post-Training Verification**
```python
# From apex_compliance_checker.py
results = verify_apex_compliance(model, eval_env)
assert results['max_drawdown'] <= 2500
assert results['overnight_positions'] == 0
```

### What to Expect During Training

**Early Training (0-1M steps)**:
- Learning position management basics
- Higher variance than Phase 1 (more actions)
- approx_kl: 0.02-0.04
- clip_fraction: 0.10-0.18
- Reward: Volatile as agent experiments with SL/TP

**Mid Training (1M-3M steps)**:
- Agent masters break-even stops
- Begins using trailing stops effectively
- approx_kl: 0.01-0.03
- clip_fraction: 0.05-0.12
- Reward: Improving, less volatile

**Late Training (3M-5M steps)**:
- Sophisticated risk management
- Optimal SL/TP adjustment timing
- approx_kl: 0.005-0.02
- clip_fraction: 0.02-0.08
- Reward: Stable, consistently positive

### Expected Outcomes
- **Win Rate**: 55-60% (improved vs Phase 1)
- **Sharpe Ratio**: 2.0-2.5 (target: >2.5 for Apex)
- **Max Drawdown**: <5% (well below $2,500 limit)
- **Profit Factor**: 1.5-2.0
- **Avg Trade Duration**: 50-150 bars

### TensorBoard Metrics to Monitor
Same as Phase 1, plus:
```
custom/apex_violations          # Should be 0
custom/trailing_stop_activations  # Should increase over time
custom/breakeven_stop_moves     # Frequency of BE adjustments
custom/position_hold_time       # Average bars per trade
```

### Success Criteria
✅ **Phase 2 Complete When**:
- Sharpe ratio > 2.0 on validation data
- Max drawdown < $1,500 (buffer below Apex limit)
- No Apex compliance violations
- explained_variance > 0.75
- Model saved to `models/phase2_position_mgmt_final.zip`

---

## Phase 3: Extended RL (261D Observations)

### Purpose
**CURRENT MODE: PURE RL (LLM DISABLED)**

Phase 3 enhances the agent with **extended market context features** originally designed for LLM reasoning but proven valuable for pure RL pattern recognition.

### 🎯 Key Changes in Phase 3 (v2.0 - Pure RL Mode)

**WHAT CHANGED**:
- ✅ LLM completely disabled (no model loading, no queries)
- ✅ 261D observations retained (extended features help RL directly)
- ✅ Mock mode enabled for LLM module (bypasses dependencies)
- ✅ Clean shutdown (no more "terminate called" crashes)
- ✅ Episode termination diagnostics added

**WHAT STAYED THE SAME**:
- Same action space as Phase 2 (6 actions)
- Same transfer learning from Phase 2
- Same Apex compliance enforcement
- Same training duration (5M timesteps)

### Configuration
- **Total Timesteps**: 5,000,000 (5M)
- **Parallel Environments**: 8-20 (LLM overhead removed in pure RL mode)
- **Learning Rate**: 3e-4
- **Batch Size**: 256
- **Action Space**: 6 actions (same as Phase 2)
- **LLM Mode**: DISABLED (mock_mode=True)

### Observation Space (261 Dimensions)

**Base Observations** (228D):
- Same as Phase 2 (Phase 1 features + action masking)

**Extended LLM Features** (33D):
These features were originally designed for LLM context but help RL learn patterns directly:

1. **ADX Trend Slope** (1D): Rate of trend strength change
2. **VWAP Distance** (1D): Price deviation from volume-weighted average
3. **Multi-Timeframe SMAs** (4D): SMA-50, SMA-200, and their slopes
4. **Multi-Timeframe RSI** (2D): RSI on 15-min and 60-min timeframes
5. **Volume Ratios** (2D): Current volume vs 5-min and 20-min averages
6. **Support/Resistance** (4D): Nearest 20-bar support/resistance levels + distance
7. **Price Change Metrics** (4D): 5-min, 15-min, 30-min, 60-min returns
8. **Pattern Recognition** (8D):
   - Higher high / lower low flags
   - Double top/bottom signals
   - Breakout/breakdown indicators
   - Consolidation range detection
9. **Risk Context** (4D):
   - Unrealized P&L
   - Distance to DD limit
   - Recent win/loss streak
   - Current volatility regime
10. **Session Features** (3D):
    - Time of day (normalized)
    - Distance from session open/close
    - Volume profile

### Transfer Learning from Phase 2
Phase 3 uses **adapter network architecture** to bridge the observation space gap:

```python
# Phase 2: 228D observations
# Phase 3: 261D observations
# Gap: 33D extra features

# Solution: Small adapter network
adapter = nn.Linear(261, 228)  # Projects 261D → 228D
phase3_model = load_phase2_and_add_adapter(phase2_model, adapter)
```

**Warmup Strategy**:
1. **Steps 0-10K**: Only train adapter (Phase 2 weights frozen)
2. **Steps 10K+**: Unfreeze all weights for full training

This allows the adapter to learn optimal feature projection before fine-tuning the main policy.

### What to Expect During Training

**Adapter Warmup (0-10K steps)**:
- Low training variance (most weights frozen)
- approx_kl: ~0.0001-0.001 (minimal policy change)
- clip_fraction: 0.01-0.05 (small updates)
- explained_variance: Inherited from Phase 2 (~0.7-0.8)

**Early Training (10K-1M steps)**:
- Full network training begins
- Learning to use extended features
- approx_kl: 0.02-0.04 (similar to Phase 2 start)
- clip_fraction: 0.08-0.15
- Reward: May dip initially as network adjusts

**Mid Training (1M-3M steps)**:
- Agent integrates extended context
- Better pattern recognition than Phase 2
- approx_kl: 0.01-0.03
- clip_fraction: 0.05-0.12
- Reward: Should exceed Phase 2 baseline

**Late Training (3M-5M steps)**:
- Optimal use of 261D feature space
- Sophisticated multi-timeframe awareness
- approx_kl: 0.005-0.02
- clip_fraction: 0.02-0.08
- Reward: Stable, best of all phases

### Expected Outcomes vs Phase 2
| Metric | Phase 2 | Phase 3 Target | Improvement |
|--------|---------|----------------|-------------|
| Win Rate | 55-60% | 60-65% | +5-10% |
| Sharpe Ratio | 2.0-2.5 | 2.5-3.0 | +20% |
| Max Drawdown | <5% | <4% | -20% |
| Profit Factor | 1.5-2.0 | 2.0-2.5 | +25% |
| Avg Win Size | $150 | $180 | +20% |
| False Signal Reduction | Baseline | -30% | Better entries |

### TensorBoard Metrics to Monitor
Same as Phase 2, plus:
```
custom/multi_timeframe_agreement  # How often 15min/60min RSI align
custom/vwap_crossover_success     # Win rate when crossing VWAP
custom/pattern_detection_accuracy  # Success rate of pattern signals
custom/adapter_activation_mean    # Adapter layer statistics
```

### Success Criteria
✅ **Phase 3 Complete When**:
- Sharpe ratio > 2.5 on validation data
- Outperforms Phase 2 by >10% on key metrics
- explained_variance > 0.80
- approx_kl < 0.015 (more stable than Phase 2)
- No Apex compliance violations
- Model saved to `models/phase3_extended_rl_final.zip`

---

## Training Expectations

### Hardware Requirements

**Minimum (All Phases)**:
- CPU: 8+ cores
- RAM: 16GB
- GPU: Optional (CUDA 11.8+)
- Disk: 20GB free

**Recommended (Phase 3)**:
- CPU: 16+ cores (for parallel envs)
- RAM: 32GB
- GPU: RTX 3060+ (6GB+ VRAM)
- Disk: 50GB free

### Training Times

| Phase | Test Mode | Production Mode | GPU Speedup |
|-------|-----------|-----------------|-------------|
| Phase 1 | 5-10 min (50K steps) | 6-8 hours (2M steps) | 2-3× |
| Phase 2 | 10-15 min (50K steps) | 8-10 hours (5M steps) | 2-3× |
| Phase 3 | 15-20 min (50K steps) | 12-16 hours (5M steps) | 2-3× |
| **Total Pipeline** | **30-45 minutes** | **26-34 hours** | **2-3×** |

### Disk Space Usage

```
models/                    # Model checkpoints
├── phase1_*.zip          # ~50MB per checkpoint
├── phase2_*.zip          # ~60MB per checkpoint
├── phase3_*.zip          # ~65MB per checkpoint
└── vecnormalize/         # ~5MB total

tensorboard_logs/         # Training logs
├── phase1/              # ~200MB
├── phase2/              # ~500MB
└── phase3/              # ~500MB

Total: ~2-3GB for complete pipeline
```

### GPU Memory Usage

| Phase | CPU Mode | GPU Mode | Notes |
|-------|----------|----------|-------|
| Phase 1 | N/A | ~2GB VRAM | Policy + value networks |
| Phase 2 | N/A | ~2.5GB VRAM | Larger network |
| Phase 3 | N/A | ~3GB VRAM | 261D input layer |

**Note**: Phase 3 no longer requires LLM (no Phi-3/FinGPT), so total VRAM is only ~3GB.

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Training Not Learning (approx_kl near zero)

**Symptoms**:
```
approx_kl: 1.24e-10
clip_fraction: 0.0
explained_variance: 0.2
```

**Root Cause**: Policy frozen (Phase 3 adapter warmup misconfigured)

**Fix**:
```python
# In train_phase3_llm.py:315
'adapter_warmup_steps': 10_000,  # Was: 100_000 (too high!)
```

---

#### Issue: Episodes Too Short (~20 bars)

**Symptoms**:
```
Episode length: 19.9 bars
Expected: 100-300 bars
```

**Diagnosis**:
```bash
# Run diagnostic script
python src/diagnostics/episode_termination_analysis.py --market NQ --episodes 100
```

**Common Causes**:
1. **Stop-loss too tight** (80% of cases)
   - Fix: Increase `initial_sl_multiplier` from 1.5 to 2.0-2.5
2. **Drawdown limit hit early**
   - Fix: Temporarily increase `trailing_drawdown_limit` to $5,000 for training
3. **Data exhausted**
   - Fix: Download 6-12 months of data (150K+ rows target)

---

#### Issue: Training Crashes on Exit

**Symptoms**:
```
terminate called without an active exception
Aborted
```

**Root Cause**: Async threads not cleaned up properly

**Status**: ✅ **FIXED in current version** (comprehensive cleanup added)

**Verification**:
```bash
# Check cleanup code exists in train_phase3_llm.py
grep -A 10 "finally:" src/train_phase3_llm.py
```

---

#### Issue: High GPU Memory Usage

**Symptoms**:
```
CUDA out of memory: tried to allocate X GB
```

**Solutions**:
1. Reduce batch size:
   ```python
   'batch_size': 128,  # From 256
   ```
2. Reduce parallel environments:
   ```python
   'n_envs': 4,  # From 8
   ```
3. Use CPU mode (slower but no VRAM):
   ```python
   'device': 'cpu',
   ```

---

#### Issue: Poor Validation Performance (Overfitting)

**Symptoms**:
```
Train reward: 1.79
Val reward: 0.244
Gap: 1.55 (7.3× worse!)
```

**Causes**:
1. Train/val data from different market regimes
2. Episodes too short (agent only learns entries, not full lifecycle)
3. Insufficient data diversity

**Fixes**:
1. Ensure train/val split by time (not random):
   ```python
   train_end = int(len(data) * 0.8)
   train_data = data.iloc[:train_end]
   val_data = data.iloc[train_end:]
   ```
2. Fix episode length (see above)
3. Expand dataset to 150K+ rows

---

## Performance Targets

### Evaluation Metrics Explained

**Sharpe Ratio** (Risk-adjusted returns):
```
Sharpe = (Mean Return - Risk-Free Rate) / Std Dev of Returns
Target: >2.5 for Apex funding
```

**Max Drawdown** (Worst peak-to-trough decline):
```
Max DD = (Peak Balance - Trough Balance) / Peak Balance
Apex Limit: $2,500 (5% of $50K account)
```

**Profit Factor** (Gross profit / gross loss):
```
PF = Sum(Winning Trades) / Sum(Losing Trades)
Target: >1.5 (win $1.50 for every $1.00 lost)
```

**Win Rate** (Percentage of profitable trades):
```
Win Rate = Winning Trades / Total Trades
Target: >55% (with 2:1+ reward:risk, 45% can work)
```

### Phase-by-Phase Targets

| Metric | Phase 1 | Phase 2 | Phase 3 | Apex Req |
|--------|---------|---------|---------|----------|
| **Sharpe Ratio** | 1.0-1.5 | 2.0-2.5 | 2.5-3.0 | >2.5 |
| **Win Rate** | 50-55% | 55-60% | 60-65% | N/A |
| **Max DD** | 10-15% | <5% | <4% | <5% |
| **Profit Factor** | 1.2-1.5 | 1.5-2.0 | 2.0-2.5 | >1.5 |
| **Avg R:R** | 1.5:1 | 2:1 | 2.5:1 | N/A |
| **Daily Return** | 0.5-1% | 1-2% | 2-3% | N/A |

### Apex Trader Funding Requirements

✅ **Must Meet ALL Requirements**:
- ✅ Max Trailing Drawdown: $2,500 (from peak balance)
- ✅ All positions closed by 4:59 PM ET
- ✅ No overnight positions
- ✅ Minimum 7 trading days
- ✅ Profit target: $3,000 for $50K account (6%)

**Verification Command**:
```bash
python src/apex_compliance_checker.py --model models/phase3_extended_rl_final --market NQ
```

---

## Quick Reference Commands

### Data Processing
```bash
# Full reprocessing
python src/update_training_data.py --market NQ

# Incremental update (10× faster)
python src/incremental_data_updater.py --market NQ
```

### Training
```bash
# Test Mode (50K steps, ~15 minutes)
python src/train_phase1.py --test --market NQ
python src/train_phase2.py --test --market NQ
python src/train_phase3_llm.py --test --market NQ

# Production Mode (full training)
python src/train_phase1.py --market NQ
python src/train_phase2.py --market NQ
python src/train_phase3_llm.py --market NQ

# Complete Pipeline (via menu)
python main.py
# Select: 3. Training Model → 1. Complete Training Pipeline (Test/Production)
```

### Continue Training
```bash
# Auto-detects newest checkpoint
python src/train_phase1.py --continue

# Specify checkpoint
python src/train_phase1.py --continue --model-path models/phase1_checkpoint_1M.zip
```

### Evaluation
```bash
python src/evaluate_phase2.py --model models/phase2_position_mgmt_final.zip
python src/evaluate_phase3_llm.py --model models/phase3_extended_rl_final --market NQ
```

### Diagnostics
```bash
# Episode termination analysis
python src/diagnostics/episode_termination_analysis.py --market NQ --episodes 100 --save-csv

# Environment validation
python src/diagnose_environment.py

# LLM setup verification (if re-enabling LLM)
python src/verify_llm_setup.py
```

### Monitoring
```bash
# TensorBoard (real-time training metrics)
tensorboard --logdir tensorboard_logs/

# View in browser
# Navigate to: http://localhost:6006
```

---

## Next Steps

After completing Phase 3 training:

1. **✅ Evaluate Performance**
   ```bash
   python src/evaluate_phase3_llm.py --model models/phase3_extended_rl_final --market NQ
   ```

2. **✅ Verify Apex Compliance**
   ```bash
   python src/apex_compliance_checker.py --model models/phase3_extended_rl_final --market NQ
   ```

3. **✅ Run Backtest on Unseen Data**
   - Use data from different time period
   - Verify generalization

4. **✅ Paper Trading** (if compliant)
   - Deploy to NinjaTrader 8
   - Monitor real-time performance
   - Track slippage and execution

5. **✅ Live Trading** (if paper trading successful)
   - Start with micro contracts (MNQ, MES)
   - Scale up after consistent profitability
   - Maintain strict risk management

---

## Additional Resources

- **Apex Rules**: See [docs/Apex-Rules.md](Apex-Rules.md)
- **Data Processing**: See [docs/QUICK_START_DATA_PROCESSING.md](QUICK_START_DATA_PROCESSING.md)
- **NinjaTrader Integration**: See [docs/NINJATRADER8_INTEGRATION_REQUIREMENTS.md](NINJATRADER8_INTEGRATION_REQUIREMENTS.md)
- **Project Overview**: See [CLAUDE.md](../CLAUDE.md)

---

**Questions? Issues?**
- Check logs in `logs/` directory
- Run diagnostics: `python src/diagnostics/episode_termination_analysis.py`
- Review recent fixes: `docs/FIXES_SUMMARY.md`

**Happy Training! 🚀**
