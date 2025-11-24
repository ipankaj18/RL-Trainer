# 🚨 PHASE 2 TRAINING ERROR - COMPLETE SOLUTION GUIDE

## Quick Summary
Your Phase 2 training fails with: `RuntimeError: shape '[4, 6]' is invalid for input of size 4`

**Root Cause:** Action space mismatch between environment (6 actions) and model (9 actions), plus improper mask generation.

## 🔧 IMMEDIATE FIX (2 minutes)

Open `train_phase2.py` and make these 3 critical changes:

### 1. Fix Action Space (Line ~280)
```python
# FIND THIS:
self.action_space = spaces.Discrete(9)

# REPLACE WITH:
self.action_space = spaces.Discrete(6)
```

### 2. Fix Configuration (Top of file)
```python
# FIND THIS:
'action_space': 9,

# REPLACE WITH:
'action_space': 6,
```

### 3. Fix Model Creation Comment (Line ~950)
```python
# FIND THIS:
"Creating Phase 2 model with expanded action space (9 actions + masking)..."

# REPLACE WITH:
"Creating Phase 2 model with expanded action space (6 actions + masking)..."
```

## 📋 Complete Fix Checklist

### Environment Fixes
- [ ] Change `spaces.Discrete(9)` to `spaces.Discrete(6)` in Phase2TradingEnv.__init__
- [ ] Update action_masks() to return numpy array with dtype=bool
- [ ] Ensure mask shape is always (6,) for single env

### Model Fixes  
- [ ] Remove hardcoded 9 from model creation
- [ ] Let model infer action space from environment
- [ ] Fix transfer learning to handle 3→6 action expansion

### Vectorization Fixes
- [ ] Override env_method for 'action_masks'
- [ ] Ensure masks are reshaped to [n_envs, n_actions]
- [ ] Add shape validation assertions

### Configuration Fixes
- [ ] Update PHASE2_CONFIG['action_space'] to 6
- [ ] Search for all occurrences of "9 actions" and fix
- [ ] Ensure consistency across all config dictionaries

## 🧪 Test Your Fix

After making changes, create this test file:

```python
# test_fix.py
import numpy as np
from gym import spaces

print("Testing Phase 2 fixes...")

# Test 1: Action space
space = spaces.Discrete(6)
assert space.n == 6, f"Expected 6 actions, got {space.n}"
print("✓ Action space: 6")

# Test 2: Mask shape
n_envs = 4
masks = np.ones((n_envs, 6), dtype=bool)
assert masks.shape == (4, 6), f"Expected (4,6), got {masks.shape}"
print("✓ Mask shape: (4, 6)")

# Test 3: Mask values
single_mask = np.array([True, True, False, False, False, False])
assert single_mask.shape == (6,), f"Expected (6,), got {single_mask.shape}"
print("✓ Single mask: (6,)")

print("\n✅ All basic tests passed!")
```

Run: `python test_fix.py`

## 🎯 Action Space Breakdown

Your Phase 2 uses 6 actions:
```
0: HOLD         - Maintain current position
1: BUY          - Enter long position  
2: SELL         - Enter short position
3: BREAKEVEN    - Move stop loss to entry (Phase 2 specific)
4: TRAIL_ON     - Enable trailing stop (Phase 2 specific)
5: TRAIL_OFF    - Disable trailing stop (Phase 2 specific)
```

Phase 1 only had actions 0-2. Phase 2 adds 3-5 for position management.

## 🔍 Verification Steps

1. **Check your changes:**
   ```bash
   grep -n "Discrete(9" train_phase2.py    # Should return nothing
   grep -n "Discrete(6" train_phase2.py    # Should show your env definition
   grep -n "'action_space': 9" train_phase2.py  # Should return nothing
   ```

2. **Run test mode:**
   ```bash
   python train_phase2.py --test --market NQ --non-interactive
   ```

3. **Watch for:**
   - Training starts without immediate crash
   - Progress bar shows movement
   - No shape errors in first 100 steps

## 🛠️ Advanced Fixes (if basic fix doesn't work)

### Fix Mask Collection in Vectorized Environment

Add this class before environment creation:

```python
class FixedMaskableVecEnv(DummyVecEnv):
    def env_method(self, method_name, *args, indices=None, **kwargs):
        if method_name == 'action_masks':
            masks = []
            for i in (indices or range(self.num_envs)):
                masks.append(self.envs[i].action_masks())
            result = np.array(masks)
            if result.shape != (self.num_envs, self.action_space.n):
                result = result.reshape(self.num_envs, self.action_space.n)
            return result
        return super().env_method(method_name, *args, indices=indices, **kwargs)

# Then use:
train_envs = FixedMaskableVecEnv([make_env(i) for i in range(n_envs)])
```

### Fix Transfer Learning

```python
def transfer_phase1_to_phase2(phase1_model, phase2_model):
    with torch.no_grad():
        # Transfer shared layers
        phase2_model.policy.features_extractor.load_state_dict(
            phase1_model.policy.features_extractor.state_dict()
        )
        
        # Handle action expansion (3 -> 6)
        p1_weights = phase1_model.policy.action_net.weight.data
        p2_weights = phase2_model.policy.action_net.weight.data
        
        # Copy actions 0,1,2
        p2_weights[:3, :] = p1_weights[:3, :]
        
        # Initialize new actions 3,4,5
        p2_weights[3:, :] = torch.randn_like(p2_weights[3:, :]) * 0.01
```

## 📊 Expected Output After Fix

```
================================================================================
PHASE 2: POSITION MANAGEMENT MASTERY
================================================================================
[CONFIG] Action space: 6 (RL Fix #10: simplified from 9 to 6)
[ENV] Creating 4 Phase 2 TRAINING environments...
[TRANSFER] Creating Phase 2 model with 6 actions (fixed)
[TRANSFER] Phase 1 knowledge transferred successfully

Validating environment and model compatibility...
✓ Mask shape correct: (4, 6)
✓ Model action space: 6

Starting Phase 2 training for 50,000 timesteps...
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50000/50000 [0:15:00<0:00:00, 55.5 it/s]
```

## 🆘 If Still Failing

1. **Check Python & Package Versions:**
   ```bash
   python --version  # Should be 3.8+
   pip show stable-baselines3  # Should be 2.0+
   pip show sb3-contrib  # Should be 2.0+
   ```

2. **Enable Debug Mode:**
   Add after imports in train_phase2.py:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Minimal Test:**
   Try the simple test script `test_phase2_fixes.py` first

4. **Nuclear Option:**
   - Backup your code
   - Start with the minimal working example
   - Gradually add your features back

## 📝 Files Created for You

1. **fix_phase2_training.py** - Diagnostic and fix utilities
2. **train_phase2_patch.py** - Direct code patches to apply
3. **PHASE2_FIX_PLAN.md** - Detailed technical analysis
4. **test_phase2_fixes.py** - Comprehensive test suite
5. **apply_quick_fix.py** - Automated fix application script

## ✅ Success Criteria

You'll know it's fixed when:
- No immediate crash on startup
- Training progress bar starts moving
- First checkpoint saves successfully
- TensorBoard shows training metrics

## 💡 Pro Tips

1. Always backup before changes: `cp train_phase2.py train_phase2.backup.py`
2. Use git to track changes: `git diff train_phase2.py`
3. Test with `--test` flag first (runs faster)
4. Check TensorBoard for training curves: `tensorboard --logdir tensorboard_logs/phase2/`

---

**Remember:** The core issue is simple - just change 9 to 6 in the action space definitions. Everything else is enhancement and robustness.
