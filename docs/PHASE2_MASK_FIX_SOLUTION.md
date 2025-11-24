## Phase 2 Action Mask Fix

### Problem
- Phase 2 training crashed with `RuntimeError: [MASK] Unexpected mask width 4 (expected 6)` after only a few steps.
- The `ActionMaskGymnasiumWrapper` was receiving `info["action_mask"] = None`, so mask fetchers downgraded to scalar falsy masks and `stable_mask_fetch` aborted.
- `TradingEnvironmentPhase2` never cached the latest 6-action mask, so every info dict exposed `None` and downstream wrappers could not retrieve valid masks.

### Fix
1. **Environment caching**  
   - `TradingEnvironmentPhase2.action_masks()` now stores the current boolean mask (`self.last_action_mask`) and returns defensive copies.
   - `reset()` primes the cache on reset so the very first observation has a 6-action mask available.
   - `info` dicts only include `action_mask` when a cached mask exists, preventing `None` from being broadcast into VecEnv wrappers.
2. **Wrapper resilience**  
   - `ActionMaskGymnasiumWrapper` ignores `info["action_mask"]` entries that are explicitly `None`, falling back to the environment hook.

### Validation
- Added `test_phase2_mask_fix.py` to instantiate the Phase 2 environment, wrap it with the same masking stack used during training, and assert every retrieval path returns `(n_envs, 6)` boolean masks.
- Running `python test_phase2_mask_fix.py` now prints `ALL TESTS PASSED` and surfaces the raw mask tensors for easier debugging.

### Next Steps
1. Run `python test_phase2_mask_fix.py` to confirm action masks behave before training.
2. Execute `python src/train_phase2.py --test --market NQ --non-interactive` to verify the end-to-end training loop now proceeds past the previous 4-step failure.
3. Follow `PHASE2_EXECUTION_PLAN.md` for a full 10M timestep training run once the quick test succeeds.
