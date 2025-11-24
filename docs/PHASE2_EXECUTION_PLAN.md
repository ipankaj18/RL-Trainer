## Phase 2 Execution Plan

1. **Environment self-check (1 min)**
   ```bash
   python test_phase2_mask_fix.py
   ```
   - Confirms `TradingEnvironmentPhase2`, `ActionMaskGymnasiumWrapper`, and `ActionMaskVecEnvWrapper` all emit `(num_envs, 6)` masks.
   - Test fails fast if a wrapper returns a scalar or 3-action Phase 1 mask.

2. **Quick pipeline test (10–15 min)**
   ```bash
   python src/train_phase2.py --test --market NQ --non-interactive
   ```
   - Runs 50k steps with 4 envs, disables early stopping, and validates checkpoints/logging.
   - Success criteria: no `[MASK]` errors, checkpoints saved at 50k, evaluation every 10k steps.

3. **Full training run (12–16 hrs on RTX 4000 Ada)**
   ```bash
   python src/train_phase2.py --market NQ
   ```
   - 10M steps, 80 envs, adaptive checkpoints. Monitor via TensorBoard:
     ```bash
     tensorboard --logdir tensorboard_logs/phase2/
     ```
   - Track Sharpe ratio (>0.8), mean reward (>0.5), and position-management action usage (>20%).

4. **Post-training validation**
   - Run `python src/testing_framework.py --mode pipeline --market NQ` for regression coverage.
   - Archive logs (tensorboard screenshots, CLI summaries) and update `docs/TESTING_CHECKLIST.md` before promotion to Phase 3.
