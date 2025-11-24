# Phase 1 & Phase 2 Training Summary

_Data sources reviewed:_

- TensorBoard scalar logs under `tensorboard_logs/phase1/PPO_1` and `tensorboard_logs/phase2/PPO_1`.
- Evaluation checkpoints in `logs/phase1/evaluations.npz` and `logs/phase2/evaluations.npz`.
- Phase 2 training configuration and environment code in `src/train_phase2.py` and `src/environment_phase2.py`.

## Phase 1 Findings

- **Training dynamics:** Rollout reward steadily improved from roughly `-2.22` to `-1.47` while episode length expanded from ~230 bars to ~2.7k. `train/explained_variance` climbed from `-0.35` to ~`0.95`, and KL / entropy curves behaved normally. The PPO optimizer therefore converged on the simplified entry curriculum.
- **Evaluation outcome:** The only stored checkpoint (4M timesteps) reports `mean_reward = -1.65` over a deterministic 163‑bar episode (std = 0). Despite healthy training metrics, the exported policy still fails the fixed validation slice.
- **Implication:** Phase 2 transfer learning is seeded with a policy that never achieved positive reward on held‑out data. Improving Phase 1 evaluation performance is prerequisite to a stronger foundation.

## Phase 2 Findings

- **Training dynamics:** Rollout reward ranges between +3 and +115 with stable KL (~0.005), entropy (~‑0.36), and explained variance (~0.93). Losses, value loss, and policy gradient terms trend downward, showing the model keeps learning under the short randomized episodes.
- **Evaluation checkpoints:** Two checkpoints (4M and 8M steps) were logged. At 4M, every eval episode lost ~`‑1191` over exactly 1,846 bars; at 8M, losses improved to ~`‑282` but lengths remain deterministic at 327 bars. The agent repeats the same failure path each evaluation despite strong rollout returns.
- **Failure mode:** Training episodes average 20–30 steps (fast, stochastic resets), whereas validation forces a long, fixed slice with strict Apex rules. The policy overfits to the short-horizon distribution and never experiences the compliance event that kills performance in evaluation.
- **Log retention issue:** `EvalCallback` writes to `logs/phase2/evaluations.npz` without versioning, so reruns overwrite earlier checkpoints. Diagnosing regressions becomes difficult and only the final two snapshots remain.

## What’s Going Well

1. PPO stability: Both phases exhibit bounded KL, non-collapsed entropy, and increasing explained variance, confirming the optimizer is tuned reasonably.
2. Training throughput: Phase 2 sustains ~1.6k FPS on 80 parallel envs, validating the virtualization strategy and action-masking integration.

## Detected Issues

1. **Train/eval mismatch:** The deterministic evaluation slice (long contiguous bars) bears little resemblance to the short randomized rollouts. Success in training does not translate to the evaluation regime.
2. **Insufficient evaluation coverage:** Only one Phase 1 checkpoint and two Phase 2 checkpoints exist, each with zero variance in episode length. Overwriting the same NPZ file removes historical context.
3. **Weak Phase 1 foundation:** Since the Phase 1 checkpoint never beats the validation slice, Phase 2 transfer starts from a suboptimal policy.
4. **Potential compliance breach:** Constant evaluation lengths and large losses suggest repeated trailing drawdown or late-entry violations that training episodes never expose.

## Recommendations

### Improve Evaluation Fidelity
1. **Version evaluation logs:** Incorporate a timestamp or run ID into `EvalCallback`’s `log_path` in `src/train_phase2.py` so every run produces its own NPZ file. This preserves progression data and prevents accidental overwrites.
2. **Multiple validation slices:** Instead of a single deterministic window, evaluate across several fixed slices (different dates or markets) and/or randomize start offsets during evaluation to capture variance.

### Align Training With Evaluation
3. **Long-horizon rollouts:** Increase `min_episode_bars` and occasionally disable random offsets in `make_env` so some training envs replay contiguous long paths just like the evaluation slice.
4. **Curriculum sampling:** Mix in dedicated “compliance” episodes that mimic the eval path (same market hours, drawdown rules) so the agent practices the failure scenario it currently never sees.

### Strengthen Base Policy
5. **Revisit Phase 1 reward scaling:** Adjust the entry-focused reward or curriculum so that the evaluation mean reward approaches zero or higher before exporting the checkpoint for transfer.
6. **Inspect deterministic failures:** Replay the evaluation trajectories (using `logs/phase2/evaluations.npz`) and log `done_reason`, `action_mask`, and compliance flags from `TradingEnvironmentPhase2` to confirm whether trailing DD, late trades, or invalid action penalties cause the crash. Add targeted masking or penalties once verified.

### Monitoring Enhancements
7. **Action distribution telemetry:** Enable the existing TensorBoard logging hook (`src/train_phase2.py:_log_action_distribution`) to ensure the agent isn’t spamming a single management action that performs well in short rollouts but fails in long horizons.
8. **Run metadata:** Store run configuration (timesteps, markets, seeds) alongside evaluation logs to ensure reproducibility while comparing checkpoints.

By aligning the training distribution with evaluation, preserving rich evaluation history, and ensuring Phase 1 checkpoints truly generalize, Phase 2 should begin to convert its strong rollout metrics into consistent validation performance. Once the evaluation gap shrinks, move on to Phase 3 with confidence that the base policy handles the deterministic compliance scenarios it will face in production.
