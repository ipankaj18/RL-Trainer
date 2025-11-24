"""
Utilities for adjusting training configuration between production and test runs.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict


def tune_eval_schedule(
    config: Dict[str, Any],
    *,
    test_mode: bool,
    label: str,
    eval_updates: int,
    min_eval_episodes: int,
    printer: Callable[[str], None] | None = None,
) -> None:
    """
    Auto-scale evaluation frequency and episode count so production runs
    evaluate less often (higher throughput) while test mode keeps the
    lightweight schedule configured in the CLI overrides.

    Notes
    -----
    SB3's ``EvalCallback`` interprets ``eval_freq`` as the number of callback
    calls (one call per ``env.step``), not global timesteps. When using
    vectorized environments, each call advances ``num_envs`` timesteps, so we
    must downscale the desired timestep cadence by ``num_envs`` to ensure
    evaluations actually trigger.
    """

    def _log(message: str) -> None:
        if printer is not None:
            printer(message)
        else:
            print(message)

    num_envs = max(1, int(config.get("num_envs", config.get("n_envs", 1))))
    calls_per_update = max(1, int(config.get("n_steps", 0)))  # callback calls per PPO update
    if calls_per_update <= 0:
        return

    total_timesteps = int(config.get("total_timesteps", 0))
    total_calls = math.ceil(total_timesteps / num_envs) if total_timesteps > 0 else 0

    desired_step_freq = int(config.get("eval_freq", calls_per_update * num_envs))
    desired_call_freq = max(1, math.ceil(desired_step_freq / num_envs))

    # Target cadence in callback-call units (one call per env.step())
    if test_mode:
        target_calls = max(desired_call_freq, calls_per_update)
    else:
        target_calls = max(desired_call_freq, calls_per_update * max(1, eval_updates))

    # Ensure at least one evaluation happens within the run
    if total_calls:
        target_calls = min(target_calls, total_calls)

    if target_calls != desired_call_freq:
        approx_updates = target_calls / calls_per_update
        approx_timesteps = target_calls * num_envs
        mode_label = "Test-mode" if test_mode else "Eval frequency auto-scaled"
        _log(
            f"[CONFIG] ({label}) {mode_label} to {target_calls:,} env steps "
            f"(~{approx_timesteps:,} timesteps across {num_envs} envs, ~{approx_updates:.1f} updates)"
        )

    config["eval_freq"] = target_calls

    current_eps = int(config.get("n_eval_episodes", min_eval_episodes))
    if current_eps < min_eval_episodes:
        config["n_eval_episodes"] = min_eval_episodes
        _log(
            f"[CONFIG] ({label}) Eval episodes increased to {min_eval_episodes} "
            "for more stable metrics"
        )
