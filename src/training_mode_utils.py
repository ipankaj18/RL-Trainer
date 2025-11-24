"""
Utilities for adjusting training configuration between production and test runs.
"""

from __future__ import annotations

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
    """

    def _log(message: str) -> None:
        if printer is not None:
            printer(message)
        else:
            print(message)

    steps_per_update = config.get("n_steps", 0) * max(
        1, config.get("num_envs", config.get("n_envs", 1))
    )
    if steps_per_update <= 0:
        return

    eval_freq = int(config.get("eval_freq", steps_per_update))
    total_timesteps = int(config.get("total_timesteps", eval_freq))

    if test_mode:
        limited = min(max(eval_freq, steps_per_update), total_timesteps)
        if limited != eval_freq:
            config["eval_freq"] = limited
            _log(f"[CONFIG] ({label}) Test-mode eval frequency set to {limited:,} steps")
        return

    target_freq = max(eval_freq, steps_per_update * max(1, eval_updates))
    if target_freq != eval_freq:
        config["eval_freq"] = target_freq
        approx_updates = target_freq / steps_per_update if steps_per_update else 0
        _log(
            f"[CONFIG] ({label}) Eval frequency auto-scaled to "
            f"{target_freq:,} steps (~{approx_updates:.1f} updates)"
        )

    current_eps = int(config.get("n_eval_episodes", min_eval_episodes))
    if current_eps < min_eval_episodes:
        config["n_eval_episodes"] = min_eval_episodes
        _log(
            f"[CONFIG] ({label}) Eval episodes increased to {min_eval_episodes} "
            "for more stable metrics"
        )
