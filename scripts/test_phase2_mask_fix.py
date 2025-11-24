"""
Standalone verification for the Phase 2 action mask pipeline.

Usage:
    python test_phase2_mask_fix.py

This script builds a lightweight TradingEnvironmentPhase2 instance with
synthetic market data, wraps it the same way as the training loop
(ActionMaskGymnasiumWrapper -> ActionMasker -> DummyVecEnv -> ActionMaskVecEnvWrapper),
and ensures every layer reports 6-action masks.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

REPO_ROOT = pathlib.Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from action_mask_utils import ActionMaskGymnasiumWrapper, ActionMaskVecEnvWrapper
from environment_phase2 import TradingEnvironmentPhase2
from train_phase2 import mask_fn, stable_mask_fetch


def _build_synthetic_data(rows: int = 600) -> pd.DataFrame:
    """Create minimal market data with the columns Phase 2 expects."""
    idx = pd.date_range(
        "2025-01-02 09:30:00",
        periods=rows,
        freq="T",
        tz="America/New_York",
    )
    base = np.linspace(15000, 15050, rows)
    data = pd.DataFrame(
        {
            "open": base,
            "high": base + 5,
            "low": base - 5,
            "close": base + np.sin(np.linspace(0, 5, rows)),
            "volume": np.full(rows, 1000.0),
            "sma_5": base,
            "sma_20": base,
            "rsi": np.full(rows, 55.0),
            "macd": np.zeros(rows),
            "momentum": np.ones(rows),
            "atr": np.full(rows, 15.0),
        },
        index=idx,
    )
    return data


def _make_env_factory(data: pd.DataFrame):
    """Return a DummyVecEnv-compatible factory building the wrapped env."""

    def _factory():
        base_env = TradingEnvironmentPhase2(
            data=data,
            window_size=20,
            initial_balance=50_000,
            second_data=None,
            randomize_start_offsets=False,
            min_episode_bars=100,
        )
        wrapped = ActionMaskGymnasiumWrapper(base_env)
        return ActionMasker(wrapped, mask_fn)

    return _factory


def run_all_checks() -> None:
    """Execute all mask assertions, raising on failure."""
    data = _build_synthetic_data()

    # Direct environment checks
    base_env = TradingEnvironmentPhase2(
        data=data,
        window_size=20,
        initial_balance=50_000,
        second_data=None,
        randomize_start_offsets=False,
        min_episode_bars=100,
    )
    obs, info = base_env.reset()
    assert obs.shape[-1] == base_env.observation_space.shape[0]
    assert info.get("action_mask") is not None, "Reset info missing action_mask"
    assert len(info["action_mask"]) == 6
    direct_mask = base_env.action_masks()
    assert direct_mask.shape == (6,), "Environment mask should expose 6 actions"

    # Wrapped single-env checks
    wrapped_env = ActionMaskGymnasiumWrapper(base_env)
    mask_from_wrapper = wrapped_env.get_action_mask()
    assert mask_from_wrapper.shape == (6,)

    # VecEnv-style pipeline checks
    env_fn = _make_env_factory(data)
    vec_env = DummyVecEnv([env_fn])
    vec_env = ActionMaskVecEnvWrapper(vec_env)
    vec_env.reset()
    vec_masks = vec_env.action_masks()
    assert vec_masks.shape == (1, 6), f"VecEnv masks wrong shape: {vec_masks.shape}"
    raw_env_masks = vec_env.env_method("action_masks")
    assert all(mask.shape == (6,) for mask in raw_env_masks), raw_env_masks
    stable_masks = stable_mask_fetch(vec_env)
    assert stable_masks.shape == (1, 6), f"stable_mask_fetch returned {stable_masks.shape}"
    assert stable_masks.dtype == bool

    print("[OK] Direct env mask:", direct_mask)
    print("[OK] VecEnv mask tensor shape:", vec_masks.shape)
    print("[OK] stable_mask_fetch tensor:\n", stable_masks)
    print("ALL TESTS PASSED!")


if __name__ == "__main__":
    run_all_checks()
