import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from action_mask_utils import (  # noqa: E402
    ActionMaskGymnasiumWrapper,
    ActionMaskVecEnvWrapper,
    phase2_action_mask,
)
from environment_phase2 import TradingEnvironmentPhase2  # noqa: E402


def _build_phase2_data(rows: int = 600) -> pd.DataFrame:
    """Minimal market dataset covering the required columns."""
    index = pd.date_range(
        "2025-01-02 09:30:00",
        periods=rows,
        freq="min",
        tz="America/New_York",
    )
    base = np.linspace(15000, 15050, rows)
    return pd.DataFrame(
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
        index=index,
    )


@dataclass
class Phase2EnvFactory:
    """Pickle-safe callable for VecEnv creation."""

    data: pd.DataFrame

    def __call__(self):
        base_env = TradingEnvironmentPhase2(
            data=self.data,
            window_size=20,
            initial_balance=50_000,
            second_data=None,
            randomize_start_offsets=False,
            min_episode_bars=100,
        )
        wrapped = ActionMaskGymnasiumWrapper(base_env)
        return ActionMasker(wrapped, phase2_action_mask)


def test_phase2_action_mask_dummy_vec_env():
    data = _build_phase2_data()
    env = DummyVecEnv([Phase2EnvFactory(data)])
    env = ActionMaskVecEnvWrapper(env)
    try:
        env.reset()
        masks = env.action_masks()
    finally:
        env.close()
    assert masks.shape == (1, 6)
    assert masks.dtype == bool


@pytest.mark.skipif(os.name == "nt", reason="SubprocVecEnv not used on Windows pipeline")
def test_phase2_action_mask_subproc_picklable():
    data = _build_phase2_data()
    env_fns = [Phase2EnvFactory(data) for _ in range(2)]
    env = SubprocVecEnv(env_fns)
    env = ActionMaskVecEnvWrapper(env)
    try:
        env.reset()
        masks = env.action_masks()
    finally:
        env.close()
    assert masks.shape == (2, 6)
    assert masks.dtype == bool
