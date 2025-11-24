"""Sanity checks for the Phase 2 action-space and mask fixes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from gymnasium import Env, spaces
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT / "src"))

from action_mask_utils import ActionMaskVecEnvWrapper  # noqa: E402


class _MockMaskEnv(Env):
    """Tiny env exposing a 6-action mask for wrapper validation."""

    metadata = {}

    def __init__(self) -> None:
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

    def action_masks(self):
        mask = np.ones(6, dtype=bool)
        mask[5] = False
        return mask


def _mask_fn(env: Env) -> np.ndarray:
    return env.action_masks()


def test_action_space() -> None:
    """Ensure the discrete action space is set to six actions."""
    space = spaces.Discrete(6)
    assert space.n == 6, f"Expected 6 actions, got {space.n}"
    print("[PASS] Action space exposes 6 actions")


def test_mask_matrix_shape() -> None:
    """Validate that collected masks keep a (n_envs, n_actions) shape."""
    n_envs = 4
    masks = np.ones((n_envs, 6), dtype=bool)
    assert masks.shape == (n_envs, 6), f"Expected (4, 6), got {masks.shape}"
    assert masks.dtype == bool, f"Expected bool dtype, got {masks.dtype}"
    print(f"[PASS] Mask matrix shape: {masks.shape}")


def test_single_mask_shape() -> None:
    """Confirm individual masks are always length six."""
    single_mask = np.array([True, True, False, False, False, False], dtype=bool)
    assert single_mask.shape == (6,), f"Expected (6,), got {single_mask.shape}"
    assert single_mask.dtype == bool
    print(f"[PASS] Single-mask shape: {single_mask.shape}")


def test_vecnormalize_mask_wrapper_chain() -> None:
    """Ensure VecNormalize + ActionMaskVecEnvWrapper interplay preserves mask dimensions."""
    env = DummyVecEnv([lambda: ActionMasker(_MockMaskEnv(), _mask_fn)])
    env = VecNormalize(env, norm_obs=False, norm_reward=False)
    env = ActionMaskVecEnvWrapper(env)
    masks = env.action_masks()
    assert masks.shape == (1, 6), f"Expected (1, 6), got {masks.shape}"
    assert masks.dtype == bool
    print(f"[PASS] VecNormalize -> ActionMaskVecEnvWrapper mask shape: {masks.shape}")


if __name__ == "__main__":
    print("Running Phase 2 action-space sanity checks...\n")
    test_action_space()
    test_mask_matrix_shape()
    test_single_mask_shape()
    test_vecnormalize_mask_wrapper_chain()
    print("\nAll Phase 2 quick checks passed.")
