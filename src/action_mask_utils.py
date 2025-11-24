"""
Action mask utilities and wrappers.

Provides pickle-safe wrappers for vectorized environments as well as helpers
for retrieving and validating action masks without relying on deprecated
Gymnasium attribute traversal.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvObs,
    VecEnvStepReturn,
)


def _as_bool_array(mask: Any, width: Optional[int] = None) -> np.ndarray:
    """Convert an arbitrary mask object into a boolean numpy array."""
    if isinstance(mask, np.ndarray):
        arr = mask.astype(bool, copy=True)
    elif isinstance(mask, (list, tuple)):
        arr = np.asarray(mask, dtype=bool)
    else:
        arr = np.asarray(mask, dtype=bool)

    if width is not None and arr.ndim == 1 and arr.size != width:
        raise ValueError(f"Mask width mismatch: expected {width}, got {arr.size}")
    return arr


class ActionMaskVecEnvWrapper(VecEnvWrapper):
    """
    VecEnv wrapper that exposes a pickle-friendly action_masks() method.

    Stable Baselines' MaskablePPO expects env.action_masks() to exist on the
    vectorized environment. This wrapper centralizes the lookup logic and
    avoids injecting lambda closures (which break pickling) directly into
    other wrappers such as VecNormalize.
    """

    def __init__(self, venv: VecEnv):
        super().__init__(venv)
        self._cached_masks: Optional[np.ndarray] = None

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self._cached_masks = None
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        result = self.venv.step_wait()
        self._cached_masks = None
        return result

    def env_method(self, method_name: str, *args, **kwargs):
        if method_name in {"action_masks", "get_action_mask"}:
            raw = self.venv.env_method(method_name, *args, **kwargs)
            formatted = self._format_masks(raw)
            return [mask.copy() for mask in formatted]
        return super().env_method(method_name, *args, **kwargs)
    def action_masks(self) -> np.ndarray:
        """
        Return current action masks for all sub-environments as a boolean array
        with shape (num_envs, action_dim).
        """
        if self._cached_masks is None:
            raw_masks = self._fetch_masks()
            self._cached_masks = self._format_masks(raw_masks)
        return self._cached_masks.copy()

    def _fetch_masks(self) -> Sequence[Any]:
        """Retrieve masks via env_method or direct attribute access."""
        for method_name in ("get_action_mask", "action_masks"):
            try:
                masks = self.venv.env_method(method_name)
                if masks is not None:
                    return masks
            except (AttributeError, NotImplementedError, AssertionError):
                continue

        # Fallback: direct access to underlying envs (DummyVecEnv)
        envs = getattr(self.venv, "envs", None)
        if envs:
            return [self._extract_mask_from_env(env) for env in envs]

        # Last resort: assume all actions valid
        default_mask = np.ones(self.action_space.n, dtype=bool)
        return [default_mask for _ in range(self.num_envs)]

    def _extract_mask_from_env(self, env: Any) -> np.ndarray:
        """Extract a mask from a raw environment instance."""
        for attr in ("get_action_mask", "action_masks"):
            maybe_callable = getattr(env, attr, None)
            if callable(maybe_callable):
                try:
                    mask = maybe_callable()
                    if mask is not None:
                        return _as_bool_array(mask, self.action_space.n)
                except Exception:
                    continue

        base = getattr(env, "unwrapped", None)
        if base and base is not env:
            return self._extract_mask_from_env(base)

        return np.ones(self.action_space.n, dtype=bool)

    def _format_masks(self, masks: Sequence[Any]) -> np.ndarray:
        """Ensure masks are shaped consistently."""
        if isinstance(masks, np.ndarray):
            if masks.ndim == 1:
                mask_iterable = [masks]
            else:
                mask_iterable = list(masks)
        else:
            mask_iterable = list(masks)

        arr = np.array(
            [_as_bool_array(mask, self.action_space.n) for mask in mask_iterable],
            dtype=bool,
        )

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.shape[0] != self.num_envs:
            # Broadcast single mask across all envs when necessary
            arr = np.repeat(arr, self.num_envs, axis=0)

        return arr

    def __getstate__(self) -> Dict[str, Any]:
        """Drop cached masks before pickling."""
        state = self.__dict__.copy()
        state["_cached_masks"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore wrapper state after unpickling."""
        self.__dict__.update(state)
        self._cached_masks = None


class ActionMaskGymnasiumWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that exposes get_action_mask/action_mask attributes.

    This wrapper keeps the action mask accessible via env.get_wrapper_attr
    and avoids relying on deprecated direct attribute traversal.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._action_mask: Optional[np.ndarray] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._action_mask = self._extract_mask(info)
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = result
        self._action_mask = self._extract_mask(info)
        return obs, reward, terminated, truncated, info

    def _extract_mask(self, info: Optional[Dict[str, Any]]) -> np.ndarray:
        """Derive current mask from info dict or environment hooks."""
        if isinstance(info, dict):
            mask_from_info = info.get("action_mask")
            if mask_from_info is not None:
                return _as_bool_array(mask_from_info, self.action_space.n)
        return self._call_env_for_mask()

    def _call_env_for_mask(self) -> np.ndarray:
        for attr in ("get_action_mask", "action_masks"):
            fn = getattr(self.env, attr, None)
            if callable(fn):
                mask = fn()
                if mask is not None:
                    return _as_bool_array(mask, self.action_space.n)

        base = getattr(self.env, "unwrapped", None)
        if base and base is not self.env:
            for attr in ("get_action_mask", "action_masks"):
                fn = getattr(base, attr, None)
                if callable(fn):
                    mask = fn()
                    if mask is not None:
                        return _as_bool_array(mask, self.action_space.n)

        return np.ones(self.action_space.n, dtype=bool)


def phase2_mask_fn(env) -> np.ndarray:
    """
    Resolve the underlying TradingEnvironmentPhase2 instance (if present)
    and return its six-action mask. Designed to be pickle-safe by living in
    a dedicated module that SubprocVecEnv workers can import.
    """
    try:
        from environment_phase2 import TradingEnvironmentPhase2
    except ImportError:  # pragma: no cover - defensive
        TradingEnvironmentPhase2 = None  # type: ignore[assignment]

    current = env
    visited: set[int] = set()

    # Search for the concrete Phase 2 environment first
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if TradingEnvironmentPhase2 is not None and isinstance(current, TradingEnvironmentPhase2):
            masks = current.action_masks()
            if masks is not None:
                result = np.asarray(masks, dtype=bool)
                if result.shape == (6,):
                    return result
        current = getattr(current, "env", None)

    # Fallback: try generic action_masks hooks on wrapper chain
    current = env
    visited.clear()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        action_masks_fn = getattr(current, "action_masks", None)
        if callable(action_masks_fn):
            masks = action_masks_fn()
            if masks is not None:
                result = np.asarray(masks, dtype=bool)
                if result.shape == (6,):
                    return result
        current = getattr(current, "env", None)

    # Last resort: allow all actions
    return np.ones(6, dtype=bool)

    def get_wrapper_attr(self, name: str) -> Any:
        if name == "action_mask":
            return self.get_action_mask()
        if name == "get_action_mask":
            return self.get_action_mask
        return super().get_wrapper_attr(name)

    def get_action_mask(self) -> np.ndarray:
        if self._action_mask is None:
            self._action_mask = self._call_env_for_mask()
        return self._action_mask.copy()

    def action_masks(self) -> np.ndarray:
        return self.get_action_mask()


def get_action_masks(env: Any) -> np.ndarray:
    """
    Retrieve action masks from any wrapped environment without deprecated
    attribute traversal.
    """
    action_masks_fn = getattr(env, "action_masks", None)
    if callable(action_masks_fn):
        return _as_bool_array(action_masks_fn())

    env_method = getattr(env, "env_method", None)
    if callable(env_method):
        for method in ("get_action_mask", "action_masks"):
            try:
                masks = env_method(method)
                if masks is not None:
                    return np.asarray(masks, dtype=bool)
            except AttributeError:
                continue

    get_wrapper_attr = getattr(env, "get_wrapper_attr", None)
    if callable(get_wrapper_attr):
        attr = get_wrapper_attr("action_mask")
        if attr is not None:
            return _as_bool_array(attr)

    base_env = getattr(env, "unwrapped", None) or env
    for attr_name in ("get_action_mask", "action_masks"):
        attr = getattr(base_env, attr_name, None)
        if callable(attr):
            masks = attr()
            if masks is not None:
                return _as_bool_array(masks)

    raise AttributeError("Environment does not expose action masks")


def validate_action_masks(env: VecEnv, actions: np.ndarray) -> bool:
    """
    Ensure that the provided actions respect the current action masks.
    """
    try:
        masks = env.action_masks()
    except AttributeError:
        masks = env.env_method("get_action_mask")

    valid = True
    for idx, (action, mask) in enumerate(zip(actions, masks)):
        if action is None:
            continue
        if not mask[int(action)]:
            print(f"[ACTION MASK] Invalid action {action} detected in env {idx}")
            valid = False
    return valid


def get_valid_action_probabilities(
    action_probs: np.ndarray,
    action_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply an action mask to probability vectors and re-normalize them.
    """
    masked = action_probs * action_mask
    sums = masked.sum(axis=-1, keepdims=True)
    sums = np.maximum(sums, 1e-8)
    return masked / sums


class ActionMaskMonitor:
    """
    Lightweight tracker for invalid action attempts and mask sparsity.
    """

    def __init__(self, log_freq: int = 100):
        self.log_freq = log_freq
        self.invalid_action_count = 0
        self.total_actions = 0
        self.mask_history: List[float] = []

    def update(self, actions: Iterable[int], masks: Sequence[np.ndarray]) -> None:
        for action, mask in zip(actions, masks):
            self.total_actions += 1
            if not mask[int(action)]:
                self.invalid_action_count += 1
            self.mask_history.append(mask.mean())

        if self.total_actions and self.total_actions % self.log_freq == 0:
            self.log_statistics()

    def log_statistics(self) -> None:
        if not self.total_actions:
            return
        invalid_ratio = self.invalid_action_count / self.total_actions
        recent = self.mask_history[-self.log_freq :] or self.mask_history
        avg_valid = float(np.mean(recent)) if recent else 0.0
        print(
            f"[ACTION MASK] Invalid actions: {invalid_ratio:.2%} | "
            f"Avg valid options: {avg_valid:.2%}"
        )


__all__ = [
    "ActionMaskVecEnvWrapper",
    "ActionMaskGymnasiumWrapper",
    "ActionMaskMonitor",
    "get_action_masks",
    "get_valid_action_probabilities",
    "validate_action_masks",
]
