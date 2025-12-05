import importlib.util
import os
import shutil

import pytest

HAS_JAX = importlib.util.find_spec("jax") is not None
pytestmark = pytest.mark.skipif(not HAS_JAX, reason="jax is not installed; skipping JAX path guard tests")

if HAS_JAX:
    from src.jax_migration.evaluate_phase2_jax import (
        _assert_checkpoint_path_safe_for_loading,
        _is_windows_unc_path,
        _resolve_checkpoint_dir,
    )


def test_is_windows_unc_path_detects_unc_on_windows():
    assert _is_windows_unc_path(r"\\wsl.localhost\Ubuntu\home\user", platform_name="nt")
    assert not _is_windows_unc_path(r"C:\temp\ckpt", platform_name="nt")
    assert not _is_windows_unc_path("/home/user", platform_name="posix")


def test_assert_checkpoint_path_safe_for_loading_raises_on_unc():
    with pytest.raises(ValueError):
        _assert_checkpoint_path_safe_for_loading(r"\\wsl.localhost\Ubuntu\home\user", platform_name="nt")


def test_resolve_checkpoint_dir_prefers_safe_path(tmp_path):
    preferred = tmp_path / "ckpt_dir"
    path, cleanup_needed = _resolve_checkpoint_dir(str(preferred), platform_name="nt")
    assert path == str(preferred)
    assert cleanup_needed is False


def test_resolve_checkpoint_dir_falls_back_for_unc_and_cleans_up():
    path, cleanup_needed = _resolve_checkpoint_dir(r"\\wsl.localhost\Ubuntu\home\user\ckpt", platform_name="nt")
    try:
        assert cleanup_needed is True
        assert not _is_windows_unc_path(path, platform_name="nt")
        assert os.path.isdir(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)
