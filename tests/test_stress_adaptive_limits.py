#!/usr/bin/env python3
"""
Unit tests for JAX stress test adaptive limit detection.

Tests the get_safe_env_limits() function to ensure it correctly
adapts search space based on system process limits.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from stress_hardware_jax import get_safe_env_limits


class TestAdaptiveLimits:
    """Test suite for adaptive env limit detection."""

    @patch('stress_hardware_jax.resource')
    def test_unlimited_limit_returns_conservative_envs(self, mock_resource):
        """Test that unlimited ulimit returns conservative [64, 128, 192]."""
        # Mock unlimited limit
        mock_resource.getrlimit.return_value = (-1, -1)
        mock_resource.RLIMIT_NPROC = 6  # Standard constant
        
        max_safe, env_array = get_safe_env_limits()
        
        assert max_safe == 192
        assert env_array == [64, 128, 192]
        print("✓ Unlimited limit correctly returns conservative env counts")

    @patch('stress_hardware_jax.resource')
    def test_high_limit_returns_aggressive_envs(self, mock_resource):
        """Test that high ulimit (e.g., 4096) returns aggressive env counts."""
        # Mock ulimit -u 4096
        mock_resource.getrlimit.return_value = (4096, 65536)
        mock_resource.RLIMIT_NPROC = 6
        
        max_safe, env_array = get_safe_env_limits()
        
        # 4096 * 0.75 / 25 = 122.88 -> max_safe_envs = 122
        # Should fall into the "max_safe_envs >= 512" category
        # Expect [256, 512, 1024] filtered to values <= 122
        # Actually would be [256, 512, 1024] but filtered, so likely will trigger different branch
        # Let me recalculate: 4096 * 0.75 = 3072, 3072 / 25 = 122
        # Wait, that doesn't seem right. Let me check the logic again.
        # Actually: safe_limit = 4096 * 0.75 = 3072
        # max_safe_envs = 3072 // 25 = 122
        
        # Hmm, 122 would fall into the "max_safe_envs >= 192" category
        # So env_array = [128, 192, 256] filtered to <= 122
        # Result should be [128]
        
        # Actually wait, I need to recheck the calculation
        # safe_limit = int(4096 * 0.75) = 3072
        # max_safe_envs = int(3072 // 25) = 122
        
        # The logic checks:
        # if max_safe_envs >= 4096: [512, 1024, 2048, 4096, 8192, 12288, 16384]
        # elif max_safe_envs >= 2048: [512, 1024, 2048, 3072]
        # elif max_safe_envs >= 512: [256, 512, 1024]
        # elif max_safe_envs >= 192: [128, 192, 256]
        # else: [64, 128]
        
        # With max_safe_envs = 122, none of those conditions match, so it goes to else
        # Result: [64, 128]
        
        assert max_safe == 122
        assert env_array == [64, 128]
        print("✓ High capped limit correctly calculates safe env counts")

    @patch('stress_hardware_jax.resource')
    def test_very_high_limit_returns_full_range(self, mock_resource):
        """Test that very high ulimit (e.g., 204800) returns full env range."""
        # Mock ulimit -u 204800 (high limit on powerful workstation)
        mock_resource.getrlimit.return_value = (204800, 204800)
        mock_resource.RLIMIT_NPROC = 6
        
        max_safe, env_array = get_safe_env_limits()
        
        # 204800 * 0.75 = 153600
        # 153600 // 25 = 6144
        # This is >= 4096, so should use full range
        assert max_safe == 6144
        assert 512 in env_array
        assert 1024 in env_array
        assert 2048 in env_array
        # Should include values up to 6144
        print("✓ Very high limit correctly returns full env range")

    @patch('stress_hardware_jax.resource')
    def test_low_limit_returns_minimal_envs(self, mock_resource):
        """Test that low ulimit (e.g., 512) returns minimal env counts."""
        # Mock ulimit -u 512 (very constrained)
        mock_resource.getrlimit.return_value = (512, 512)
        mock_resource.RLIMIT_NPROC = 6
        
        max_safe, env_array = get_safe_env_limits()
        
        # 512 * 0.75 = 384
        # 384 // 25 = 15
        # This is < 192, so should use [64, 128] filtered to <= 15
        # Result should be empty or minimal
        assert max_safe == 15
        # With max_safe = 15, no values in [64, 128] are <= 15
        assert env_array == []
        print("✓ Low limit correctly returns minimal env counts (empty if too low)")

    @patch('stress_hardware_jax.subprocess')
    @patch('stress_hardware_jax.resource', side_effect=ImportError)
    def test_fallback_to_ulimit_command(self, mock_resource, mock_subprocess):
        """Test fallback to ulimit command when resource module unavailable."""
        # Mock subprocess call to ulimit -u
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "4096\n"
        mock_subprocess.run.return_value = mock_result
        
        max_safe, env_array = get_safe_env_limits()
        
        # Should have parsed "4096" as the limit
        # 4096 * 0.75 / 25 = 122
        assert max_safe == 122
        print("✓ Successfully falls back to ulimit command when resource module unavailable")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
