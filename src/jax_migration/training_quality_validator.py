"""
Training Quality Validator for JAX PPO

Validates that training quality remains acceptable during hardware stress testing.
Detects training divergence, policy collapse, and poor learning.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class QualityMetrics:
    """Training quality metrics snapshot."""
    mean_return: float = 0.0
    entropy: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    win_rate: float = 0.0  # If available from environment


class TrainingQualityValidator:
    """
    Validates training quality during stress testing.

    Checks for:
    - Entropy collapse (deterministic policy)
    - Training divergence (NaN/Inf losses)
    - Poor learning (returns not improving)
    - Excessive KL divergence (policy changing too fast)

    Usage:
        validator = TrainingQualityValidator()
        validator.add_metrics(mean_return=10.5, entropy=0.15, ...)
        is_good = validator.is_quality_acceptable()
        score = validator.calculate_quality_score()
    """

    def __init__(
        self,
        entropy_threshold: float = 0.05,
        kl_threshold: float = 0.5,
        min_return_threshold: float = -1000.0,
        window_size: int = 10
    ):
        """
        Initialize quality validator.

        Args:
            entropy_threshold: Minimum acceptable entropy (below = collapse)
            kl_threshold: Maximum acceptable KL divergence (above = unstable)
            min_return_threshold: Minimum acceptable mean return
            window_size: Number of recent metrics to track
        """
        self.entropy_threshold = entropy_threshold
        self.kl_threshold = kl_threshold
        self.min_return_threshold = min_return_threshold
        self.window_size = window_size

        self._metrics_history: List[QualityMetrics] = []
        self._has_diverged = False
        self._divergence_reason: Optional[str] = None

    def add_metrics(
        self,
        mean_return: float,
        entropy: float,
        policy_loss: float = 0.0,
        value_loss: float = 0.0,
        approx_kl: float = 0.0,
        clip_fraction: float = 0.0,
        win_rate: float = 0.0
    ):
        """
        Add new training metrics.

        Args:
            mean_return: Mean episode return
            entropy: Policy entropy
            policy_loss: Policy loss value
            value_loss: Value function loss
            approx_kl: Approximate KL divergence
            clip_fraction: Fraction of clipped ratios
            win_rate: Win rate (if available)
        """
        metrics = QualityMetrics(
            mean_return=mean_return,
            entropy=entropy,
            policy_loss=policy_loss,
            value_loss=value_loss,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
            win_rate=win_rate
        )

        # Check for divergence
        if self._check_divergence(metrics):
            self._has_diverged = True

        self._metrics_history.append(metrics)

        # Keep only recent window
        if len(self._metrics_history) > self.window_size:
            self._metrics_history.pop(0)

    def _check_divergence(self, metrics: QualityMetrics) -> bool:
        """Check if training has diverged."""
        # NaN or Inf in any metric
        values = [
            metrics.mean_return,
            metrics.entropy,
            metrics.policy_loss,
            metrics.value_loss,
            metrics.approx_kl
        ]

        for val in values:
            if not np.isfinite(val):
                self._divergence_reason = f"Non-finite value detected: {val}"
                return True

        # Extreme KL divergence
        if metrics.approx_kl > self.kl_threshold:
            self._divergence_reason = f"Excessive KL divergence: {metrics.approx_kl:.4f}"
            return True

        # Catastrophic return collapse
        if metrics.mean_return < self.min_return_threshold:
            self._divergence_reason = f"Return collapse: {metrics.mean_return:.2f}"
            return True

        return False

    def is_quality_acceptable(self) -> bool:
        """
        Check if training quality is acceptable.

        Returns:
            True if quality passes all checks, False otherwise
        """
        if not self._metrics_history:
            return True  # No data yet

        if self._has_diverged:
            return False

        latest = self._metrics_history[-1]

        # Entropy collapse check
        if latest.entropy < self.entropy_threshold:
            return False

        # Check for NaN/Inf
        if not np.isfinite(latest.mean_return):
            return False

        if not np.isfinite(latest.entropy):
            return False

        return True

    def calculate_quality_score(self) -> float:
        """
        Calculate normalized quality score (0-1 scale).

        Components:
        - Entropy health (0.3 weight)
        - Return stability (0.3 weight)
        - Loss convergence (0.2 weight)
        - KL stability (0.2 weight)

        Returns:
            Quality score between 0.0 (poor) and 1.0 (excellent)
        """
        if not self._metrics_history:
            return 0.0

        if self._has_diverged:
            return 0.0

        latest = self._metrics_history[-1]

        # 1. Entropy health (0.3 weight)
        # Scale: 0.0-0.2 entropy -> 0.0-1.0 score
        entropy_score = min(latest.entropy / 0.2, 1.0)

        # 2. Return stability (0.3 weight)
        # Positive returns = 1.0, negative returns scaled
        if latest.mean_return >= 0:
            return_score = 1.0
        else:
            # Scale negative returns: -100 = 0.5, -1000 = 0.0
            return_score = max(0.0, 1.0 + latest.mean_return / 1000.0)

        # 3. Loss convergence (0.2 weight)
        # Lower losses are better (but not too low, which might indicate collapse)
        # Typical range: 0.01-1.0, optimal around 0.1-0.5
        policy_loss = latest.policy_loss
        if 0.1 <= policy_loss <= 0.5:
            loss_score = 1.0
        elif policy_loss < 0.01:
            loss_score = 0.5  # Too low, might be collapsed
        elif policy_loss > 1.0:
            loss_score = max(0.0, 1.0 - (policy_loss - 1.0) / 10.0)
        else:
            loss_score = 0.8

        # 4. KL stability (0.2 weight)
        # Optimal KL: 0.01-0.05, too low = no learning, too high = instability
        kl = latest.approx_kl
        if 0.01 <= kl <= 0.05:
            kl_score = 1.0
        elif kl < 0.01:
            kl_score = 0.7  # Learning might be too slow
        elif kl > self.kl_threshold:
            kl_score = 0.0  # Unstable
        else:
            kl_score = max(0.0, 1.0 - (kl - 0.05) / self.kl_threshold)

        # Weighted average
        total_score = (
            0.3 * entropy_score +
            0.3 * return_score +
            0.2 * loss_score +
            0.2 * kl_score
        )

        return np.clip(total_score, 0.0, 1.0)

    def get_diagnostics(self) -> Dict[str, any]:
        """
        Get detailed diagnostics.

        Returns:
            Dictionary with diagnostic information
        """
        if not self._metrics_history:
            return {
                'has_data': False,
                'num_samples': 0
            }

        latest = self._metrics_history[-1]

        # Calculate trends if we have enough history
        trends = {}
        if len(self._metrics_history) >= 3:
            returns = [m.mean_return for m in self._metrics_history[-3:]]
            entropies = [m.entropy for m in self._metrics_history[-3:]]

            trends['return_trend'] = 'improving' if returns[-1] > returns[0] else 'declining'
            trends['entropy_trend'] = 'increasing' if entropies[-1] > entropies[0] else 'decreasing'

        return {
            'has_data': True,
            'num_samples': len(self._metrics_history),
            'latest_metrics': {
                'mean_return': latest.mean_return,
                'entropy': latest.entropy,
                'policy_loss': latest.policy_loss,
                'value_loss': latest.value_loss,
                'approx_kl': latest.approx_kl,
                'clip_fraction': latest.clip_fraction,
                'win_rate': latest.win_rate
            },
            'quality_acceptable': self.is_quality_acceptable(),
            'quality_score': self.calculate_quality_score(),
            'has_diverged': self._has_diverged,
            'divergence_reason': self._divergence_reason,
            'trends': trends,
            'warnings': self._get_warnings(latest)
        }

    def _get_warnings(self, latest: QualityMetrics) -> List[str]:
        """Get list of current warnings."""
        warnings = []

        if latest.entropy < self.entropy_threshold:
            warnings.append(f"Low entropy ({latest.entropy:.4f} < {self.entropy_threshold})")

        if latest.approx_kl > self.kl_threshold * 0.8:
            warnings.append(f"High KL divergence ({latest.approx_kl:.4f})")

        if latest.mean_return < -100:
            warnings.append(f"Poor returns ({latest.mean_return:.2f})")

        if latest.clip_fraction > 0.5:
            warnings.append(f"High clip fraction ({latest.clip_fraction:.2f})")

        return warnings

    def reset(self):
        """Reset validator state."""
        self._metrics_history.clear()
        self._has_diverged = False
        self._divergence_reason = None


if __name__ == "__main__":
    # Test quality validator
    print("Testing Training Quality Validator...")

    validator = TrainingQualityValidator()

    # Simulate good training
    print("\n1. Testing good training progression:")
    for i in range(5):
        validator.add_metrics(
            mean_return=10.0 + i * 5,
            entropy=0.15 - i * 0.01,
            policy_loss=0.3,
            value_loss=0.5,
            approx_kl=0.03,
            clip_fraction=0.2
        )

    print(f"  Quality acceptable: {validator.is_quality_acceptable()}")
    print(f"  Quality score: {validator.calculate_quality_score():.3f}")

    # Simulate entropy collapse
    print("\n2. Testing entropy collapse:")
    validator.reset()
    validator.add_metrics(
        mean_return=20.0,
        entropy=0.02,  # Too low
        policy_loss=0.3,
        approx_kl=0.03
    )
    print(f"  Quality acceptable: {validator.is_quality_acceptable()}")
    print(f"  Quality score: {validator.calculate_quality_score():.3f}")

    # Simulate divergence
    print("\n3. Testing divergence:")
    validator.reset()
    validator.add_metrics(
        mean_return=float('nan'),
        entropy=0.15,
        policy_loss=0.3,
        approx_kl=0.03
    )
    print(f"  Quality acceptable: {validator.is_quality_acceptable()}")
    print(f"  Has diverged: {validator._has_diverged}")
    print(f"  Reason: {validator._divergence_reason}")

    # Full diagnostics
    print("\n4. Full diagnostics:")
    validator.reset()
    for i in range(10):
        validator.add_metrics(
            mean_return=5.0 + i * 2,
            entropy=0.12,
            policy_loss=0.25,
            value_loss=0.4,
            approx_kl=0.02,
            clip_fraction=0.15
        )

    diag = validator.get_diagnostics()
    print(f"  Samples: {diag['num_samples']}")
    print(f"  Latest return: {diag['latest_metrics']['mean_return']:.2f}")
    print(f"  Quality score: {diag['quality_score']:.3f}")
    print(f"  Warnings: {diag['warnings']}")

    print("\n✓ Quality validator test complete")
