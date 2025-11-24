"""
Metric Forecaster - Time-series analysis and prediction for training metrics

This module uses Kalman Filtering to forecast training metrics, detect trends,
and identify anomalies that may require corrective action.

Key Features:
- Kalman Filter implementation for metric prediction
- Trend detection (improving, plateau, diverging, oscillating)
- Multi-step ahead forecasting (1-step and 5-step)
- Alert level classification (none, warning, critical)
- Confidence scores for recommendations

Author: Javier (@javiertradess)
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, asdict
from collections import deque
import logging


TrendType = Literal['improving', 'plateau', 'diverging', 'oscillating', 'insufficient_data']
AlertLevel = Literal['none', 'warning', 'critical']
Recommendation = Literal['none', 'reduce_lr', 'increase_ent', 'rollback', 'stop']


@dataclass
class ForecastResult:
    """Result from metric forecasting"""
    trend: TrendType
    confidence: float
    forecast_next: float
    forecast_5_steps: List[float]
    alert_level: AlertLevel
    recommendation: Recommendation
    reason: str

    # Historical data
    current_value: float
    previous_value: Optional[float]
    mean_value: float
    std_value: float

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class KalmanFilter:
    """
    Simple Kalman Filter for 1D time-series prediction.

    State model:
        x = [metric, velocity]

    Prediction: metric_next = metric + velocity * dt
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.05,
        initial_estimate: float = 0.0
    ):
        """
        Initialize Kalman Filter.

        Args:
            process_noise: Process noise covariance (Q)
            measurement_noise: Measurement noise covariance (R)
            initial_estimate: Initial state estimate
        """
        # State: [metric, velocity]
        self.x = np.array([initial_estimate, 0.0])

        # State covariance
        self.P = np.eye(2) * 1.0

        # Process noise covariance
        self.Q = np.array([
            [process_noise, 0],
            [0, process_noise * 0.1]  # Lower noise for velocity
        ])

        # Measurement noise covariance
        self.R = np.array([[measurement_noise]])

        # State transition matrix (dt = 1)
        self.F = np.array([
            [1.0, 1.0],  # metric_new = metric_old + velocity
            [0.0, 1.0]   # velocity_new = velocity_old (constant velocity)
        ])

        # Measurement matrix (we only observe metric, not velocity)
        self.H = np.array([[1.0, 0.0]])

    def predict(self) -> Tuple[float, float]:
        """
        Predict next state (a priori estimate).

        Returns:
            Tuple of (predicted_metric, predicted_velocity)
        """
        # Predict state
        self.x = self.F @ self.x

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x[0], self.x[1]

    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Update state with new measurement (a posteriori estimate).

        Args:
            measurement: Observed metric value

        Returns:
            Tuple of (updated_metric, updated_velocity)
        """
        # Innovation (measurement residual)
        z = np.array([measurement])
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

        return self.x[0], self.x[1]

    def forecast(self, n_steps: int = 1) -> List[float]:
        """
        Forecast N steps ahead.

        Args:
            n_steps: Number of steps to forecast

        Returns:
            List of forecasted values
        """
        forecasts = []
        x_forecast = self.x.copy()

        for _ in range(n_steps):
            x_forecast = self.F @ x_forecast
            forecasts.append(x_forecast[0])

        return forecasts


class MetricForecaster:
    """
    Time-series forecaster for training metrics using Kalman Filtering.

    Tracks metric history, detects trends, and provides recommendations
    for corrective actions.
    """

    def __init__(
        self,
        metric_name: str = 'sharpe_ratio',
        history_size: int = 50,
        process_noise: float = 0.01,
        measurement_noise: float = 0.05,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the metric forecaster.

        Args:
            metric_name: Name of metric to track (e.g., 'sharpe_ratio')
            history_size: Number of historical observations to keep
            process_noise: Kalman filter process noise
            measurement_noise: Kalman filter measurement noise
            logger: Optional logger instance
        """
        self.metric_name = metric_name
        self.history_size = history_size
        self.logger = logger or logging.getLogger(__name__)

        # Historical data
        self.history: deque = deque(maxlen=history_size)
        self.timesteps_history: deque = deque(maxlen=history_size)

        # Kalman filter
        self.kf = KalmanFilter(
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )

        # Trend detection parameters
        self.plateau_threshold = 0.02  # 2% change = plateau
        self.divergence_threshold = -0.10  # -10% change = diverging
        self.min_samples = 5  # Minimum samples for trend detection

    def update(self, value: float, timesteps: int) -> ForecastResult:
        """
        Update forecaster with new metric value.

        Args:
            value: New metric value
            timesteps: Training timesteps at this measurement

        Returns:
            ForecastResult with trend analysis and forecast
        """
        # Add to history
        self.history.append(value)
        self.timesteps_history.append(timesteps)

        # Update Kalman filter
        if len(self.history) == 1:
            # First measurement - initialize filter
            self.kf.x[0] = value
            self.kf.update(value)
        else:
            # Predict then update
            self.kf.predict()
            self.kf.update(value)

        # Generate forecast
        forecast_result = self._generate_forecast()

        self.logger.debug(
            f"Metric update: {self.metric_name}={value:.3f}, "
            f"Trend={forecast_result.trend}, Alert={forecast_result.alert_level}"
        )

        return forecast_result

    def _generate_forecast(self) -> ForecastResult:
        """
        Generate forecast and trend analysis.

        Returns:
            ForecastResult with predictions and recommendations
        """
        if len(self.history) < 2:
            return ForecastResult(
                trend='insufficient_data',
                confidence=0.0,
                forecast_next=self.history[-1] if self.history else 0.0,
                forecast_5_steps=[],
                alert_level='none',
                recommendation='none',
                reason='Insufficient data for forecasting',
                current_value=self.history[-1] if self.history else 0.0,
                previous_value=None,
                mean_value=self.history[-1] if self.history else 0.0,
                std_value=0.0
            )

        # Current statistics
        current_value = self.history[-1]
        previous_value = self.history[-2]
        mean_value = np.mean(self.history)
        std_value = np.std(self.history)

        # Multi-step forecast
        forecast_5_steps = self.kf.forecast(n_steps=5)
        forecast_next = forecast_5_steps[0]

        # Detect trend
        trend, confidence = self._detect_trend()

        # Determine alert level and recommendation
        alert_level, recommendation, reason = self._classify_alert(
            trend, current_value, forecast_next, confidence
        )

        return ForecastResult(
            trend=trend,
            confidence=confidence,
            forecast_next=forecast_next,
            forecast_5_steps=forecast_5_steps,
            alert_level=alert_level,
            recommendation=recommendation,
            reason=reason,
            current_value=current_value,
            previous_value=previous_value,
            mean_value=mean_value,
            std_value=std_value
        )

    def _detect_trend(self) -> Tuple[TrendType, float]:
        """
        Detect trend in metric history.

        Returns:
            Tuple of (trend_type, confidence)
        """
        if len(self.history) < self.min_samples:
            return 'insufficient_data', 0.0

        # Get recent history (last 10 samples or all if fewer)
        recent_window = min(10, len(self.history))
        recent_values = list(self.history)[-recent_window:]

        # Calculate linear regression slope
        x = np.arange(len(recent_values))
        y = np.array(recent_values)

        # Handle edge case of constant values
        if np.std(y) < 1e-6:
            return 'plateau', 0.95

        # Linear fit
        slope, intercept = np.polyfit(x, y, 1)

        # Normalize slope by mean value
        mean_val = np.mean(y)
        if abs(mean_val) > 1e-6:
            normalized_slope = slope / abs(mean_val)
        else:
            normalized_slope = 0.0

        # Calculate R² for confidence
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-6 else 0.0
        confidence = max(0.0, min(1.0, r_squared))

        # Classify trend based on normalized slope
        if normalized_slope > self.plateau_threshold:
            return 'improving', confidence
        elif normalized_slope < self.divergence_threshold:
            return 'diverging', confidence
        elif abs(normalized_slope) <= self.plateau_threshold:
            # Check for oscillation
            if self._is_oscillating(recent_values):
                return 'oscillating', confidence * 0.8
            else:
                return 'plateau', confidence
        else:
            # Slight negative trend (between plateau and divergence)
            return 'plateau', confidence * 0.9

    def _is_oscillating(self, values: List[float]) -> bool:
        """
        Detect if values are oscillating (not monotonic).

        Args:
            values: List of metric values

        Returns:
            True if oscillating pattern detected
        """
        if len(values) < 4:
            return False

        # Count direction changes
        diffs = np.diff(values)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)

        # If more than 40% of transitions change direction -> oscillating
        return sign_changes > len(diffs) * 0.4

    def _classify_alert(
        self,
        trend: TrendType,
        current_value: float,
        forecast_next: float,
        confidence: float
    ) -> Tuple[AlertLevel, Recommendation, str]:
        """
        Classify alert level and generate recommendation.

        Args:
            trend: Detected trend type
            current_value: Current metric value
            forecast_next: Forecasted next value
            confidence: Trend confidence score

        Returns:
            Tuple of (alert_level, recommendation, reason)
        """
        # Check for NaN or Inf
        if not np.isfinite(current_value) or not np.isfinite(forecast_next):
            return (
                'critical',
                'stop',
                'Metric has NaN or Inf values - training unstable'
            )

        # Diverging trend with high confidence
        if trend == 'diverging' and confidence > 0.7:
            # Check magnitude of divergence
            if len(self.history) >= 5:
                recent_max = max(list(self.history)[-5:])
                drop_pct = (current_value - recent_max) / abs(recent_max) if recent_max != 0 else 0

                if drop_pct < -0.20:  # >20% drop
                    return (
                        'critical',
                        'rollback',
                        f'Sharpe dropped {drop_pct*100:.1f}% - rollback recommended'
                    )
                elif drop_pct < -0.10:  # >10% drop
                    return (
                        'warning',
                        'reduce_lr',
                        f'Diverging trend detected ({drop_pct*100:.1f}% drop) - reduce learning rate'
                    )

            return (
                'warning',
                'reduce_lr',
                'Diverging trend detected - consider reducing learning rate'
            )

        # Plateau with high confidence
        if trend == 'plateau' and confidence > 0.75:
            if len(self.history) >= 10:
                # Check if plateaued for a while
                recent_std = np.std(list(self.history)[-10:])
                if recent_std < self.plateau_threshold:
                    return (
                        'warning',
                        'reduce_lr',
                        'Plateau detected - reduce LR or increase exploration'
                    )

        # Oscillating pattern
        if trend == 'oscillating' and confidence > 0.6:
            return (
                'warning',
                'increase_ent',
                'Oscillating performance - increase entropy for exploration'
            )

        # Improving trend - all good
        if trend == 'improving':
            return (
                'none',
                'none',
                'Metric improving - no action needed'
            )

        # Default - no alert
        return (
            'none',
            'none',
            f'Trend: {trend} (confidence: {confidence:.2f})'
        )

    def get_history(self) -> Dict[str, List]:
        """
        Get historical data.

        Returns:
            Dictionary with timesteps and values
        """
        return {
            'timesteps': list(self.timesteps_history),
            'values': list(self.history),
            'metric_name': self.metric_name
        }

    def reset(self) -> None:
        """Reset forecaster state"""
        self.history.clear()
        self.timesteps_history.clear()
        self.kf = KalmanFilter(
            process_noise=self.kf.Q[0, 0],
            measurement_noise=self.kf.R[0, 0]
        )
        self.logger.info(f"Reset forecaster for {self.metric_name}")
