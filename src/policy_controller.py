"""
Policy Controller - Dynamic hyperparameter adjustment for RL training

This module implements intelligent hyperparameter adjustment based on training
metrics and forecast trends. It acts as a control loop that responds to
plateau, divergence, or oscillation in performance metrics.

Key Features:
- Learning rate reduction on plateau/divergence
- Entropy increase on oscillation (exploration boost)
- Clip range adjustment on policy divergence
- Cooldown mechanism to prevent rapid changes
- Comprehensive logging and action tracking

Author: Javier (@javiertradess)
Version: 1.0.0
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

import yaml
from stable_baselines3.common.callbacks import BaseCallback

from metric_forecaster import ForecastResult, TrendType
from model_utils import resolve_learning_rate


@dataclass
class ControlAction:
    """Record of a control action taken"""
    timestamp: str
    timesteps: int
    action_type: str  # 'adjust_lr', 'adjust_ent', 'adjust_clip'
    trigger: TrendType
    parameters: Dict[str, Any]
    reason: str

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class PolicyController(BaseCallback):
    """
    Dynamic policy controller that adjusts hyperparameters based on metric forecasts.

    This callback monitors forecast results from MetricForecaster and automatically
    adjusts:
    - Learning rate (reduced on plateau/divergence)
    - Entropy coefficient (increased on oscillation)
    - Clip range (reduced on policy divergence)

    Features:
    - Configurable triggers and adjustment factors
    - Cooldown period to prevent rapid changes
    - Action history tracking
    - TensorBoard logging

    Usage:
        controller = PolicyController(
            metric_tracker=eval_tracker,
            config_path='config/checkpoint_config.yaml',
            verbose=True
        )
        model.learn(..., callback=[eval_callback, controller])
    """

    def __init__(
        self,
        metric_tracker: Any,
        config_path: str = 'config/checkpoint_config.yaml',
        verbose: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize PolicyController.

        Args:
            metric_tracker: MetricTrackingEvalCallback with forecaster
            config_path: Path to configuration YAML
            verbose: Print control actions to console
            logger: Optional logger instance
        """
        super().__init__(verbose=0)  # We handle our own logging

        self.metric_tracker = metric_tracker
        self.verbose_mode = verbose
        self.app_logger = logger or logging.getLogger(__name__)

        # Load configuration
        self.config = self._load_config(config_path)
        self.enabled = self.config.get('enabled', True)

        if not self.enabled:
            self.app_logger.info("PolicyController is disabled in config")
            return

        # Get sub-configs
        self.lr_config = self.config.get('learning_rate', {})
        self.ent_config = self.config.get('entropy', {})
        self.clip_config = self.config.get('clip_range', {})
        self.cooldown_steps = self.config.get('cooldown_steps', 100000)

        # State tracking
        self.last_adjustment_timestep = 0
        self.adjustment_count = 0
        self.action_history: List[ControlAction] = []

        # Original hyperparameters (saved for reference)
        self.original_lr = None
        self.original_ent = None
        self.original_clip = None

    def _load_config(self, config_path: str) -> Dict:
        """Load policy controller configuration"""
        try:
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            return full_config.get('policy_controller', {})
        except Exception as e:
            self.app_logger.warning(f"Failed to load config from {config_path}: {e}")
            return {'enabled': False}

    def _init_callback(self) -> None:
        """Called when training starts"""
        if not self.enabled:
            return

        # Save original hyperparameters
        if hasattr(self.model, 'learning_rate'):
            self.original_lr = resolve_learning_rate(getattr(self.model, 'learning_rate', None))
            if self.original_lr is not None:
                self.app_logger.info(f"PolicyController initialized - Original LR: {self.original_lr:.2e}")

        if hasattr(self.model, 'ent_coef'):
            self.original_ent = float(self.model.ent_coef)
            self.app_logger.info(f"PolicyController initialized - Original Entropy: {self.original_ent:.3f}")

        if hasattr(self.model, 'clip_range'):
            # Handle both float and callable clip_range
            if callable(self.model.clip_range):
                self.original_clip = self.model.clip_range(1.0)
            else:
                self.original_clip = float(self.model.clip_range)
            self.app_logger.info(f"PolicyController initialized - Original Clip Range: {self.original_clip:.3f}")

    def _on_step(self) -> bool:
        """
        Called at each training step.

        Returns:
            True to continue training
        """
        if not self.enabled:
            return True

        # Check if we have a forecast result
        if not hasattr(self.metric_tracker, 'last_forecast'):
            return True

        forecast = self.metric_tracker.last_forecast
        if forecast is None:
            return True

        # Check cooldown
        if self.num_timesteps - self.last_adjustment_timestep < self.cooldown_steps:
            return True

        # Check if action is needed
        if forecast.recommendation == 'none':
            return True

        # Take action based on trend and recommendation
        self._take_action(forecast)

        return True

    def force_adjustment(self, forecast: ForecastResult) -> bool:
        """
        Force an immediate adjustment bypassing cooldown (called by CorrectiveActionManager).

        Args:
            forecast: ForecastResult from MetricForecaster

        Returns:
            True if adjustment was made, False otherwise
        """
        if not self.enabled:
            return False

        # Take action immediately (bypass cooldown)
        self._take_action(forecast)

        # Check if any adjustments were actually made
        return self.adjustment_count > 0

    def _take_action(self, forecast: ForecastResult) -> None:
        """
        Take control action based on forecast.

        Args:
            forecast: ForecastResult from MetricForecaster
        """
        trend = forecast.trend
        actions_taken = []

        # Learning rate adjustment
        if self.lr_config.get('enabled', False):
            if trend in self.lr_config.get('trigger_on', []):
                if self._adjust_learning_rate(trend, forecast):
                    actions_taken.append('learning_rate')

        # Entropy adjustment
        if self.ent_config.get('enabled', False):
            if trend in self.ent_config.get('trigger_on', []):
                if self._adjust_entropy(trend, forecast):
                    actions_taken.append('entropy')

        # Clip range adjustment
        if self.clip_config.get('enabled', False):
            if trend in self.clip_config.get('trigger_on', []):
                if self._adjust_clip_range(trend, forecast):
                    actions_taken.append('clip_range')

        # Update adjustment tracking
        if actions_taken:
            self.last_adjustment_timestep = self.num_timesteps
            self.adjustment_count += 1

            if self.verbose_mode:
                print(f"\n{'='*60}")
                print(f"POLICY CONTROLLER - Action Taken at {self.num_timesteps:,} steps")
                print(f"{'='*60}")
                print(f"Trigger: {trend} trend (confidence: {forecast.confidence:.2f})")
                print(f"Adjusted: {', '.join(actions_taken)}")
                print(f"Reason: {forecast.reason}")
                print(f"{'='*60}\n")

    def _adjust_learning_rate(self, trend: TrendType, forecast: ForecastResult) -> bool:
        """
        Adjust learning rate based on trend.

        Args:
            trend: Detected trend type
            forecast: Forecast result

        Returns:
            True if adjustment was made
        """
        if not hasattr(self.model, 'learning_rate'):
            return False

        current_lr = resolve_learning_rate(getattr(self.model, 'learning_rate', None))
        if current_lr is None:
            return False
        reduction_factor = self.lr_config.get('reduction_factor', 0.5)
        min_lr = self.lr_config.get('min_lr', 1e-5)

        new_lr = max(current_lr * reduction_factor, min_lr)

        if new_lr < current_lr:
            # Apply adjustment
            self.model.learning_rate = new_lr

            if hasattr(self.model, 'lr_schedule'):
                # Replace schedule with constant returning the new LR
                self.model.lr_schedule = lambda _: new_lr

            # Record action
            action = ControlAction(
                timestamp=datetime.now().isoformat(),
                timesteps=self.num_timesteps,
                action_type='adjust_lr',
                trigger=trend,
                parameters={
                    'old_lr': current_lr,
                    'new_lr': new_lr,
                    'reduction_factor': reduction_factor
                },
                reason=f"Learning rate reduced due to {trend} trend"
            )
            self.action_history.append(action)

            # Log to TensorBoard
            self.logger.record('policy_controller/learning_rate', new_lr)
            self.logger.record('policy_controller/lr_adjustment_count', self.adjustment_count)

            self.app_logger.info(
                f"LR adjusted: {current_lr:.2e} -> {new_lr:.2e} "
                f"(trigger: {trend}, confidence: {forecast.confidence:.2f})"
            )
            return True

        return False

    def _adjust_entropy(self, trend: TrendType, forecast: ForecastResult) -> bool:
        """
        Adjust entropy coefficient to increase exploration.

        Args:
            trend: Detected trend type
            forecast: Forecast result

        Returns:
            True if adjustment was made
        """
        if not hasattr(self.model, 'ent_coef'):
            return False

        current_ent = float(self.model.ent_coef)
        increase_factor = self.ent_config.get('increase_factor', 2.0)
        max_ent = self.ent_config.get('max_ent', 0.1)

        new_ent = min(current_ent * increase_factor, max_ent)

        if new_ent > current_ent:
            # Apply adjustment
            self.model.ent_coef = new_ent

            # Record action
            action = ControlAction(
                timestamp=datetime.now().isoformat(),
                timesteps=self.num_timesteps,
                action_type='adjust_ent',
                trigger=trend,
                parameters={
                    'old_ent': current_ent,
                    'new_ent': new_ent,
                    'increase_factor': increase_factor
                },
                reason=f"Entropy increased to boost exploration (trigger: {trend})"
            )
            self.action_history.append(action)

            # Log to TensorBoard
            self.logger.record('policy_controller/entropy_coef', new_ent)

            self.app_logger.info(
                f"Entropy adjusted: {current_ent:.3f} -> {new_ent:.3f} "
                f"(trigger: {trend}, confidence: {forecast.confidence:.2f})"
            )
            return True

        return False

    def _adjust_clip_range(self, trend: TrendType, forecast: ForecastResult) -> bool:
        """
        Adjust PPO clip range for policy stability.

        Args:
            trend: Detected trend type
            forecast: Forecast result

        Returns:
            True if adjustment was made
        """
        if not hasattr(self.model, 'clip_range'):
            return False

        # Get current clip range
        if callable(self.model.clip_range):
            current_clip = self.model.clip_range(1.0)
        else:
            current_clip = float(self.model.clip_range)

        reduction_factor = self.clip_config.get('reduction_factor', 0.8)
        min_clip = self.clip_config.get('min_clip', 0.05)

        new_clip = max(current_clip * reduction_factor, min_clip)

        if new_clip < current_clip:
            # Apply adjustment (convert to constant if it was a function)
            self.model.clip_range = new_clip

            # Record action
            action = ControlAction(
                timestamp=datetime.now().isoformat(),
                timesteps=self.num_timesteps,
                action_type='adjust_clip',
                trigger=trend,
                parameters={
                    'old_clip': current_clip,
                    'new_clip': new_clip,
                    'reduction_factor': reduction_factor
                },
                reason=f"Clip range reduced for policy stability (trigger: {trend})"
            )
            self.action_history.append(action)

            # Log to TensorBoard
            self.logger.record('policy_controller/clip_range', new_clip)

            self.app_logger.info(
                f"Clip range adjusted: {current_clip:.3f} -> {new_clip:.3f} "
                f"(trigger: {trend}, confidence: {forecast.confidence:.2f})"
            )
            return True

        return False

    def get_action_history(self) -> List[Dict]:
        """
        Get history of all control actions.

        Returns:
            List of action dictionaries
        """
        return [action.to_dict() for action in self.action_history]

    def save_action_history(self, filepath: Optional[Path] = None) -> None:
        """
        Save action history to JSON file.

        Args:
            filepath: Path to save file (default: logs/policy_controller_actions.json)
        """
        if filepath is None:
            filepath = Path("logs/policy_controller_actions.json")

        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'total_adjustments': self.adjustment_count,
            'actions': self.get_action_history(),
            'config': {
                'learning_rate': self.lr_config,
                'entropy': self.ent_config,
                'clip_range': self.clip_config,
                'cooldown_steps': self.cooldown_steps
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.app_logger.info(f"Saved {self.adjustment_count} control actions to {filepath}")

    def _on_training_end(self) -> None:
        """Called when training ends"""
        if not self.enabled:
            return

        # Save action history
        self.save_action_history()

        # Print summary
        if self.verbose_mode and self.adjustment_count > 0:
            print(f"\n{'='*60}")
            print(f"POLICY CONTROLLER - Training Summary")
            print(f"{'='*60}")
            print(f"Total Adjustments: {self.adjustment_count}")

            if self.original_lr is not None and hasattr(self.model, 'learning_rate'):
                final_lr = resolve_learning_rate(getattr(self.model, 'learning_rate', None))
                if final_lr is not None:
                    print(f"Learning Rate: {self.original_lr:.2e} -> {final_lr:.2e}")

            if self.original_ent and hasattr(self.model, 'ent_coef'):
                final_ent = float(self.model.ent_coef)
                print(f"Entropy Coef: {self.original_ent:.3f} -> {final_ent:.3f}")

            if self.original_clip and hasattr(self.model, 'clip_range'):
                if callable(self.model.clip_range):
                    final_clip = self.model.clip_range(1.0)
                else:
                    final_clip = float(self.model.clip_range)
                print(f"Clip Range: {self.original_clip:.3f} -> {final_clip:.3f}")

            print(f"{'='*60}\n")
