"""
Corrective Action Manager - Multi-level intervention system for training quality control

This module implements a sophisticated intervention system that monitors training
health and takes corrective actions when problems are detected:
- Level 1 (Warning): Log alerts
- Level 2 (Adjust): Modify hyperparameters via PolicyController
- Level 3 (Rollback): Load previous checkpoint and resume
- Level 4 (Stop): Halt training for manual intervention

Key Features:
- Automatic rollback on Sharpe drop >20%
- NaN/Inf detection and recovery
- Sustained divergence monitoring
- Integration with CheckpointRegistry
- Comprehensive intervention logging

Author: Javier (@javiertradess)
Version: 1.0.0
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque
import logging

import yaml
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from metric_forecaster import ForecastResult, AlertLevel
from checkpoint_registry import CheckpointRegistry
from model_utils import load_model_auto, load_vecnormalize, resolve_learning_rate


InterventionLevel = Literal['warning', 'adjust', 'rollback', 'stop']


@dataclass
class InterventionRecord:
    """Record of an intervention action"""
    timestamp: str
    timesteps: int
    level: InterventionLevel
    trigger: str
    reason: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class CorrectiveActionManager(BaseCallback):
    """
    Multi-level intervention system for automatic training quality control.

    Monitors training metrics and forecasts to detect problems early and
    take appropriate corrective actions:

    Level 1 (Warning): Log alerts for minor issues
    Level 2 (Adjust): Trigger PolicyController for hyperparameter tuning
    Level 3 (Rollback): Load best recent checkpoint and resume
    Level 4 (Stop): Halt training for manual review

    Rollback Triggers:
    - Sharpe ratio drop >20% from recent best
    - NaN or Inf rewards detected
    - Sustained divergence with high confidence

    Stop Triggers:
    - Sharpe drop >50% (critical)
    - Apex compliance violations
    - Repeated rollback failures

    Usage:
        registry = CheckpointRegistry()
        manager = CorrectiveActionManager(
            metric_tracker=eval_tracker,
            registry=registry,
            config_path='config/checkpoint_config.yaml',
            verbose=True
        )
        model.learn(..., callback=[eval_callback, policy_controller, manager])
    """

    def __init__(
        self,
        metric_tracker: Any,
        registry: CheckpointRegistry,
        market: str,
        phase: int,
        config_path: str = 'config/checkpoint_config.yaml',
        policy_controller: Optional[Any] = None,
        verbose: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize CorrectiveActionManager.

        Args:
            metric_tracker: MetricTrackingEvalCallback with forecaster
            registry: CheckpointRegistry for finding best checkpoints
            market: Market symbol (ES, NQ, etc.)
            phase: Training phase (1, 2, or 3)
            config_path: Path to configuration YAML
            policy_controller: Optional PolicyController instance
            verbose: Print intervention actions to console
            logger: Optional logger instance
        """
        super().__init__(verbose=0)  # We handle our own logging

        self.metric_tracker = metric_tracker
        self.registry = registry
        self.market = market
        self.phase = phase
        self.policy_controller = policy_controller
        self.verbose_mode = verbose
        self.app_logger = logger or logging.getLogger(__name__)

        # Load configuration
        self.config = self._load_config(config_path)
        self.enabled = self.config.get('enabled', True)

        if not self.enabled:
            self.app_logger.info("CorrectiveActionManager is disabled in config")
            return

        # Level configurations
        self.level_config = self.config.get('levels', {})
        self.rollback_triggers = self.config.get('rollback_triggers', {})
        self.stop_triggers = self.config.get('stop_triggers', {})

        # State tracking
        self.intervention_history: List[InterventionRecord] = []
        self.warning_count = 0
        self.adjustment_count = 0
        self.rollback_count = 0
        self.stop_requested = False

        # Rollback tracking
        self.max_rollbacks = self.level_config.get('rollback', {}).get('max_rollbacks', 2)
        self.max_adjustments = self.level_config.get('adjust', {}).get('max_adjustments', 5)

        # Metric tracking for trend detection
        self.sharpe_history = deque(maxlen=20)  # Track recent Sharpe values
        self.best_sharpe = float('-inf')
        self.divergence_streak = 0  # Consecutive diverging evaluations

    def _load_config(self, config_path: str) -> Dict:
        """Load corrective actions configuration"""
        try:
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            return full_config.get('corrective_actions', {})
        except Exception as e:
            self.app_logger.warning(f"Failed to load config from {config_path}: {e}")
            return {'enabled': False}

    def _init_callback(self) -> None:
        """Called when training starts"""
        if not self.enabled:
            return

        self.app_logger.info(
            f"CorrectiveActionManager initialized - "
            f"Max rollbacks: {self.max_rollbacks}, Max adjustments: {self.max_adjustments}"
        )

    def _on_step(self) -> bool:
        """
        Called at each training step.

        Returns:
            False if stop requested, True otherwise
        """
        if not self.enabled:
            return True

        # Check if stop was requested
        if self.stop_requested:
            self.app_logger.error("Training stopped by CorrectiveActionManager")
            return False

        # Check for forecast results
        if not hasattr(self.metric_tracker, 'last_forecast'):
            return True

        forecast = self.metric_tracker.last_forecast
        if forecast is None:
            return True

        # Update Sharpe history
        current_sharpe = self.metric_tracker.last_metrics.get('sharpe_ratio', 0.0)
        self.sharpe_history.append(current_sharpe)
        if current_sharpe > self.best_sharpe:
            self.best_sharpe = current_sharpe

        # Check for NaN/Inf
        if self._check_nan_inf(forecast):
            return self._handle_nan_inf()

        # Check for critical Sharpe drop
        if self._check_critical_sharpe_drop():
            return self._handle_critical_drop()

        # Check for rollback triggers
        if self._check_rollback_triggers(forecast):
            return self._handle_rollback(forecast)

        # Check for sustained divergence
        if forecast.trend == 'diverging' and forecast.confidence > 0.7:
            self.divergence_streak += 1
        else:
            self.divergence_streak = 0

        sustained_div_config = self.rollback_triggers.get('sustained_divergence', {})
        if sustained_div_config.get('enabled', True):
            min_confidence = sustained_div_config.get('min_confidence', 0.75)
            consecutive_evals = sustained_div_config.get('consecutive_evals', 3)

            if self.divergence_streak >= consecutive_evals and forecast.confidence >= min_confidence:
                self.app_logger.warning(
                    f"Sustained divergence detected: {self.divergence_streak} consecutive evals "
                    f"with confidence {forecast.confidence:.2f}"
                )
                return self._handle_rollback(forecast)

        # Check for warning/adjustment level interventions
        if forecast.alert_level == 'warning':
            self._handle_warning(forecast)
        elif forecast.alert_level == 'critical':
            # Critical alert -> try adjustment first, then rollback
            if self.adjustment_count < self.max_adjustments:
                self._handle_adjustment(forecast)
            else:
                self.app_logger.warning("Max adjustments reached, escalating to rollback")
                return self._handle_rollback(forecast)

        return True

    def _check_nan_inf(self, forecast: ForecastResult) -> bool:
        """Check if metrics contain NaN or Inf"""
        if not self.rollback_triggers.get('nan_inf_detection', True):
            return False

        current_value = forecast.current_value
        forecast_next = forecast.forecast_next

        return not (math.isfinite(current_value) and math.isfinite(forecast_next))

    def _check_critical_sharpe_drop(self) -> bool:
        """Check if Sharpe dropped critically (>50%)"""
        if len(self.sharpe_history) < 2:
            return False

        critical_threshold = self.stop_triggers.get('critical_sharpe_drop', 0.50)
        current_sharpe = self.sharpe_history[-1]

        if self.best_sharpe > 0:
            drop_pct = (self.best_sharpe - current_sharpe) / abs(self.best_sharpe)
            return drop_pct > critical_threshold

        return False

    def _check_rollback_triggers(self, forecast: ForecastResult) -> bool:
        """Check if any rollback trigger conditions are met"""
        if len(self.sharpe_history) < 5:
            return False

        # Get Sharpe drop threshold
        sharpe_threshold = self.rollback_triggers.get('sharpe_drop_threshold', 0.20)

        # Calculate recent best Sharpe (last 10 evaluations)
        recent_history = list(self.sharpe_history)[-10:]
        recent_best = max(recent_history)
        current_sharpe = self.sharpe_history[-1]

        if recent_best > 0:
            drop_pct = (recent_best - current_sharpe) / abs(recent_best)

            if drop_pct > sharpe_threshold:
                self.app_logger.warning(
                    f"Sharpe drop detected: {recent_best:.2f} â†’ {current_sharpe:.2f} "
                    f"({drop_pct*100:.1f}% drop)"
                )
                return True

        return False

    def _handle_nan_inf(self) -> bool:
        """Handle NaN/Inf detection"""
        self.app_logger.error("NaN or Inf detected in metrics - triggering rollback")

        record = InterventionRecord(
            timestamp=datetime.now().isoformat(),
            timesteps=self.num_timesteps,
            level='rollback',
            trigger='nan_inf_detection',
            reason='Metrics contain NaN or Inf values',
            details={
                'last_sharpe': float(self.sharpe_history[-1]) if self.sharpe_history else 0.0
            }
        )
        self.intervention_history.append(record)

        return self._execute_rollback('nan_inf_detection')

    def _handle_critical_drop(self) -> bool:
        """Handle critical Sharpe drop (>50%)"""
        self.app_logger.error("Critical Sharpe drop detected - stopping training")

        record = InterventionRecord(
            timestamp=datetime.now().isoformat(),
            timesteps=self.num_timesteps,
            level='stop',
            trigger='critical_sharpe_drop',
            reason=f'Sharpe dropped >50% from best ({self.best_sharpe:.2f})',
            details={
                'best_sharpe': self.best_sharpe,
                'current_sharpe': float(self.sharpe_history[-1])
            }
        )
        self.intervention_history.append(record)

        self.stop_requested = True
        return False  # Stop training

    def _handle_warning(self, forecast: ForecastResult) -> None:
        """Handle warning level intervention"""
        warning_config = self.level_config.get('warning', {})
        if not warning_config.get('enabled', True):
            return

        self.warning_count += 1

        record = InterventionRecord(
            timestamp=datetime.now().isoformat(),
            timesteps=self.num_timesteps,
            level='warning',
            trigger=forecast.trend,
            reason=forecast.reason,
            details={
                'sharpe': float(self.sharpe_history[-1]) if self.sharpe_history else 0.0,
                'confidence': forecast.confidence
            }
        )
        self.intervention_history.append(record)

        # Log to file if enabled
        if warning_config.get('log_to_file', True):
            self._log_intervention(record, warning_config.get('log_path'))

        if self.verbose_mode:
            print(f"\n[WARNING] {forecast.reason}")

    def _handle_adjustment(self, forecast: ForecastResult) -> None:
        """Handle adjustment level intervention (delegate to PolicyController)"""
        adjust_config = self.level_config.get('adjust', {})
        if not adjust_config.get('enabled', True):
            return

        self.adjustment_count += 1

        # Actually trigger PolicyController adjustment (if available)
        adjustment_made = False
        if self.policy_controller is not None:
            adjustment_made = self.policy_controller.force_adjustment(forecast)

        record = InterventionRecord(
            timestamp=datetime.now().isoformat(),
            timesteps=self.num_timesteps,
            level='adjust',
            trigger=forecast.trend,
            reason=f'Hyperparameter adjustment triggered by {forecast.trend} trend',
            details={
                'forecast_recommendation': forecast.recommendation,
                'sharpe': float(self.sharpe_history[-1]) if self.sharpe_history else 0.0,
                'adjustment_applied': adjustment_made
            }
        )
        self.intervention_history.append(record)

        if adjustment_made:
            self.app_logger.info(f"Adjustment #{self.adjustment_count} applied by PolicyController")
        else:
            self.app_logger.warning(f"Adjustment #{self.adjustment_count} requested but PolicyController made no changes")

    def _handle_rollback(self, forecast: ForecastResult) -> bool:
        """
        Handle rollback level intervention.

        Returns:
            False to stop training (rollback failed), True if successful
        """
        rollback_config = self.level_config.get('rollback', {})
        if not rollback_config.get('enabled', True):
            return True

        # Check rollback limit
        if self.rollback_count >= self.max_rollbacks:
            self.app_logger.error(f"Max rollbacks ({self.max_rollbacks}) reached - stopping training")
            self.stop_requested = True
            return False

        self.rollback_count += 1

        record = InterventionRecord(
            timestamp=datetime.now().isoformat(),
            timesteps=self.num_timesteps,
            level='rollback',
            trigger=forecast.trend,
            reason=forecast.reason,
            details={
                'rollback_number': self.rollback_count,
                'current_sharpe': float(self.sharpe_history[-1]) if self.sharpe_history else 0.0,
                'best_sharpe': self.best_sharpe
            }
        )
        self.intervention_history.append(record)

        return self._execute_rollback(forecast.trend)

    def _execute_rollback(self, trigger: str) -> bool:
        """
        Execute checkpoint rollback and resume training.

        Args:
            trigger: Trigger reason for rollback

        Returns:
            False to stop training (rollback failed), True if successful
        """
        try:
            # Find best recent checkpoint
            lookback_steps = self.level_config.get('rollback', {}).get('lookback_timesteps', 500000)
            min_timesteps = max(0, self.num_timesteps - lookback_steps)

            best_checkpoint = self.registry.get_best_checkpoint(
                metric='sharpe_ratio',
                market=self.market,
                phase=self.phase,
                min_timesteps=min_timesteps,
                max_timesteps=self.num_timesteps
            )

            if best_checkpoint is None:
                self.app_logger.error("No suitable checkpoint found for rollback - stopping training")
                self.stop_requested = True
                return False

            checkpoint_path = Path(best_checkpoint.path)

            if self.verbose_mode:
                print(f"\n{'='*60}")
                print(f"ROLLBACK TRIGGERED - Checkpoint Restore")
                print(f"{'='*60}")
                print(f"Trigger: {trigger}")
                print(f"Rollback #{self.rollback_count}/{self.max_rollbacks}")
                print(f"Restoring checkpoint: {checkpoint_path.name}")
                print(f"Checkpoint Sharpe: {best_checkpoint.sharpe_ratio:.2f}")
                print(f"Current Sharpe: {self.sharpe_history[-1]:.2f}")
                print(f"{'='*60}\n")

            # Load checkpoint model (load_model_auto returns tuple: (model, model_type))
            load_result = load_model_auto(checkpoint_path)
            if load_result is None:
                raise ValueError(f"Failed to load model from {checkpoint_path}")

            # Unpack tuple return
            loaded_model, model_type = load_result
            self.app_logger.info(f"Loaded {model_type} model from {checkpoint_path.name}")

            # Load VecNormalize state if available
            vecnorm_path = checkpoint_path.parent / (checkpoint_path.stem + "_vecnormalize.pkl")
            if vecnorm_path.exists() and hasattr(self.training_env, 'load'):
                # load_vecnormalize requires (path, env) parameters
                vec_env = load_vecnormalize(vecnorm_path, self.training_env)
                if vec_env is not None:
                    # Copy normalization stats to current environment
                    if isinstance(self.training_env, VecNormalize):
                        self.training_env.obs_rms = vec_env.obs_rms
                        self.training_env.ret_rms = vec_env.ret_rms
                        self.app_logger.info("VecNormalize state restored")

            # Transfer learned weights to current model
            self.model.policy.load_state_dict(loaded_model.policy.state_dict())
            self.model.num_timesteps = best_checkpoint.timesteps

            # Apply conservative hyperparameters after rollback
            if hasattr(self.model, 'learning_rate'):
                original_lr = resolve_learning_rate(getattr(self.model, 'learning_rate', None))
                if original_lr is not None:
                    reduced_lr = original_lr * 0.5  # 50% reduction post-rollback
                    self.model.learning_rate = reduced_lr
                    if hasattr(self.model, 'lr_schedule'):
                        self.model.lr_schedule = lambda _: reduced_lr
                    self.app_logger.info(
                        f"Learning rate reduced post-rollback: {original_lr:.2e} -> {reduced_lr:.2e}"
                    )

            self.app_logger.info(f"Rollback successful - restored to timestep {best_checkpoint.timesteps:,}")
            return True

        except Exception as e:
            self.app_logger.error(f"Rollback failed: {e}")
            self.stop_requested = True
            return False

    def _log_intervention(self, record: InterventionRecord, log_path: Optional[str] = None) -> None:
        """Log intervention to JSON file"""
        if log_path is None:
            log_path = "logs/corrective_actions.json"

        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing log
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = {'interventions': []}

        # Append new record
        log_data['interventions'].append(record.to_dict())

        # Save updated log
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    def get_intervention_summary(self) -> Dict[str, Any]:
        """Get summary of all interventions"""
        return {
            'total_warnings': self.warning_count,
            'total_adjustments': self.adjustment_count,
            'total_rollbacks': self.rollback_count,
            'stop_requested': self.stop_requested,
            'intervention_history': [rec.to_dict() for rec in self.intervention_history]
        }

    def _on_training_end(self) -> None:
        """Called when training ends"""
        if not self.enabled:
            return

        # Print summary
        if self.verbose_mode:
            print(f"\n{'='*60}")
            print(f"CORRECTIVE ACTION MANAGER - Summary")
            print(f"{'='*60}")
            print(f"Warnings: {self.warning_count}")
            print(f"Adjustments: {self.adjustment_count}")
            print(f"Rollbacks: {self.rollback_count}/{self.max_rollbacks}")
            print(f"Training Stopped: {self.stop_requested}")
            print(f"{'='*60}\n")

        # Save intervention history
        summary = self.get_intervention_summary()
        summary_path = Path("logs/corrective_action_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.app_logger.info(f"Intervention summary saved to {summary_path}")
