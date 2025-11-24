"""
Dynamic Checkpoint Management System
Intelligent, event-driven checkpoint saves for RL training

Features:
- Adaptive save intervals (grows as training stabilizes)
- Event-driven triggers (best metric, phase end, interrupt)
- Rich metadata (metrics, runtime info, phase data)
- Disk-full resilience
- Automatic VecNormalize state saving
"""

import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import yaml
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from metadata_utils import write_checkpoint_metadata
from model_utils import resolve_learning_rate
from checkpoint_registry import CheckpointRegistry
from metric_forecaster import MetricForecaster


class DynamicCheckpointManager(BaseCallback):
    """
    Intelligent checkpoint manager with adaptive save intervals and event-driven triggers.

    Features:
    - Adaptive interval: Starts frequent (base_interval), grows to max as training progresses
    - Event-driven saves: New best metric, phase boundaries, interrupts
    - Rich naming: {market}_ts-{steps}_evt-{event}_val-{reward}_sharpe-{sharpe}_seed-{seed}
    - Automatic VecNormalize saving
    - Metadata JSON for every checkpoint
    - Disk-full resilience

    Usage:
        manager = DynamicCheckpointManager(
            market='NQ',
            phase=1,
            seed=42,
            config_path='config/checkpoint_config.yaml',
            metric_tracker=eval_tracker
        )
        model.learn(..., callback=manager)
    """

    def __init__(
        self,
        market: str,
        phase: int,
        seed: int,
        config_path: str = 'config/checkpoint_config.yaml',
        metric_tracker: Optional[Any] = None,
        target_timesteps: Optional[int] = None,
        verbose: bool = True,
        registry: Optional[CheckpointRegistry] = None
    ):
        """
        Initialize DynamicCheckpointManager.

        Args:
            market: Market symbol (ES, NQ, YM, etc.)
            phase: Training phase (1, 2, or 3)
            seed: Random seed for reproducibility
            config_path: Path to checkpoint config YAML
            metric_tracker: Object exposing .best_metric and .last_metrics
            target_timesteps: Override target timesteps (defaults from config)
            verbose: Print checkpoint events to console
            registry: Optional CheckpointRegistry for global tracking
        """
        super().__init__(verbose=0)  # We handle our own logging

        self.market = market
        self.phase = phase
        self.seed = seed
        self.verbose_mode = verbose
        self.registry = registry

        # Load configuration
        self.config = self._load_config(config_path)
        self.phase_config = self.config['checkpointing'][f'phase{phase}']

        # Save intervals
        self.base_interval = self.phase_config['base_interval']
        self.max_multiplier = self.phase_config['max_interval_multiplier']
        self.target_timesteps = target_timesteps or self.phase_config['target_timesteps']

        # Event configuration
        self.event_config = self.config['checkpointing']['events']
        self.metric_threshold = self.event_config['metric_improvement_threshold']

        # Paths
        path_template = self.config['checkpointing']['paths'][f'phase{phase}']
        self.checkpoint_dir = Path(path_template.format(market=market))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Naming
        self.naming_config = self.config['checkpointing']['naming']

        # Metric tracking
        self.metric_tracker = metric_tracker
        self._best_metric = float('-inf')
        self._last_save_timestep = 0
        self._save_counter = 0
        self._disable_saves = False  # Disk-full protection
        self._awaiting_first_eval_notice = False

        # Runtime tracking
        self._training_start_time = None
        self._total_eval_episodes = 0

        if self.verbose_mode:
            self._print(f"[CHECKPOINT] Initialized for {market} Phase {phase}")
            self._print(f"[CHECKPOINT] Save dir: {self.checkpoint_dir}")
            self._print(f"[CHECKPOINT] Base interval: {self.base_interval:,} steps")

    def _init_callback(self) -> None:
        """Initialize callback at start of training."""
        self._training_start_time = time.time()

        if self.verbose_mode:
            self._print(f"[CHECKPOINT] Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load checkpoint configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _next_interval(self) -> int:
        """
        Calculate adaptive save interval based on training progress.

        Returns interval that grows from base_interval to (base_interval * max_multiplier)
        as training progresses from 0% to 100%.

        Examples:
            Progress 0%: base_interval * 1.0
            Progress 50%: base_interval * 3.0
            Progress 100%: base_interval * 5.0
        """
        progress = min(1.0, self.num_timesteps / self.target_timesteps)
        scale = 1.0 + (self.max_multiplier - 1.0) * progress
        return int(self.base_interval * scale)

    def _checkpoint_name(self, event_tag: str, metrics: Dict[str, float]) -> str:
        """
        Generate rich checkpoint filename with metrics embedded.

        Format: {market}_ts-{timesteps:07d}_evt-{event}_val-{val_reward:+.3f}_sharpe-{sharpe:+.2f}_seed-{seed}.zip

        Args:
            event_tag: Event type (periodic, best, phase_end, interrupt, etc.)
            metrics: Dictionary with val_reward, sharpe_ratio, etc.

        Returns:
            Formatted checkpoint filename (without extension)
        """
        val_reward = metrics.get('val_reward', float('nan'))
        sharpe = metrics.get('sharpe_ratio', float('nan'))

        name = self.naming_config['format'].format(
            market=self.market,
            timesteps=self.num_timesteps,
            event=event_tag,
            val_reward=val_reward,
            sharpe=sharpe,
            seed=self.seed
        )

        return name

    def _get_free_disk_space_gb(self) -> float:
        """Get free disk space in GB for checkpoint directory."""
        try:
            stat = shutil.disk_usage(self.checkpoint_dir)
            return stat.free / (1024 ** 3)  # Convert to GB
        except Exception as e:
            self._print(f"[WARNING] Could not check disk space: {e}")
            return float('inf')  # Assume space available

    def _save_checkpoint(self, event_tag: str, metrics: Dict[str, float]) -> bool:
        """
        Save checkpoint with model, VecNormalize, and metadata.

        Args:
            event_tag: Event type (periodic, best, phase_end, etc.)
            metrics: Metrics to include in filename and metadata

        Returns:
            True if save succeeded, False otherwise
        """
        # Check if saves are disabled (disk-full protection)
        if self._disable_saves:
            return False

        # Check disk space
        free_space_gb = self._get_free_disk_space_gb()
        min_space = self.config['checkpointing']['performance']['min_disk_space_gb']

        if free_space_gb < min_space:
            self._print(f"[ERROR] Low disk space ({free_space_gb:.2f} GB < {min_space:.2f} GB). Disabling checkpoint saves.")
            self._disable_saves = True
            return False

        try:
            # Generate filenames
            base_name = self._checkpoint_name(event_tag, metrics)
            model_path = self.checkpoint_dir / (base_name + self.naming_config['extension'])
            vecnorm_path = self.checkpoint_dir / (base_name + self.naming_config['vecnormalize_suffix'])
            metadata_path = self.checkpoint_dir / (base_name + self.naming_config['metadata_suffix'])

            # Save model
            if self.verbose_mode:
                self._print(f"[CHECKPOINT] Saving {event_tag} checkpoint at {self.num_timesteps:,} steps...")

            self.model.save(str(model_path))

            # Save VecNormalize if available
            if hasattr(self.training_env, 'save'):
                self.training_env.save(str(vecnorm_path))

            # Collect and save metadata
            metadata = self._collect_metadata(event_tag, metrics)
            write_checkpoint_metadata(str(model_path.with_suffix('')), metadata)

            # Register checkpoint in global registry
            if self.registry is not None:
                try:
                    self.registry.register_checkpoint(model_path, metadata)
                except Exception as e:
                    self._print(f"[WARNING] Failed to register checkpoint in registry: {e}")

            # Log to TensorBoard if enabled
            if self.config['checkpointing']['logging']['tensorboard_logging']:
                self.logger.record('checkpoint/timesteps', self.num_timesteps)
                self.logger.record('checkpoint/event', event_tag)
                self.logger.record(f'checkpoint/{event_tag}_count', self._save_counter)

                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.logger.record(f'checkpoint/{key}', value)

            self._save_counter += 1

            if self.verbose_mode:
                self._print(f"[CHECKPOINT] Γ£ô Saved: {model_path.name}")
                self._print(f"[CHECKPOINT]   Val Reward: {metrics.get('val_reward', 0.0):+.3f} | Sharpe: {metrics.get('sharpe_ratio', 0.0):+.2f}")

            return True

        except OSError as e:
            if "No space left" in str(e) or "Disk quota exceeded" in str(e):
                self._print(f"[ERROR] Disk full. Disabling checkpoint saves: {e}")
                self._disable_saves = True
            else:
                self._print(f"[ERROR] Failed to save checkpoint: {e}")
            return False
        except Exception as e:
            self._print(f"[ERROR] Unexpected error saving checkpoint: {e}")
            return False

    def _collect_metadata(self, event_tag: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Collect comprehensive metadata for checkpoint.

        Args:
            event_tag: Event type
            metrics: Current metrics dict

        Returns:
            Metadata dictionary
        """
        metadata = {
            # Standard fields
            'phase': self.phase,
            'market': self.market,
            'timesteps': self.num_timesteps,
            'seed': self.seed,
            'event_tag': event_tag,
            'timestamp': datetime.now().isoformat(),

            # Metrics fields
            'val_reward': metrics.get('val_reward', None),
            'sharpe_ratio': metrics.get('sharpe_ratio', None),
            'win_rate': metrics.get('win_rate', None),
            'max_drawdown': metrics.get('max_drawdown', None),
            'total_return': metrics.get('total_return', None),

            # Runtime fields
            'training_elapsed_seconds': time.time() - self._training_start_time if self._training_start_time else 0,
            'eval_episodes_run': self._total_eval_episodes,
            'save_counter': self._save_counter,
        }

        # Add hyperparameters if available
        if hasattr(self.model, 'learning_rate'):
            lr_value = resolve_learning_rate(getattr(self.model, 'learning_rate', None))
            if lr_value is not None:
                metadata['learning_rate'] = lr_value
        if hasattr(self.model, 'n_envs'):
            metadata['n_envs'] = self.model.n_envs

        # Phase 3 LLM-specific metrics (if available)
        if self.phase == 3:
            llm_metrics = ['reasoning_usage_rate', 'llm_confidence_avg', 'rl_llm_agreement_rate',
                          'llm_error_count', 'fusion_override_count', 'risk_veto_count']
            for metric in llm_metrics:
                if metric in metrics:
                    metadata[metric] = metrics[metric]

        return metadata

    def _on_step(self) -> bool:
        """
        Called at each training step. Handles periodic and event-driven checkpoints.

        Returns:
            True to continue training, False to stop
        """
        # Check if it's time for periodic checkpoint
        next_interval = self._next_interval()
        steps_since_save = self.num_timesteps - self._last_save_timestep

        if steps_since_save >= next_interval:
            metrics = self._get_current_metrics()
            if self._save_checkpoint('periodic', metrics):
                self._last_save_timestep = self.num_timesteps

        # Check for metric improvement (event-driven)
        if self.event_config['enabled'] and self.metric_tracker is not None:
            evals_run = getattr(self.metric_tracker, 'eval_count', 0)
            if evals_run > 0:
                current_metrics = self._get_current_metrics()
                current_val_reward = current_metrics.get('val_reward', float('-inf'))

                # Trigger save on significant improvement
                if current_val_reward > self._best_metric * (1 + self.metric_threshold):
                    if self._save_checkpoint('best', current_metrics):
                        self._best_metric = current_val_reward
            elif self.verbose_mode and not self._awaiting_first_eval_notice:
                self._print("[CHECKPOINT] Waiting for first evaluation before tracking best-model saves")
                self._awaiting_first_eval_notice = True

        return True

    def _get_current_metrics(self) -> Dict[str, float]:
        """
        Get current metrics from metric tracker.

        Returns:
            Dictionary of current metrics
        """
        if self.metric_tracker is None:
            return {}

        if hasattr(self.metric_tracker, 'last_metrics'):
            return self.metric_tracker.last_metrics.copy()

        # Fallback: minimal metrics
        return {
            'val_reward': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0
        }

    def on_phase_end(self) -> None:
        """
        Explicitly save checkpoint at phase boundary.
        Called manually from training script when phase completes.
        """
        metrics = self._get_current_metrics()
        self._save_checkpoint('phase_end', metrics)

        if self.verbose_mode:
            self._print(f"[CHECKPOINT] Phase {self.phase} completed. Final checkpoint saved.")

    def on_interrupt(self) -> None:
        """
        Save checkpoint on training interruption (Ctrl+C, exception).
        Called manually from training script exception handler.
        """
        metrics = self._get_current_metrics()
        self._save_checkpoint('interrupt', metrics)

        if self.verbose_mode:
            self._print(f"[CHECKPOINT] Training interrupted. Emergency checkpoint saved.")

    def on_phase_boundary(self, boundary_name: str) -> None:
        """
        Save checkpoint at curriculum transition points.

        Args:
            boundary_name: Description of boundary (e.g., 'instrument_switch_ES_to_NQ')
        """
        metrics = self._get_current_metrics()
        self._save_checkpoint(f'phase_boundary_{boundary_name}', metrics)

        if self.verbose_mode:
            self._print(f"[CHECKPOINT] Phase boundary '{boundary_name}' reached. Checkpoint saved.")

    def _print(self, message: str) -> None:
        """Print message if verbose mode enabled."""
        if self.verbose_mode:
            print(message)


class MetricTrackingEvalCallback:
    """
    Wrapper around EvalCallback that exposes metrics to DynamicCheckpointManager.

    Tracks:
    - Best validation reward
    - Last evaluation metrics (reward, Sharpe, win rate, etc.)
    - Evaluation episode count
    - Optional metric forecasting for trend detection

    Usage:
        forecaster = MetricForecaster(metric_name='sharpe_ratio')
        eval_tracker = MetricTrackingEvalCallback(forecaster=forecaster)
        eval_callback = EvalCallback(..., callback_after_eval=eval_tracker.update)
        checkpoint_manager = DynamicCheckpointManager(..., metric_tracker=eval_tracker)
    """

    def __init__(self, forecaster: Optional[MetricForecaster] = None):
        self.best_metric = float('-inf')
        self.last_metrics = {
            'val_reward': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0
        }
        self.eval_count = 0
        self.seed = 42  # Default seed, should be set externally
        self.forecaster = forecaster
        self.last_forecast = None  # Store last forecast result
        self.timesteps = 0  # Track timesteps for forecaster
        self.last_episode_count = 0

    def update(self, locals_dict: Dict[str, Any], globals_dict: Dict[str, Any]) -> None:
        """
        Update metrics after evaluation.

        Called by EvalCallback via callback_after_eval parameter.

        Args:
            locals_dict: Local variables from EvalCallback
            globals_dict: Global variables from EvalCallback
        """
        rewards = self._extract_episode_rewards(locals_dict)
        if rewards is None:
            print('[WARNING] Eval metrics unavailable - episode_rewards missing')
            return

        if rewards.size == 0:
            print('[WARNING] Eval metrics unavailable - empty reward array')
            return

        episode_count = rewards.size
        self.last_episode_count = episode_count
        if episode_count < 2:
            print("[EVAL] Warning: evaluation ran fewer than 2 episodes; variance may be zero")

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        sharpe = mean_reward / std_reward if std_reward > 1e-8 else 0.0
        win_rate = float(np.mean(rewards > 0))

        cumulative = np.cumsum(rewards)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        total_return = float(np.sum(rewards))

        self.last_metrics = {
            'val_reward': mean_reward,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'std_reward': std_reward
        }

        if mean_reward > self.best_metric:
            self.best_metric = mean_reward

        self.eval_count += 1

        # Overfitting Check: Compare with recent training rewards
        train_reward = 0.0
        if 'model' in locals_dict:
            model = locals_dict['model']
            if hasattr(model, 'ep_info_buffer') and len(model.ep_info_buffer) > 0:
                train_rewards = [ep_info['r'] for ep_info in model.ep_info_buffer]
                train_reward = float(np.mean(train_rewards))
                
                gap = train_reward - mean_reward
                gap_pct = (gap / abs(mean_reward) * 100) if abs(mean_reward) > 1e-6 else 0.0
                
                # Store in metrics for checkpoint metadata
                self.last_metrics['train_reward'] = train_reward
                self.last_metrics['overfitting_gap'] = gap
                
                # Print warning if significant overfitting detected
                if gap > 0 and gap_pct > 20:
                    print("[EVAL] WARNING OVERFITTING DETECTED: Train={:.2f} vs Val={:.2f} (Gap: {:.2f}, {:.1f}%)".format(
                        train_reward, mean_reward, gap, gap_pct))
                else:
                    print("[EVAL] Overfitting Check: Train={:.2f} vs Val={:.2f} (Gap: {:.2f})".format(
                        train_reward, mean_reward, gap))

        if self.forecaster is not None:
            if 'model' in locals_dict and hasattr(locals_dict['model'], 'num_timesteps'):
                self.timesteps = locals_dict['model'].num_timesteps
            else:
                self.timesteps += 1

            try:
                self.last_forecast = self.forecaster.update(sharpe, self.timesteps)
            except Exception as e:
                print(f"[WARNING] Forecaster update failed: {e}")

        print(f"[EVAL] Metrics updated: reward={mean_reward:.2f}, sharpe={sharpe:.2f}, win_rate={win_rate:.2%}, episodes={episode_count}")

    @staticmethod
    def _extract_episode_rewards(locals_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract evaluation rewards from EvalCallback locals.

        Accepts multiple key names and coerces to a 1D float64 array.
        """
        potential_keys = ('episode_rewards', 'rewards', 'eval_episode_rewards')
        rewards: Optional[np.ndarray] = None

        for key in potential_keys:
            if key in locals_dict:
                rewards = np.asarray(locals_dict[key], dtype=np.float64).reshape(-1)
                break

        if rewards is None:
            return None

        return rewards

    def set_seed(self, seed: int) -> None:
        """Set random seed for metadata."""
        self.seed = seed
class EvalMetricHook(BaseCallback):
    """
    Callback hook that updates MetricTrackingEvalCallback after each evaluation
    and optionally chains additional callbacks that must run under EvalCallback.
    """

    def __init__(
        self,
        metric_tracker: MetricTrackingEvalCallback,
        extra_callbacks: Optional[List[BaseCallback]] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.metric_tracker = metric_tracker
        self.extra_callbacks: list[BaseCallback] = [cb for cb in (extra_callbacks or []) if cb is not None]

    def _init_callback(self) -> None:
        for callback in self.extra_callbacks:
            callback.parent = self.parent
            callback.init_callback(self.model)

    def _on_training_start(self) -> None:
        for callback in self.extra_callbacks:
            callback.on_training_start(self.locals, self.globals)

    def _on_rollout_start(self) -> None:
        for callback in self.extra_callbacks:
            callback.on_rollout_start()

    def _on_rollout_end(self) -> None:
        for callback in self.extra_callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        for callback in self.extra_callbacks:
            callback.on_training_end()

    def _on_step(self) -> bool:
        continue_training = True

        for callback in self.extra_callbacks:
            callback.parent = self.parent
            continue_training = callback.on_step() and continue_training

        if self.parent is not None and hasattr(self.parent, 'evaluations_results'):
            if len(self.parent.evaluations_results) > 0:
                tracker_locals = {
                    'episode_rewards': self.parent.evaluations_results[-1],
                    'model': self.model,
                }
                self.metric_tracker.update(tracker_locals, {})

        return continue_training

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        for callback in self.extra_callbacks:
            callback.update_locals(locals_)
# Backward compatibility alias
SafeCheckpointCallback = DynamicCheckpointManager
