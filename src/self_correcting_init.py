"""
Self-Correcting System Initialization

Factory functions to initialize self-correcting components from configuration.
Ensures all components read from config/checkpoint_config.yaml instead of
hard-coded values.

Author: Javier (@javiertradess)
Version: 1.0.0
"""

from pathlib import Path
from typing import Optional, Tuple, Any
import yaml

from checkpoint_registry import CheckpointRegistry
from metric_forecaster import MetricForecaster
from policy_controller import PolicyController
from corrective_action_manager import CorrectiveActionManager
from checkpoint_manager import MetricTrackingEvalCallback


def load_config(config_path: str = 'config/checkpoint_config.yaml') -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[WARNING] Failed to load config from {config_path}: {e}")
        return {}


def init_checkpoint_registry(config_path: str = 'config/checkpoint_config.yaml') -> Optional[CheckpointRegistry]:
    """
    Initialize CheckpointRegistry from configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        CheckpointRegistry instance or None if disabled
    """
    config = load_config(config_path)
    registry_config = config.get('checkpoint_registry', {})

    if not registry_config.get('enabled', True):
        print("[SELF-CORRECT] Checkpoint registry disabled in config")
        return None

    registry_path = registry_config.get('registry_path', './models/registry.json')

    registry = CheckpointRegistry(
        registry_path=Path(registry_path),
        logger=None
    )

    print(f"[SELF-CORRECT] Checkpoint registry initialized: {registry_path}")
    return registry


def init_metric_forecaster(config_path: str = 'config/checkpoint_config.yaml') -> Optional[MetricForecaster]:
    """
    Initialize MetricForecaster from configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        MetricForecaster instance or None if disabled
    """
    config = load_config(config_path)
    forecaster_config = config.get('metric_forecaster', {})

    if not forecaster_config.get('enabled', True):
        print("[SELF-CORRECT] Metric forecaster disabled in config")
        return None

    # Extract configuration parameters
    metric_name = forecaster_config.get('metric_name', 'sharpe_ratio')
    history_size = forecaster_config.get('history_size', 50)

    # Kalman filter parameters
    kalman_config = forecaster_config.get('kalman', {})
    process_noise = kalman_config.get('process_noise', 0.01)
    measurement_noise = kalman_config.get('measurement_noise', 0.05)

    forecaster = MetricForecaster(
        metric_name=metric_name,
        history_size=history_size,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        logger=None
    )

    # Apply trend detection configuration
    trend_config = forecaster_config.get('trend_detection', {})
    if trend_config:
        forecaster.plateau_threshold = trend_config.get('plateau_threshold', 0.02)
        forecaster.divergence_threshold = trend_config.get('divergence_threshold', -0.10)
        forecaster.min_samples = trend_config.get('min_samples', 5)

    print(f"[SELF-CORRECT] Metric forecaster initialized (Kalman Filter)")
    print(f"[SELF-CORRECT]   Metric: {metric_name}, History: {history_size}")
    print(f"[SELF-CORRECT]   Process noise: {process_noise}, Measurement noise: {measurement_noise}")

    return forecaster


def init_self_correcting_system(
    market: str,
    phase: int,
    config_path: str = 'config/checkpoint_config.yaml',
    verbose: bool = True
) -> Tuple[
    Optional[CheckpointRegistry],
    Optional[MetricForecaster],
    Optional[MetricTrackingEvalCallback],
    Optional[PolicyController],
    Optional[CorrectiveActionManager]
]:
    """
    Initialize complete self-correcting system from configuration.

    Args:
        market: Market symbol (ES, NQ, etc.)
        phase: Training phase (1, 2, or 3)
        config_path: Path to configuration file
        verbose: Print initialization messages

    Returns:
        Tuple of (registry, forecaster, metric_tracker, policy_controller, corrective_manager)
        Any component may be None if disabled in config
    """
    if verbose:
        print("\n" + "="*60)
        print("SELF-CORRECTING SYSTEM INITIALIZATION")
        print("="*60)

    # Initialize registry
    registry = init_checkpoint_registry(config_path)

    # Initialize forecaster
    forecaster = init_metric_forecaster(config_path)

    # Initialize metric tracker (with forecaster if available)
    metric_tracker = MetricTrackingEvalCallback(forecaster=forecaster)

    # Initialize policy controller
    policy_controller = None
    config = load_config(config_path)
    policy_config = config.get('policy_controller', {})

    if policy_config.get('enabled', True):
        policy_controller = PolicyController(
            metric_tracker=metric_tracker,
            config_path=config_path,
            verbose=verbose,
            logger=None
        )
        if verbose:
            print("[SELF-CORRECT] Policy controller initialized")
            if policy_config.get('learning_rate', {}).get('enabled', True):
                print("[SELF-CORRECT]   LR adjustment: ENABLED")
            if policy_config.get('entropy', {}).get('enabled', True):
                print("[SELF-CORRECT]   Entropy adjustment: ENABLED")
    else:
        if verbose:
            print("[SELF-CORRECT] Policy controller disabled in config")

    # Initialize corrective action manager
    corrective_manager = None
    corrective_config = config.get('corrective_actions', {})

    if corrective_config.get('enabled', True) and registry is not None:
        corrective_manager = CorrectiveActionManager(
            metric_tracker=metric_tracker,
            registry=registry,
            market=market,
            phase=phase,
            config_path=config_path,
            policy_controller=policy_controller,
            verbose=verbose,
            logger=None
        )
        if verbose:
            print("[SELF-CORRECT] Corrective action manager initialized")

            rollback_config = corrective_config.get('levels', {}).get('rollback', {})
            if rollback_config.get('enabled', True):
                max_rollbacks = rollback_config.get('max_rollbacks', 2)
                print(f"[SELF-CORRECT]   Rollback: ENABLED (max: {max_rollbacks})")

            triggers = corrective_config.get('rollback_triggers', {})
            trigger_list = []
            if triggers.get('sharpe_drop_threshold'):
                threshold = triggers['sharpe_drop_threshold']
                trigger_list.append(f"Sharpe drop >{threshold*100:.0f}%")
            if triggers.get('nan_inf_detection'):
                trigger_list.append("NaN/Inf detection")
            if triggers.get('sustained_divergence', {}).get('enabled'):
                trigger_list.append("Sustained divergence")

            if trigger_list:
                print(f"[SELF-CORRECT]   Triggers: {', '.join(trigger_list)}")
    else:
        if verbose:
            if not corrective_config.get('enabled', True):
                print("[SELF-CORRECT] Corrective action manager disabled in config")
            elif registry is None:
                print("[SELF-CORRECT] Corrective action manager requires registry (disabled)")

    if verbose:
        print("="*60 + "\n")

    return registry, forecaster, metric_tracker, policy_controller, corrective_manager
