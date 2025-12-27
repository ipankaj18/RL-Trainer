#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: Position Management Training with Transfer Learning
- Load Phase 1 weights
- Expand action space from 3 -> 6
- Train dynamic SL/TP management
- Duration: 15M timesteps (~12-16 hours on RTX 4000 Ada 20GB)

Based on: Transfer Learning + OpenAI Spinning Up PPO
Optimized for: RunPod RTX 4000 Ada deployment
"""

import os
import wandb
from wandb.integration.sb3 import WandbCallback

from kaggle_secrets import UserSecretsClient
import wandb

secrets = UserSecretsClient()
wandb_api_key = secrets.get_secret("WANDB_API_KEY")
wandb.login(key=wandb_api_key)
# Limit math/BLAS thread pools before heavy numerical imports to avoid exhausting
# pthread limits on constrained systems.
try:
    _THREAD_LIMIT_INT = max(1, int(os.environ.get("TRAINER_MAX_BLAS_THREADS", "1")))
except ValueError:
    _THREAD_LIMIT_INT = 1
_THREAD_LIMIT_STR = str(_THREAD_LIMIT_INT)

for _env_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_env_var, _THREAD_LIMIT_STR)

import sys
import glob
import torch
import numpy as np
import pandas as pd
import yaml
from datetime import datetime

_TORCH_THREADS_OVERRIDE = os.environ.get("PYTORCH_NUM_THREADS")
try:
    if _TORCH_THREADS_OVERRIDE is not None:
        torch.set_num_threads(max(1, int(_TORCH_THREADS_OVERRIDE)))
    else:
        torch.set_num_threads(_THREAD_LIMIT_INT)
except (TypeError, ValueError):
    torch.set_num_threads(1)
from stable_baselines3 import PPO
# RL FIX #4: Import MaskablePPO for action masking support
from sb3_contrib import MaskablePPO
# ACTION MASKING FIX: Import ActionMasker wrapper to enable action masking during training
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable import utils as sb3_maskable_utils
import sb3_contrib.ppo_mask.ppo_mask as sb3_ppo_mask_module
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
from environment_phase2 import TradingEnvironmentPhase2
from kl_callback import KLDivergenceCallback
from feature_engineering import add_market_regime_features
from model_utils import detect_models_in_folder, detect_available_markets, select_market_for_training
from market_specs import get_market_spec
from metadata_utils import read_metadata, write_metadata
from action_mask_utils import (
    ActionMaskGymnasiumWrapper,
    ActionMaskVecEnvWrapper,
    get_action_masks as local_action_masks,
    phase2_mask_fn,
)
from training_mode_utils import tune_eval_schedule

# CHECKPOINT MANAGEMENT
from checkpoint_manager import DynamicCheckpointManager, MetricTrackingEvalCallback, EvalMetricHook
from checkpoint_retention import CheckpointRetentionManager

# SELF-CORRECTING SYSTEM
from self_correcting_init import init_self_correcting_system

# Set UTF-8 encoding for Windows compatibility
if os.name == 'nt':  # Windows
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
    except:
        pass

# Suppress Gymnasium deprecation warnings for action_masks
# These are cosmetic warnings from Gymnasium about deprecated API for accessing action masks
# The functionality still works correctly - this just reduces console noise
import warnings
warnings.filterwarnings('ignore', message='.*action_mask.*', category=UserWarning)


def safe_print(message=""):
    """Print with fallback for encoding errors on Windows."""
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback: replace Unicode characters with ASCII equivalents
        replacements = {
            '✓': '[OK]',
            '✅': '[OK]',
            '✗': '[X]',
            '❌': '[X]',
            '→': '->',
            '⚠': '[WARN]',
            '⚠️': '[WARN]',
            '—': '-',
            '–': '-',
            '’': "'",
            '“': '"',
            '”': '"',
        }
        ascii_message = message
        for src, target in replacements.items():
            ascii_message = ascii_message.replace(src, target)
        ascii_message = ascii_message.encode('ascii', errors='ignore').decode('ascii')
        print(ascii_message)


def get_effective_num_envs(requested_envs: int) -> int:
    """Determine a safe number of parallel environments for this host."""
    override = os.environ.get("TRAINER_NUM_ENVS")
    if override:
        try:
            value = max(1, int(override))
            safe_print(f"[CONFIG] TRAINER_NUM_ENVS override detected: {value}")
            return value
        except ValueError:
            safe_print(f"[WARN] Ignoring invalid TRAINER_NUM_ENVS='{override}'")

    cpu_count = os.cpu_count()
    if cpu_count:
        cpu_aligned = max(1, cpu_count)
        if requested_envs > cpu_aligned:
            safe_print(
                f"[SYSTEM] Reducing parallel envs to {cpu_aligned} (requested {requested_envs}, {cpu_count} CPU cores)"
            )
            return cpu_aligned

    return requested_envs


def ensure_action_space(env, expected_actions: int, label: str = "ENV") -> None:
    """
    Ensure the provided environment exposes the expected number of discrete actions.
    """
    action_space = getattr(env, "action_space", None)
    action_count = getattr(action_space, "n", None) if action_space is not None else None
    if action_count != expected_actions:
        raise RuntimeError(
            f"[{label}] Action space mismatch: expected {expected_actions}, found {action_count}"
        )
    safe_print(f"[{label}] Action space verified: {action_count} actions")


def log_action_mask_shape(env, expected_actions: int, label: str = "ENV") -> None:
    """
    Fetch action masks from a wrapped environment and verify their shape.
    """
    try:
        masks = local_action_masks(env)
    except AttributeError as err:
        raise RuntimeError(f"[{label}] Unable to retrieve action masks: {err}") from err

    masks = np.asarray(masks, dtype=bool)
    if masks.ndim == 1:
        masks = masks.reshape(1, -1)

    if masks.shape[1] != expected_actions:
        raise RuntimeError(
            f"[{label}] Action mask width mismatch: expected {expected_actions}, found {masks.shape[1]}"
        )

    safe_print(f"[{label}] Mask shape verified: {masks.shape}")


mask_fn = phase2_mask_fn


def stable_mask_fetch(env) -> np.ndarray:
    """
    Replacement for sb3's get_action_masks that guarantees a 2D boolean tensor.
    """
    raw_vec_masks = None
    raw_vec_get_masks = None
    if hasattr(env, "env_method"):
        try:
            raw_vec_masks = env.env_method("action_masks")
        except Exception:
            raw_vec_masks = None
        try:
            raw_vec_get_masks = env.env_method("get_action_mask")
        except Exception:
            raw_vec_get_masks = None

    masks = np.asarray(local_action_masks(env), dtype=bool)
    if masks.ndim == 1:
        masks = masks.reshape(1, -1)

    num_envs = getattr(env, "num_envs", masks.shape[0])
    if masks.shape[0] != num_envs and num_envs > 1:
        if masks.shape[0] == 1:
            masks = np.repeat(masks, num_envs, axis=0)
        else:
            masks = masks.reshape(num_envs, -1)

    action_space = getattr(env, "action_space", None)
    action_dim = getattr(action_space, "n", masks.shape[-1])
    if masks.shape[-1] != action_dim:
        # If mask only stores scalar validity (e.g., shape (n_envs,)), broadcast per action.
        if masks.shape[-1] == 1:
            masks = np.repeat(masks, action_dim, axis=-1)
        else:
            if raw_vec_masks is not None:
                safe_print(f"[MASK][DEBUG] env_method masks: {raw_vec_masks}")
            if raw_vec_get_masks is not None:
                safe_print(f"[MASK][DEBUG] env_method get_action_mask: {raw_vec_get_masks}")
            safe_print(
                f"[MASK][DEBUG] Raw mask shape {masks.shape} | action_dim={action_dim} | num_envs={num_envs}"
            )
            safe_print(f"[MASK][DEBUG] Mask sample: {masks}")
            raise RuntimeError(
                f"[MASK] Unexpected mask width {masks.shape[-1]} (expected {action_dim})"
            )

    return masks.astype(bool, copy=False)


# Monkey-patch sb3 to use the stable mask fetcher everywhere
sb3_maskable_utils.get_action_masks = stable_mask_fetch
sb3_ppo_mask_module.get_action_masks = stable_mask_fetch
# Local alias for logging convenience
sb3_get_action_masks = stable_mask_fetch


# Check progress bar availability
PROGRESS_BAR_AVAILABLE = False
try:
    import tqdm
    import rich
    PROGRESS_BAR_AVAILABLE = True
except ImportError:
    safe_print("[WARNING] tqdm or rich not installed - progress bar will be disabled")
    safe_print("[WARNING] Install with: pip install 'stable-baselines3[extra]'")
    safe_print("[WARNING] Training will continue without progress bar visualization")

# COMPREHENSIVE OVERFITTING FIXES + RTX 4000 Ada OPTIMIZATIONS
# STRATEGY B: Data-aware training with early stopping
PHASE2_CONFIG = {
    # Training - Data-constrained budget (22,984 unique episodes)
    # ACTION MASKING FIX: Increased to 10M for better convergence with action masking
    'total_timesteps': 10_000_000,  # 10M - extended training for proper action learning
    'num_envs': 64,  # OPTIMIZED: Sweet spot for CPU/GPU balance with SubprocVecEnv
    'action_space': 6,  # RL FIX #10: explicit 6-action configuration

    # ACTION MASKING FIX: Observation space is now 228 (was 225)
    # 220 market + 5 position + 3 validity features (can_enter, can_manage, has_position)

    # Network architecture - REDUCED to prevent overfitting
    'policy_layers': [512, 256, 128],  # DOWN from [1024,512,256,128,64,32] - less capacity = less memorization

    # PPO parameters - MULTIPLE FIXES APPLIED
    'learning_rate': 3e-4,  # FIX #7: UP from 1e-4 - match Phase 1, allow faster adaptation
    'n_steps': 2048,
    # OPTIMIZED batch size for RTX 4000 Ada
    # Calculation: 64 envs × 2048 = 131,072 samples per update
    # 131,072 / 512 = 256 minibatches (optimal balance)
    # Larger batch = better GPU utilization while maintaining generalization
    'batch_size': 512,  # INCREASED from 256 for 20GB GPU
    'n_epochs': 5,  # FIX: DOWN from 10 - reduce overfitting on same batch
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.06,  # FIXED: Tuned to keep exploration healthy across 6 discrete actions
    'vf_coef': 0.25,  # FIX: DOWN from 0.5 - reduce value function overfitting
    'max_grad_norm': 0.5,

    # Environment parameters
    'window_size': 20,
    'initial_balance': 50000,
    'initial_sl_multiplier': 1.5,
    'initial_tp_ratio': 3.0,
    'position_size': 1.0,  # Full size (Phase 1 now also uses 1 contract)
    'trailing_dd_limit': 15000,  # TRAINING: Match Phase 1 ($15K) for consistency (Apex eval uses $2,500)
    'tighten_sl_step': 0.5,
    'extend_tp_step': 1.0,
    'min_episode_bars': 600,  # Short episodes (20% of training) - reduced for variety
    'long_episode_min_bars': 2000,  # CRITICAL FIX: Long episodes (80%) - match eval horizon (327-1846 bars)
    'randomize_start_offsets': True,
    'deterministic_env_offsets': False,
    'start_offset_seed': 23,

    # Market specifications (optional override)
    'commission_override': None,  # None = use market default, or set custom value (e.g., 1.50)

    # Evaluation - More frequent for better early stopping
    'eval_freq': 50_000,  # Base frequency before auto-scaling
    'n_eval_episodes': 20,  # INCREASED from 10 for better statistics
    'eval_min_episode_bars': 2_000,
    'eval_randomize_start_offsets': True,  # FIXED: Enable randomization for diverse evaluation
    'eval_interval_updates': 4,  # Target: once every 4 PPO updates in production
    'min_eval_episodes': 8,

    # Early stopping - Aggressive to prevent overfitting with limited data
    'use_early_stopping': True,
    'early_stop_max_no_improvement': 5,  # DOWN from 10 - stop faster
    'early_stop_min_evals': 3,  # DOWN from 5 - require fewer evals

    # Device configuration
    # SWITCHED TO GPU FOR TESTING (heavy environment bottleneck detected)
    # Heavy feature engineering (33 features) + dual data sources (minute + second-level)
    # CPU environment overhead (~80-100ms) > GPU data transfer overhead (~20-30ms)
    # GPU freed CPU to handle environment simulation better
    'device': 'cuda',  # Testing GPU for heavy environment scenarios

    # Transfer learning
    'phase1_model_path': 'models/phase1_foundational_final.zip',
    'phase1_vecnorm_path': 'models/phase1_vecnorm.pkl',

    # Small-World Rewiring (Watts-Strogatz inspired)
    # Based on: Watts & Strogatz (1998) "Collective dynamics of 'small-world' networks"
    # Theory: 1-5% random rewiring creates small-world properties
    # - Preserves local clustering (Phase 1 patterns)
    # - Creates shortcuts (faster Phase 2 adaptation)
    # DISABLED: Preserve Phase 1 knowledge until Phase 2 is profitable
    'use_smallworld_rewiring': False,  # Disabled to preserve Phase 1 entry patterns
    'rewiring_probability': 0.05,  # 5% of weights rewired (optimal per Watts-Strogatz)

    # NEW: Early stopping with KL monitoring
    'target_kl': 0.01,
    'use_kl_callback': True,

    # NEW: Learning rate scheduling (matching Phase 1)
    'use_lr_schedule': True,
    'lr_final_fraction': 0.2,  # End at 20% of initial LR

    # RL FIX #7: Entropy decay schedule for exploration->exploitation
    # NOTE: MaskablePPO doesn't support entropy schedules - use fixed value instead
    'use_ent_schedule': False,  # Disabled for MaskablePPO compatibility
    'ent_coef_initial': 0.02,
    'ent_coef_final': 0.005
}

def create_lr_schedule(initial_lr, final_fraction, total_timesteps):
    """Create linear learning rate schedule."""
    def lr_schedule(progress):
        # progress goes from 0 to 1
        return initial_lr * (1 - progress * (1 - final_fraction))
    return lr_schedule


def get_learning_rate(config):
    """Get learning rate, with schedule if enabled."""
    learning_rate = config["learning_rate"]
    if config.get("use_lr_schedule", False):
        learning_rate = create_lr_schedule(
            config["learning_rate"],
            config["lr_final_fraction"],
            config["total_timesteps"]
        )
    return learning_rate


def compute_env_start_index(data_length: int, config: dict, env_id: int) -> int:
    """Helper to compute deterministic episode start when randomization disabled."""
    min_start = config.get('window_size', 20)
    min_episode_bars = max(config.get('min_episode_bars', 1500), 10)
    max_start = data_length - min_episode_bars
    if max_start <= min_start:
        return min_start

    if config.get('deterministic_env_offsets', False):
        n_envs = max(1, config.get('num_envs', 1))
        spacing = max(1, (max_start - min_start) // n_envs or 1)
        start = min_start + spacing * env_id
        return min(max(start, min_start), max_start)

    seed = config.get('start_offset_seed')
    rng = np.random.default_rng(seed + env_id if seed is not None else None)
    return int(rng.integers(min_start, max_start + 1))



def create_entropy_schedule(initial_ent, final_ent, total_timesteps):
    """
    Create linear entropy coefficient schedule.

    RL FIX #7: Entropy decay for better exploration->exploitation transition.
    Start with high exploration (0.02), end with low exploration (0.005).

    Args:
        initial_ent: Starting entropy coefficient (higher = more exploration)
        final_ent: Final entropy coefficient (lower = more exploitation)
        total_timesteps: Total training timesteps

    Returns:
        Callable schedule function
    """
    def ent_schedule(progress):
        # Linear decay from initial_ent to final_ent
        return initial_ent * (1 - progress) + final_ent * progress
    return ent_schedule


def load_hardware_profile(profile_path: str) -> dict:
    """Loads a hardware profile from a YAML file."""
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Hardware profile not found at: {profile_path}")
    safe_print(f"[CONFIG] Loading hardware profile from: {profile_path}")
    with open(profile_path, 'r') as f:
        profile = yaml.safe_load(f)
    return profile


def apply_hardware_profile(config: dict, profile_data: dict, test_mode: bool):
    """Applies hardware profile settings to the training configuration."""
    safe_print("[CONFIG] Applying hardware profile settings...")
    for key, value in profile_data.items():
        if key == 'test_mode_overrides' and test_mode:
            safe_print("[CONFIG] Applying test mode overrides from hardware profile...")
            for test_key, test_value in value.items():
                if test_key in config:
                    safe_print(f"         - Overriding {test_key}: {config[test_key]} -> {test_value}")
                    config[test_key] = test_value
                else:
                    safe_print(f"         - Adding new config {test_key}: {test_value}")
                    config[test_key] = test_value
        elif key in config:
            safe_print(f"         - Overriding {key}: {config[key]} -> {value}")
            config[key] = value
        else:
            safe_print(f"         - Adding new config {key}: {value}")
            config[key] = value
    safe_print("[CONFIG] Hardware profile applied.")


def find_data_file(market=None):
    """Find training data file with priority order.
    
    TRAIN/TEST SPLIT: Prefers *_train.csv files for training.

    Args:
        market: Market identifier (e.g., 'ES', 'NQ') or None for auto-detect

    Returns:
        Path to data file
    """
    # Get project root directory (parent of src/)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(script_dir, 'data')

    # TRAIN/TEST SPLIT: Try train-specific file first
    if market and market != 'GENERIC':
        train_file = os.path.join(data_dir, f'{market}_D1M_train.csv')
        if os.path.exists(train_file):
            safe_print(f"[DATA] Using TRAIN data (80%): {train_file}")
            return train_file
        # Fall back to full market file
        market_file = os.path.join(data_dir, f'{market}_D1M.csv')
        if os.path.exists(market_file):
            return market_file

    filenames_to_try = [
        'D1M.csv',  # New generic 1-minute data format
        'ES_D1M.csv',  # Instrument-prefixed 1-minute data
        'es_training_data_CORRECTED_CLEAN.csv',  # Legacy ES format
        'es_training_data_CORRECTED.csv',
        'databento_es_training_data_processed_cleaned.csv',
        'databento_es_training_data_processed.csv'
    ]

    for filename in filenames_to_try:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return path

    # TRAIN/TEST SPLIT: Search for train files first
    train_pattern = os.path.join(data_dir, '*_D1M_train.csv')
    train_files = sorted(glob.glob(train_pattern))
    if train_files:
        safe_print(f"[DATA] Using TRAIN data (80%): {train_files[0]}")
        return train_files[0]

    # Fall back to full data files
    pattern = os.path.join(data_dir, '*_D1M.csv')
    instrument_files = sorted(glob.glob(pattern))
    if instrument_files:
        return instrument_files[0]

    raise FileNotFoundError(
        f"Training data not found in {data_dir}. "
        f"Expected one of: {filenames_to_try} or any '*_D1M.csv' file"
    )


def _ensure_datetime_index(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Guarantee a tz-aware DatetimeIndex so downstream code can safely use .tz.
    Mirrors the Phase 1 helper to avoid AttributeError when indexes are plain Index.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    time_cols = [col for col in ("datetime", "timestamp", "ts_event") if col in df.columns]
    df_local = df.copy()

    for col in time_cols:
        parsed = pd.to_datetime(df_local[col], errors="coerce")
        if parsed.notna().any():
            safe_print(f"[DATA] Normalizing {source_label} index using '{col}' column")
            df_local[col] = parsed
            df_local = df_local.set_index(col)
            break
    else:
        parsed_index = pd.to_datetime(df_local.index, errors="coerce")
        if parsed_index.isna().all():
            raise ValueError(f"Unable to parse datetime index for {source_label}")
        safe_print(f"[DATA] Normalizing {source_label} index from existing index values")
        df_local.index = parsed_index

    if not isinstance(df_local.index, pd.DatetimeIndex):
        raise ValueError(f"Failed to convert index to datetime for {source_label}")

    return df_local


def load_data(train_split=0.7, market=None):
    """
    Load and prepare training data with proper train/val split.

    FIX #1: CRITICAL - Separate train and validation data to prevent overfitting

    Args:
        train_split: Fraction of data to use for training (default 0.7 = 70%)
        market: Market to load data for (e.g., 'ES', 'NQ', or None for auto-detect)

    Returns:
        train_data, val_data, train_second_data, val_second_data
    """
    data_path = find_data_file(market=market)
    safe_print(f"[DATA] Loading minute-level data from {data_path}")

    try:
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    except Exception:
        df_tmp = pd.read_csv(data_path)
        time_col = 'timestamp' if 'timestamp' in df_tmp.columns else 'datetime'
        df_tmp[time_col] = pd.to_datetime(df_tmp[time_col])
        data = df_tmp.set_index(time_col)

    data = _ensure_datetime_index(data, "minute-level data")

    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC').tz_convert('America/New_York')
    elif str(data.index.tz) != 'America/New_York':
        data.index = data.index.tz_convert('America/New_York')

    safe_print(f"[DATA] Loaded {len(data):,} rows")
    safe_print(f"[DATA] Full date range: {data.index.min()} to {data.index.max()}")

    # FIX #1: Chronological train/val split
    train_end_idx = int(len(data) * train_split)
    train_data = data.iloc[:train_end_idx].copy()
    val_data = data.iloc[train_end_idx:].copy()

    safe_print(f"\n[SPLIT] Train/Val Split Applied:")
    safe_print(f"[SPLIT] Train: {len(train_data):,} bars ({train_split*100:.0f}%) - {train_data.index.min()} to {train_data.index.max()}")
    safe_print(f"[SPLIT] Val:   {len(val_data):,} bars ({(1-train_split)*100:.0f}%) - {val_data.index.min()} to {val_data.index.max()}")

    # Add market regime features to BOTH splits separately (prevent leakage)
    safe_print("[DATA] Adding market regime features to train data...")
    train_data = add_market_regime_features(train_data)
    safe_print("[DATA] Adding market regime features to val data...")
    val_data = add_market_regime_features(val_data)
    safe_print(f"[DATA] Feature count: {len(train_data.columns)}")

    # Load and split second-level data
    train_second_data = None
    val_second_data = None
    # Get project root directory (parent of src/)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    second_data_candidates = []

    # Try market-specific file first, then generic
    if market and market != 'GENERIC':
        second_data_candidates.append(os.path.join(script_dir, 'data', f'{market}_D1S.csv'))

    # Add generic file
    second_data_candidates.append(os.path.join(script_dir, 'data', 'D1S.csv'))

    # Also add ES as fallback (legacy)
    second_data_candidates.append(os.path.join(script_dir, 'data', 'ES_D1S.csv'))

    # Check for wildcards if none of the above exist
    if not any(os.path.exists(path) for path in second_data_candidates):
        pattern = os.path.join(script_dir, 'data', '*_D1S.csv')
        instrument_seconds = sorted(glob.glob(pattern))
        second_data_candidates.extend(instrument_seconds)

    second_data_path = next((path for path in second_data_candidates if os.path.exists(path)), None)

    if second_data_path and os.path.exists(second_data_path):
        try:
            safe_print(f"[DATA] Loading second-level data from {second_data_path}")
            second_data = pd.read_csv(second_data_path, index_col='ts_event', parse_dates=True)
            second_data = _ensure_datetime_index(second_data, "second-level data")
            if second_data.index.tz is None:
                second_data.index = second_data.index.tz_localize('UTC').tz_convert('America/New_York')
            elif str(second_data.index.tz) != 'America/New_York':
                second_data.index = second_data.index.tz_convert('America/New_York')

            # Split second-level data by time range
            train_second_data = second_data[(second_data.index >= train_data.index[0]) &
                                           (second_data.index <= train_data.index[-1])].copy()
            val_second_data = second_data[(second_data.index >= val_data.index[0]) &
                                         (second_data.index <= val_data.index[-1])].copy()

            safe_print(f"[SPLIT] Train second-level: {len(train_second_data):,} bars")
            safe_print(f"[SPLIT] Val second-level: {len(val_second_data):,} bars")
        except Exception as e:
            safe_print(f"[DATA] Warning: Could not load second-level data: {e}")
    else:
        missing_path = os.path.join(script_dir, 'data', 'D1S.csv')
        safe_print(f"[DATA] Second-level data not found (looked for D1S/ES_D1S/_D1S.csv). Optional feature. Missing reference: {missing_path}")

    return train_data, val_data, train_second_data, val_second_data


class ActionDistributionCallback(CheckpointCallback):
    """
    ACTION MASKING FIX: Custom callback to log action distribution during training.

    Monitors which actions are being selected by the model to verify that:
    1. Invalid actions are not being selected (should be 0% with ActionMasker)
    2. HOLD action dominates when position is FLAT (should be 80-90%)
    3. Position management actions only occur when in position

    Logs to TensorBoard for visualization.
    """

    def __init__(self, save_freq, save_path, name_prefix, verbose=0):
        super().__init__(save_freq, save_path, name_prefix, verbose)
        self.action_counts = np.zeros(6, dtype=np.int64)
        self.invalid_action_count = 0
        self.total_steps = 0
        self.log_freq = 10000  # Log every 10K steps

    def _on_step(self) -> bool:
        """Called at each step. Track action selections."""
        # Get actions from training buffer if available
        if hasattr(self.locals.get('self', None), 'env'):
            env = self.locals['self'].env
            # This is called during collection, so we don't have direct access to actions
            # We'll log distribution periodically from the model's action probabilities
            pass

        self.total_steps += 1

        # Periodic logging
        if self.total_steps % self.log_freq == 0:
            self._log_action_distribution()

        # Call parent to handle checkpointing
        return super()._on_step()

    def _log_action_distribution(self):
        """Log action distribution to TensorBoard."""
        if self.total_steps > 0:
            # Calculate percentages
            total = self.action_counts.sum()
            if total > 0:
                percentages = (self.action_counts / total) * 100

                # Log to TensorBoard
                for action_idx, pct in enumerate(percentages):
                    self.logger.record(f"action_dist/action_{action_idx}_pct", pct)

                # Log invalid action rate
                invalid_pct = (self.invalid_action_count / self.total_steps) * 100 if self.total_steps > 0 else 0
                self.logger.record("action_dist/invalid_action_pct", invalid_pct)

                # Reset counters for next interval
                self.action_counts.fill(0)
                self.invalid_action_count = 0


def make_env(data, second_data, env_id, config, market_spec):
    """
    Create Phase 2 environment factory with mixed episode lengths.

    IMPROVEMENT: 80% long-form episodes (like evaluation) + 20% short randomized
    This fixes the train/eval distribution mismatch.

    Long episodes: Deterministic start, run to completion (like evaluation)
    Short episodes: Random start, standard min_episode_bars

    ACTION MASKING FIX: Ensures ActionMasker is the outermost wrapper so MaskablePPO can
    access action masks without deprecated wrapper traversal.
    """
    def _init():
        # CRITICAL FIX: 80% long-form episodes to match evaluation distribution
        is_long_episode = (env_id % 5 != 0)  # 80% long (env 0,1,2,3 long, env 4 short)

        if is_long_episode:
            # LONG EPISODE MODE (like evaluation)
            # CRITICAL FIX: Use randomized start to prevent overfitting to specific slices
            # Previous implementation used deterministic env_id * 100 offsets,
            # which meant each worker replayed the same slice forever (only 64 unique sequences)
            start_idx = None  # Let environment choose randomly
            randomize = True  # FIXED: Randomize to prevent overfitting
            min_bars = config.get('long_episode_min_bars', 500)  # Configurable, default 500-2000 range
        else:
            # SHORT EPISODE MODE (20% for robustness)
            start_idx = None
            randomize = True  # Randomized start
            min_bars = config.get('min_episode_bars', 1500)

        env = TradingEnvironmentPhase2(
            data=data,
            window_size=config['window_size'],
            initial_balance=config['initial_balance'],
            second_data=second_data,
            market_spec=market_spec,
            commission_override=config.get('commission_override', None),
            initial_sl_multiplier=config['initial_sl_multiplier'],
            initial_tp_ratio=config['initial_tp_ratio'],
            position_size_contracts=config['position_size'],
            trailing_drawdown_limit=config['trailing_dd_limit'],
            tighten_sl_step=config['tighten_sl_step'],
            extend_tp_step=config['extend_tp_step'],
            start_index=start_idx,
            randomize_start_offsets=randomize,
            min_episode_bars=min_bars
        )

        # Validation: Ensure environment has randomization support
        assert hasattr(env, 'randomize_start_offsets'), \
            "Environment missing randomization support"
        assert hasattr(env, 'min_episode_bars'), \
            "Environment missing min_episode_bars attribute"
        assert hasattr(env, '_determine_episode_start'), \
            "Environment missing _determine_episode_start method"

        # Wrap with Gymnasium-compatible action mask helper before monitoring
        env = ActionMaskGymnasiumWrapper(env)
        env = Monitor(env)

        # IMPORTANT: ActionMasker must be the outermost wrapper so SB3's
        # env.get_wrapper_attr('action_masks') finds it without hitting the
        # deprecated env.action_masks traversal.
        return ActionMasker(env, mask_fn)

    return _init


def apply_smallworld_rewiring(weight_tensor, bias_tensor, rewiring_prob, device='cuda'):
    """
    Apply Watts-Strogatz inspired rewiring to neural network weights.

    Theory (Watts & Strogatz 1998):
    - Regular lattice: High clustering, long paths
    - Random rewiring: Creates shortcuts while preserving local structure
    - Result: Small-world properties = fast propagation + pattern preservation

    For neural networks:
    - Keep (1-p)% of learned connections (clustering = Phase 1 patterns)
    - Rewire p% randomly (shortcuts = faster Phase 2 adaptation)

    Args:
        weight_tensor: Original weight matrix from Phase 1
        bias_tensor: Original bias vector from Phase 1
        rewiring_prob: Probability of rewiring each weight (0.01-0.10 typical)
        device: torch device

    Returns:
        Rewired weight tensor, original bias tensor
    """
    rewired_weight = weight_tensor.clone()

    # Create rewiring mask: True = rewire, False = keep
    rewiring_mask = torch.rand_like(rewired_weight) < rewiring_prob
    num_rewired = rewiring_mask.sum().item()
    total_weights = rewiring_mask.numel()

    # Rewire selected connections with random values
    # Use Xavier/Glorot initialization for rewired connections
    fan_in, fan_out = rewired_weight.shape[1], rewired_weight.shape[0]
    std = np.sqrt(2.0 / (fan_in + fan_out))

    # Generate random rewiring values
    random_weights = torch.randn_like(rewired_weight) * std

    # Apply rewiring: keep original where mask=False, use random where mask=True
    rewired_weight = torch.where(rewiring_mask, random_weights, weight_tensor)

    return rewired_weight, bias_tensor, num_rewired, total_weights


def load_phase1_and_transfer(config, env):
    """
    Load Phase 1 model and perform transfer learning with optional small-world rewiring.

    Strategy:
    1. Check if Phase 1 model exists
    2. Create new Phase 2 model with expanded action space (6 actions)
    3. Load Phase 1 weights into shared layers
    4. OPTIONAL: Apply small-world rewiring (Watts-Strogatz)
       - Preserves 95% of Phase 1 connections (high clustering)
       - Rewires 5% randomly (creates shortcuts)
       - Result: Faster adaptation while preventing catastrophic forgetting
    5. Initialize new action heads with small random weights
    6. Return Phase 2 model ready for training

    Returns:
        PPO model with transferred Phase 1 knowledge
    """
    phase1_path = config['phase1_model_path']

    # Auto-detect newest Phase 1 model if configured path doesn't exist
    if not os.path.exists(phase1_path):
        safe_print(f"\n[INFO] Configured Phase 1 model not found at {phase1_path}")
        safe_print("[INFO] Auto-detecting newest Phase 1 model...")

        # Detect all Phase 1 models
        phase1_models = detect_models_in_folder(phase='phase1')

        if not phase1_models:
            safe_print("[WARNING] No Phase 1 models found in models directory")
            safe_print("[WARNING] Starting Phase 2 from scratch (not recommended)")
            safe_print("[WARNING] For best results, train Phase 1 first!")
            return None

        # Use the newest model (already sorted by modification time)
        newest_model = phase1_models[0]
        phase1_path = newest_model['path']

        safe_print(f"[INFO] Found {len(phase1_models)} Phase 1 model(s)")
        safe_print(f"[INFO] Using newest: {newest_model['name']}")
        safe_print(f"[INFO] Modified: {newest_model['modified_str']}")

        if len(phase1_models) > 1:
            safe_print(f"[INFO] Other available models:")
            for model in phase1_models[1:4]:  # Show up to 3 more
                safe_print(f"       - {model['name']} ({model['modified_str']})")
            if len(phase1_models) > 4:
                safe_print(f"       ... and {len(phase1_models) - 4} more")

    safe_print(f"\n[TRANSFER] Loading Phase 1 model from {phase1_path}")

    target_market = config.get('target_market')

    # CRITICAL FIX: Check model compatibility before loading
    from pipeline.phase_guard import PhaseGuard
    compatible, compat_msg = PhaseGuard.check_model_compatibility(phase1_path, phase2_env_obs_space=228)
    if not compatible:
        raise RuntimeError(f"Phase 1 model incompatible with Phase 2: {compat_msg}")
    safe_print(f"[TRANSFER] {compat_msg}")

    try:
        # CRITICAL FIX: Load Phase 1 MaskablePPO model (not PPO)
        # Phase 1 uses MaskablePPO, so we must use MaskablePPO.load()
        # Add custom_objects to handle incompatible parameters
        try:
            phase1_model = MaskablePPO.load(
                phase1_path,
                device=config['device'],
                custom_objects={
                    'use_sde': False,  # Handle incompatible SDE args
                    'clip_range_vf': None,  # Handle optional vf clipping
                    'target_kl': None,  # Handle optional KL target
                }
            )
            safe_print("[TRANSFER] [OK] Phase 1 MaskablePPO model loaded")
        except TypeError as te:
            # If custom_objects still fails, try without it and let SB3 handle defaults
            safe_print(f"[TRANSFER] [WARN] Custom objects failed ({te}), trying default load...")
            phase1_model = MaskablePPO.load(phase1_path, device=config['device'])
            safe_print("[TRANSFER] [OK] Phase 1 MaskablePPO model loaded (default parameters)")
        except Exception as load_err:
            raise RuntimeError(
                f"Failed to load Phase 1 model from {phase1_path}: {load_err}\n"
                f"Remediation: Ensure Phase 1 was trained with MaskablePPO and saved correctly.\n"
                f"If using an old model, retrain Phase 1 with the current codebase."
            )

        phase1_meta = read_metadata(phase1_path)
        if phase1_meta:
            meta_market = phase1_meta.get('market')
            if target_market and meta_market and meta_market.upper() != target_market.upper():
                raise RuntimeError(
                    f"Phase 1 model market mismatch: expected {target_market}, found {meta_market}"
                )
            if phase1_meta.get('test_mode'):
                raise RuntimeError(
                    "Phase 1 model metadata indicates test mode; retrain Phase 1 with full timesteps."
                )
        else:
            safe_print("[TRANSFER] [WARN] Phase 1 model metadata not found; cannot verify market alignment")

        vecnorm_meta = read_metadata(config['phase1_vecnorm_path']) if os.path.exists(config['phase1_vecnorm_path']) else None
        if vecnorm_meta:
            meta_market = vecnorm_meta.get('market')
            if target_market and meta_market and meta_market.upper() != target_market.upper():
                raise RuntimeError(
                    f"Phase 1 VecNormalize market mismatch: expected {target_market}, found {meta_market}"
                )
            if vecnorm_meta.get('test_mode'):
                raise RuntimeError(
                    "Phase 1 VecNormalize metadata indicates test mode; retrain Phase 1 with full timesteps."
                )
        elif os.path.exists(config['phase1_vecnorm_path']):
            safe_print("[TRANSFER] [WARN] Phase 1 VecNormalize metadata not found; cannot verify market alignment")

        # Create new Phase 2 model (6 actions with action masking)
        safe_print("[TRANSFER] Creating Phase 2 model with expanded action space (6 actions + masking)...")
        safe_print("[TRANSFER] RL FIX #4: Using MaskablePPO for efficient exploration")

        learning_rate = get_learning_rate(config)

        # RL FIX #7: Use fixed entropy coefficient (MaskablePPO doesn't support schedules)
        ent_coef = 0.06  # FIXED: Keeps exploration balanced across the 6-action head

        # RL FIX #4: Use MaskablePPO instead of standard PPO
        phase2_model = MaskablePPO(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_range=config['clip_range'],
            ent_coef=ent_coef,  # Use schedule instead of fixed value
            vf_coef=config['vf_coef'],
            max_grad_norm=config['max_grad_norm'],
            policy_kwargs={
                'net_arch': dict(
                    pi=config['policy_layers'],
                    vf=config['policy_layers']
                ),
                'activation_fn': torch.nn.ReLU
            },
            device=config['device'],
            verbose=1,
            tensorboard_log='./tensorboard_logs/phase2/'
        )

        # Transfer weights from Phase 1 to Phase 2
        use_rewiring = config.get('use_smallworld_rewiring', False)
        rewiring_prob = config.get('rewiring_probability', 0.05)

        if use_rewiring:
            safe_print("[TRANSFER] Transferring Phase 1 knowledge with SMALL-WORLD REWIRING...")
            safe_print(f"[TRANSFER] Rewiring probability: {rewiring_prob:.1%} (Watts-Strogatz model)")
            safe_print(f"[TRANSFER] This preserves {(1-rewiring_prob)*100:.1f}% of learned patterns")
        else:
            safe_print("[TRANSFER] Transferring Phase 1 knowledge (standard copy)...")

        total_rewired = 0
        total_weights_transferred = 0

        with torch.no_grad():
            # Get extractors
            phase1_extractor = phase1_model.policy.mlp_extractor
            phase2_extractor = phase2_model.policy.mlp_extractor

            # Transfer policy network weights (shared layers)
            try:
                for i, (p1_layer, p2_layer) in enumerate(zip(
                    phase1_extractor.policy_net,
                    phase2_extractor.policy_net
                )):
                    if hasattr(p1_layer, 'weight'):
                        if p1_layer.weight.shape == p2_layer.weight.shape:
                            if use_rewiring:
                                # Apply small-world rewiring
                                rewired_weight, bias, num_rewired, total_w = apply_smallworld_rewiring(
                                    p1_layer.weight,
                                    p1_layer.bias,
                                    rewiring_prob,
                                    device=config['device']
                                )
                                p2_layer.weight.copy_(rewired_weight)
                                p2_layer.bias.copy_(bias)
                                total_rewired += num_rewired
                                total_weights_transferred += total_w
                                safe_print(f"  [REWIRE] Policy layer {i}: {p1_layer.weight.shape} "
                                      f"({num_rewired:,}/{total_w:,} rewired = {num_rewired/total_w*100:.2f}%)")
                            else:
                                # Standard transfer (copy all)
                                p2_layer.weight.copy_(p1_layer.weight)
                                p2_layer.bias.copy_(p1_layer.bias)
                                safe_print(f"  [OK] Transferred policy layer {i}: {p1_layer.weight.shape}")
            except Exception as e:
                safe_print(f"  [!] Warning: Could not transfer all policy layers: {e}")

            # Transfer value network weights (shared layers)
            try:
                for i, (p1_layer, p2_layer) in enumerate(zip(
                    phase1_extractor.value_net,
                    phase2_extractor.value_net
                )):
                    if hasattr(p1_layer, 'weight'):
                        if p1_layer.weight.shape == p2_layer.weight.shape:
                            if use_rewiring:
                                # Apply small-world rewiring
                                rewired_weight, bias, num_rewired, total_w = apply_smallworld_rewiring(
                                    p1_layer.weight,
                                    p1_layer.bias,
                                    rewiring_prob,
                                    device=config['device']
                                )
                                p2_layer.weight.copy_(rewired_weight)
                                p2_layer.bias.copy_(bias)
                                total_rewired += num_rewired
                                total_weights_transferred += total_w
                                safe_print(f"  [REWIRE] Value layer {i}: {p1_layer.weight.shape} "
                                      f"({num_rewired:,}/{total_w:,} rewired = {num_rewired/total_w*100:.2f}%)")
                            else:
                                # Standard transfer (copy all)
                                p2_layer.weight.copy_(p1_layer.weight)
                                p2_layer.bias.copy_(p1_layer.bias)
                                safe_print(f"  [OK] Transferred value layer {i}: {p1_layer.weight.shape}")
            except Exception as e:
                safe_print(f"  [!] Warning: Could not transfer all value layers: {e}")

            # Transfer action head weights (partial transfer for shared actions)
            # Phase 1: 3 actions (Hold, Buy, Sell)
            # Phase 2: 6 actions (Hold, Buy, Sell, MoveBE, TrailOn, TrailOff)
            # We transfer weights for indices 0, 1, 2 to preserve entry policy
            try:
                p1_action_net = phase1_model.policy.action_net
                p2_action_net = phase2_model.policy.action_net
                
                # Indices for common actions: 0=Hold, 1=Buy, 2=Sell
                common_indices = [0, 1, 2]
                
                if use_rewiring:
                    # Apply small-world rewiring to action head
                    # We process the subset of weights for common actions
                    subset_weights = p1_action_net.weight[common_indices, :]
                    subset_bias = p1_action_net.bias[common_indices]
                    
                    rewired_weight, bias, num_rewired, total_w = apply_smallworld_rewiring(
                        subset_weights,
                        subset_bias,
                        rewiring_prob,
                        device=config['device']
                    )
                    
                    p2_action_net.weight[common_indices, :] = rewired_weight
                    p2_action_net.bias[common_indices] = bias
                    
                    total_rewired += num_rewired
                    total_weights_transferred += total_w
                    safe_print(f"  [REWIRE] Action head (partial): {subset_weights.shape} "
                          f"({num_rewired:,}/{total_w:,} rewired = {num_rewired/total_w*100:.2f}%)")
                else:
                    # Standard transfer (copy subset)
                    p2_action_net.weight[common_indices, :] = p1_action_net.weight[common_indices, :]
                    p2_action_net.bias[common_indices] = p1_action_net.bias[common_indices]
                    safe_print(f"  [OK] Transferred action head (indices 0-2): {p1_action_net.weight.shape[1]} inputs")
                    
            except Exception as e:
                safe_print(f"  [!] Warning: Could not transfer action head: {e}")

        safe_print("[TRANSFER] [OK] Transfer learning complete!")
        if use_rewiring:
            safe_print(f"[TRANSFER] Small-world rewiring applied: {total_rewired:,}/{total_weights_transferred:,} "
                  f"weights rewired ({total_rewired/total_weights_transferred*100:.2f}%)")
            safe_print("[TRANSFER] Phase 1 patterns preserved in high-clustering regions")
            safe_print("[TRANSFER] Random shortcuts created for faster Phase 2 adaptation")
        else:
            safe_print("[TRANSFER] Phase 1 knowledge (entry patterns) preserved")
        
        # Initialize new action heads (indices 3, 4, 5) with negative bias
        # This encourages the agent to stick to Phase 1 actions (0-2) initially
        try:
            p2_action_net = phase2_model.policy.action_net
            new_action_indices = [3, 4, 5]
            
            # Set weights to small random values (already done by default init, but good to be safe)
            # Set biases to large negative value to suppress these actions initially
            with torch.no_grad():
                p2_action_net.bias[new_action_indices] = -5.0
                safe_print(f"[TRANSFER] New actions (3-5) suppressed with bias=-5.0 (gradual exploration)")
        except Exception as e:
            safe_print(f"[TRANSFER] [WARN] Could not initialize new action biases: {e}")

        safe_print("[TRANSFER] New actions (3-5) initialized for learning")

        return phase2_model

    except Exception as e:
        safe_print(f"\n[ERROR] Transfer learning failed: {e}")
        safe_print("[ERROR] Starting Phase 2 from scratch...")
        return None


def train_phase2(market_override=None, non_interactive=False, test_mode=False, hardware_profile=None):
    """Execute Phase 2 training with transfer learning.

    Args:
        market_override: Optional market symbol to use (ES, NQ, etc.). If None, will prompt interactively.
        non_interactive: If True, run in non-interactive mode (no prompts, use defaults)
        test_mode: If True, skip PhaseGuard for quick testing
    """
    # Import PhaseGuard at function level (used multiple times)
    from pipeline.phase_guard import PhaseGuard, print_gate_banner

    # Apply hardware profile if provided
    if hardware_profile:
        profile_data = load_hardware_profile(hardware_profile)
        apply_hardware_profile(PHASE2_CONFIG, profile_data, test_mode)

    # PHASE GUARD: Validate Phase 1 completion before proceeding
    if not test_mode:  # Skip gate in test mode for quick iteration

        safe_print("\n" + "=" * 80)
        safe_print("PHASE GUARD: Validating Phase 1 Completion")
        safe_print("=" * 80 + "\n")

        # CRITICAL FIX: Detect Phase 1 model path for metadata-based test detection
        phase1_model_path = PHASE2_CONFIG['phase1_model_path']
        if not os.path.exists(phase1_model_path):
            # Auto-detect newest Phase 1 model
            from model_utils import detect_models_in_folder
            phase1_models = detect_models_in_folder(phase='phase1')
            if phase1_models:
                phase1_model_path = phase1_models[0]['path']
                safe_print(f"[GUARD] Auto-detected Phase 1 model: {phase1_models[0]['name']}")
            else:
                phase1_model_path = None
                safe_print("[GUARD] WARNING: No Phase 1 model found for metadata check")

        # Pass model_path so metadata-based test detection runs
        passed, message, metrics = PhaseGuard.validate_phase1(model_path=phase1_model_path)
        safe_print(message)

        if not passed:
            print_gate_banner(False, "Phase 1")
            safe_print("\n⛔ PHASE 2 TRAINING BLOCKED\n")
            safe_print("Recommendations:")
            safe_print("  1. Retrain Phase 1 with increased timesteps (10M)")
            safe_print("  2. Ensure mean_reward > 0.0 in evaluation")
            safe_print("  3. Check logs/phase1/evaluations.npz for details")
            safe_print("\nTo override (NOT recommended):")
            safe_print("  Edit train_phase2.py and set override=True in PhaseGuard.validate_phase1()")

            # Log gate decision
            PhaseGuard.log_gate_decision('phase1', False, message, metrics)
            return

        print_gate_banner(True, "Phase 1")
        PhaseGuard.log_gate_decision('phase1', True, message, metrics)
        safe_print("\n✅ Proceeding to Phase 2 training...\n")

    safe_print("=" * 80)
    safe_print("PHASE 2: POSITION MANAGEMENT MASTERY")
    safe_print("=" * 80)

    tune_eval_schedule(
        PHASE2_CONFIG,
        test_mode=test_mode,
        label="Phase 2",
        eval_updates=PHASE2_CONFIG.get('eval_interval_updates', 4),
        min_eval_episodes=PHASE2_CONFIG.get('min_eval_episodes', 8),
        printer=safe_print,
    )

    market_name = "NQ"  # Example market name; replace as needed
    # =========================
    # Weights & Biases Init
    # =========================
    wandb_run = wandb.init(
        project="rl-trading-phase2",
        group=f"phase2-{market_name}",
        config={
            **PHASE2_CONFIG,
            "phase": 2,
            "market": market_name,
            "algo": "MaskablePPO",
            "action_masking": True,
            "num_envs": 64,
        },
        sync_tensorboard=True,   # IMPORTANT
        monitor_gym=True,        # Captures episode rewards
        save_code=True,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=0,   # Avoid heavy overhead
        model_save_path=None,   # You already handle checkpoints
        verbose=0,
    )

    # Detect and select market
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(script_dir, 'data')
    available_markets = detect_available_markets(data_dir)

    # If market override provided via CLI, use it directly
    if market_override:
        from market_specs import get_market_spec
        market_spec = get_market_spec(market_override.upper())
        if market_spec is None:
            safe_print(f"\n[ERROR] Invalid market symbol: {market_override}")
            safe_print("[ERROR] Valid markets: ES, NQ, YM, RTY, MNQ, MES, M2K, MYM")
            return
        # Find the matching market data
        selected_market = next((m for m in available_markets if m['market'] == market_override.upper()), None)
        if selected_market is None:
            safe_print(f"\n[ERROR] No data found for market: {market_override}")
            safe_print(f"[ERROR] Available markets: {', '.join([m['market'] for m in available_markets])}")
            return
        market_name = market_override.upper()
        safe_print(f"\n[CLI] Market specified via --market: {market_name}")
    else:
        # Interactive market selection
        selected_market, market_spec = select_market_for_training(available_markets, safe_print)
        if selected_market is None or market_spec is None:
            safe_print("\n[INFO] Training cancelled - no market selected")
            return  # User cancelled or no data
        market_name = selected_market['market']

    safe_print(f"\n[TRAINING] Market: {market_name}")
    PHASE2_CONFIG['target_market'] = market_name

    safe_print(f"[CONFIG] Total timesteps: {PHASE2_CONFIG['total_timesteps']:,}")
    requested_envs = PHASE2_CONFIG['num_envs']
    num_envs = get_effective_num_envs(requested_envs)
    safe_print(f"[CONFIG] Parallel envs (requested): {requested_envs}")
    if num_envs != requested_envs:
        safe_print(f"[CONFIG] Parallel envs (effective): {num_envs} (adjusted for host limits)")
    else:
        safe_print(f"[CONFIG] Parallel envs (effective): {num_envs}")
    safe_print(f"[SYSTEM] BLAS threads per process: {os.environ.get('OPENBLAS_NUM_THREADS', 'unknown')}")
    safe_print(f"[CONFIG] Network: {PHASE2_CONFIG['policy_layers']}")
    safe_print(
        f"[CONFIG] Action space: {PHASE2_CONFIG['action_space']} (RL Fix #10: simplified from 9 to 6)"
    )
    safe_print(f"[CONFIG] Device: {PHASE2_CONFIG['device']}")
    safe_print(f"[CONFIG] Position size: {PHASE2_CONFIG['position_size']} contracts (full)")
    safe_print(f"[CONFIG] Trailing DD: ${PHASE2_CONFIG['trailing_dd_limit']:,} training (eval: $2,500 Apex)")
    safe_print()

    # Create directories
    # FIX: Use market-specific directory
    model_dir = f'models/phase2_{market_name.lower()}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f'{model_dir}/checkpoints', exist_ok=True)
    os.makedirs('logs/phase2', exist_ok=True)
    os.makedirs('tensorboard_logs/phase2', exist_ok=True)

    # Load data with train/val split - FIX #1
    train_data, val_data, train_second_data, val_second_data = load_data(train_split=0.7, market=market_name)

    # Create vectorized training environments
    safe_print(f"[ENV] Creating {num_envs} parallel training environments...")

    # Phase 2 uses SubprocVecEnv for parallel processing across CPU cores
    # This dramatically improves throughput by running environments in parallel
    # (DummyVecEnv ran sequentially, causing 90%+ CPU idle time)
    env_factories = [
        make_env(
            train_data,
            train_second_data,
            i,
            PHASE2_CONFIG,
            market_spec
        )
        for i in range(num_envs)
    ]

    env = SubprocVecEnv(env_factories, start_method='forkserver')
    safe_print(f"[ENV] Using SubprocVecEnv ({num_envs} environments, multi-process)")
    safe_print(f"[ENV] Rationale: Parallel processing maximizes CPU utilization for heavy feature engineering")

    # Wrap with action mask support
    env = ActionMaskVecEnvWrapper(env)
    safe_print("[ENV] Action masking enabled for 6-action space")

    # Wrap with VecNormalize
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    safe_print("[ENV] Training environments created with VecNormalize")

    safe_print("[VALIDATION] Verifying training environment action space and mask shapes...")
    ensure_action_space(env, PHASE2_CONFIG['action_space'], label="TRAIN")
    log_action_mask_shape(env, PHASE2_CONFIG['action_space'], label="TRAIN")
    try:
        sb3_masks = sb3_get_action_masks(env)
        safe_print(f"[TRAIN] SB3 mask tensor shape: {sb3_masks.shape}")
    except Exception as mask_err:
        safe_print(f"[TRAIN][ERROR] Unable to fetch SB3 masks: {mask_err}")

    # Create VALIDATION environment (use VAL data - CRITICAL FIX)
    safe_print("[EVAL] Creating VALIDATION environment (unseen data)...")
    def _make_eval_env():
        base_env = TradingEnvironmentPhase2(
            data=val_data,  # CHANGED: Use validation data
            window_size=PHASE2_CONFIG['window_size'],
            initial_balance=PHASE2_CONFIG['initial_balance'],
            second_data=val_second_data,  # CHANGED: Use val second-level data
            market_spec=market_spec,  # NEW: Pass market spec
            commission_override=PHASE2_CONFIG.get('commission_override', None),  # NEW
            initial_sl_multiplier=PHASE2_CONFIG['initial_sl_multiplier'],
            initial_tp_ratio=PHASE2_CONFIG['initial_tp_ratio'],
            position_size_contracts=PHASE2_CONFIG['position_size'],
            trailing_drawdown_limit=15000,  # EVAL: RELAXED to $15K (match training) to allow learning
            start_index=PHASE2_CONFIG['window_size'],
            randomize_start_offsets=PHASE2_CONFIG.get('eval_randomize_start_offsets', False),
            min_episode_bars=PHASE2_CONFIG.get(
                'eval_min_episode_bars',
                PHASE2_CONFIG.get('min_episode_bars', 1500),
            )
        )
        wrapped_env = ActionMaskGymnasiumWrapper(base_env)
        monitored_env = Monitor(wrapped_env)
        return ActionMasker(monitored_env, mask_fn)

    eval_env = DummyVecEnv([_make_eval_env])
    # Order matters: apply action-mask wrapper before VecNormalize so the top-level
    # env remains VecNormalize for normalization syncing with the training env.
    eval_env = ActionMaskVecEnvWrapper(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    ensure_action_space(eval_env, PHASE2_CONFIG['action_space'], label="EVAL")
    log_action_mask_shape(eval_env, PHASE2_CONFIG['action_space'], label="EVAL")
    try:
        eval_sb3_masks = sb3_get_action_masks(eval_env)
        safe_print(f"[EVAL] SB3 mask tensor shape: {eval_sb3_masks.shape}")
    except Exception as mask_err:
        safe_print(f"[EVAL][ERROR] Unable to fetch SB3 masks: {mask_err}")
    # Transfer learning from Phase 1
    model = load_phase1_and_transfer(PHASE2_CONFIG, env)

    if model is None:
        learning_rate = get_learning_rate(PHASE2_CONFIG)
        safe_print("\n[MODEL] Creating Phase 2 model from scratch...")
        safe_print("[MODEL] RL FIX #4: Using MaskablePPO for efficient exploration")

        # RL FIX #7: Use fixed entropy coefficient (MaskablePPO doesn't support schedules)
        ent_coef = 0.06  # FIXED: Keeps exploration balanced across the 6-action head

        # RL FIX #4: Use MaskablePPO instead of standard PPO
        model = MaskablePPO(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            n_steps=PHASE2_CONFIG['n_steps'],
            batch_size=PHASE2_CONFIG['batch_size'],
            n_epochs=PHASE2_CONFIG['n_epochs'],
            gamma=PHASE2_CONFIG['gamma'],
            gae_lambda=PHASE2_CONFIG['gae_lambda'],
            clip_range=PHASE2_CONFIG['clip_range'],
            ent_coef=ent_coef,  # Use schedule
            vf_coef=PHASE2_CONFIG['vf_coef'],
            max_grad_norm=PHASE2_CONFIG['max_grad_norm'],
            policy_kwargs={
                'net_arch': dict(
                    pi=PHASE2_CONFIG['policy_layers'],
                    vf=PHASE2_CONFIG['policy_layers']
                ),
                'activation_fn': torch.nn.ReLU
            },
            device=PHASE2_CONFIG['device'],
            verbose=1,
            tensorboard_log='./tensorboard_logs/phase2/'
        )

    # Callbacks
    # STRATEGY B: Aggressive early stopping to prevent overfitting with limited data
    if PHASE2_CONFIG.get('use_early_stopping', False):
        early_stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=PHASE2_CONFIG['early_stop_max_no_improvement'],
            min_evals=PHASE2_CONFIG['early_stop_min_evals'],
            verbose=1
        )
        safe_print(f"\n[TRAIN] Early stopping enabled:")
        safe_print(f"        - Stop after {PHASE2_CONFIG['early_stop_max_no_improvement']} evals with no improvement")
        safe_print(f"        - Minimum {PHASE2_CONFIG['early_stop_min_evals']} evals required")
        eval_cadence_steps = PHASE2_CONFIG['eval_freq'] * num_envs
        safe_print(
            f"        - Evaluation every {eval_cadence_steps:,} timesteps "
            f"(~{PHASE2_CONFIG['eval_freq']:,} env steps across {num_envs} envs)"
        )
        min_stop_evals = PHASE2_CONFIG['early_stop_min_evals'] + PHASE2_CONFIG['early_stop_max_no_improvement']
        safe_print(f"        - Could stop as early as: {min_stop_evals * eval_cadence_steps:,} timesteps")
    else:
        early_stop_callback = None
        safe_print("[TRAIN] Early stopping disabled")

    # IMPROVEMENT: Add timestamp to evaluation logs to prevent overwriting
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_log_path = f'./logs/phase2/eval_{eval_timestamp}'
    os.makedirs(eval_log_path, exist_ok=True)

    safe_print(f"[EVAL] Evaluation logs will be saved to: {eval_log_path}")

    # Initialize self-correcting system from configuration
    registry, forecaster, metric_tracker, policy_controller, corrective_manager = init_self_correcting_system(
        market=market_name,
        phase=2,
        config_path='config/checkpoint_config.yaml',
        verbose=True
    )

    # Set seed for metric tracker
    metric_tracker.set_seed(42)  # Match training seed

    eval_metric_hook = EvalMetricHook(metric_tracker)

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=early_stop_callback,  # Proper placement for early stopping
        callback_after_eval=eval_metric_hook,      # Must be BaseCallback for EventCallback API
        eval_freq=PHASE2_CONFIG['eval_freq'],
        n_eval_episodes=PHASE2_CONFIG['n_eval_episodes'],
        best_model_save_path=f'./models/phase2_{market_name.lower()}/',
        log_path=eval_log_path,  # IMPROVEMENT: Versioned log path
        deterministic=True,
        render=False,
        verbose=1
    )

    # CRITICAL FIX: Link hook to parent so it can access evaluation results
    # Without this link, checkpoint_manager.py:600-606 cannot access evaluations_results
    # and evaluations.npz will NOT be created, blocking Phase 2 training
    eval_metric_hook.parent = eval_callback
    assert eval_metric_hook.parent is not None, "CRITICAL: eval_metric_hook.parent must be set!"
    assert hasattr(eval_metric_hook.parent, 'evaluations_results'), \
        "CRITICAL: parent must have evaluations_results attribute!"

    # IMPROVED: Dynamic checkpoint manager with adaptive intervals and event-driven saves
    checkpoint_manager = DynamicCheckpointManager(
        market=market_name,
        phase=2,
        seed=42,
        config_path='config/checkpoint_config.yaml',
        metric_tracker=metric_tracker,
        target_timesteps=PHASE2_CONFIG['total_timesteps'],
        verbose=False,
        registry=registry  # Connect to global registry
    )
    safe_print(f"[CHECKPOINT] Dynamic checkpoint manager initialized")
    safe_print(f"[CHECKPOINT] Base interval: 50K steps (adaptive)")
    safe_print(f"[CHECKPOINT] Event triggers: periodic, best, phase_end, interrupt")

    # Build callbacks list (order matters: eval -> checkpoint -> controller -> corrective)
    callbacks = [eval_callback, checkpoint_manager]

    # Add optional self-correcting callbacks (if enabled in config)
    if policy_controller is not None:
        callbacks.append(policy_controller)
    if corrective_manager is not None:
        callbacks.append(corrective_manager)
    
    if PHASE2_CONFIG.get('use_kl_callback', False):
        kl_callback = KLDivergenceCallback(
            target_kl=PHASE2_CONFIG['target_kl'],
            verbose=1,
            log_freq=1000
        )
        callbacks.append(kl_callback)
        safe_print("[TRAIN] KL divergence monitoring enabled")

    # Train Phase 2
    safe_print("\n" + "=" * 80)
    safe_print(f"[TRAIN] Starting Phase 2 training for {PHASE2_CONFIG['total_timesteps']:,} timesteps...")
    safe_print("=" * 80)
    safe_print("\n[TRAIN] Position management actions (RL Fix #10 - Simplified):")
    safe_print("        Action 3: Move SL to break-even (risk-free)")
    safe_print("        Action 4: Enable trailing stop (explicit enable)")
    safe_print("        Action 5: Disable trailing stop (explicit disable)")
    safe_print("\n[TRAIN] Monitor progress:")
    safe_print("        - TensorBoard: tensorboard --logdir tensorboard_logs/phase2/")
    safe_print("        - Logs: logs/phase2/evaluations.npz")
    safe_print()

    import time
    start_time = time.time()

    # Use progress bar if available, otherwise disable
    use_progress_bar = PROGRESS_BAR_AVAILABLE
    if use_progress_bar:
        safe_print("[TRAIN] Progress bar enabled (tqdm + rich available)")
    else:
        safe_print("[TRAIN] Progress bar disabled (missing tqdm/rich)")
        safe_print("[TRAIN] Monitor via TensorBoard: tensorboard --logdir tensorboard_logs/phase2/")

    try:
        model.learn(
            total_timesteps=PHASE2_CONFIG['total_timesteps'],
            callback=callbacks,
            progress_bar=use_progress_bar
        )
        # Save phase_end checkpoint after successful training
        checkpoint_manager.on_phase_end()
    except KeyboardInterrupt:
        safe_print("\n[TRAIN] Training interrupted by user")
        safe_print("[TRAIN] Saving interrupt checkpoint...")
        checkpoint_manager.on_interrupt()
    except Exception as e:
        safe_print(f"\n[ERROR] Training failed: {e}")
        # Save interrupt checkpoint on exception
        try:
            checkpoint_manager.on_interrupt()
        except Exception:
            pass
        raise

    elapsed = time.time() - start_time

    # Run checkpoint retention cleanup
    safe_print("\n[CHECKPOINT] Running checkpoint retention cleanup...")
    try:
        retention_manager = CheckpointRetentionManager('config/checkpoint_config.yaml')
        checkpoint_dir = f'./models/phase2/{market_name}/checkpoints/'
        retention_manager.prune_checkpoints(checkpoint_dir, dry_run=False, verbose=False)
        safe_print("[CHECKPOINT] Retention cleanup completed")
    except Exception as e:
        safe_print(f"[WARNING] Checkpoint retention cleanup failed: {e}")

    # Save final model
    safe_print("\n[SAVE] Saving final Phase 2 model...")
    final_model_base = 'models/phase2_position_mgmt_final'
    final_vecnorm_path = 'models/phase2_position_mgmt_final_vecnorm.pkl'

    if test_mode:
        final_model_base = 'models/phase2_position_mgmt_test'
        final_vecnorm_path = 'models/phase2_position_mgmt_test_vecnorm.pkl'
        safe_print("[SAVE] Test mode run detected - using test-specific artifact names")

    model.save(final_model_base)
    env.save(final_vecnorm_path)

    metadata_common = {
        'phase': 2,
        'market': market_name,
        'total_timesteps': int(model.num_timesteps),
        'timesteps_target': int(PHASE2_CONFIG['total_timesteps']),
        'test_mode': bool(test_mode),
    }

    write_metadata(final_model_base, {**metadata_common, 'artifact': 'model'})
    write_metadata(final_vecnorm_path, {**metadata_common, 'artifact': 'vecnormalize'})

    # CRITICAL FIX: Create legacy symlink for backward compatibility
    # This allows PhaseGuard and other tools to find evaluation results at the legacy path
    legacy_path = PhaseGuard.create_legacy_symlink('logs/phase2', use_copy=False)
    if legacy_path:
        safe_print(f"[COMPAT] Created legacy evaluation link: {legacy_path}")

    safe_print("\n" + "=" * 80)
    safe_print("PHASE 2 TRAINING COMPLETE!")
    safe_print("=" * 80)
    safe_print(f"[RESULTS] Training time: {elapsed/3600:.2f} hours")
    safe_print(f"[SAVE] Model: {final_model_base}.zip")
    safe_print(f"[SAVE] VecNorm: {final_vecnorm_path}")
    safe_print(f"[SAVE] Best model: models/phase2/best_model.zip")
    safe_print()
    safe_print("[SUCCESS] Phase 2 training complete!")
    safe_print()
    safe_print("[NEXT STEPS]")
    safe_print("  1. Evaluate Phase 2: python3 evaluate_phase2.py")
    safe_print("  2. Train Phase 3 (Hybrid RL + LLM): python3 src/train_phase3_llm.py")
    safe_print("  3. Compare TensorBoard: tensorboard --logdir tensorboard_logs/")
    safe_print("  4. Review position management usage in logs")
    safe_print()


if __name__ == '__main__':
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Phase 2 Training')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with reduced timesteps (50K for quick local testing)')
    parser.add_argument('--market', type=str, default=None,
                       help='Market to train on (ES, NQ, YM, RTY, MNQ, MES, M2K, MYM). If not specified, will prompt interactively.')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Run in non-interactive mode (no prompts, use defaults)')
    parser.add_argument('--hardware-profile', type=str, help='Path to hardware profile yaml to apply')
    args = parser.parse_args()

    # Override config for test mode (quick local testing)
    if args.test:
        safe_print("\n" + "=" * 80)
        safe_print("TEST MODE ENABLED - Quick Local Testing")
        safe_print("=" * 80)
        PHASE2_CONFIG['total_timesteps'] = 50_000  # Small test run
        PHASE2_CONFIG['num_envs'] = 32  # OPTIMIZED: Better hardware utilization with SubprocVecEnv
        PHASE2_CONFIG['eval_freq'] = 50_000  # Single evaluation near the end
        PHASE2_CONFIG['n_eval_episodes'] = 20  # INCREASED from 1 for better statistics
        PHASE2_CONFIG['eval_min_episode_bars'] = 800
        PHASE2_CONFIG['eval_randomize_start_offsets'] = True  # Short randomized evals

        # Disable early stopping in test mode (run full 50K)
        PHASE2_CONFIG['use_early_stopping'] = False

        safe_print(f"[TEST] Timesteps:       5M -> {PHASE2_CONFIG['total_timesteps']:,} (1% for testing)")
        safe_print(f"[TEST] Parallel envs:   80 -> {PHASE2_CONFIG['num_envs']} (optimized for parallel processing)")
        safe_print(f"[TEST] Eval frequency:  Every {PHASE2_CONFIG['eval_freq']:,} steps")
        safe_print(f"[TEST] Eval episodes:   1 -> {PHASE2_CONFIG['n_eval_episodes']} (diverse evaluation)")
        safe_print(f"[TEST] Early stopping:  DISABLED (test mode)")
        safe_print(f"[TEST] Expected time:   ~2-3 minutes (30x faster with SubprocVecEnv)")
        safe_print(f"[TEST] Purpose:         Verify pipeline works before full training")
        safe_print("=" * 80 + "\n")

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    train_phase2(
        market_override=args.market,
        non_interactive=args.non_interactive,
        test_mode=args.test,
        hardware_profile=args.hardware_profile
    )
