#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: Foundational Trading Patterns Training

Optimized training approach with:
1. Simplified reward function focusing on entry quality
2. Relaxed constraints (trailing DD, daily loss limit, profit target)
3. Simplified observation space (removed complex features)
4. Better reward magnitudes for stronger learning signal
5. Enhanced exploration parameters

Designed to learn quality entry signals before advancing to Phase 2 position management.
"""

import os
import wandb
from wandb.integration.sb3 import WandbCallback

from kaggle_secrets import UserSecretsClient
import wandb

secrets = UserSecretsClient()
wandb_api_key = secrets.get_secret("WANDB_API_KEY")
wandb.login(key=wandb_api_key)
# Limit math/BLAS thread pools before importing numpy/torch
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
):pass
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: Foundational Trading Patterns Training

Optimized training approach with:
1. Simplified reward function focusing on entry quality
2. Relaxed constraints (trailing DD, daily loss limit, profit target)
3. Simplified observation space (removed complex features)
4. Better reward magnitudes for stronger learning signal
5. Enhanced exploration parameters

Designed to learn quality entry signals before advancing to Phase 2 position management.
"""

import os

# Limit math/BLAS thread pools before importing numpy/torch
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
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import yaml
from datetime import datetime

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

_TORCH_THREADS_OVERRIDE = os.environ.get("PYTORCH_NUM_THREADS")
try:
    if _TORCH_THREADS_OVERRIDE is not None:
        torch.set_num_threads(max(1, int(_TORCH_THREADS_OVERRIDE)))
    else:
        torch.set_num_threads(_THREAD_LIMIT_INT)
except (TypeError, ValueError):
    torch.set_num_threads(1)

from sb3_contrib import MaskablePPO  # Using MaskablePPO for action masking
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

# IMPORT PHASE 1 ENVIRONMENT
from environment_phase1 import TradingEnvironmentPhase1
from kl_callback import KLDivergenceCallback
from feature_engineering import add_market_regime_features
from model_utils import get_model_save_name, detect_available_markets, select_market_for_training
from market_specs import get_market_spec
from metadata_utils import write_metadata

# CHECKPOINT MANAGEMENT
from checkpoint_manager import DynamicCheckpointManager, MetricTrackingEvalCallback, EvalMetricHook
from checkpoint_retention import CheckpointRetentionManager

# SELF-CORRECTING SYSTEM
from self_correcting_init import init_self_correcting_system
from action_mask_utils import ActionMaskVecEnvWrapper, ActionMaskGymnasiumWrapper
from training_mode_utils import tune_eval_schedule

# Set UTF-8 encoding for Windows compatibility
if os.name == 'nt':
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
            '—': '-',
            '–': '-',
            '’': "'",
        }
        ascii_message = message
        for src, target in replacements.items():
            ascii_message = ascii_message.replace(src, target)
        ascii_message = ascii_message.encode('ascii', errors='ignore').decode('ascii')
        print(ascii_message)


def get_effective_num_envs(requested_envs: int) -> int:
    """Determine a safe number of parallel environments for the host."""
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

# MaskablePPO Hyperparameters - Optimized for Phase 1 with Action Masking
# Training approach:
# - 5M timesteps for production (100K for test mode)
# - Simplified reward function for entry quality
# - Enhanced exploration parameters
PHASE1_CONFIG = {
    # Training - UPGRADED to 10M timesteps for better baseline
    # IMPROVEMENT: Increased from 5M to 10M for stronger Phase 1 foundation
    'total_timesteps': 10_000_000,  # 10M for production (PhaseGuard requirement)
    'num_envs': 64,  # OPTIMIZED: Sweet spot for CPU/GPU balance with SubprocVecEnv

    # Network architecture - MAINTAINED for capacity
    'policy_layers': [512, 256, 128],

    # MaskablePPO parameters - ENHANCED for exploration
    'learning_rate': 3e-4,
    'n_steps': 4096,  # Rollout length
    'batch_size': 1024,
    'n_epochs': 8,  # Optimization epochs per update
    'gamma': 0.99,  # Discount factor
    'gae_lambda': 0.95,  # GAE parameter
    'clip_range': 0.2,  # Clip range
    'ent_coef': 0.02,  # INCREASED from 0.01 for more exploration
    'vf_coef': 0.5,  # Value function coefficient
    'max_grad_norm': 0.5,  # Gradient clipping

    # Learning rate scheduling
    'use_lr_schedule': True,
    'lr_final_fraction': 0.2,  # End at 20% of initial LR

    # Early stopping with KL monitoring
    'target_kl': 0.01,
    'use_kl_callback': False,

    # Environment parameters - IMPROVED for longer episodes
    'window_size': 20,
    'initial_balance': 50000,
    'initial_sl_multiplier': 1.5,
    'initial_tp_ratio': 3.0,
    'position_size': 1.0,  # Futures exchanges require whole contracts
    'trailing_dd_limit': 15000,  # RELAXED from 5000 (was too restrictive)
    'episode_length': 10000,  # 1 trading day
    'min_episode_bars': 5000,  # IMPROVED: Increased from 1500 for longer episodes
    'randomize_start_offsets': True,
    'deterministic_env_offsets': False,
    'start_offset_seed': 11,

    # Market specifications
    'commission_override': None,

    # Evaluation - IMPROVED: More frequent and better coverage
    'eval_freq': 100_000,  # Every 100K steps (10M / 100K = 100 evaluations)
    'n_eval_episodes': 10,
    'eval_interval_updates': 6,  # Target evaluations every ~6 PPO updates
    'min_eval_episodes': 8,

    # Early stopping - More patient for 10M timestep training
    'use_early_stopping': True,
    'early_stop_max_no_improvement': 10,  # INCREASED for longer training
    'early_stop_min_evals': 5,

    # Device configuration
    'device': 'cuda'
}

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

def create_lr_schedule(initial_lr, final_fraction, total_timesteps):
    """Create linear learning rate schedule."""
    def lr_schedule(progress):
        return initial_lr * (1 - progress * (1 - final_fraction))
    return lr_schedule


def find_data_file(market=None):
    """Find training data file with priority order."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(script_dir, 'data')

    if market and market != 'GENERIC':
        market_file = os.path.join(data_dir, f'{market}_D1M.csv')
        if os.path.exists(market_file):
            return market_file

    filenames_to_try = [
        'D1M.csv',
        'es_training_data_CORRECTED_CLEAN.csv',
        'es_training_data_CORRECTED.csv',
        'databento_es_training_data_processed_cleaned.csv',
        'databento_es_training_data_processed.csv'
    ]

    for filename in filenames_to_try:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return path

    pattern = os.path.join(data_dir, '*_D1M.csv')
    instrument_files = sorted(glob.glob(pattern))
    if instrument_files:
        return instrument_files[0]

    raise FileNotFoundError(
        f"Training data not found in {data_dir}. "
        f"Expected one of: {filenames_to_try} or any '*_D1M.csv' file"
    )


def _ensure_datetime_index(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Convert a DataFrame index to a timezone-aware DatetimeIndex if needed."""
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
    """
    data_path = find_data_file(market=market)
    safe_print(f"[DATA] Loading minute-level data from {data_path}")

    # Load minute-level data
    try:
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    except Exception:
        df_tmp = pd.read_csv(data_path)
        time_col = 'timestamp' if 'timestamp' in df_tmp.columns else 'datetime'
        df_tmp[time_col] = pd.to_datetime(df_tmp[time_col])
        data = df_tmp.set_index(time_col)

    data = _ensure_datetime_index(data, "minute-level data")

    # Convert to America/New_York timezone
    if data.index.tz is None:
        safe_print("[DATA] Converting timezone: UTC -> America/New_York")
        data.index = data.index.tz_localize('UTC').tz_convert('America/New_York')
    elif str(data.index.tz) != 'America/New_York':
        safe_print(f"[DATA] Converting timezone: {data.index.tz} -> America/New_York")
        data.index = data.index.tz_convert('America/New_York')

    safe_print(f"[DATA] Loaded {len(data):,} rows")
    safe_print(f"[DATA] Full date range: {data.index.min()} to {data.index.max()}")

    # Chronological train/val split
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
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    second_data_candidates = []

    if market and market != 'GENERIC':
        second_data_candidates.append(os.path.join(script_dir, 'data', f'{market}_D1S.csv'))

    second_data_candidates.append(os.path.join(script_dir, 'data', 'D1S.csv'))

    filename = os.path.basename(data_path)
    if filename.endswith('_D1M.csv') and len(filename) > len('_D1M.csv'):
        prefix = filename[:-len('_D1M.csv')]
        if prefix:
            second_data_candidates.append(os.path.join(script_dir, 'data', f"{prefix}_D1S.csv"))

    seen = set()
    unique_candidates = []
    for candidate in second_data_candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    second_data_candidates = unique_candidates

    second_data_path = None
    for candidate in second_data_candidates:
        if os.path.exists(candidate):
            second_data_path = candidate
            break

    if second_data_path:
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
        fallback_list = " or ".join(second_data_candidates)
        safe_print(f"[DATA] Second-level data not found (expected {fallback_list}) - continuing without it")

    return train_data, val_data, train_second_data, val_second_data


def make_env(data, second_data, env_id, config, market_spec):
    """Create environment factory leveraging internal episode randomization."""
    def _init():
        randomize = config.get('randomize_start_offsets', True)
        deterministic_start = None
        if not randomize:
            deterministic_start = compute_env_start_index(len(data), config, env_id)

        env = TradingEnvironmentPhase1(
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
            start_index=deterministic_start,
            randomize_start_offsets=randomize,
            min_episode_bars=config.get('min_episode_bars', 1500),
            enable_daily_loss_limit=False,
            enable_profit_target=False,
            enable_4pm_rule=True,  # Keep this for safety
            max_episode_bars=config.get('episode_length', 390),
        )

        # Validation: Ensure environment has randomization support
        assert hasattr(env, 'randomize_start_offsets'), \
            "Environment missing randomization support"
        assert hasattr(env, 'min_episode_bars'), \
            "Environment missing min_episode_bars attribute"
        assert hasattr(env, '_determine_episode_start'), \
            "Environment missing _determine_episode_start method"

        env = ActionMaskGymnasiumWrapper(env)
        return Monitor(env)

    return _init


def train_phase1(
    continue_training=False,
    model_path=None,
    market_override=None,
    non_interactive=False,
    test_mode=False,
    hardware_profile=None,
):
    """Execute Phase 1 training with simplified reward and relaxed constraints."""
    # Apply hardware profile if provided
    if hardware_profile:
        profile_data = load_hardware_profile(hardware_profile)
        apply_hardware_profile(PHASE1_CONFIG, profile_data, test_mode)
    if continue_training:
        safe_print("=" * 80)
        safe_print("PHASE 1: CONTINUING TRAINING FROM EXISTING MODEL")
        safe_print("=" * 80)
        safe_print(f"[MODEL] Loading model from: {model_path}")
    else:
        safe_print("=" * 80)
        safe_print("PHASE 1: FOUNDATIONAL TRADING PATTERNS")
        safe_print("=" * 80)
        safe_print("[TRAINING APPROACH]")

        safe_print("  ✓ Simplified reward function (entry quality focus)")
        safe_print("  ✓ Relaxed constraints (trailing DD: $5K → $15K)")
        safe_print("  ✓ Disabled daily loss limit for Phase 1")
        safe_print("  ✓ Disabled profit target for Phase 1")
        safe_print("  ✓ Enhanced exploration (ent_coef: 0.01 → 0.02)")
        safe_print("  ✓ Increased patience (early stopping: 5 → 8 evals)")
        safe_print("=" * 80)

    tune_eval_schedule(
        PHASE1_CONFIG,
        test_mode=test_mode,
        label="Phase 1",
        eval_updates=PHASE1_CONFIG.get('eval_interval_updates', 6),
        min_eval_episodes=PHASE1_CONFIG.get('min_eval_episodes', 8),
        printer=safe_print,
    )

    market_name = "NQ"  # Example market name; replace as needed
    # =========================
    # Weights & Biases Init
    # =========================
    wandb_run = wandb.init(
        project="rl-trading-phase1",
        group=f"phase1-{market_name}",
        config={
            **PHASE1_CONFIG,
            "phase": 1,
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

    if market_override:
        from market_specs import get_market_spec
        market_spec = get_market_spec(market_override.upper())
        if market_spec is None:
            safe_print(f"\n[ERROR] Invalid market symbol: {market_override}")
            safe_print("[ERROR] Valid markets: ES, NQ, YM, RTY, MNQ, MES, M2K, MYM")
            return
        selected_market = next((m for m in available_markets if m['market'] == market_override.upper()), None)
        if selected_market is None:
            safe_print(f"\n[ERROR] No data found for market: {market_override}")
            safe_print(f"[ERROR] Available markets: {', '.join([m['market'] for m in available_markets])}")
            return
        market_name = market_override.upper()
        safe_print(f"\n[CLI] Market specified via --market: {market_name}")
    else:
        selected_market, market_spec = select_market_for_training(available_markets, safe_print)
        if selected_market is None or market_spec is None:
            safe_print("\n[INFO] Training cancelled - no market selected")
            return
        market_name = selected_market['market']

    safe_print(f"\n[TRAINING] Market: {market_name}")

    safe_print(f"[CONFIG] Total timesteps: {PHASE1_CONFIG['total_timesteps']:,}")
    requested_envs = PHASE1_CONFIG['num_envs']
    num_envs = get_effective_num_envs(requested_envs)
    safe_print(f"[CONFIG] Parallel envs (requested): {requested_envs}")
    if num_envs != requested_envs:
        safe_print(f"[CONFIG] Parallel envs (effective): {num_envs} (adjusted for host limits)")
    else:
        safe_print(f"[CONFIG] Parallel envs (effective): {num_envs}")
    safe_print(f"[SYSTEM] BLAS threads per process: {os.environ.get('OPENBLAS_NUM_THREADS', 'unknown')}")
    safe_print(f"[CONFIG] Network: {PHASE1_CONFIG['policy_layers']}")
    safe_print(f"[CONFIG] Device: {PHASE1_CONFIG['device']}")
    safe_print(f"[CONFIG] Fixed SL: {PHASE1_CONFIG['initial_sl_multiplier']}x ATR")
    safe_print(f"[CONFIG] Fixed TP: {PHASE1_CONFIG['initial_tp_ratio']}x SL")
    safe_print(f"[CONFIG] Entropy coef: {PHASE1_CONFIG['ent_coef']} (INCREASED for exploration)")
    safe_print(f"[CONFIG] Trailing DD limit: ${PHASE1_CONFIG['trailing_dd_limit']:,} (RELAXED)")
    safe_print("")

    # Create directories
    os.makedirs('models', exist_ok=True)  # FIX: Ensure base models directory exists
    os.makedirs('models/phase1', exist_ok=True)
    os.makedirs('models/phase1/checkpoints', exist_ok=True)
    os.makedirs('logs/phase1', exist_ok=True)
    os.makedirs('tensorboard_logs/phase1', exist_ok=True)

    # Load data with train/val split
    train_data, val_data, train_second_data, val_second_data = load_data(train_split=0.7, market=market_name)

    # Create vectorized training environments
    safe_print(f"[ENV] Creating {num_envs} parallel training environments...")

    # Use SubprocVecEnv for Phase 1 (parallel processing for better hardware utilization)
    env_factories = [
        make_env(
            train_data,
            train_second_data,
            i,
            PHASE1_CONFIG,
            market_spec
        )
        for i in range(num_envs)
    ]

    env = SubprocVecEnv(env_factories, start_method='spawn')
    safe_print(f"[ENV] Using SubprocVecEnv ({num_envs} environments, multi-process)")

    # Wrap with action mask support
    env = ActionMaskVecEnvWrapper(env)
    safe_print("[ENV] Action masking enabled for 3-action space")

    # Add normalization
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )

    safe_print("[ENV] Training environments created with VecNormalize")

    # Create evaluation environment
    safe_print("[EVAL] Creating VALIDATION environment (unseen data)...")
    safe_print("[EVAL] Using randomized episode start points for diverse evaluation")
    eval_env = DummyVecEnv([
        lambda: Monitor(ActionMaskGymnasiumWrapper(TradingEnvironmentPhase1(
            data=val_data,
            window_size=PHASE1_CONFIG['window_size'],
            initial_balance=PHASE1_CONFIG['initial_balance'],
            second_data=val_second_data,
            market_spec=market_spec,
            commission_override=PHASE1_CONFIG.get('commission_override', None),
            initial_sl_multiplier=PHASE1_CONFIG['initial_sl_multiplier'],
            initial_tp_ratio=PHASE1_CONFIG['initial_tp_ratio'],
            position_size_contracts=PHASE1_CONFIG['position_size'],
            trailing_drawdown_limit=PHASE1_CONFIG['trailing_dd_limit'],
            start_index=None,
            randomize_start_offsets=True,
            min_episode_bars=PHASE1_CONFIG.get('min_episode_bars', 1500),
            enable_daily_loss_limit=False,
            enable_profit_target=False,
            enable_4pm_rule=True,
        )))
    ])
    eval_env = ActionMaskVecEnvWrapper(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create learning rate schedule
    learning_rate = PHASE1_CONFIG['learning_rate']
    if PHASE1_CONFIG.get('use_lr_schedule', False):
        learning_rate = create_lr_schedule(
            PHASE1_CONFIG['learning_rate'],
            PHASE1_CONFIG['lr_final_fraction'],
            PHASE1_CONFIG['total_timesteps']
        )
        safe_print("[TRAIN] Using linear learning rate schedule")
    
    # Create or load MaskablePPO model (with action masking)
    if continue_training and model_path:
        safe_print("\n[MODEL] Loading existing MaskablePPO model...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        model = MaskablePPO.load(model_path, device=PHASE1_CONFIG['device'])
        model.set_env(env)
        model.tensorboard_log = './tensorboard_logs/phase1/'
        safe_print(f"[MODEL] Model loaded successfully from {model_path}")
        safe_print(f"[MODEL] Current timesteps: {model.num_timesteps:,}")
        safe_print(f"[MODEL] Will train for additional {PHASE1_CONFIG['total_timesteps']:,} timesteps")
    else:
        safe_print("\n[MODEL] Creating NEW MaskablePPO model (with action masking)...")
        safe_print(f"[MODEL] Policy network: {PHASE1_CONFIG['policy_layers']}")
        safe_print(f"[MODEL] Initial learning rate: {PHASE1_CONFIG['learning_rate']}")
        safe_print(f"[MODEL] Batch size: {PHASE1_CONFIG['batch_size']}")
        safe_print(f"[MODEL] Clip range: {PHASE1_CONFIG['clip_range']}")
        safe_print(f"[MODEL] Entropy coefficient: {PHASE1_CONFIG['ent_coef']} (INCREASED for exploration)")
        safe_print(f"[MODEL] Action masking: ENABLED (forces HOLD when in position)")

        model = MaskablePPO(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            n_steps=PHASE1_CONFIG['n_steps'],
            batch_size=PHASE1_CONFIG['batch_size'],
            n_epochs=PHASE1_CONFIG['n_epochs'],
            gamma=PHASE1_CONFIG['gamma'],
            gae_lambda=PHASE1_CONFIG['gae_lambda'],
            clip_range=PHASE1_CONFIG['clip_range'],
            ent_coef=PHASE1_CONFIG['ent_coef'],  # INCREASED
            vf_coef=PHASE1_CONFIG['vf_coef'],
            max_grad_norm=PHASE1_CONFIG['max_grad_norm'],
            policy_kwargs={
                'net_arch': dict(
                    pi=PHASE1_CONFIG['policy_layers'],
                    vf=PHASE1_CONFIG['policy_layers']
                ),
                'activation_fn': torch.nn.ReLU
            },
            device=PHASE1_CONFIG['device'],
            verbose=1,
            tensorboard_log='./tensorboard_logs/phase1/'
        )

        safe_print("[MODEL] MaskablePPO model created with action masking enabled")

    # Callbacks
    callbacks_list = []

    if PHASE1_CONFIG.get('use_early_stopping', False):
        early_stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=PHASE1_CONFIG['early_stop_max_no_improvement'],
            min_evals=PHASE1_CONFIG['early_stop_min_evals'],
            verbose=1
        )
        safe_print(f"[TRAIN] Early stopping enabled:")
        safe_print(f"        - Stop after {PHASE1_CONFIG['early_stop_max_no_improvement']} evals with no improvement")
        safe_print(f"        - Minimum {PHASE1_CONFIG['early_stop_min_evals']} evals required")
        eval_cadence_steps = PHASE1_CONFIG['eval_freq'] * num_envs
        safe_print(
            f"        - Evaluation every {eval_cadence_steps:,} timesteps "
            f"(~{PHASE1_CONFIG['eval_freq']:,} env steps across {num_envs} envs)"
        )
    else:
        early_stop_callback = None
        safe_print("[TRAIN] Early stopping disabled")

    # IMPROVEMENT: Add timestamp to evaluation logs to prevent overwriting
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_log_path = f'./logs/phase1/eval_{eval_timestamp}'
    os.makedirs(eval_log_path, exist_ok=True)

    safe_print(f"[EVAL] Evaluation logs will be saved to: {eval_log_path}")

    # Initialize self-correcting system from configuration
    registry, forecaster, metric_tracker, policy_controller, corrective_manager = init_self_correcting_system(
        market=market_name,
        phase=1,
        config_path='config/checkpoint_config.yaml',
        verbose=True
    )

    # Set seed for metric tracker
    metric_tracker.set_seed(42)  # Match training seed

    eval_metric_hook = EvalMetricHook(metric_tracker)

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=early_stop_callback,  # Proper placement for early stopping
        callback_after_eval=eval_metric_hook,      # Must be a BaseCallback per SB3 API
        eval_freq=PHASE1_CONFIG['eval_freq'],
        n_eval_episodes=PHASE1_CONFIG['n_eval_episodes'],
        best_model_save_path='./models/phase1/',
        log_path=eval_log_path,  # IMPROVEMENT: Versioned log path
        deterministic=True,
        render=False,
        verbose=1
    )

    # CRITICAL FIX: Link hook to parent so it can access evaluation results
    eval_metric_hook.parent = eval_callback

    # IMPROVED: Dynamic checkpoint manager
    checkpoint_manager = DynamicCheckpointManager(
        market=market_name,
        phase=1,
        seed=42,
        config_path='config/checkpoint_config.yaml',
        metric_tracker=metric_tracker,
        target_timesteps=PHASE1_CONFIG['total_timesteps'],
        verbose=False,
        registry=registry
    )

    # Build callbacks list
    callbacks = [wandb_callback, eval_callback, checkpoint_manager]

    if policy_controller:
        callbacks.append(policy_controller)
    if corrective_manager:
        callbacks.append(corrective_manager)

    if PHASE1_CONFIG.get('use_kl_callback', False):
        kl_callback = KLDivergenceCallback(
            target_kl=PHASE1_CONFIG['target_kl'],
            verbose=1,
            log_freq=1000
        )
        callbacks.append(kl_callback)

    # Train
    safe_print("\n" + "=" * 80)
    safe_print(f"[TRAIN] Starting Phase 1 training for {PHASE1_CONFIG['total_timesteps']:,} timesteps...")
    safe_print("=" * 80)

    import time
    start_time = time.time()
    use_progress_bar = PROGRESS_BAR_AVAILABLE

    try:
        model.learn(
            total_timesteps=PHASE1_CONFIG['total_timesteps'],
            callback=callbacks,
            progress_bar=use_progress_bar,
            reset_num_timesteps=not continue_training
        )
        checkpoint_manager.on_phase_end()
    except KeyboardInterrupt:
        safe_print("\n[TRAIN] Training interrupted by user")
        checkpoint_manager.on_interrupt()
    except Exception as e:
        safe_print(f"\n[ERROR] Training failed: {e}")
        try:
            checkpoint_manager.on_interrupt()
        except:
            pass
        raise

    elapsed = time.time() - start_time

    # Save final model
    safe_print("\n[SAVE] Saving final Phase 1 model...")
    if continue_training:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        default_name = f"{base_name}_continued_fixed"
    else:
        default_name = "phase1_foundational_fixed"

    if test_mode and not default_name.endswith('_test'):
        default_name = f"{default_name}_test"

    save_name = default_name
    if not non_interactive and sys.stdin.isatty():
        try:
            save_name = get_model_save_name(default_name)
        except:
            pass

    model_save_path = f'models/{save_name}'
    vecnorm_save_path = f'models/{save_name}_vecnorm.pkl'

    model.save(model_save_path)
    env.save(vecnorm_save_path)

    write_metadata(model_save_path, {'phase': 1, 'market': market_name, 'total_timesteps': int(model.num_timesteps)})
    write_metadata(vecnorm_save_path, {'phase': 1, 'market': market_name, 'artifact': 'vecnormalize'})

    safe_print("\n" + "=" * 80)
    safe_print("PHASE 1 TRAINING COMPLETE!")
    safe_print(f"[RESULTS] Training time: {elapsed/3600:.2f} hours")
    safe_print("=" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Phase 1 Training - Foundational Trading Patterns')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--continue', dest='continue_training', action='store_true', help='Continue training')
    parser.add_argument('--model-path', type=str, default=None, help='Path to model to continue from')
    parser.add_argument('--market', type=str, default=None, help='Market symbol')
    parser.add_argument('--non-interactive', action='store_true', help='No prompts')
    parser.add_argument('--hardware-profile', type=str, help='Path to hardware profile yaml')
    
    args = parser.parse_args()

    # Override config for test mode
    if args.test:
        safe_print("\n[TEST] Test mode enabled")
        PHASE1_CONFIG['total_timesteps'] = 100_000
        PHASE1_CONFIG['num_envs'] = 32
        PHASE1_CONFIG['eval_freq'] = 25_000
        PHASE1_CONFIG['n_eval_episodes'] = 20
        PHASE1_CONFIG['use_early_stopping'] = False

    if args.continue_training and not args.model_path:
        safe_print("\n[ERROR] --continue requires --model-path")
        sys.exit(1)

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    train_phase1(
        continue_training=args.continue_training,
        model_path=args.model_path,
        market_override=args.market,
        non_interactive=args.non_interactive,
        test_mode=args.test,
        hardware_profile=args.hardware_profile
    )
def compute_env_start_index(data_length: int, config: dict, env_id: int) -> int:
    """Deterministic start offset when randomization is disabled."""
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
