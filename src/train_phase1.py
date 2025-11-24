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
    'n_steps': 2048,  # Rollout length
    'batch_size': 512,
    'n_epochs': 10,  # Optimization epochs per update
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
    'use_kl_callback': True,

    # Environment parameters - IMPROVED for longer episodes
    'window_size': 20,
    'initial_balance': 50000,
    'initial_sl_multiplier': 1.5,
    'initial_tp_ratio': 3.0,
    'position_size': 1.0,  # Futures exchanges require whole contracts
    'trailing_dd_limit': 15000,  # RELAXED from 5000 (was too restrictive)
    'episode_length': 390,  # 1 trading day
    'min_episode_bars': 3000,  # IMPROVED: Increased from 1500 for longer episodes
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
):
    """Execute Phase 1 training with simplified reward and relaxed constraints."""
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

    env = SubprocVecEnv(env_factories, start_method='forkserver')
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
        safe_print(f"        - Evaluation every {PHASE1_CONFIG['eval_freq']:,} timesteps")
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
    # Without this link, checkpoint_manager.py:600-606 cannot access evaluations_results
    # and evaluations.npz will NOT be created, blocking Phase 2 training
    eval_metric_hook.parent = eval_callback
    assert eval_metric_hook.parent is not None, "CRITICAL: eval_metric_hook.parent must be set!"
    assert hasattr(eval_metric_hook.parent, 'evaluations_results'), \
        "CRITICAL: parent must have evaluations_results attribute!"

    # IMPROVED: Dynamic checkpoint manager with adaptive intervals and event-driven saves
    checkpoint_manager = DynamicCheckpointManager(
        market=market_name,
        phase=1,
        seed=42,
        config_path='config/checkpoint_config.yaml',
        metric_tracker=metric_tracker,
        target_timesteps=PHASE1_CONFIG['total_timesteps'],
        verbose=False,
        registry=registry  # Connect to global registry
    )
    safe_print(f"[CHECKPOINT] Dynamic checkpoint manager initialized")
    safe_print(f"[CHECKPOINT] Base interval: 25K steps (adaptive)")
    safe_print(f"[CHECKPOINT] Event triggers: periodic, best, phase_end, interrupt")

    # Build callbacks list (order matters: eval -> checkpoint -> controller -> corrective)
    callbacks = [eval_callback, checkpoint_manager]

    # Add optional self-correcting callbacks (if enabled in config)
    if policy_controller is not None:
        callbacks.append(policy_controller)
    if corrective_manager is not None:
        callbacks.append(corrective_manager)

    if PHASE1_CONFIG.get('use_kl_callback', False):
        kl_callback = KLDivergenceCallback(
            target_kl=PHASE1_CONFIG['target_kl'],
            verbose=1,
            log_freq=1000
        )
        callbacks.append(kl_callback)
        safe_print("[TRAIN] KL divergence monitoring enabled")

    # Train
    safe_print("\n" + "=" * 80)
    safe_print(f"[TRAIN] Starting Phase 1 training for {PHASE1_CONFIG['total_timesteps']:,} timesteps...")
    safe_print("=" * 80)
    safe_print("\n[TRAIN] Monitor progress:")
    safe_print("        - TensorBoard: tensorboard --logdir tensorboard_logs/phase1/")
    safe_print("        - Logs: logs/phase1/evaluations.npz")
    safe_print("        - Checkpoints: models/phase1/checkpoints/")
    safe_print("")

    import time
    start_time = time.time()

    use_progress_bar = PROGRESS_BAR_AVAILABLE
    if use_progress_bar:
        safe_print("[TRAIN] Progress bar enabled (tqdm + rich available)")
    else:
        safe_print("[TRAIN] Progress bar disabled (missing tqdm/rich)")
        safe_print("[TRAIN] Monitor via TensorBoard: tensorboard --logdir tensorboard_logs/phase1/")

    try:
        model.learn(
            total_timesteps=PHASE1_CONFIG['total_timesteps'],
            callback=callbacks,
            progress_bar=use_progress_bar,
            reset_num_timesteps=not continue_training
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
        checkpoint_dir = f'./models/phase1/{market_name}/checkpoints/'
        retention_manager.prune_checkpoints(checkpoint_dir, dry_run=False, verbose=False)
        safe_print("[CHECKPOINT] Retention cleanup completed")
    except Exception as e:
        safe_print(f"[WARNING] Checkpoint retention cleanup failed: {e}")

    # Save final model
    safe_print("\n[SAVE] Saving final Phase 1 model...")

    # Determine default name
    if continue_training:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        default_name = f"{base_name}_continued_fixed"
    else:
        default_name = "phase1_foundational_fixed"

    if test_mode and not default_name.endswith('_test'):
        default_name = f"{default_name}_test"
        safe_print("[SAVE] Test mode run detected - using '_test' suffix for artifacts")

    # Get save name
    is_interactive = (not non_interactive and
                      sys.stdin.isatty() and
                      hasattr(sys.stdin, 'readable'))

    if is_interactive:
        try:
            if sys.stdin.readable():
                save_name = get_model_save_name(default_name)
            else:
                save_name = default_name
                safe_print(f"[SAVE] Non-interactive mode - using default name: {save_name}")
        except (EOFError, OSError):
            save_name = default_name
            safe_print(f"[SAVE] Non-interactive mode - using default name: {save_name}")
    else:
        save_name = default_name
        safe_print(f"[SAVE] Non-interactive mode - using default name: {save_name}")

    model_save_path = f'models/{save_name}'
    vecnorm_save_path = f'models/{save_name}_vecnorm.pkl'

    model.save(model_save_path)
    env.save(vecnorm_save_path)

    metadata_common = {
        'phase': 1,
        'market': market_name,
        'total_timesteps': int(model.num_timesteps),
        'timesteps_target': int(PHASE1_CONFIG['total_timesteps']),
        'test_mode': bool(test_mode),
        'continue_training': bool(continue_training),
        'fixed_version': True,  # Mark as fixed version
    }

    write_metadata(model_save_path, {**metadata_common, 'artifact': 'model'})
    write_metadata(vecnorm_save_path, {**metadata_common, 'artifact': 'vecnormalize'})

    # CRITICAL FIX: Create legacy symlink for backward compatibility
    # This allows PhaseGuard and other tools to find evaluation results at the legacy path
    safe_print("[COMPAT] Creating legacy evaluation symlink for PhaseGuard...")
    try:
        from pipeline.phase_guard import PhaseGuard
        legacy_path = PhaseGuard.create_legacy_symlink('logs/phase1', use_copy=False)
        if legacy_path:
            safe_print(f"[COMPAT] ✓ Created legacy evaluation link: {legacy_path}")
        else:
            # Fallback: Create symlink manually if PhaseGuard method failed
            safe_print("[COMPAT] PhaseGuard method returned None, creating symlink manually...")
            log_path = Path('logs/phase1')
            eval_folders = sorted(log_path.glob("eval_*/"))
            if eval_folders:
                newest_folder = eval_folders[-1]
                source_file = newest_folder / "evaluations.npz"
                legacy_link = log_path / "evaluations.npz"

                if source_file.exists():
                    # Remove old symlink if exists
                    if legacy_link.exists() or legacy_link.is_symlink():
                        legacy_link.unlink()
                    # Create new symlink
                    legacy_link.symlink_to(source_file.name, target_is_directory=False)
                    # Use relative path for symlink
                    legacy_link.unlink()
                    legacy_link.symlink_to(f"{newest_folder.name}/evaluations.npz")
                    safe_print(f"[COMPAT] ✓ Manually created symlink: logs/phase1/evaluations.npz -> {newest_folder.name}/evaluations.npz")
                else:
                    safe_print(f"[COMPAT] ✗ Warning: Evaluation file not found at {source_file}")
            else:
                safe_print("[COMPAT] ✗ Warning: No timestamped eval folders found")
    except Exception as e:
        safe_print(f"[COMPAT] ✗ Warning: Failed to create legacy symlink: {e}")
        safe_print("[COMPAT] Phase 2 may need manual symlink creation")

    safe_print("\n" + "=" * 80)
    safe_print("PHASE 1 TRAINING COMPLETE!")
    safe_print("=" * 80)
    safe_print(f"[RESULTS] Training time: {elapsed/3600:.2f} hours")
    safe_print(f"[RESULTS] Total timesteps: {model.num_timesteps:,}")
    safe_print(f"[SAVE] Model: {model_save_path}.zip")
    safe_print(f"[SAVE] VecNorm: {vecnorm_save_path}")
    safe_print(f"[SAVE] Best model: models/phase1/best_model.zip")
    safe_print("")
    safe_print("[NEXT STEPS]")
    safe_print("  1. Review training logs: logs/phase1/evaluations.npz")
    safe_print("  2. Check TensorBoard: tensorboard --logdir tensorboard_logs/phase1/")
    safe_print("  3. Evaluate performance: python evaluate_phase1.py")
    safe_print("  4. Run Phase 2: python train_phase2.py")
    safe_print("")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Phase 1 Training - Foundational Trading Patterns')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with reduced timesteps (100K for better testing)')
    parser.add_argument('--continue', dest='continue_training', action='store_true',
                       help='Continue training from an existing model')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to the model to continue training from')
    parser.add_argument('--market', type=str, default=None,
                       help='Market to train on (ES, NQ, YM, RTY, MNQ, MES, M2K, MYM)')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Run in non-interactive mode (no prompts, use defaults)')
    args = parser.parse_args()

    # Override config for test mode
    if args.test:
        safe_print("\n" + "=" * 80)
        safe_print("TEST MODE ENABLED - Quick Local Testing")
        safe_print("=" * 80)
        PHASE1_CONFIG['total_timesteps'] = 100_000  # INCREASED from 30K
        PHASE1_CONFIG['num_envs'] = 32  # OPTIMIZED: Better hardware utilization with SubprocVecEnv
        PHASE1_CONFIG['eval_freq'] = 25_000  # Eval 4 times
        PHASE1_CONFIG['n_eval_episodes'] = 20  # INCREASED from 5 for better statistics
        PHASE1_CONFIG['use_early_stopping'] = False  # Disable for test mode

        safe_print(f"[TEST] Timesteps:       5M → {PHASE1_CONFIG['total_timesteps']:,} (2% for testing)")
        safe_print(f"[TEST] Parallel envs:   80 → {PHASE1_CONFIG['num_envs']} (optimized for parallel processing)")
        safe_print(f"[TEST] Eval frequency:  Every {PHASE1_CONFIG['eval_freq']:,} steps")
        safe_print(f"[TEST] Eval episodes:   5 → {PHASE1_CONFIG['n_eval_episodes']} (diverse evaluation)")
        safe_print(f"[TEST] Early stopping:  DISABLED (test mode)")
        safe_print(f"[TEST] Expected time:   ~3-5 minutes (30x faster with SubprocVecEnv)")
        safe_print("=" * 80 + "\n")

    # Validate continuation arguments
    if args.continue_training and not args.model_path:
        safe_print("\n[ERROR] --continue requires --model-path to be specified")
        sys.exit(1)

    if args.continue_training:
        safe_print(f"\n[INFO] Continuation mode enabled")
        safe_print(f"[INFO] Will load model from: {args.model_path}")

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    train_phase1(
        continue_training=args.continue_training,
        model_path=args.model_path,
        market_override=args.market,
        non_interactive=args.non_interactive,
        test_mode=args.test,
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
