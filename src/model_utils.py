"""
Model Management Utilities for RL Training System

This module provides utilities for detecting, loading, and managing
trained models in the models directory.
"""

import os
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import torch
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize

from metadata_utils import read_metadata


def resolve_learning_rate(lr_attr: Any, progress: float = 1.0) -> Optional[float]:
    """
    Safely resolve a learning rate attribute to a float value.

    Handles constant floats, numpy/tensor scalars, or callable schedules that
    expect a progress value (0.0–1.0). Returns None when the rate cannot be
    evaluated.
    """
    if lr_attr is None:
        return None

    # Callables (e.g., SB3 schedules) may expect a progress argument
    if callable(lr_attr):
        for arg in (progress,):
            try:
                return float(lr_attr(arg))
            except TypeError:
                continue
            except Exception:
                return None
        try:
            return float(lr_attr())
        except Exception:
            return None

    try:
        return float(lr_attr)
    except (TypeError, ValueError):
        return None


def detect_models_in_folder(model_dir: str = 'models', phase: Optional[str] = None) -> List[Dict]:
    """
    Detect all trained models in the models directory.

    Args:
        model_dir: Path to models directory (default: 'models')
        phase: Filter by phase ('phase1' or 'phase2'). If None, return all models.

    Returns:
        List of dicts with model information:
        [
            {
                'path': 'models/phase1_foundational_final.zip',
                'name': 'phase1_foundational_final',
                'type': 'phase1',
                'size_mb': 6.97,
                'modified': datetime object,
                'modified_str': '2025-10-28 14:30:15',
                'vecnorm_path': 'models/phase1_vecnorm.pkl' or None
            },
            ...
        ]
    """
    models = []

    # Search for all .zip files in models directory
    search_pattern = os.path.join(model_dir, '**', '*.zip')
    model_files = glob.glob(search_pattern, recursive=True)

    for model_path in model_files:
        # Get basic file info
        path_obj = Path(model_path)
        name = path_obj.stem  # filename without extension
        size_bytes = path_obj.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        modified_timestamp = path_obj.stat().st_mtime
        modified_dt = datetime.fromtimestamp(modified_timestamp)

        # Determine model type from path or name
        model_type = None
        path_lower = model_path.lower()
        if 'phase3' in path_lower or 'hybrid' in path_lower:
            model_type = 'phase3'
        elif 'phase2' in path_lower:
            model_type = 'phase2'
        elif 'phase1' in path_lower:
            model_type = 'phase1'
        else:
            # Try to infer from directory structure
            if 'phase1' in str(path_obj.parent):
                model_type = 'phase1'
            elif 'phase2' in str(path_obj.parent):
                model_type = 'phase2'
            elif 'phase3' in str(path_obj.parent):
                model_type = 'phase3'

        # Skip if we're filtering by phase and this doesn't match
        if phase and model_type != phase:
            continue

        # Look for corresponding VecNormalize file
        vecnorm_path = None

        # Check for vecnorm file with same base name
        vecnorm_candidate1 = str(path_obj.parent / f"{name}_vecnorm.pkl")
        vecnorm_candidate2 = str(path_obj.parent / f"{model_type}_vecnorm.pkl")
        vecnorm_candidate3 = os.path.join(model_dir, f"{model_type}_vecnorm.pkl")

        for candidate in [vecnorm_candidate1, vecnorm_candidate2, vecnorm_candidate3]:
            if os.path.exists(candidate):
                vecnorm_path = candidate
                break

        metadata = None
        metadata_path = model_path[:-4] if model_path.lower().endswith('.zip') else model_path
        metadata = read_metadata(metadata_path)

        models.append({
            'path': model_path,
            'name': name,
            'type': model_type or 'unknown',
            'size_mb': size_mb,
            'modified': modified_dt,
            'modified_str': modified_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'vecnorm_path': vecnorm_path,
            'metadata': metadata,
            'market': (metadata or {}).get('market'),
            'phase': (metadata or {}).get('phase')
        })

    # Sort by modification time (newest first)
    models.sort(key=lambda x: x['modified'], reverse=True)

    return models


def load_model_auto(model_path: str, device: str = 'auto') -> Tuple[object, str]:
    """
    Load a model automatically detecting its type (Phase 1 PPO or Phase 2 MaskablePPO).

    Args:
        model_path: Path to .zip model file
        device: Device to load model on ('cuda', 'cpu', or 'auto')

    Returns:
        Tuple of (model, model_type) where model_type is 'phase1' or 'phase2'

    Raises:
        ValueError: If model cannot be loaded
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Auto-detect device if needed
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Try loading as Phase 2 first (MaskablePPO)
    try:
        print(f"Attempting to load as Phase 2 (MaskablePPO) model...")
        model = MaskablePPO.load(model_path, device=device)
        print(f"✓ Successfully loaded as Phase 2 model")
        return model, 'phase2'
    except Exception as e:
        print(f"  Not a Phase 2 model: {str(e)[:50]}...")

    # Try loading as Phase 1 (PPO)
    try:
        print(f"Attempting to load as Phase 1 (PPO) model...")
        model = PPO.load(model_path, device=device)
        print(f"✓ Successfully loaded as Phase 1 model")
        return model, 'phase1'
    except Exception as e:
        print(f"  Not a Phase 1 model: {str(e)[:50]}...")

    raise ValueError(f"Could not load model from {model_path}. Model may be corrupted or incompatible.")


def load_vecnormalize(vecnorm_path: str, env) -> VecNormalize:
    """
    Load VecNormalize statistics from a .pkl file.

    Args:
        vecnorm_path: Path to VecNormalize .pkl file
        env: Environment to wrap with VecNormalize

    Returns:
        VecNormalize wrapped environment

    Raises:
        FileNotFoundError: If vecnorm file doesn't exist
    """
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")

    print(f"Loading VecNormalize stats from {vecnorm_path}...")
    env = VecNormalize.load(vecnorm_path, env)
    print("✓ VecNormalize stats loaded successfully")

    return env


def display_model_selection(models: List[Dict], phase_filter: Optional[str] = None) -> int:
    """
    Display a list of models and get user selection.

    Args:
        models: List of model info dicts from detect_models_in_folder()
        phase_filter: Optional phase filter to show in title

    Returns:
        Index of selected model in the models list

    Raises:
        ValueError: If user input is invalid
    """
    if not models:
        print("\n⚠ No models found in the models directory.")
        if phase_filter:
            print(f"   (Filtered for: {phase_filter})")
        return -1

    # Display header
    print("\n" + "="*70)
    if phase_filter:
        print(f"Available {phase_filter.upper()} Models:")
    else:
        print("Available Models:")
    print("="*70)

    # Display each model
    for i, model_info in enumerate(models, 1):
        type_label = model_info['type'].upper() if model_info['type'] != 'unknown' else 'UNKNOWN'
        vecnorm_status = "✓" if model_info['vecnorm_path'] else "✗"
        metadata = model_info.get('metadata') or {}
        market = metadata.get('market')
        test_mode = metadata.get('test_mode')
        timesteps = metadata.get('total_timesteps')

        print(f"\n{i}. {model_info['name']}")
        print(f"   Type: {type_label} | Size: {model_info['size_mb']:.2f} MB | VecNorm: {vecnorm_status}")
        print(f"   Modified: {model_info['modified_str']}")
        if market or timesteps or test_mode is not None:
            details = []
            if market:
                details.append(f"Market: {market}")
            if timesteps:
                details.append(f"Timesteps: {timesteps:,}")
            if test_mode is not None:
                details.append("Mode: TEST" if test_mode else "Mode: PROD")
            if details:
                print(f"   {' | '.join(details)}")
        print(f"   Path: {model_info['path']}")

    print("\n" + "="*70)
    print(f"0. Cancel")
    print("="*70)

    # Get user input
    while True:
        try:
            choice = input(f"\nSelect model to continue training (0-{len(models)}): ").strip()
            choice_num = int(choice)

            if choice_num == 0:
                print("Operation cancelled.")
                return -1
            elif 1 <= choice_num <= len(models):
                return choice_num - 1  # Convert to 0-indexed
            else:
                print(f"⚠ Please enter a number between 0 and {len(models)}")
        except ValueError:
            print("⚠ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            return -1


def get_model_save_name(default_name: str) -> str:
    """
    Prompt user for a model save name.

    Args:
        default_name: Suggested default name

    Returns:
        User-chosen model name (without .zip extension)
    """
    print("\n" + "="*70)
    print("Model Training Complete!")
    print("="*70)
    print(f"\nDefault save name: {default_name}")
    print("Enter a custom name, or press Enter to use the default.")
    print("Note: Do not include .zip extension (it will be added automatically)")
    print("="*70)

    while True:
        try:
            user_input = input("\nModel save name: ").strip()

            # Use default if empty
            if not user_input:
                return default_name

            # Remove .zip extension if user included it
            if user_input.endswith('.zip'):
                user_input = user_input[:-4]

            # Validate name (no special characters that would break filesystem)
            if any(char in user_input for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']):
                print("⚠ Model name contains invalid characters. Please use only letters, numbers, hyphens, and underscores.")
                continue

            return user_input

        except KeyboardInterrupt:
            print(f"\n\nUsing default name: {default_name}")
            return default_name


def validate_model_environment_compatibility(model_type: str, environment_type: str) -> bool:
    """
    Validate that a model type is compatible with an environment type.

    Args:
        model_type: 'phase1' or 'phase2'
        environment_type: 'phase1' or 'phase2'

    Returns:
        True if compatible, False otherwise
    """
    if model_type == environment_type:
        return True

    print(f"\n⚠ ERROR: Model type ({model_type}) does not match environment type ({environment_type})")
    print(f"   Phase 1 models can only continue training in Phase 1 environments.")
    print(f"   Phase 2 models can only continue training in Phase 2 environments.")
    print(f"   For transfer learning from Phase 1 to Phase 2, use 'Training Pod' instead.")

    return False


def detect_available_markets(data_dir='data'):
    """
    Detect all available market data files in the data directory.

    Looks for files matching patterns:
    - {MARKET}_D1M.csv (minute data)
    - D1M.csv (generic)

    Returns:
        list: List of dicts with market info
        Example: [
            {'market': 'ES', 'minute_file': 'ES_D1M.csv', 'second_file': 'ES_D1S.csv', 'has_second': True},
            {'market': 'NQ', 'minute_file': 'NQ_D1M.csv', 'second_file': None, 'has_second': False}
        ]
    """
    markets = []

    # Convert to absolute path
    data_dir = os.path.abspath(data_dir)

    # Pattern 1: Look for {MARKET}_D1M.csv files
    pattern = os.path.join(data_dir, '*_D1M.csv')
    minute_files = glob.glob(pattern)

    for minute_file in minute_files:
        basename = os.path.basename(minute_file)
        market = basename.replace('_D1M.csv', '')

        # Check for corresponding second-level file
        second_file = os.path.join(data_dir, f'{market}_D1S.csv')
        has_second = os.path.exists(second_file)

        markets.append({
            'market': market,
            'minute_file': basename,
            'second_file': f'{market}_D1S.csv' if has_second else None,
            'has_second': has_second,
            'path': os.path.abspath(minute_file)
        })

    # Pattern 2: Check for generic D1M.csv (no market prefix)
    generic_minute = os.path.join(data_dir, 'D1M.csv')
    if os.path.exists(generic_minute) and not any(m['market'] == 'GENERIC' for m in markets):
        second_file = os.path.join(data_dir, 'D1S.csv')
        has_second = os.path.exists(second_file)

        markets.append({
            'market': 'GENERIC',
            'minute_file': 'D1M.csv',
            'second_file': 'D1S.csv' if has_second else None,
            'has_second': has_second,
            'path': os.path.abspath(generic_minute)
        })

    return markets


def select_market_for_training(markets, safe_print_func=print):
    """
    Prompt user to select a market for training.

    Args:
        markets: List of market dicts from detect_available_markets()
        safe_print_func: Print function to use (for Windows compatibility)

    Returns:
        Tuple of (selected market dict, MarketSpecification object), or (None, None) if cancelled
    """
    from market_specs import get_market_spec

    if not markets:
        safe_print_func("\n[ERROR] No market data files found!")
        safe_print_func("[ERROR] Please run data processing first to create training data.")
        return None, None

    if len(markets) == 1:
        # Only one dataset - use it automatically
        market = markets[0]
        market_spec = get_market_spec(market['market'])

        safe_print_func(f"\n[DATA] Auto-detected: {market['market']} (only dataset available)")
        safe_print_func(f"[DATA] Using: {market['minute_file']}")
        if market['has_second']:
            safe_print_func(f"[DATA] Second-level: {market['second_file']}")

        if market_spec:
            safe_print_func(f"[MARKET] {market_spec.name}")
            safe_print_func(f"[MARKET] Multiplier: ${market_spec.contract_multiplier} | "
                          f"Tick: {market_spec.tick_size} | "
                          f"Tick Value: ${market_spec.tick_value:.2f} | "
                          f"Commission: ${market_spec.commission}")

        return market, market_spec

    # Multiple datasets - prompt user
    safe_print_func("\n" + "=" * 80)
    safe_print_func("MARKET SELECTION")
    safe_print_func("=" * 80)
    safe_print_func(f"\nDetected {len(markets)} market datasets:\n")

    for i, market in enumerate(markets, 1):
        second_status = "[OK]" if market['has_second'] else "[MINUTE ONLY]"
        market_spec = get_market_spec(market['market'])

        if market_spec:
            spec_info = f"(${market_spec.contract_multiplier} x {market_spec.tick_size} tick = ${market_spec.tick_value:.2f})"
        else:
            spec_info = "(unknown specs)"

        safe_print_func(f"  {i}. {market['market']:<8} - {market['minute_file']:<20} {second_status:<15} {spec_info}")

    safe_print_func("\n" + "=" * 80)

    while True:
        try:
            choice = input("\nSelect market number (or 'q' to quit): ").strip()

            if choice.lower() == 'q':
                safe_print_func("\n[INFO] Training cancelled by user")
                return None, None

            idx = int(choice) - 1
            if 0 <= idx < len(markets):
                selected = markets[idx]
                market_spec = get_market_spec(selected['market'])

                safe_print_func(f"\n[DATA] Selected: {selected['market']}")
                safe_print_func(f"[DATA] Minute data: {selected['minute_file']}")
                if selected['has_second']:
                    safe_print_func(f"[DATA] Second data: {selected['second_file']}")
                else:
                    safe_print_func(f"[DATA] Note: No second-level data available for {selected['market']}")

                if market_spec:
                    safe_print_func(f"\n[MARKET] {market_spec.name}")
                    safe_print_func(f"[MARKET] Contract Multiplier: ${market_spec.contract_multiplier}")
                    safe_print_func(f"[MARKET] Tick Size: {market_spec.tick_size} points")
                    safe_print_func(f"[MARKET] Tick Value: ${market_spec.tick_value:.2f}")
                    safe_print_func(f"[MARKET] Default Commission: ${market_spec.commission}/side")
                    safe_print_func(f"[MARKET] Slippage Model: {market_spec.slippage_ticks} tick(s)")
                else:
                    safe_print_func(f"[WARNING] Unknown market specs for {selected['market']}, using ES defaults")

                return selected, market_spec
            else:
                safe_print_func(f"[ERROR] Invalid choice. Please enter 1-{len(markets)}")
        except ValueError:
            safe_print_func("[ERROR] Invalid input. Please enter a number or 'q'")
        except KeyboardInterrupt:
            safe_print_func("\n[INFO] Training cancelled by user")
            return None, None


if __name__ == "__main__":
    # Test the detection function
    print("Testing model detection...")
    models = detect_models_in_folder()

    if models:
        print(f"\nFound {len(models)} model(s):")
        for model in models:
            print(f"  - {model['name']} ({model['type']}) - {model['modified_str']}")
    else:
        print("No models found.")

    # Test display
    if models:
        display_model_selection(models)
