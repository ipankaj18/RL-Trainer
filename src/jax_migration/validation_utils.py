"""Validation utilities for JAX training compatibility."""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, List
from pathlib import Path


def validate_checkpoint_compatibility(
    checkpoint_path: str,
    expected_obs_shape: Tuple[int, ...],
    num_actions: int
) -> Dict[str, any]:
    """
    Validate checkpoint compatibility with current training config.
    
    Args:
        checkpoint_path: Path to the Phase 1 checkpoint
        expected_obs_shape: Expected observation shape for Phase 2, e.g. (228,)
        num_actions: Expected number of actions in Phase 2 (usually 6)
    
    Returns:
        Dict with keys: 'compatible', 'obs_shape', 'num_actions', 'warnings', 'errors'
    """
    from flax.training import checkpoints
    import flax.core
    
    try:
        # Try multiple methods to load checkpoint
        ckpt = None
        
        # Method 1: Try as a directory
        import os
        if os.path.isdir(checkpoint_path):
            ckpt = checkpoints.restore_checkpoint(checkpoint_path, target=None)
        
        # Method 2: Try with prefix (e.g., "models/phase1_jax/phase1_jax_final")
        if ckpt is None and not os.path.isdir(checkpoint_path):
            # Extract directory and prefix
            parts = Path(checkpoint_path)
            if parts.parent.exists():
                prefix = parts.name
                ckpt = checkpoints.restore_checkpoint(
                    ckpt_dir=str(parts.parent),
                    target=None,
                    prefix=prefix
                )
        
        # Method 3: Try as exact file path
        if ckpt is None and os.path.isfile(checkpoint_path):
            import pickle
            with open(checkpoint_path, 'rb') as f:
                ckpt = pickle.load(f)
        
        if not ckpt or 'params' not in ckpt:
            # Provide helpful error message
            available_files = []
            if os.path.exists(str(Path(checkpoint_path).parent)):
                import glob
                pattern = str(Path(checkpoint_path).parent / "checkpoint_*")
                available_files = glob.glob(pattern)
            
            error_msg = f'Invalid checkpoint structure at {checkpoint_path}'
            if available_files:
                error_msg += f'. Found checkpoints: {", ".join([Path(f).name for f in available_files[:3]])}'
            
            return {
                'compatible': False,
                'error': error_msg,
                'warnings': []
            }
        
        params = ckpt['params']
        
        # Handle nested params
        if 'params' in params and isinstance(params['params'], (dict, flax.core.FrozenDict)):
            params = params['params']
        
        # Check Dense_0 (observation input layer)
        if 'Dense_0' not in params:
            return {
                'compatible': False,
                'error': 'Missing Dense_0 layer in checkpoint',
                'warnings': []
            }
        
        kernel_shape = params['Dense_0']['kernel'].shape
        ckpt_obs_dim, hidden_dim = kernel_shape
        
        # Check action head
        if 'Dense_3' not in params:
            return {
                'compatible': False,
                'error': 'Missing Dense_3 (action head) in checkpoint',
                'warnings': []
            }
        
        action_kernel = params['Dense_3']['kernel'].shape
        ckpt_num_actions = action_kernel[1]
        
        # Determine compatibility
        # Can pad if checkpoint has fewer observation dims
        obs_compatible = (ckpt_obs_dim <= expected_obs_shape[0])
        
        # Can expand if checkpoint has fewer actions
        action_compatible = (ckpt_num_actions <= num_actions)
        
        warnings = []
        if ckpt_obs_dim != expected_obs_shape[0]:
            if ckpt_obs_dim < expected_obs_shape[0]:
                warnings.append(
                    f"Observation shape mismatch: checkpoint={ckpt_obs_dim}, "
                    f"expected={expected_obs_shape[0]}. Will pad with {expected_obs_shape[0] - ckpt_obs_dim} new dims."
                )
            else:
                return {
                    'compatible': False,
                    'error': f"Checkpoint has more observation dims ({ckpt_obs_dim}) than expected ({expected_obs_shape[0]})",
                    'warnings': warnings
                }
        
        if ckpt_num_actions != num_actions:
            if ckpt_num_actions < num_actions:
                warnings.append(
                    f"Action space mismatch: checkpoint={ckpt_num_actions}, "
                    f"expected={num_actions}. Will expand to {num_actions} actions."
                )
            else:
                return {
                    'compatible': False,
                    'error': f"Checkpoint has more actions ({ckpt_num_actions}) than expected ({num_actions})",
                    'warnings': warnings
                }
        
        return {
            'compatible': obs_compatible and action_compatible,
            'obs_shape': (ckpt_obs_dim,),
            'num_actions': ckpt_num_actions,
            'warnings': warnings,
            'kernel_shape': kernel_shape
        }
    
    except Exception as e:
        return {
            'compatible': False,
            'error': f"Failed to load checkpoint: {str(e)}",
            'warnings': []
        }


def compute_observation_shape(env_params) -> int:
    """
    Compute expected observation shape from environment parameters.
    
    This ensures consistency across training scripts.
    
    Args:
        env_params: EnvParams or EnvParamsPhase2 instance
        
    Returns:
        Total observation dimension (int)
    """
    # Market window features
    market_dims = env_params.window_size * env_params.num_features
    
    # Position features (always 5 in both phases)
    # 1. position type, 2. entry ratio, 3. SL distance,
    # 4. TP distance, 5. time in position
    position_dims = 5
    
    # Phase 2 additional features (if present)
    phase2_dims = 0
    if hasattr(env_params, 'trail_activation_mult'):
        # Has Phase 2 specific params
        # FIX (2025-12-10): Updated from 6 to 8 for Apex compliance features
        # 8 features total:
        #   - 5 PM state: trailing_stop_active, unrealized_pnl, be_move_count, 
        #                 drawdown_ratio (NEW), drawdown_room (NEW)
        #   - 3 validity: can_enter, can_manage, has_position
        phase2_dims = 8
    
    total = market_dims + position_dims + phase2_dims
    return total


def validate_training_config(
    env_params,
    config,
    phase1_checkpoint: Optional[str] = None
) -> Dict[str, any]:
    """
    Comprehensive validation of training configuration.
    
    Args:
        env_params: Environment parameters
        config: PPO configuration
        phase1_checkpoint: Optional Phase 1 checkpoint path for transfer learning
        
    Returns:
        Dict with validation results and warnings
    """
    warnings = []
    errors = []
    
    # Validate observation shape computation
    expected_obs_dim = compute_observation_shape(env_params)
    
    # Check if batch size is reasonable
    batch_size = config.num_envs * config.num_steps
    if config.total_timesteps < batch_size:
        warnings.append(
            f"Total timesteps ({config.total_timesteps}) < Batch size ({batch_size}). "
            f"Will run only {max(1, config.total_timesteps // batch_size)} update(s)."
        )
    
    # Validate checkpoint compatibility if provided
    ckpt_validation = None
    if phase1_checkpoint:
        # Determine expected number of actions
        num_actions = 6  # Phase 2 has 6 actions
        
        ckpt_validation = validate_checkpoint_compatibility(
            phase1_checkpoint,
            expected_obs_shape=(expected_obs_dim,),
            num_actions=num_actions
        )
        
        if not ckpt_validation['compatible']:
            errors.append(f"Checkpoint incompatible: {ckpt_validation.get('error', 'Unknown error')}")
        else:
            warnings.extend(ckpt_validation.get('warnings', []))
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'expected_obs_dim': expected_obs_dim,
        'checkpoint_validation': ckpt_validation
    }
