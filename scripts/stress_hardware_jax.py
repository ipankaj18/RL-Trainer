#!/usr/bin/env python3
"""
JAX Hardware Stress Test and Auto-tuner - ENHANCED VERSION

Finds optimal GPU utilization (90%+) without sacrificing training quality.

Features:
- Real-time GPU monitoring (utilization, memory, temperature)
- Training quality validation (entropy, returns, losses)
- Composite scoring (GPU + quality + SPS)
- Patience-based early stopping
- Phase 1 & Phase 2 support
- Multiple profile generation (balanced, max-gpu, max-quality)
- Detailed CSV logging

Usage:
    # Quick test with dummy data
    python scripts/stress_hardware_jax.py --phase 1 --max-runs 5

    # Phase 2 with real data
    python scripts/stress_hardware_jax.py --phase 2 --market NQ --use-real-data

    # Validate existing profile
    python scripts/stress_hardware_jax.py --validate-profile config/hardware_profiles/balanced.yaml
"""

import os
import sys

# ============================================================================
# CRITICAL: XLA/TensorFlow Thread Control + Backend Selection
# ============================================================================
# Must be set BEFORE JAX import to prevent pthread_create failures during
# compilation. Even with "unlimited" ulimit, kernel limits apply:
# - /proc/sys/kernel/threads-max (system-wide thread limit)
# - /proc/sys/kernel/pid_max (max PIDs, threads count as PIDs)
#
# JAX/XLA creates large temporary thread pools during optimization passes
# (HloConstantFolding, LLVM compilation, etc.) that can exceed limits.
#
# Conservative limit of 4 threads prevents pthread failures on all platforms
# while maintaining acceptable compilation performance.
#
# Backend Selection:
# - Use 'cuda' (not 'gpu') to explicitly target NVIDIA GPUs
# - Generic 'gpu' can trigger ROCm (AMD) detection on mixed systems
# - This project is designed for NVIDIA GPUs (RTX series)
# ============================================================================

_MAX_TF_THREADS = 4  # Conservative limit for stability

os.environ['TF_NUM_INTEROP_THREADS'] = str(_MAX_TF_THREADS)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(_MAX_TF_THREADS)
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false'
os.environ['JAX_PLATFORMS'] = 'cuda'  # Force NVIDIA CUDA backend explicitly

# Standard thread pool control
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import csv
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Import JAX modules
try:
    import jax
    import jax.numpy as jnp
    from jax_migration.train_ppo_jax_fixed import train, PPOConfig
    from jax_migration.train_phase2_jax import train_phase2
    from jax_migration.env_phase1_jax import EnvParams
    from jax_migration.env_phase2_jax import EnvParamsPhase2
    from jax_migration.data_loader import MarketData, load_market_data
    from jax_migration.gpu_monitor import GPUMonitor
    from jax_migration.training_quality_validator import TrainingQualityValidator
    from jax_migration.gpu_name_utils import get_gpu_name
except ImportError as e:
    print(f"Error importing JAX modules: {e}")
    print("Ensure you have installed the JAX dependencies.")
    print("Install missing packages: pip install pynvml>=11.5.0")
    sys.exit(1)

# Extended search combo: (num_envs, num_steps, num_minibatches, num_epochs)
SearchCombo = Tuple[int, int, int, int]

def get_safe_env_limits() -> Tuple[int, List[int]]:
    """
    Detect system process limits and recommend safe env counts.
    
    This function adapts the search space based on system configuration:
    - Cloud platforms (unlimited ulimit):
        - High-End (>32 cores): Aggressive [1024, ..., 16384]
        - Mid-Range (16-32 cores): Standard [512, ..., 4096]
        - Low-End (<16 cores): Conservative [128, ..., 512]
    - Workstations (capped ulimit): Aggressive [512, 1024, ...] based on limit
    
    Returns:
        Tuple of (max_safe_envs, recommended_env_array)
    """
    import subprocess
    try:
        import resource
        has_resource = True
    except ImportError:
        has_resource = False
    
    # Get CPU core count for hardware tier detection
    try:
        cpu_count = os.cpu_count() or 4
    except:
        cpu_count = 4

    # Get current process limits
    soft_limit = None
    try:
        if has_resource:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NPROC)
        else:
            # Fallback: use ulimit command
            result = subprocess.run(['bash', '-c', 'ulimit -u'], 
                                  capture_output=True, text=True, timeout=1)
            if result.returncode == 0 and result.stdout.strip() != 'unlimited':
                soft_limit = int(result.stdout.strip())
    except Exception as e:
        print(f"[WARN] Could not determine process limit: {e}")
        print(f"[WARN] Defaulting to conservative env counts")
        return (192, [64, 128, 192])
    
    # Check if unlimited (common on cloud platforms)
    if soft_limit is None or soft_limit == -1 or soft_limit > 1000000:
        print(f"\n[PLATFORM DETECTION]")
        print(f"Process limit: unlimited (cloud platform detected)")
        print(f"CPU Cores: {cpu_count}")
        
        # Smart Hardware Tier Detection
        if cpu_count > 32:
            # High-End (e.g., H100/A100 instances, Threadripper)
            print(f"Hardware Tier: HIGH-PERFORMANCE (Aggressive Scaling)")
            print(f"Safe max envs: 16384")
            print(f"Search space: [1024, 2048, 4096, 8192, 16384]")
            return (16384, [1024, 2048, 4096, 8192, 16384])
            
        elif cpu_count >= 16:
            # Mid-Range (e.g., Standard Cloud Instances, Desktop)
            print(f"Hardware Tier: MID-RANGE (Standard Scaling)")
            print(f"Safe max envs: 4096")
            print(f"Search space: [512, 1024, 2048, 4096]")
            return (4096, [512, 1024, 2048, 4096])
            
        else:
            # Low-End (e.g., Laptops, Small Instances)
            print(f"Hardware Tier: CONSUMER/ENTRY (Conservative Scaling)")
            print(f"Safe max envs: 512")
            print(f"Search space: [128, 256, 512]")
            print(f"Note: Limited by CPU core count ({cpu_count}) to prevent system freeze")
            return (512, [128, 256, 512])
    
    # Calculate safe max based on actual limit
    # Rule of thumb: ~25 threads per env, use 75% of limit for safety
    safe_limit = int(soft_limit * 0.75)
    max_safe_envs = int(safe_limit // 25)
    
    # Build adaptive env array based on max_safe_envs
    if max_safe_envs >= 4096:
        env_array = [512, 1024, 2048, 4096, 8192, 12288, 16384]
    elif max_safe_envs >= 2048:
        env_array = [512, 1024, 2048, 3072]
    elif max_safe_envs >= 512:
        env_array = [256, 512, 1024]
    elif max_safe_envs >= 192:
        env_array = [128, 192, 256]
    else:
        env_array = [64, 128]
    
    # Filter to only include values <= max_safe_envs
    env_array = [e for e in env_array if e <= max_safe_envs]
    
    print(f"\n[PLATFORM DETECTION]")
    print(f"Process limit: {soft_limit}")
    print(f"Safe max envs: {max_safe_envs}")
    print(f"Search space: {env_array} (based on system limit)\n")
    
    return (max_safe_envs, env_array)

def build_search_space(
    num_envs_options: List[int],
    num_steps_options: List[int] = None,
    num_minibatches_options: List[int] = None,
    num_epochs_options: List[int] = None
) -> List[SearchCombo]:
    """
    Return an ordered search space from lighter to heavier loads.
    JAX can handle much larger batch sizes than PyTorch.

    Args:
        num_envs_options: Environment counts to test (default: [512, 1024, ..., 16384])
        num_steps_options: Steps per rollout (default: [128, 256, 384, 512])
        num_minibatches_options: Minibatch splits (default: [4, 8, 16])
        num_epochs_options: Training epochs (default: [4, 8])

    Returns:
        List of (num_envs, num_steps, num_minibatches, num_epochs) tuples
    """
    # num_envs_options is now REQUIRED - caller must provide adaptive array

    if num_steps_options is None:
        num_steps_options = [128, 256, 384, 512]

    if num_minibatches_options is None:
        num_minibatches_options = [4, 8, 16]

    if num_epochs_options is None:
        num_epochs_options = [4, 8]

    # Generate all combinations, sorted by rough computational load
    combos = []
    for envs in num_envs_options:
        for steps in num_steps_options:
            for minibatches in num_minibatches_options:
                for epochs in num_epochs_options:
                    combos.append((envs, steps, minibatches, epochs))

    # Sort by ascending load (batch_size * epochs)
    combos.sort(key=lambda x: x[0] * x[1] * x[3])

    return combos

def create_dummy_data(num_timesteps: int = 100000, phase: int = 1) -> MarketData:
    """
    Create dummy market data for testing.

    Args:
        num_timesteps: Number of timesteps to generate
        phase: Training phase (1 or 2) - Phase 2 needs additional fields

    Returns:
        MarketData with dummy values
    """
    key = jax.random.key(0)
    num_features = 8  # Default for EnvParams

    data = MarketData(
        features=jax.random.normal(key, (num_timesteps, num_features)),
        prices=jnp.abs(jax.random.normal(key, (num_timesteps, 4))) * 100 + 5000,
        atr=jnp.abs(jax.random.normal(key, (num_timesteps,))) * 10 + 5,
        time_features=jax.random.uniform(key, (num_timesteps, 3)),
        trading_mask=jnp.ones(num_timesteps),
        timestamps_hour=jnp.linspace(9.5, 16.9, num_timesteps),
        # Phase 2 additional fields
        rth_indices=jnp.arange(60, num_timesteps - 100) if phase == 2 else jnp.array([60]),
        low_s=jnp.abs(jax.random.normal(key, (num_timesteps,))) * 100 + 4990,
        high_s=jnp.abs(jax.random.normal(key, (num_timesteps,))) * 100 + 5010,
    )

    return data


def calculate_gpu_score(gpu_util: float, target: float = 90.0) -> float:
    """
    Calculate GPU utilization score.

    Target is 90% utilization. Penalize both under-utilization and over-saturation.

    Args:
        gpu_util: GPU utilization percentage (0-100)
        target: Target utilization percentage

    Returns:
        Score between 0.0 and 1.0
    """
    if gpu_util < 80.0:
        # Under-utilized: linear penalty
        return gpu_util / target
    elif gpu_util <= 95.0:
        # Sweet spot: full score
        return 1.0
    else:
        # Over-saturated: exponential penalty
        penalty = (gpu_util - 95.0) / 5.0  # 95-100% range
        return max(0.0, 1.0 - penalty * 2.0)


def normalize_sps(sps: float, reference_sps: float = 50000.0) -> float:
    """
    Normalize SPS to 0-1 scale.

    Args:
        sps: Steps per second achieved
        reference_sps: Reference SPS for scaling (default 50k)

    Returns:
        Normalized score (0-1, can exceed 1.0 for exceptional performance)
    """
    return min(sps / reference_sps, 1.5)


def normalize_quality(mean_return: float, entropy: float) -> float:
    """
    Normalize training quality to 0-1 scale.

    Args:
        mean_return: Mean episode return
        entropy: Policy entropy

    Returns:
        Normalized quality score (0-1)
    """
    # Entropy component (0.6 weight)
    # Target: 0.05-0.2 is good
    if entropy < 0.05:
        entropy_score = 0.0  # Collapsed
    elif entropy <= 0.2:
        entropy_score = 1.0  # Ideal
    else:
        entropy_score = max(0.0, 1.0 - (entropy - 0.2) / 0.3)

    # Return component (0.4 weight)
    if mean_return >= 0:
        return_score = 1.0
    else:
        # Scale: -100 = 0.5, -1000 = 0.0
        return_score = max(0.0, 1.0 + mean_return / 1000.0)

    return 0.6 * entropy_score + 0.4 * return_score


def test_process_limits_before_training(num_envs: int) -> Dict[str, any]:
    """
    Test if system can handle the thread/process load BEFORE training.
    
    JAX/LLVM creates many threads during compilation. This pre-check prevents:
    - pthread_create failures
    - fork bombs
    - Resource exhaustion crashes
    
    Args:
        num_envs: Number of parallel environments to test
    
    Returns:
        Dict with 'safe': bool, 'max_recommended_envs': int, 'reason': str
    """
    import subprocess
    try:
        import resource
        has_resource = True
    except ImportError:
        has_resource = False
    
    # Get current process limits
    soft_limit = None
    try:
        if has_resource:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NPROC)
        else:
            # Fallback: use ulimit command
            result = subprocess.run(['bash', '-c', 'ulimit -u'], 
                                  capture_output=True, text=True, timeout=1)
            if result.returncode == 0 and result.stdout.strip() != 'unlimited':
                soft_limit = int(result.stdout.strip())
    except Exception as e:
        print(f"  [WARN] Could not determine process limit: {e}")
    
    # Estimate threads needed
    # Rule of thumb: JAX uses ~20 threads per environment during compilation
    # LLVM also creates worker threads: ~num_envs * 1.5 additional threads
    estimated_threads = int(num_envs * 25)  # Conservative estimate
    
    # Check if unlimited
    if soft_limit is None or soft_limit == -1 or soft_limit > 1000000:
        # Smart Hardware Check
        try:
            cpu_count = os.cpu_count() or 4
        except:
            cpu_count = 4
            
        # If high-end hardware (>32 cores), allow massive scaling
        if cpu_count > 32:
             print(f"  [INFO] Unlimited process limit + High-End CPU ({cpu_count} cores)")
             print(f"  [INFO] Allowing aggressive scaling (estimated {estimated_threads} threads)")
             return {
                'safe': True,
                'max_recommended_envs': num_envs,
                'reason': f'High-Performance Tier ({cpu_count} cores, unlimited processes)'
            }
        
        # If mid-range (16-32 cores), allow standard scaling up to 4096
        elif cpu_count >= 16:
            if num_envs <= 4096:
                return {
                    'safe': True,
                    'max_recommended_envs': num_envs,
                    'reason': f'Mid-Range Tier ({cpu_count} cores)'
                }
            else:
                 return {
                    'safe': False,
                    'max_recommended_envs': 4096,
                    'reason': f'Exceeds safe limit for Mid-Range CPU ({cpu_count} cores). Max: 4096'
                }
                
        # If low-end, cap at 512 for safety
        else:
            if num_envs <= 512:
                 return {
                    'safe': True,
                    'max_recommended_envs': num_envs,
                    'reason': f'Consumer Tier ({cpu_count} cores)'
                }
            else:
                return {
                    'safe': False,
                    'max_recommended_envs': 512,
                    'reason': f'Exceeds safe limit for Consumer CPU ({cpu_count} cores). Max: 512'
                }
    
    # Check if we're within 75% of limit (safety margin)
    safe_limit = int(soft_limit * 0.75)
    
    if estimated_threads > safe_limit:
        max_safe_envs = int(safe_limit // 25)
        print(f"  [FAIL] Process limit ({soft_limit}) too low for {num_envs} envs!")
        print(f"  [FAIL] Estimated {estimated_threads} threads > {safe_limit} safe limit")
        print(f"  [FAIL] Increase limit: ulimit -u {estimated_threads * 2}")
        return {
            'safe': False,
            'max_recommended_envs': max_safe_envs,
            'reason': f'Would exceed process limit (ulimit -u: {soft_limit}, need ~{estimated_threads} threads)'
        }
    
    return {
        'safe': True, 
        'max_recommended_envs': num_envs,
        'reason': f'OK (limit: {soft_limit}, estimated: {estimated_threads} threads)'
    }

def run_combo(
    combo: SearchCombo,
    data: MarketData,
    phase: int = 1,
    market: str = "TEST",
    timeout: int = 300
) -> Dict[str, float]:
    """
    Execute a single validation run with full monitoring.

    Args:
        combo: (num_envs, num_steps, num_minibatches, num_epochs)
        data: Market data
        phase: Training phase (1 or 2)
        market: Market symbol for logging
        timeout: Maximum runtime in seconds

    Returns:
        Dictionary with all metrics and scores
    """
    num_envs, num_steps, num_minibatches, num_epochs = combo

    # CRITICAL FIX: Run 200 updates to detect thread/process exhaustion
    # Real training runs 400-600 updates. Stress test needs to be realistic.
    # Previous 50 updates only tested 8-12% of workload, missing pthread_create failures
    # that manifest at ~130 updates when LLVM compiler thread pool is exhausted
    total_timesteps = num_envs * num_steps * 200

    config = PPOConfig(
        num_envs=num_envs,
        num_steps=num_steps,
        total_timesteps=total_timesteps,
        normalize_obs=True,
        num_minibatches=num_minibatches,
        num_epochs=num_epochs
    )

    # Select environment params based on phase
    if phase == 1:
        env_params = EnvParams(
            rth_start_count=int(data.rth_indices.shape[0])
        )
        train_fn = train
    else:
        env_params = EnvParamsPhase2()
        train_fn = lambda cfg, params, d, seed, market, checkpoint_dir: train_phase2(
            cfg, params, d, phase1_checkpoint=None,
            checkpoint_dir=checkpoint_dir, market=market, seed=seed
        )[0]  # Return only runner_state

    print(f"  Running: Envs={num_envs}, Steps={num_steps}, MB={num_minibatches}, Epochs={num_epochs}...")

    # NEW: Pre-check process/thread limits BEFORE training
    limit_check = test_process_limits_before_training(num_envs)
    if not limit_check['safe']:
        print(f"  [SKIP] {limit_check['reason']}")
        print(f"  [SKIP] Recommended max: {limit_check['max_recommended_envs']} envs")
        return {
            "num_envs": num_envs,
            "num_steps": num_steps,
            "num_minibatches": num_minibatches,
            "num_epochs": num_epochs,
            "gpu_util": 0.0,
            "peak_memory_gb": 0.0,
            "total_memory_gb": 0.0,
            "avg_temperature": 0.0,
            "mean_return": 0.0,
            "entropy": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "approx_kl": 0.0,
            "sps": 0.0,
            "duration": 0.0,
            "gpu_score": 0.0,
            "quality_score": 0.0,
            "sps_score": 0.0,
            "final_score": -100.0,
            "status": "skipped_thread_limit",
            "quality_acceptable": False,
            "error": limit_check['reason']
        }
    else:
        print(f"  [OK] {limit_check['reason']}")

    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(device_id=0, interval=0.5)
    quality_validator = TrainingQualityValidator()

    start_time = time.time()
    status = "success"
    error_msg = None

    try:
        # Start GPU monitoring
        gpu_monitor.start_monitoring()

        # Run training
        if phase == 1:
            trained_state, normalizer, metrics = train(
                config, env_params, data, seed=42,
                market=market, checkpoint_dir="/tmp/jax_stress_test"
            )
        else:
            # Phase 2 training
            from pathlib import Path
            temp_dir = Path("/tmp/jax_stress_test_phase2")
            temp_dir.mkdir(parents=True, exist_ok=True)

            runner_state = train_phase2(
                config, env_params, data,
                phase1_checkpoint=None,  # No transfer for stress test
                checkpoint_dir=str(temp_dir),
                market=market,
                seed=42
            )
            # Extract metrics (Phase 2 doesn't return metrics list yet)
            metrics = []

        # Stop GPU monitoring
        gpu_monitor.stop_monitoring()

        duration = time.time() - start_time

        # Check timeout
        if duration > timeout:
            status = "timeout"
            print(f"  [WARN] Run exceeded timeout ({timeout}s)")

        # Get GPU stats
        gpu_stats = gpu_monitor.get_stats()

        # Extract training metrics
        if metrics:
            final_metric = metrics[-1]
            mean_return = final_metric.get('mean_return', 0.0)
            entropy = final_metric.get('entropy', 0.0)
            policy_loss = final_metric.get('policy_loss', 0.0)
            value_loss = final_metric.get('value_loss', 0.0)
            approx_kl = final_metric.get('approx_kl', 0.0)
            sps = final_metric.get('sps', 0.0)

            # Validate quality
            quality_validator.add_metrics(
                mean_return=mean_return,
                entropy=entropy,
                policy_loss=policy_loss,
                value_loss=value_loss,
                approx_kl=approx_kl
            )
        else:
            # No metrics available (Phase 2 or failure)
            mean_return = 0.0
            entropy = 0.0
            policy_loss = 0.0
            value_loss = 0.0
            approx_kl = 0.0
            sps = total_timesteps / duration if duration > 0 else 0.0

        quality_acceptable = quality_validator.is_quality_acceptable()
        quality_score_val = quality_validator.calculate_quality_score()

        # Calculate composite score
        gpu_util = gpu_stats['avg_utilization']
        gpu_score_val = calculate_gpu_score(gpu_util)
        sps_score_val = normalize_sps(sps)

        # Penalties
        oom_penalty = 0.0
        divergence_penalty = 0.0 if quality_acceptable else 50.0
        
        # FIXED: More conservative memory threshold (was 0.95)
        # Long training runs accumulate more memory than short stress tests
        memory_penalty = 0.0
        # GUARD: Prevent division by zero when GPU monitoring fails
        if gpu_stats['total_memory_gb'] > 0:
            if gpu_stats['peak_memory_gb'] > gpu_stats['total_memory_gb'] * 0.92:
                memory_penalty = 20.0  # Aggressive penalty - too close to limit
                print(f"  [WARN] Memory usage critical: {gpu_stats['peak_memory_gb']:.1f}/{gpu_stats['total_memory_gb']:.1f} GB")
            elif gpu_stats['peak_memory_gb'] > gpu_stats['total_memory_gb'] * 0.85:
                memory_penalty = 5.0   # Mild penalty - getting tight
        else:
            # GPU monitoring failed - no memory penalty
            print(f"  [WARN] GPU monitoring failed (total_memory_gb=0), skipping memory penalty")

        # Final composite score
        final_score = (
            0.4 * gpu_score_val +
            0.4 * quality_score_val +
            0.2 * sps_score_val -
            oom_penalty - divergence_penalty - memory_penalty
        )

        return {
            # Config
            "num_envs": num_envs,
            "num_steps": num_steps,
            "num_minibatches": num_minibatches,
            "num_epochs": num_epochs,
            # GPU stats
            "gpu_util": gpu_util,
            "peak_memory_gb": gpu_stats['peak_memory_gb'],
            "total_memory_gb": gpu_stats['total_memory_gb'],
            "avg_temperature": gpu_stats['avg_temperature'],
            # Training metrics
            "mean_return": mean_return,
            "entropy": entropy,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "approx_kl": approx_kl,
            "sps": sps,
            "duration": duration,
            # Scores
            "gpu_score": gpu_score_val,
            "quality_score": quality_score_val,
            "sps_score": sps_score_val,
            "final_score": final_score,
            # Status
            "status": status,
            "quality_acceptable": quality_acceptable
        }

    except Exception as e:
        gpu_monitor.stop_monitoring()
        error_msg = str(e)
        print(f"  [ERROR] Run failed: {e}")

        # Check for specific error types
        if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
            status = "oom"
        elif "pthread_create" in error_msg.lower() or ("resource temporarily unavailable" in error_msg.lower() and "llvm" in error_msg.lower()):
            status = "thread_limit_exceeded"
            print(f"  [CRITICAL] Hit system thread/process limit!")
            print(f"  [CRITICAL] Reduce num_envs or increase 'ulimit -u'")
            print(f"  [CRITICAL] Current config too heavy for system resources")
        else:
            status = "error"

        return {
            "num_envs": num_envs,
            "num_steps": num_steps,
            "num_minibatches": num_minibatches,
            "num_epochs": num_epochs,
            "gpu_util": 0.0,
            "peak_memory_gb": 0.0,
            "total_memory_gb": 0.0,
            "avg_temperature": 0.0,
            "mean_return": 0.0,
            "entropy": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "approx_kl": 0.0,
            "sps": 0.0,
            "duration": time.time() - start_time,
            "gpu_score": 0.0,
            "quality_score": 0.0,
            "sps_score": 0.0,
            "final_score": -100.0,
            "status": status,
            "quality_acceptable": False,
            "error": error_msg
        }

def save_profile(name: str, stats: Dict[str, float], phase: int = 1, gpu_prefix: str = "") -> Path:
    """
    Persist hardware profile to YAML.

    Args:
        name: Profile name (e.g., "balanced", "max_gpu", "max_quality")
        stats: Statistics dictionary from run_combo
        phase: Training phase (1 or 2)
        gpu_prefix: GPU name prefix (e.g., "RTXA6000") - will be prepended to filename
        stats: Statistics dictionary from run_combo
        phase: Training phase (1 or 2)

    Returns:
        Path to saved profile
    """
    profiles_dir = PROJECT_ROOT / "config" / "hardware_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    # Add GPU prefix to filename if provided
    if gpu_prefix:
        filename = f"{gpu_prefix}_{name}.yaml"
    else:
        filename = f"{name}.yaml"
    
    path = profiles_dir / filename

    # FIXED v2: No safety margin needed - stress test now runs 200 updates (realistic)
    # With proper thread limit checking and longer tests, the tested value IS safe
    tested_num_envs = int(stats.get("num_envs", 1024))
    safe_num_envs = tested_num_envs  # Use tested value directly

    payload = {
        "mode": "jax_hardware_maximized",
        "phase": phase,
        "num_envs": safe_num_envs,
        "tested_num_envs": tested_num_envs,
        "stress_test_updates": 200,  # Now tests 200 updates (33-50% of real workload)
        "safety_margin_applied": 1.0,  # No reduction - tested value is safe
        "num_steps": int(stats.get("num_steps", 128)),
        "num_minibatches": int(stats.get("num_minibatches", 4)),
        "num_epochs": int(stats.get("num_epochs", 4)),
        "device": "gpu" if jax.default_backend() == "gpu" else "cpu",
        # Performance metrics
        "expected_sps": float(stats.get("sps", 0)),
        "expected_gpu_util": float(stats.get("gpu_util", 0)),
        "expected_memory_gb": float(stats.get("peak_memory_gb", 0)),
        # Quality metrics
        "quality_score": float(stats.get("quality_score", 0)),
        "mean_return": float(stats.get("mean_return", 0)),
        "entropy": float(stats.get("entropy", 0)),
        # Metadata
        "notes": f"Auto-tuned via stress_hardware_jax.py on {datetime.now().strftime('%Y-%m-%d')}",
        "final_score": float(stats.get("final_score", 0))
    }

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    return path


def save_csv_log(results: List[Dict[str, float]], market: str, phase: int):
    """
    Save all test results to CSV for analysis.

    Args:
        results: List of result dictionaries from run_combo
        market: Market symbol
        phase: Training phase
    """
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = results_dir / f"jax_stress_test_{market}_phase{phase}_{timestamp}.csv"

    if not results:
        print(f"[WARN] No results to save")
        return

    # CSV columns
    fieldnames = [
        "combo", "num_envs", "num_steps", "num_minibatches", "num_epochs",
        "gpu_util", "peak_memory_gb", "avg_temperature",
        "mean_return", "entropy", "policy_loss", "value_loss", "approx_kl",
        "sps", "duration", "gpu_score", "quality_score", "sps_score",
        "final_score", "status", "quality_acceptable"
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for i, result in enumerate(results):
            row = result.copy()
            row['combo'] = i + 1
            writer.writerow(row)

    print(f"\n[SAVE] CSV log saved to: {csv_path}")

def format_table_row(text: str, width: int = 78, align: str = "left") -> str:
    """
    Format a table row with proper padding and borders.

    Args:
        text: The text content (without borders)
        width: Total width including padding (default 78)
        align: Alignment ('left', 'center', 'right')

    Returns:
        Formatted row with borders: "║ <padded text> ║"
    """
    content_width = width - 2  # Account for spaces after borders

    if align == "center":
        padded = text.center(content_width)
    elif align == "right":
        padded = text.rjust(content_width)
    else:  # left
        padded = text.ljust(content_width)

    return f"║ {padded} ║"


def print_summary_table(profiles: Dict[str, Dict[str, float]]):
    """Print beautiful summary table with all profiles."""
    TABLE_WIDTH = 78

    print("\n" + "╔" + "═" * TABLE_WIDTH + "╗")
    print(format_table_row("OPTIMAL PROFILES SUMMARY", TABLE_WIDTH, "center"))
    print("╠" + "═" * TABLE_WIDTH + "╣")

    for profile_name, stats in profiles.items():
        # Profile name header
        print(format_table_row(profile_name.upper(), TABLE_WIDTH, "center"))
        print("╟" + "─" * TABLE_WIDTH + "╢")

        # Config row
        config_text = (f"Config: Envs={stats['num_envs']:,} | Steps={stats['num_steps']} | "
                      f"MB={stats['num_minibatches']} | Epochs={stats['num_epochs']}")
        print(format_table_row(config_text, TABLE_WIDTH, "left"))

        # GPU row
        gpu_text = (f"GPU:    {stats['gpu_util']:5.1f}% util | {stats['peak_memory_gb']:4.1f} GB mem | "
                   f"{stats['avg_temperature']:4.1f}°C temp")
        print(format_table_row(gpu_text, TABLE_WIDTH, "left"))

        # Training row
        train_text = (f"Train:  Return={stats['mean_return']:6.1f} | Entropy={stats['entropy']:5.3f} | "
                     f"SPS={stats['sps']:,.0f}")
        print(format_table_row(train_text, TABLE_WIDTH, "left"))

        # Score row
        score_text = (f"Score:  {stats['final_score']:5.2f} (GPU={stats['gpu_score']:.2f} | "
                     f"Quality={stats['quality_score']:.2f} | SPS={stats['sps_score']:.2f})")
        print(format_table_row(score_text, TABLE_WIDTH, "left"))

        print("╟" + "─" * TABLE_WIDTH + "╢")

    print("╚" + "═" * TABLE_WIDTH + "╝")


def main():
    parser = argparse.ArgumentParser(
        description="JAX Hardware Stress Test & Auto-tuner - Find optimal GPU utilization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1 quick test
  python scripts/stress_hardware_jax.py --phase 1 --max-runs 5

  # Phase 2 with real data
  python scripts/stress_hardware_jax.py --phase 2 --market NQ --use-real-data --max-runs 10

  # Validate existing profile
  python scripts/stress_hardware_jax.py --validate-profile config/hardware_profiles/balanced.yaml
        """
    )

    # Core arguments
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1,
                        help="Training phase to test (1 or 2)")
    parser.add_argument("--market", type=str, default="TEST",
                        help="Market symbol (e.g., NQ, ES)")
    parser.add_argument("--use-real-data", action="store_true",
                        help="Use real market data instead of dummy data")
    parser.add_argument("--data-path", type=str,
                        help="Path to market data CSV (auto-detects if --use-real-data)")

    # Search parameters
    parser.add_argument("--max-runs", type=int, default=10,
                        help="Maximum number of combos to test")
    parser.add_argument("--patience", type=int, default=3,
                        help="Stop after N runs without improvement")
    parser.add_argument("--min-gain", type=float, default=0.5,
                        help="Minimum score improvement to reset patience")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per run in seconds")

    # Profile management
    parser.add_argument("--save-profiles", action="store_true", default=True,
                        help="Save top 3 profiles (balanced, max-gpu, max-quality)")
    parser.add_argument("--validate-profile", type=str,
                        help="Validate an existing profile YAML file")
    parser.add_argument("--gpu-name-override", type=str,
                        help="Override detected GPU name (also respects GPU_NAME_OVERRIDE env var)")

    args = parser.parse_args()

    # Validation mode
    if args.validate_profile:
        print("="*80)
        print("PROFILE VALIDATION MODE")
        print("="*80)
        print(f"Loading profile: {args.validate_profile}")

        try:
            with open(args.validate_profile, 'r') as f:
                profile = yaml.safe_load(f)

            print(f"Profile config:")
            print(f"  Envs: {profile.get('num_envs')}")
            print(f"  Steps: {profile.get('num_steps')}")
            print(f"  Minibatches: {profile.get('num_minibatches', 4)}")
            print(f"  Epochs: {profile.get('num_epochs', 4)}")

            # Run single test
            phase = profile.get('phase', 1)
            data = create_dummy_data(100000, phase)

            combo = (
                profile.get('num_envs', 1024),
                profile.get('num_steps', 128),
                profile.get('num_minibatches', 4),
                profile.get('num_epochs', 4)
            )

            print(f"\nRunning validation test...")
            result = run_combo(combo, data, phase, args.market, args.timeout)

            print(f"\nValidation Results:")
            print(f"  GPU Utilization: {result['gpu_util']:.1f}% (expected: {profile.get('expected_gpu_util', 0):.1f}%)")
            print(f"  Memory: {result['peak_memory_gb']:.1f} GB (expected: {profile.get('expected_memory_gb', 0):.1f} GB)")
            print(f"  SPS: {result['sps']:,.0f} (expected: {profile.get('expected_sps', 0):,.0f})")
            print(f"  Status: {result['status']}")
            print(f"  Quality: {'✓' if result['quality_acceptable'] else '✗'}")

            return

        except Exception as e:
            print(f"[ERROR] Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return

    # Normal stress test mode
    print("="*80)
    print("JAX HARDWARE STRESS TEST & AUTO-TUNER - ENHANCED VERSION")
    print("="*80)
    print(f"JAX Backend: {jax.default_backend().upper()}")
    print(f"Devices: {jax.devices()}")
    print(f"Phase: {args.phase}")
    print(f"Market: {args.market}")
    print(f"Max runs: {args.max_runs}")
    print(f"Patience: {args.patience} (min gain: {args.min_gain})")
    print("-" * 80)
    
    # AUTO-FIX: Set reasonable process limit if unlimited
    # This prevents the conservative [64, 128, 192] search space on cloud platforms
    try:
        import resource
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NPROC)
        
        # If unlimited or extremely high, set to 500000 for this process
        if soft_limit == resource.RLIM_INFINITY or soft_limit > 100000:
            recommended_limit = 500000
            print(f"\n[AUTO-CONFIG] Process limit is unlimited")
            print(f"[AUTO-CONFIG] Setting to {recommended_limit} for optimal stress testing")
            
            # Try to set the limit
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (recommended_limit, hard_limit))
                # Verify it was set
                new_soft, _ = resource.getrlimit(resource.RLIMIT_NPROC)
                print(f"[AUTO-CONFIG] ✓ Process limit set to {new_soft}")
                print(f"[AUTO-CONFIG] This enables testing with 512-16384 envs for better GPU utilization")
            except (ValueError, OSError) as e:
                print(f"[AUTO-CONFIG] ⚠ Could not set process limit: {e}")
                print(f"[AUTO-CONFIG] Continuing with current limit (search space will be conservative)")
        else:
            print(f"\n[AUTO-CONFIG] Process limit: {soft_limit} (user-configured)")
            
        print()  # Blank line for readability
    except Exception as e:
        print(f"\n[AUTO-CONFIG] Could not detect/set process limit: {e}")
        print(f"[AUTO-CONFIG] Continuing with system defaults\n")

    # Prepare data
    if args.use_real_data:
        if not args.data_path:
            # Auto-detect data path
            data_path = PROJECT_ROOT / "data" / f"{args.market}_D1M.csv"
            if not data_path.exists():
                print(f"[ERROR] Market data not found: {data_path}")
                print("  Please specify --data-path or use --market with existing data")
                sys.exit(1)
            args.data_path = str(data_path)

        print(f"Loading real market data from {args.data_path}...")
        second_data_path = args.data_path.replace('_D1M.csv', '_D1S.csv')
        data = load_market_data(
            args.data_path,
            second_data_path=second_data_path if Path(second_data_path).exists() else None
        )
        
        # RTH-Aware Smart Subsetting for speed (50K timesteps)
        # CRITICAL FIX: Check for valid RTH indices BEFORE subsetting
        # If RTH starts occur >= 50000 (pre-market data fills first rows),
        # we need to create an RTH-aligned subset instead
        if data.features.shape[0] > 50000:
            # Check if we have valid RTH indices in the first 50K rows
            valid_rth_in_subset = data.rth_indices[data.rth_indices < 50000]
            
            if valid_rth_in_subset.shape[0] > 0:
                # Fast path: RTH indices exist in first 50K rows
                print(f"  Using subset (50K timesteps) for faster iteration")
                print(f"  RTH indices in subset: {valid_rth_in_subset.shape[0]} valid starts")
                data = MarketData(
                    features=data.features[:50000],
                    prices=data.prices[:50000],
                    atr=data.atr[:50000],
                    time_features=data.time_features[:50000],
                    trading_mask=data.trading_mask[:50000],
                    timestamps_hour=data.timestamps_hour[:50000],
                    rth_indices=jnp.asarray(valid_rth_in_subset),
                    low_s=data.low_s[:50000],
                    high_s=data.high_s[:50000]
                )
            else:
                # No RTH indices in first 50K - create RTH-aligned subset
                first_rth = int(data.rth_indices.min())
                total_rows = data.features.shape[0]
                
                if first_rth < total_rows - 50000:
                    # Center subset around RTH start (start 100 bars before first RTH)
                    start_idx = max(0, first_rth - 100)
                    end_idx = min(total_rows, start_idx + 50000)
                    
                    print(f"  [RTH-ALIGNED] First RTH at index {first_rth}, using subset [{start_idx}:{end_idx}]")
                    
                    # Filter and recompute relative RTH indices for the subset
                    subset_rth_mask = (data.rth_indices >= start_idx) & (data.rth_indices < end_idx)
                    subset_rth = data.rth_indices[subset_rth_mask]
                    relative_rth = subset_rth - start_idx  # Shift to 0-based for subset
                    
                    print(f"  RTH indices in subset: {relative_rth.shape[0]} valid starts")
                    
                    data = MarketData(
                        features=data.features[start_idx:end_idx],
                        prices=data.prices[start_idx:end_idx],
                        atr=data.atr[start_idx:end_idx],
                        time_features=data.time_features[start_idx:end_idx],
                        trading_mask=data.trading_mask[start_idx:end_idx],
                        timestamps_hour=data.timestamps_hour[start_idx:end_idx],
                        rth_indices=jnp.asarray(relative_rth),
                        low_s=data.low_s[start_idx:end_idx],
                        high_s=data.high_s[start_idx:end_idx]
                    )
                else:
                    # RTH starts too late - use full dataset
                    print(f"  [WARN] RTH starts at index {first_rth}, too late to subset. Using full dataset.")
                    print(f"  RTH indices: {data.rth_indices.shape[0]} valid starts")
        else:
            print(f"  Using full dataset ({data.features.shape[0]} timesteps)")
            print(f"  RTH indices: {data.rth_indices.shape[0]} valid starts")
        
        # Post-validation: Ensure we have valid RTH indices
        if data.rth_indices.shape[0] == 0:
            print(f"  [CRITICAL ERROR] No RTH indices after subsetting!")
            print(f"  Cannot run stress test without valid RTH start positions.")
            print(f"  Check your data - it may not contain RTH trading hours (9:30 AM - 4:00 PM ET).")
            sys.exit(1)
    else:
        print("Generating dummy data...")
        data = create_dummy_data(100000, args.phase)

    # Detect system limits and get adaptive env counts
    max_safe_envs, adaptive_env_array = get_safe_env_limits()

    # Build search space with adaptive env counts
    search_space = build_search_space(num_envs_options=adaptive_env_array)
    print(f"Search space: {len(search_space)} combinations")

    # Run tests with patience-based early stopping
    all_results = []
    best_score = float('-inf')
    best_result = None
    no_gain_streak = 0

    for i, combo in enumerate(search_space):
        if i >= args.max_runs:
            print(f"\n[STOP] Reached maximum runs ({args.max_runs})")
            break

        print(f"\n{'='*80}")
        print(f"Test {i+1}/{min(len(search_space), args.max_runs)}")
        print(f"{'='*80}")

        result = run_combo(combo, data, args.phase, args.market, args.timeout)
        all_results.append(result)

        # Display result
        print(f"  Status: {result['status']}")
        print(f"  GPU: {result['gpu_util']:.1f}% | Memory: {result['peak_memory_gb']:.1f} GB | Temp: {result['avg_temperature']:.1f}°C")
        print(f"  Training: Return={result['mean_return']:.1f} | Entropy={result['entropy']:.3f} | SPS={result['sps']:,.0f}")
        print(f"  Score: {result['final_score']:.2f} (GPU={result['gpu_score']:.2f} | Quality={result['quality_score']:.2f} | SPS={result['sps_score']:.2f})")

        # Check for improvement
        if result['final_score'] > best_score + args.min_gain:
            best_score = result['final_score']
            best_result = result
            no_gain_streak = 0
            print(f"  [NEW BEST] Score improved by {result['final_score'] - best_score + args.min_gain:.2f}")
        else:
            no_gain_streak += 1
            print(f"  [NO GAIN] Streak: {no_gain_streak}/{args.patience}")

            if no_gain_streak >= args.patience:
                print(f"\n[STOP] Improvement plateau detected (patience={args.patience})")
                break

    # Save CSV log
    if all_results:
        save_csv_log(all_results, args.market, args.phase)

    # Generate profiles
    print("\n" + "="*80)
    print("GENERATING OPTIMAL PROFILES")
    print("="*80)

    if not all_results or not any(r['final_score'] > 0 for r in all_results):
        print("[ERROR] No successful runs completed. Cannot generate profiles.")
        return

    # Filter successful runs
    successful = [r for r in all_results if r['status'] == 'success' and r['quality_acceptable']]

    if not successful:
        print("[WARN] No successful runs with acceptable quality. Using best available.")
        successful = [r for r in all_results if r['final_score'] > -50]

    if not successful:
        print("[ERROR] All runs failed. Cannot generate profiles.")
        return

    # Identify top 3 profiles
    profiles = {}

    # 1. Balanced: Best overall score
    balanced = max(successful, key=lambda x: x['final_score'])
    profiles['balanced'] = balanced
    print(f"\n1. BALANCED: Best overall score ({balanced['final_score']:.2f})")

    # 2. Max GPU: Highest GPU util with acceptable quality
    high_gpu_candidates = [r for r in successful if r['quality_score'] > 0.7]
    if high_gpu_candidates:
        max_gpu = max(high_gpu_candidates, key=lambda x: x['gpu_util'])
        profiles['max_gpu'] = max_gpu
        print(f"2. MAX GPU: Highest GPU utilization ({max_gpu['gpu_util']:.1f}%)")
    else:
        profiles['max_gpu'] = balanced
        print(f"2. MAX GPU: Using balanced profile (no high-GPU candidates)")

    # 3. Max Quality: Best quality with good GPU util
    good_gpu_candidates = [r for r in successful if r['gpu_util'] > 80.0]
    if good_gpu_candidates:
        max_quality = max(good_gpu_candidates, key=lambda x: x['quality_score'])
        profiles['max_quality'] = max_quality
        print(f"3. MAX QUALITY: Best training quality ({max_quality['quality_score']:.2f})")
    else:
        profiles['max_quality'] = balanced
        print(f"3. MAX QUALITY: Using balanced profile (no good-GPU candidates)")

    # Save profiles
    if args.save_profiles:
        # Get GPU name with debug output to verify detection
        gpu_name = get_gpu_name(
            device_id=0,
            debug=True,
            override_name=args.gpu_name_override
        )
        print(f"\nSaving profiles with GPU prefix: {gpu_name}")
        
        for profile_name, stats in profiles.items():
            path = save_profile(profile_name, stats, args.phase, gpu_prefix=gpu_name)
            print(f"  ✓ {profile_name}: {path}")

    # Print summary table
    print_summary_table(profiles)

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("Use 'balanced' for general training (best overall performance)")
    print("Use 'max_gpu' to maximize hardware utilization (highest throughput)")
    print("Use 'max_quality' for best learning outcomes (highest quality)")
    print("\nApply with: --hardware-profile config/hardware_profiles/<name>.yaml")
    print("="*80)


def test_gpu_name_detection():
    """Test GPU name detection with common GPU models."""
    test_cases = [
        "NVIDIA RTX A6000",
        "NVIDIA RTX A4000",
        "NVIDIA GeForce RTX 5090",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 3060 Ti",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA Quadro RTX 8000",
        "NVIDIA Tesla V100",
    ]

    print("\n" + "="*80)
    print("GPU NAME DETECTION TEST")
    print("="*80)

    for test_name in test_cases:
        # Simulate sanitization (without actually calling pynvml)
        sanitized = test_name
        prefixes_to_remove = ["NVIDIA", "GeForce", "Quadro", "Tesla"]
        for prefix in prefixes_to_remove:
            sanitized = sanitized.replace(prefix, "")
        sanitized = sanitized.replace(" ", "").replace("-", "").replace("_", "")
        sanitized = ''.join(c for c in sanitized if c.isalnum())

        print(f"{test_name:45s} -> {sanitized}")

    print("="*80 + "\n")

# Uncomment to run test:
# test_gpu_name_detection()


if __name__ == "__main__":
    main()
