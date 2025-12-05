#!/usr/bin/env python3
"""
JAX Stress Test - Simple Self-Contained Version

This stress test is completely self-contained with NO external dependencies on 
complex trading environments. It uses a minimal dummy environment to test GPU 
utilization and find optimal num_envs/batch_size configurations.

Why this exists:
- The original stress_hardware_jax.py repeatedly failed with TracerIntegerConversionError
- Root cause: Complex trading environment (env_phase1_jax.py) mixed Python control flow 
  with JAX traced values in ways that break JIT compilation
- Solution: This script uses a simple dummy environment with PURE JAX operations

Key Design Principles:
1. All random keys pre-split OUTSIDE JIT using Python int for count
2. No Python [] indexing with traced arrays - use jnp.take() instead  
3. No Python if statements with traced booleans - use jnp.where() instead
4. Fixed episode lengths (no RTH sampling complexity)
5. NamedTuple state (immutable, JAX-friendly)

Usage:
    python scripts/stress_test_simple.py --max-runs 3
    python scripts/stress_test_simple.py --num-envs 64 --num-updates 10  # Quick test
"""

# =============================================================================
# Thread Control + Backend Selection (MUST be before JAX import)
# =============================================================================
import os

# Limit TensorFlow/XLA thread creation to prevent pthread_create failures
os.environ['TF_NUM_INTEROP_THREADS'] = '4'  
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false'
os.environ['JAX_PLATFORMS'] = 'cuda'  # Force NVIDIA CUDA, not ROCm

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from datetime import datetime

# Add src to path for utility imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# -----------------------------------------------------------------------------
# JAX Imports (after environment variables set)
# -----------------------------------------------------------------------------
import jax
import jax.numpy as jnp
from jax import lax, vmap

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
    print("[WARN] pynvml not available - GPU monitoring disabled")


# =============================================================================
# Dummy Environment (Pure JAX, No Trading Logic)
# =============================================================================

class DummyEnvState(NamedTuple):
    """Minimal environment state for stress testing."""
    step_idx: jnp.ndarray      # Current step in episode (scalar)
    position: jnp.ndarray      # Dummy position: -1, 0, 1 (scalar)
    episode_return: jnp.ndarray # Accumulated reward (scalar)


# Environment constants (static, not traced)
OBS_DIM = 128          # Observation dimension
NUM_ACTIONS = 3        # HOLD=0, BUY=1, SELL=2  
MAX_EPISODE_STEPS = 1000  # Fixed episode length


def dummy_reset(key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, DummyEnvState]:
    """
    Reset environment to initial state.
    
    Returns:
        obs: Random observation of shape (OBS_DIM,)
        state: Initial DummyEnvState
    """
    obs = jax.random.normal(key, shape=(OBS_DIM,))
    state = DummyEnvState(
        step_idx=jnp.array(0, dtype=jnp.int32),
        position=jnp.array(0, dtype=jnp.int32),
        episode_return=jnp.array(0.0, dtype=jnp.float32)
    )
    return obs, state


def dummy_step(
    key: jax.random.PRNGKey, 
    state: DummyEnvState, 
    action: jnp.ndarray
) -> Tuple[jnp.ndarray, DummyEnvState, jnp.ndarray, jnp.ndarray]:
    """
    Execute one environment step.
    
    Args:
        key: PRNG key for randomness
        state: Current environment state
        action: Action to take (0=HOLD, 1=BUY, 2=SELL)
    
    Returns:
        obs: New observation of shape (OBS_DIM,)
        new_state: Updated state
        reward: Reward for this step (scalar)
        done: Whether episode ended (boolean scalar)
    """
    # Random reward (simulates PnL)
    reward = jax.random.normal(key) * 0.01
    
    # Update position based on action (pure JAX, no Python if)
    # action 0 = HOLD (keep position), 1 = BUY (+1), 2 = SELL (-1)
    new_position = jnp.where(
        action == 0,
        state.position,
        jnp.where(action == 1, 1, -1)
    )
    
    # Increment step and accumulate return
    next_step = state.step_idx + 1
    new_return = state.episode_return + reward
    
    # Check if episode is done (fixed length)
    done = next_step >= MAX_EPISODE_STEPS
    
    # Generate new observation
    obs = jax.random.normal(key, shape=(OBS_DIM,))
    
    # Create new state
    new_state = DummyEnvState(
        step_idx=next_step,
        position=new_position,
        episode_return=new_return
    )
    
    return obs, new_state, reward, done


# Vectorized versions
batch_reset = vmap(dummy_reset)
batch_step = vmap(dummy_step)


# =============================================================================
# Rollout Collection (lax.scan for JIT efficiency)
# =============================================================================

def make_collect_rollouts(num_envs: int, num_steps: int):
    """
    Create a JIT-compiled rollout collection function.
    
    We generate this function with static num_envs/num_steps to avoid
    traced values for loop bounds.
    """
    
    def collect_rollouts(
        key: jax.random.PRNGKey,
        states: DummyEnvState,
        obs: jnp.ndarray
    ) -> Tuple[DummyEnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Collect num_steps of experience from num_envs environments.
        
        Returns:
            final_states: States after collection
            final_obs: Final observations
            all_obs: Collected observations (num_steps, num_envs, OBS_DIM)
            all_rewards: Collected rewards (num_steps, num_envs)
            all_dones: Episode terminations (num_steps, num_envs)
        """
        
        def step_fn(carry, key_step):
            """Single rollout step."""
            states, obs = carry
            
            # Split key for action sampling and environment steps
            key_action, key_env = jax.random.split(key_step)
            
            # Sample random actions (would be from policy in real training)
            actions = jax.random.randint(key_action, (num_envs,), 0, NUM_ACTIONS)
            
            # Split env key for each environment
            env_keys = jax.random.split(key_env, num_envs)
            
            # Take steps in all environments
            next_obs, next_states, rewards, dones = batch_step(env_keys, states, actions)
            
            # Reset done environments (pure JAX using jnp.where on each field)
            reset_keys = jax.random.split(key_env, num_envs)
            reset_obs, reset_states = batch_reset(reset_keys)
            
            # Expand dones for broadcasting with state fields
            dones_expanded = dones.astype(jnp.bool_)
            
            # Select reset or continuing state/obs
            final_obs = jnp.where(dones_expanded[:, None], reset_obs, next_obs)
            final_states = DummyEnvState(
                step_idx=jnp.where(dones_expanded, reset_states.step_idx, next_states.step_idx),
                position=jnp.where(dones_expanded, reset_states.position, next_states.position),
                episode_return=jnp.where(dones_expanded, reset_states.episode_return, next_states.episode_return)
            )
            
            return (final_states, final_obs), (obs, rewards, dones)
        
        # Pre-split keys for all steps (static count - Python int)
        step_keys = jax.random.split(key, num_steps)
        
        # Use lax.scan for efficient looping
        (final_states, final_obs), (all_obs, all_rewards, all_dones) = lax.scan(
            step_fn, (states, obs), step_keys
        )
        
        return final_states, final_obs, all_obs, all_rewards, all_dones
    
    return jax.jit(collect_rollouts)


# =============================================================================
# GPU Monitoring
# =============================================================================

def get_gpu_stats() -> Dict[str, float]:
    """Get current GPU utilization and memory stats."""
    if not HAS_PYNVML:
        return {
            'gpu_util': 0.0,
            'memory_used_gb': 0.0,
            'total_memory_gb': 0.0,
            'temperature': 0.0
        }
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        
        return {
            'gpu_util': util.gpu,
            'memory_used_gb': mem.used / (1024**3),
            'total_memory_gb': mem.total / (1024**3),
            'temperature': temp
        }
    except Exception as e:
        print(f"[WARN] GPU monitoring error: {e}")
        return {
            'gpu_util': 0.0,
            'memory_used_gb': 0.0,
            'total_memory_gb': 0.0,
            'temperature': 0.0
        }


def get_gpu_name() -> str:
    """Get sanitized GPU name for profile filenames."""
    if not HAS_PYNVML:
        return "UNKNOWN"
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        
        # Sanitize: remove NVIDIA prefix, spaces, special chars
        name = name.replace('NVIDIA', '').replace('GeForce', '')
        name = name.replace(' ', '').replace('-', '')
        name = ''.join(c for c in name if c.isalnum())
        return name or "GPU"
    except Exception:
        return "UNKNOWN"


# =============================================================================
# Stress Test Configuration
# =============================================================================

def get_safe_env_limits() -> Tuple[int, List[int]]:
    """
    Detect system limits and return safe num_envs options.
    
    Returns:
        max_safe_envs: Maximum safe environment count
        env_options: List of num_envs values to test
    """
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        
        # Check for unlimited
        if soft == resource.RLIM_INFINITY or soft is None or soft < 0 or soft > 100000:
            # Try to auto-configure for cloud platforms
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (8192, hard))
                print("[AUTO-CONFIG] Process limit set to 8192 for stress testing")
                soft = 8192
            except:
                print("[INFO] Unlimited process limit - using conservative env counts")
                return 192, [64, 128, 192]
        
        # Calculate max safe envs (estimate 25 threads per env)
        safe_limit = int(soft * 0.75)
        max_safe_envs = safe_limit // 25
        
        # Generate appropriate env options based on limit
        if max_safe_envs >= 4096:
            return max_safe_envs, [512, 1024, 2048, 4096, 8192]
        elif max_safe_envs >= 2048:
            return max_safe_envs, [512, 1024, 2048]
        elif max_safe_envs >= 512:
            return max_safe_envs, [256, 512, 1024]
        elif max_safe_envs >= 192:
            return max_safe_envs, [128, 192, 256]
        else:
            return max_safe_envs, [64, 128]
            
    except Exception as e:
        print(f"[WARN] Could not detect process limits: {e}")
        return 192, [64, 128, 192]


# =============================================================================
# Main Stress Test
# =============================================================================

def run_stress_test(
    num_envs: int,
    num_steps: int = 128,
    num_updates: int = 100,
    warmup_updates: int = 5
) -> Dict[str, float]:
    """
    Run stress test with given configuration.
    
    Args:
        num_envs: Number of parallel environments
        num_steps: Steps per rollout
        num_updates: Number of training updates to simulate
        warmup_updates: Updates to skip for timing (JIT compilation)
    
    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing: num_envs={num_envs}, num_steps={num_steps}, updates={num_updates}")
    print(f"{'='*60}")
    
    try:
        # Initialize PRNG
        master_key = jax.random.PRNGKey(42)
        
        # Pre-split keys for all environments (Python int count - NOT traced!)
        init_keys = jax.random.split(master_key, num_envs)
        
        # Initialize environments
        obs, states = batch_reset(init_keys)
        print(f"✓ Initialized {num_envs} environments")
        
        # Create JIT-compiled rollout function with STATIC num_envs/num_steps
        collect_fn = make_collect_rollouts(num_envs, num_steps)
        
        # Warmup (JIT compilation)
        print(f"⏳ Warming up ({warmup_updates} updates)...")
        for i in range(warmup_updates):
            key, subkey = jax.random.split(master_key)
            master_key = key
            states, obs, _, _, _ = collect_fn(subkey, states, obs)
        jax.block_until_ready(obs)
        print("✓ Warmup complete")
        
        # Timed test
        print(f"🚀 Running {num_updates} updates...")
        gpu_utils = []
        start_time = time.time()
        
        for i in range(num_updates):
            key, subkey = jax.random.split(master_key)
            master_key = key
            states, obs, all_obs, all_rewards, all_dones = collect_fn(subkey, states, obs)
            
            # Sample GPU stats periodically
            if i % 10 == 0:
                stats = get_gpu_stats()
                gpu_utils.append(stats['gpu_util'])
                
                # Progress update
                if i % 20 == 0:
                    print(f"  Update {i}/{num_updates} | GPU: {stats['gpu_util']:.1f}%")
        
        # Wait for all computations to complete
        jax.block_until_ready(obs)
        elapsed = time.time() - start_time
        
        # Calculate metrics
        total_steps = num_envs * num_steps * num_updates
        sps = total_steps / elapsed if elapsed > 0 else 0
        avg_gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
        final_stats = get_gpu_stats()
        
        results = {
            'status': 'success',
            'num_envs': num_envs,
            'num_steps': num_steps,
            'num_updates': num_updates,
            'elapsed_seconds': elapsed,
            'steps_per_second': sps,
            'avg_gpu_util': avg_gpu,
            'peak_memory_gb': final_stats['memory_used_gb'],
            'total_memory_gb': final_stats['total_memory_gb'],
            'temperature': final_stats['temperature']
        }
        
        print(f"\n✅ SUCCESS")
        print(f"   Steps/Second: {sps:,.0f}")
        print(f"   GPU Utilization: {avg_gpu:.1f}%")
        print(f"   Memory: {final_stats['memory_used_gb']:.1f} / {final_stats['total_memory_gb']:.1f} GB")
        print(f"   Temperature: {final_stats['temperature']}°C")
        
        return results
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ FAILED: {error_msg}")
        
        # Detect specific error types
        status = 'error'
        if 'TracerIntegerConversionError' in error_msg or '__index__' in error_msg:
            status = 'tracer_int_error'
        elif 'TracerBoolConversionError' in error_msg:
            status = 'tracer_bool_error'
        elif 'out of memory' in error_msg.lower() or 'OOM' in error_msg:
            status = 'oom'
        elif 'pthread_create' in error_msg.lower():
            status = 'thread_limit'
        
        return {
            'status': status,
            'num_envs': num_envs,
            'num_steps': num_steps,
            'error': error_msg
        }


def save_profile(name: str, config: Dict, gpu_name: str = ""):
    """Save hardware profile to YAML in training-compatible format."""
    # IMPORTANT: Save to config/hardware_profiles/ to match old stress test
    # This ensures main.py can find and load profiles for training
    profiles_dir = PROJECT_ROOT / "config" / "hardware_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{gpu_name}_{name}.yaml" if gpu_name else f"{name}.yaml"
    filepath = profiles_dir / filename
    
    # Match old stress test format for training script compatibility
    payload = {
        "mode": "jax_hardware_maximized",
        "phase": 1,  # Simple stress test is for Phase 1 only
        "num_envs": config['num_envs'],
        "tested_num_envs": config['num_envs'],
        "stress_test_updates": config.get('num_updates', 100),
        "safety_margin_applied": 1.0,  # No reduction - tested value is safe
        "num_steps": config.get('num_steps', 128),
        "num_minibatches": 4,  # Default PPO value
        "num_epochs": 4,  # Default PPO value
        "device": "gpu" if jax.default_backend() == "gpu" else "cpu",
        # Performance metrics
        "expected_sps": float(config.get('steps_per_second', 0)),
        "expected_gpu_util": float(config.get('avg_gpu_util', 0)),
        "expected_memory_gb": float(config.get('peak_memory_gb', 0)),
        # Quality metrics (not available from simple stress test)
        "quality_score": 0.0,
        "mean_return": 0.0,
        "entropy": 0.0,
        # Metadata
        "notes": f"Auto-tuned via stress_test_simple.py on {datetime.now().strftime('%Y-%m-%d')}",
        "final_score": float(config.get('avg_gpu_util', 0)) * 0.5 + (config.get('steps_per_second', 0) / 100000) * 0.5
    }
    
    import yaml
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    
    print(f"📁 Saved profile: {filepath}")
    return filepath


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="JAX Stress Test - Simple Self-Contained Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stress_test_simple.py                    # Full stress test
  python stress_test_simple.py --num-envs 64     # Test specific config
  python stress_test_simple.py --max-runs 3      # Limit test runs
"""
    )
    parser.add_argument('--num-envs', type=int, default=None,
                        help='Specific num_envs to test (default: auto-detect range)')
    parser.add_argument('--num-steps', type=int, default=128,
                        help='Steps per rollout (default: 128)')
    parser.add_argument('--num-updates', type=int, default=100,
                        help='Number of updates per test (default: 100)')
    parser.add_argument('--max-runs', type=int, default=None,
                        help='Maximum number of configurations to test')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (10 updates)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("JAX STRESS TEST - SIMPLE SELF-CONTAINED VERSION")
    print("="*60)
    
    # Check GPU availability
    devices = jax.devices()
    print(f"\n🖥️  JAX Devices: {devices}")
    gpu_name = get_gpu_name()
    print(f"🎮 GPU: {gpu_name}")
    
    # Get safe env limits
    max_safe, env_options = get_safe_env_limits()
    print(f"📊 Max Safe Envs: {max_safe}")
    print(f"📊 Test Options: {env_options}")
    
    # Quick mode settings
    num_updates = 10 if args.quick else args.num_updates
    
    # Single config test
    if args.num_envs is not None:
        result = run_stress_test(
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            num_updates=num_updates
        )
        if result['status'] == 'success':
            save_profile('test', result, gpu_name)
        return
    
    # Full stress test - try all configurations
    results = []
    for i, num_envs in enumerate(env_options):
        if args.max_runs and i >= args.max_runs:
            print(f"\n⏹️  Reached max runs ({args.max_runs})")
            break
        
        result = run_stress_test(
            num_envs=num_envs,
            num_steps=args.num_steps,
            num_updates=num_updates
        )
        results.append(result)
        
        # Early stop on critical errors
        if result['status'] in ['tracer_int_error', 'tracer_bool_error']:
            print(f"\n🛑 Critical error detected - stopping stress test")
            break
    
    # Analyze results
    successful = [r for r in results if r['status'] == 'success']
    
    if not successful:
        print("\n❌ NO SUCCESSFUL RUNS - Cannot generate profiles")
        print("Check errors above for debugging information")
        return
    
    # Generate profiles
    print("\n" + "="*60)
    print("GENERATING HARDWARE PROFILES")
    print("="*60)
    
    # Balanced: Best GPU utilization with good SPS
    balanced = max(successful, key=lambda r: r['avg_gpu_util'] * 0.5 + (r['steps_per_second'] / 100000) * 0.5)
    save_profile('balanced', balanced, gpu_name)
    
    # Max GPU: Highest GPU utilization
    max_gpu = max(successful, key=lambda r: r['avg_gpu_util'])
    save_profile('max_gpu', max_gpu, gpu_name)
    
    # Max SPS: Highest steps per second
    max_sps = max(successful, key=lambda r: r['steps_per_second'])
    save_profile('max_sps', max_sps, gpu_name)
    
    # Summary
    print("\n" + "="*60)
    print("STRESS TEST COMPLETE")
    print("="*60)
    print(f"✓ Successful configs: {len(successful)}/{len(results)}")
    print(f"✓ Best balanced: num_envs={balanced['num_envs']}, GPU={balanced['avg_gpu_util']:.1f}%")
    print(f"✓ Max GPU config: num_envs={max_gpu['num_envs']}, GPU={max_gpu['avg_gpu_util']:.1f}%")
    print(f"✓ Max SPS config: num_envs={max_sps['num_envs']}, SPS={max_sps['steps_per_second']:,.0f}")


if __name__ == "__main__":
    main()
