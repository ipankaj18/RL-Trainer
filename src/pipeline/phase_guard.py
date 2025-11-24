"""
Phase Guard - Training Phase Validation & Gating System

Prevents progression to subsequent training phases until quality gates are met.
Validates metrics, model compatibility, and data integrity.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import json
from datetime import datetime
import os


class PhaseGuard:
    """
    Validates phase completion criteria before allowing progression.

    Quality Gates:
    - Phase 1: mean_reward >= 0.0, eval_variance > 0.001
    - Phase 2: mean_reward >= -50, eval_variance > 0.01
    """

    # Gate thresholds (can be overridden via config)
    THRESHOLDS = {
        'phase1': {
            'min_mean_reward': 0.0,
            'min_eval_variance': 0.001,
            'min_episode_length': 100,
            'max_episode_length': 10000,
        },
        'phase2': {
            'min_mean_reward': -50.0,
            'min_eval_variance': 0.01,
            'min_episode_length': 200,
        },
        'phase3': {
            'min_mean_reward': 0.0,
            'min_eval_variance': 0.1,
        }
    }

    @staticmethod
    def _find_newest_eval_file(log_dir: str) -> str:
        """
        Find the newest evaluations.npz file in timestamped eval folders.

        Searches for:
        1. Timestamped folders: logs/phaseX/eval_YYYYMMDD_HHMMSS/evaluations.npz
        2. Legacy path: logs/phaseX/evaluations.npz

        Returns the path to the newest file, or legacy path if no timestamped folders exist.
        """
        log_path = Path(log_dir)

        # Search for timestamped eval folders
        eval_folders = sorted(log_path.glob("eval_*/"))

        if eval_folders:
            # Get the newest folder (sorted alphabetically = chronologically due to timestamp format)
            newest_folder = eval_folders[-1]
            eval_file = newest_folder / "evaluations.npz"

            if eval_file.exists():
                return str(eval_file)

        # Fallback to legacy path
        legacy_path = log_path / "evaluations.npz"
        return str(legacy_path)

    @staticmethod
    def _check_test_mode(log_dir: str, model_path: Optional[str] = None) -> bool:
        """
        Check if the phase was trained in test mode (--test flag).

        Uses both metadata and timestep heuristics:
        1. If metadata exists and explicitly marks test_mode, trust it
        2. If metadata exists and marks test_mode=False, trust it (even if timesteps < 1M)
        3. If no metadata, use timestep heuristic as fallback

        Args:
            log_dir: Phase log directory (e.g., "logs/phase1")
            model_path: Optional model path to check metadata

        Returns:
            True if test mode detected, False otherwise
        """
        # PRIORITY 1: Check model metadata if provided
        metadata_found = False
        if model_path:
            metadata_file = Path(model_path).parent / (Path(model_path).stem + "_metadata.json")
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        meta = json.load(f)
                        metadata_found = True
                        # Explicitly check for test_mode flag
                        if 'test_mode' in meta:
                            return bool(meta['test_mode'])  # Trust metadata
                except Exception:
                    pass

        # PRIORITY 2: If metadata was found but didn't have test_mode, assume production
        if metadata_found:
            return False  # Metadata exists without test_mode flag = production

        # FALLBACK: Use timestep heuristic only if no metadata
        eval_path = PhaseGuard._find_newest_eval_file(log_dir)
        if Path(eval_path).exists():
            try:
                data = np.load(eval_path)
                timesteps = int(data['timesteps'][-1])

                # Check timestep thresholds for test mode detection
                phase = Path(log_dir).name  # "phase1", "phase2", etc.
                test_thresholds = {
                    'phase1': 1_000_000,  # Production is 10M, test is 50K-500K
                    'phase2': 1_000_000,  # Production is 5M, test is similar
                }

                threshold = test_thresholds.get(phase, 1_000_000)
                if timesteps < threshold:
                    # Only use timestep heuristic if no metadata was found
                    return True  # Likely a test run

            except Exception:
                pass

        return False

    @staticmethod
    def create_legacy_symlink(log_dir: str, use_copy: bool = False) -> Optional[str]:
        """
        Create symlink (or copy) from newest eval file to legacy path for backward compatibility.

        Args:
            log_dir: Phase log directory (e.g., "logs/phase1")
            use_copy: If True, copy file instead of symlink (for Windows compatibility)

        Returns:
            Path to legacy file if created, None if no timestamped eval exists
        """
        log_path = Path(log_dir)
        eval_folders = sorted(log_path.glob("eval_*/"))

        if not eval_folders:
            return None  # No timestamped folders exist

        newest_folder = eval_folders[-1]
        source_file = newest_folder / "evaluations.npz"

        if not source_file.exists():
            return None  # Source file doesn't exist

        legacy_path = log_path / "evaluations.npz"

        try:
            # Remove existing legacy file/symlink if it exists
            if legacy_path.exists() or legacy_path.is_symlink():
                legacy_path.unlink()

            if use_copy:
                # Copy file (Windows-compatible)
                import shutil
                shutil.copy2(source_file, legacy_path)
            else:
                # Create symlink (Unix/WSL) using relative path for portability
                relative_target = f"{newest_folder.name}/evaluations.npz"
                legacy_path.symlink_to(relative_target)

            return str(legacy_path)

        except Exception as e:
            # Fallback to copy if symlink fails (e.g., Windows without admin)
            if not use_copy:
                try:
                    import shutil
                    shutil.copy2(source_file, legacy_path)
                    return str(legacy_path)
                except Exception:
                    pass
            return None

    @staticmethod
    def validate_phase1(
        eval_path: Optional[str] = None,
        model_path: Optional[str] = None,
        override: bool = False
    ) -> Tuple[bool, str, Dict]:
        """
        Validate Phase 1 completion before allowing Phase 2 training.

        Args:
            eval_path: Path to Phase 1 evaluation results (auto-discovers newest if None)
            model_path: Path to Phase 1 model checkpoint
            override: If True, skip validation (manual approval)

        Returns:
            (passed, message, metrics_dict)
        """
        if override:
            return True, "⚠️  Phase 1 gate OVERRIDDEN (manual approval)", {}

        # Auto-discover evaluation path if not provided
        if eval_path is None:
            eval_path = PhaseGuard._find_newest_eval_file("logs/phase1")

        # Check evaluation results exist
        if not Path(eval_path).exists():
            return False, f"❌ No Phase 1 evaluation results found at {eval_path}", {}

        # Load evaluation data
        try:
            data = np.load(eval_path)
            last_results = data['results'][-1]  # Most recent checkpoint
            last_lengths = data['ep_lengths'][-1]
            timesteps = data['timesteps'][-1]
        except Exception as e:
            return False, f"❌ Failed to load evaluation data: {e}", {}

        # Compute metrics
        metrics = {
            'mean_reward': float(last_results.mean()),
            'reward_std': float(last_results.std()),
            'reward_variance': float(last_results.var()),
            'mean_episode_length': float(last_lengths.mean()),
            'episode_length_std': float(last_lengths.std()),
            'episode_length_variance': float(last_lengths.var()),
            'min_episode_length': int(last_lengths.min()),
            'max_episode_length': int(last_lengths.max()),
            'num_eval_episodes': len(last_results),
            'timesteps_trained': int(timesteps),
        }

        # Check if this was a test mode run
        is_test_mode = PhaseGuard._check_test_mode("logs/phase1", model_path)
        if is_test_mode:
            return False, (
                "❌ PHASE 1 GATE FAILED:\n"
                "   - Phase 1 was run in TEST MODE (--test flag)\n"
                "   - Test runs use reduced timesteps and are not suitable for production\n"
                "   - Please run full Phase 1 training without --test flag"
            ), metrics

        # Check thresholds
        thresholds = PhaseGuard.THRESHOLDS['phase1']
        failures = []

        if metrics['mean_reward'] < thresholds['min_mean_reward']:
            failures.append(
                f"Mean reward {metrics['mean_reward']:.2f} < {thresholds['min_mean_reward']:.2f}"
            )

        if metrics['reward_variance'] < thresholds['min_eval_variance']:
            failures.append(
                f"Reward variance {metrics['reward_variance']:.4f} < {thresholds['min_eval_variance']:.4f} "
                "(deterministic failure - all episodes fail identically)"
            )

        if metrics['episode_length_variance'] < 1.0:
            failures.append(
                f"Episode length variance {metrics['episode_length_variance']:.2f} < 1.0 "
                "(all episodes same length - evaluation is deterministic)"
            )

        if metrics['mean_episode_length'] < thresholds['min_episode_length']:
            failures.append(
                f"Episodes too short: {metrics['mean_episode_length']:.0f} bars < {thresholds['min_episode_length']}"
            )

        # Check model exists if provided
        if model_path and not Path(model_path).exists():
            failures.append(f"Model checkpoint not found at {model_path}")

        # Generate result
        if failures:
            message = "❌ PHASE 1 GATE FAILED:\n" + "\n".join(f"   - {f}" for f in failures)
            message += f"\n\n📊 Metrics: mean_reward={metrics['mean_reward']:.2f}, "
            message += f"variance={metrics['reward_variance']:.4f}, "
            message += f"ep_length={metrics['mean_episode_length']:.0f}"
            return False, message, metrics

        message = "✅ Phase 1 PASSED all quality gates"
        message += f"\n   Mean reward: {metrics['mean_reward']:.2f}"
        message += f"\n   Episode length: {metrics['mean_episode_length']:.0f} ± {metrics['episode_length_std']:.0f} bars"
        message += f"\n   Timesteps trained: {metrics['timesteps_trained']:,}"

        return True, message, metrics

    @staticmethod
    def validate_phase2(
        eval_path: Optional[str] = None,
        override: bool = False
    ) -> Tuple[bool, str, Dict]:
        """Validate Phase 2 before Phase 3."""
        if override:
            return True, "⚠️  Phase 2 gate OVERRIDDEN", {}

        # Auto-discover evaluation path if not provided
        if eval_path is None:
            eval_path = PhaseGuard._find_newest_eval_file("logs/phase2")

        if not Path(eval_path).exists():
            return False, f"❌ No Phase 2 evaluation results at {eval_path}", {}

        try:
            data = np.load(eval_path)
            last_results = data['results'][-1]
            last_lengths = data['ep_lengths'][-1]
        except Exception as e:
            return False, f"❌ Failed to load Phase 2 evaluation: {e}", {}

        metrics = {
            'mean_reward': float(last_results.mean()),
            'reward_variance': float(last_results.var()),
            'mean_episode_length': float(last_lengths.mean()),
        }

        thresholds = PhaseGuard.THRESHOLDS['phase2']
        failures = []

        if metrics['mean_reward'] < thresholds['min_mean_reward']:
            failures.append(f"Mean reward {metrics['mean_reward']:.2f} < {thresholds['min_mean_reward']:.2f}")

        if metrics['reward_variance'] < thresholds['min_eval_variance']:
            failures.append(f"Variance {metrics['reward_variance']:.4f} too low")

        if failures:
            return False, "❌ PHASE 2 GATE FAILED:\n" + "\n".join(f"   - {f}" for f in failures), metrics

        return True, "✅ Phase 2 PASSED", metrics

    @staticmethod
    def log_gate_decision(
        phase: str,
        passed: bool,
        message: str,
        metrics: Dict,
        log_dir: str = "logs/pipeline"
    ):
        """Log gate decision to file for audit trail."""
        os.makedirs(log_dir, exist_ok=True)
        log_file = Path(log_dir) / "phase_guard.log"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "PASSED" if passed else "FAILED"

        entry = {
            'timestamp': timestamp,
            'phase': phase,
            'status': status,
            'message': message,
            'metrics': metrics
        }

        # Append to log file
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{timestamp}] {phase.upper()} GATE {status}\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{message}\n")
            if metrics:
                f.write(f"\nMetrics:\n")
                for key, value in metrics.items():
                    f.write(f"  {key}: {value}\n")

        # Also save JSON for programmatic access
        json_file = Path(log_dir) / f"{phase}_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(entry, f, indent=2)

        return log_file

    @staticmethod
    def check_model_compatibility(
        phase1_model: str,
        phase2_env_obs_space: int = 228
    ) -> Tuple[bool, str]:
        """
        Verify Phase 1 model can be loaded for Phase 2 transfer learning.

        Args:
            phase1_model: Path to Phase 1 .zip model
            phase2_env_obs_space: Expected observation space size

        Returns:
            (compatible, message)
        """
        if not Path(phase1_model).exists():
            return False, f"Model not found: {phase1_model}"

        # Check if metadata file exists
        metadata_file = phase1_model.replace('.zip', '_metadata.json')
        if Path(metadata_file).exists():
            try:
                with open(metadata_file, 'r') as f:
                    meta = json.load(f)

                # Check observation space compatibility
                if 'observation_space' in meta:
                    obs_dim = meta['observation_space'].get('shape', [0])[0]
                    if obs_dim != phase2_env_obs_space and obs_dim + 3 != phase2_env_obs_space:
                        return False, f"Incompatible obs space: Phase 1={obs_dim}, Phase 2={phase2_env_obs_space}"

                return True, f"✅ Model compatible (obs_dim={obs_dim})"
            except Exception as e:
                return True, f"⚠️  Metadata check failed ({e}), proceeding with caution"

        # No metadata, assume compatible
        return True, "⚠️  No metadata found, compatibility unknown"


def print_gate_banner(passed: bool, phase: str):
    """Print colorful gate result banner."""
    if passed:
        print("\n" + "="*80)
        print(f"  ✅  {phase.upper()} QUALITY GATE: PASSED")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print(f"  ❌  {phase.upper()} QUALITY GATE: FAILED")
        print("="*80 + "\n")


if __name__ == "__main__":
    # Test the guard
    print("Testing PhaseGuard...\n")

    passed, msg, metrics = PhaseGuard.validate_phase1()
    print(msg)
    print(f"\nMetrics: {metrics}")

    if passed:
        PhaseGuard.log_gate_decision('phase1', passed, msg, metrics)
        print("\n✅ Gate decision logged to logs/pipeline/phase_guard.log")
