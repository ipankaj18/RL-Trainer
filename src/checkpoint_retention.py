"""
Checkpoint Retention and Pruning System
Manages checkpoint lifecycle to balance disk space with training history preservation

Features:
- Keep last N periodic checkpoints
- Keep top K checkpoints by Sharpe ratio
- Preserve special events (phase_end, phase_boundary)
- Parse metrics from filenames
- Safe deletion with dry-run mode
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import yaml


@dataclass
class CheckpointInfo:
    """Information about a checkpoint file."""
    path: Path
    market: str
    timesteps: int
    event_tag: str
    val_reward: float
    sharpe_ratio: float
    seed: int
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class CheckpointRetentionManager:
    """
    Manages checkpoint retention policies to balance disk space with history preservation.

    Retention Strategy:
    1. Keep last N periodic checkpoints (sorted by timesteps)
    2. Keep top K checkpoints by Sharpe ratio
    3. Keep top K checkpoints by validation reward
    4. Always preserve special events (phase_end, phase_boundary, interrupt)
    5. Optionally prune checkpoints older than threshold

    Usage:
        manager = CheckpointRetentionManager('config/checkpoint_config.yaml')
        manager.prune_checkpoints(checkpoint_dir='models/phase1/NQ/checkpoints/', dry_run=False)
    """

    def __init__(self, config_path: str = 'config/checkpoint_config.yaml'):
        """
        Initialize retention manager.

        Args:
            config_path: Path to checkpoint configuration YAML
        """
        self.config = self._load_config(config_path)
        self.retention_config = self.config['checkpointing']['retention']
        self.naming_config = self.config['checkpointing']['naming']

        # Retention parameters
        self.keep_last_n = self.retention_config['keep_last_n']
        self.keep_top_k_sharpe = self.retention_config['keep_top_k_by_sharpe']
        self.keep_top_k_reward = self.retention_config['keep_top_k_by_reward']
        self.preserve_events = set(self.retention_config['preserve_events'])
        self.max_age_days = self.retention_config['max_age_days']

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load checkpoint configuration from YAML."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def parse_checkpoint_filename(self, filename: str) -> Optional[CheckpointInfo]:
        """
        Parse checkpoint filename to extract metadata.

        Format: {market}_ts-{timesteps:07d}_evt-{event}_val-{val_reward:+.3f}_sharpe-{sharpe:+.2f}_seed-{seed}.zip

        Args:
            filename: Checkpoint filename (with or without extension)

        Returns:
            CheckpointInfo object or None if parsing fails
        """
        # Remove extension if present
        name = Path(filename).stem

        # Regex pattern for checkpoint filename
        # Example: NQ_ts-0025000_evt-periodic_val-+0.112_sharpe-+1.45_seed-42
        pattern = r'(?P<market>[A-Z0-9]+)_ts-(?P<timesteps>\d+)_evt-(?P<event>[\w\-]+)_val-(?P<val_reward>[+-]?\d+\.\d+)_sharpe-(?P<sharpe>[+-]?\d+\.\d+)_seed-(?P<seed>\d+)'

        match = re.match(pattern, name)
        if not match:
            return None

        return CheckpointInfo(
            path=Path(filename),
            market=match.group('market'),
            timesteps=int(match.group('timesteps')),
            event_tag=match.group('event'),
            val_reward=float(match.group('val_reward')),
            sharpe_ratio=float(match.group('sharpe')),
            seed=int(match.group('seed'))
        )

    def load_checkpoint_metadata(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint metadata JSON if available.

        Args:
            checkpoint_path: Path to checkpoint .zip file

        Returns:
            Metadata dictionary or None
        """
        # Metadata is stored as {checkpoint}_metadata.json
        metadata_path = checkpoint_path.with_suffix('').with_suffix('.json')

        # Alternative location: same name with _metadata suffix
        if not metadata_path.exists():
            base = checkpoint_path.with_suffix('')
            metadata_path = Path(str(base) + '_metadata.json')

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to load metadata from {metadata_path}: {e}")

        return None

    def scan_checkpoints(self, checkpoint_dir: Path) -> List[CheckpointInfo]:
        """
        Scan directory for checkpoints and parse their info.

        Args:
            checkpoint_dir: Directory containing checkpoints

        Returns:
            List of CheckpointInfo objects
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return []

        checkpoints = []

        # Find all .zip files
        for checkpoint_file in checkpoint_dir.glob('*.zip'):
            info = self.parse_checkpoint_filename(checkpoint_file.name)

            if info is not None:
                info.path = checkpoint_file

                # Load metadata if available
                metadata = self.load_checkpoint_metadata(checkpoint_file)
                if metadata:
                    info.metadata = metadata

                    # Extract timestamp
                    if 'timestamp' in metadata:
                        try:
                            info.timestamp = datetime.fromisoformat(metadata['timestamp'])
                        except Exception:
                            pass

                checkpoints.append(info)

        return checkpoints

    def select_checkpoints_to_keep(self, checkpoints: List[CheckpointInfo]) -> List[CheckpointInfo]:
        """
        Select which checkpoints to keep based on retention policy.

        Args:
            checkpoints: List of all checkpoints

        Returns:
            List of checkpoints to KEEP
        """
        keep = set()

        # 1. Always preserve special events
        for ckpt in checkpoints:
            if ckpt.event_tag in self.preserve_events:
                keep.add(ckpt.path)

        # 2. Keep last N periodic checkpoints (sorted by timesteps)
        periodic = [c for c in checkpoints if c.event_tag == 'periodic']
        periodic_sorted = sorted(periodic, key=lambda c: c.timesteps, reverse=True)
        for ckpt in periodic_sorted[:self.keep_last_n]:
            keep.add(ckpt.path)

        # 3. Keep top K by Sharpe ratio (excluding NaN)
        valid_sharpe = [c for c in checkpoints if not (c.sharpe_ratio != c.sharpe_ratio)]  # Exclude NaN
        sharpe_sorted = sorted(valid_sharpe, key=lambda c: c.sharpe_ratio, reverse=True)
        for ckpt in sharpe_sorted[:self.keep_top_k_sharpe]:
            keep.add(ckpt.path)

        # 4. Keep top K by validation reward (excluding NaN)
        valid_reward = [c for c in checkpoints if not (c.val_reward != c.val_reward)]  # Exclude NaN
        reward_sorted = sorted(valid_reward, key=lambda c: c.val_reward, reverse=True)
        for ckpt in reward_sorted[:self.keep_top_k_reward]:
            keep.add(ckpt.path)

        # 5. Filter by age if enabled
        if self.max_age_days > 0:
            cutoff = datetime.now() - timedelta(days=self.max_age_days)
            keep = {path for path in keep
                    if any(c.timestamp and c.timestamp >= cutoff for c in checkpoints if c.path == path)}

        # Convert back to CheckpointInfo objects
        return [c for c in checkpoints if c.path in keep]

    def prune_checkpoints(
        self,
        checkpoint_dir: Path,
        dry_run: bool = True,
        verbose: bool = True
    ) -> Tuple[List[Path], List[Path]]:
        """
        Prune checkpoints based on retention policy.

        Args:
            checkpoint_dir: Directory containing checkpoints
            dry_run: If True, don't actually delete files (just report what would be deleted)
            verbose: Print detailed information

        Returns:
            Tuple of (kept_paths, deleted_paths)
        """
        checkpoint_dir = Path(checkpoint_dir)

        if verbose:
            print(f"\n{'='*60}")
            print(f"CHECKPOINT RETENTION - {'DRY RUN' if dry_run else 'LIVE MODE'}")
            print(f"{'='*60}")
            print(f"Directory: {checkpoint_dir}")

        # Scan all checkpoints
        all_checkpoints = self.scan_checkpoints(checkpoint_dir)

        if verbose:
            print(f"Found {len(all_checkpoints)} total checkpoints")

        if len(all_checkpoints) == 0:
            if verbose:
                print("No checkpoints to prune.")
            return [], []

        # Select checkpoints to keep
        to_keep = self.select_checkpoints_to_keep(all_checkpoints)
        to_keep_paths = {c.path for c in to_keep}

        # Determine checkpoints to delete
        to_delete = [c for c in all_checkpoints if c.path not in to_keep_paths]

        if verbose:
            print(f"\n{'─'*60}")
            print(f"Retention Policy:")
            print(f"  • Keep last {self.keep_last_n} periodic checkpoints")
            print(f"  • Keep top {self.keep_top_k_sharpe} by Sharpe ratio")
            print(f"  • Keep top {self.keep_top_k_reward} by validation reward")
            print(f"  • Always preserve: {', '.join(self.preserve_events)}")
            if self.max_age_days > 0:
                print(f"  • Max age: {self.max_age_days} days")
            print(f"{'─'*60}")

            print(f"\nRESULTS:")
            print(f"  ✓ Keep: {len(to_keep)} checkpoints")
            print(f"  ✗ Delete: {len(to_delete)} checkpoints")

        # Delete checkpoints (if not dry run)
        deleted_paths = []

        if not dry_run and len(to_delete) > 0:
            if verbose:
                print(f"\n{'─'*60}")
                print("Deleting checkpoints:")

            for ckpt in to_delete:
                try:
                    # Delete main checkpoint file
                    ckpt.path.unlink()
                    deleted_paths.append(ckpt.path)

                    if verbose:
                        print(f"  ✗ {ckpt.path.name}")

                    # Delete companion files (VecNormalize, metadata)
                    base_path = ckpt.path.with_suffix('')

                    vecnorm_path = Path(str(base_path) + self.naming_config['vecnormalize_suffix'])
                    if vecnorm_path.exists():
                        vecnorm_path.unlink()

                    metadata_path = Path(str(base_path) + self.naming_config['metadata_suffix'])
                    if metadata_path.exists():
                        metadata_path.unlink()

                except Exception as e:
                    print(f"[ERROR] Failed to delete {ckpt.path.name}: {e}")

            if verbose:
                print(f"{'─'*60}")
                print(f"✓ Deleted {len(deleted_paths)} checkpoints")

        elif dry_run and len(to_delete) > 0:
            if verbose:
                print(f"\n{'─'*60}")
                print("Would delete (DRY RUN):")

                for ckpt in to_delete:
                    print(f"  ✗ {ckpt.path.name}")
                    print(f"      Timesteps: {ckpt.timesteps:,} | Event: {ckpt.event_tag} | "
                          f"Sharpe: {ckpt.sharpe_ratio:+.2f} | Reward: {ckpt.val_reward:+.3f}")

                print(f"{'─'*60}")

        if verbose:
            print(f"\n{'='*60}\n")

        kept_paths = [c.path for c in to_keep]
        deleted_paths_list = [c.path for c in to_delete]

        return kept_paths, deleted_paths_list

    def get_checkpoint_summary(self, checkpoint_dir: Path) -> Dict[str, Any]:
        """
        Get summary statistics for checkpoints in directory.

        Args:
            checkpoint_dir: Directory containing checkpoints

        Returns:
            Dictionary with summary statistics
        """
        checkpoints = self.scan_checkpoints(checkpoint_dir)

        if len(checkpoints) == 0:
            return {
                'total_count': 0,
                'event_breakdown': {},
                'best_sharpe': None,
                'best_reward': None,
                'timestep_range': None
            }

        # Count by event type
        event_breakdown = {}
        for ckpt in checkpoints:
            event_breakdown[ckpt.event_tag] = event_breakdown.get(ckpt.event_tag, 0) + 1

        # Best metrics
        valid_sharpe = [c for c in checkpoints if not (c.sharpe_ratio != c.sharpe_ratio)]
        valid_reward = [c for c in checkpoints if not (c.val_reward != c.val_reward)]

        best_sharpe_ckpt = max(valid_sharpe, key=lambda c: c.sharpe_ratio) if valid_sharpe else None
        best_reward_ckpt = max(valid_reward, key=lambda c: c.val_reward) if valid_reward else None

        # Timestep range
        timesteps = [c.timesteps for c in checkpoints]
        timestep_range = (min(timesteps), max(timesteps))

        return {
            'total_count': len(checkpoints),
            'event_breakdown': event_breakdown,
            'best_sharpe': {
                'value': best_sharpe_ckpt.sharpe_ratio,
                'timesteps': best_sharpe_ckpt.timesteps,
                'filename': best_sharpe_ckpt.path.name
            } if best_sharpe_ckpt else None,
            'best_reward': {
                'value': best_reward_ckpt.val_reward,
                'timesteps': best_reward_ckpt.timesteps,
                'filename': best_reward_ckpt.path.name
            } if best_reward_ckpt else None,
            'timestep_range': timestep_range,
            'markets': list(set(c.market for c in checkpoints)),
            'seeds': list(set(c.seed for c in checkpoints))
        }


def prune_all_phase_checkpoints(
    phase: int,
    config_path: str = 'config/checkpoint_config.yaml',
    dry_run: bool = True,
    verbose: bool = True
) -> None:
    """
    Convenience function to prune checkpoints for all markets in a phase.

    Args:
        phase: Training phase (1, 2, or 3)
        config_path: Path to checkpoint config
        dry_run: If True, don't actually delete files
        verbose: Print detailed information
    """
    manager = CheckpointRetentionManager(config_path)

    # Get phase checkpoint directory template
    path_template = manager.config['checkpointing']['paths'][f'phase{phase}']

    # Determine markets to scan
    # Scan parent directory for market subdirectories
    parent_dir = Path(path_template.split('{market}')[0])

    if not parent_dir.exists():
        if verbose:
            print(f"Phase {phase} checkpoint directory does not exist: {parent_dir}")
        return

    market_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]

    if verbose:
        print(f"\n{'='*60}")
        print(f"PRUNING PHASE {phase} CHECKPOINTS")
        print(f"{'='*60}\n")

    for market_dir in market_dirs:
        checkpoint_dir = market_dir / 'checkpoints'
        if checkpoint_dir.exists():
            manager.prune_checkpoints(checkpoint_dir, dry_run=dry_run, verbose=verbose)


if __name__ == '__main__':
    """Command-line interface for checkpoint retention."""
    import argparse

    parser = argparse.ArgumentParser(description='Checkpoint Retention Manager')
    parser.add_argument('--phase', type=int, required=True, choices=[1, 2, 3],
                       help='Training phase (1, 2, or 3)')
    parser.add_argument('--market', type=str, default=None,
                       help='Market symbol (e.g., NQ, ES). If not specified, prunes all markets.')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--config', type=str, default='config/checkpoint_config.yaml',
                       help='Path to checkpoint config YAML')

    args = parser.parse_args()

    if args.market:
        # Prune specific market
        manager = CheckpointRetentionManager(args.config)
        path_template = manager.config['checkpointing']['paths'][f'phase{args.phase}']
        checkpoint_dir = Path(path_template.format(market=args.market))

        manager.prune_checkpoints(checkpoint_dir, dry_run=args.dry_run, verbose=True)
    else:
        # Prune all markets in phase
        prune_all_phase_checkpoints(args.phase, args.config, dry_run=args.dry_run, verbose=True)
