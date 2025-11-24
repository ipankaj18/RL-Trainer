"""
Checkpoint Registry - Global index for all training checkpoints

This module provides a centralized registry for tracking checkpoints across
markets, phases, and training runs. It enables fast lookup, ranking, and
cross-market comparison of model checkpoints.

Key Features:
- JSON-based storage with in-memory indexing
- Fast lookup by Sharpe, reward, market, phase, timesteps
- Automatic ranking (global, per-market, per-phase)
- Thread-safe operations
- Metadata validation and enrichment

Author: Javier (@javiertradess)
Version: 1.0.0
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging


@dataclass
class CheckpointEntry:
    """Individual checkpoint entry in the registry"""
    path: str
    market: str
    phase: int
    timesteps: int
    event_tag: str
    created_at: str

    # Performance metrics
    val_reward: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    total_return: float

    # Rankings
    rank_global: int = 0
    rank_market: int = 0
    rank_phase: int = 0

    # Optional metadata
    seed: Optional[int] = None
    learning_rate: Optional[float] = None
    training_elapsed_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointEntry':
        """Create from dictionary"""
        return cls(**data)


class CheckpointRegistry:
    """
    Global registry for all training checkpoints.

    Provides fast lookup, ranking, and filtering of checkpoints across
    markets, phases, and metrics. Uses JSON storage with in-memory indices
    for performance.

    Thread-safe with read-write locking.
    """

    SCHEMA_VERSION = "2.0"

    def __init__(self, registry_path: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the checkpoint registry.

        Args:
            registry_path: Path to registry JSON file (default: models/registry.json)
            logger: Optional logger instance
        """
        if registry_path is None:
            registry_path = Path("models/registry.json")

        self.registry_path = Path(registry_path)
        self.logger = logger or logging.getLogger(__name__)

        # Thread safety
        self._lock = threading.RLock()

        # In-memory data structures
        self.entries: List[CheckpointEntry] = []
        self.indices: Dict[str, Any] = {
            'by_sharpe': [],      # Sorted paths by Sharpe ratio (descending)
            'by_reward': [],      # Sorted paths by val_reward (descending)
            'by_timesteps': [],   # Sorted paths by timesteps (ascending)
            'by_market': {},      # Market -> [paths]
            'by_phase': {},       # Phase -> [paths]
            'by_event': {},       # Event tag -> [paths]
        }

        # Load existing registry
        self._load()

    def _load(self) -> None:
        """Load registry from JSON file"""
        with self._lock:
            if not self.registry_path.exists():
                self.logger.info(f"Creating new registry at {self.registry_path}")
                self._save()
                return

            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)

                # Validate schema version
                if data.get('schema_version') != self.SCHEMA_VERSION:
                    self.logger.warning(
                        f"Registry schema version mismatch: "
                        f"expected {self.SCHEMA_VERSION}, got {data.get('schema_version')}"
                    )

                # Load entries
                self.entries = [
                    CheckpointEntry.from_dict(entry_data)
                    for entry_data in data.get('checkpoints', [])
                ]

                # Rebuild indices
                self._rebuild_indices()

                self.logger.info(f"Loaded {len(self.entries)} checkpoints from registry")

            except Exception as e:
                self.logger.error(f"Failed to load registry: {e}")
                self.entries = []
                self._rebuild_indices()

    def _save(self) -> None:
        """Save registry to JSON file"""
        with self._lock:
            # Ensure directory exists
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'schema_version': self.SCHEMA_VERSION,
                'last_updated': datetime.now().isoformat(),
                'total_checkpoints': len(self.entries),
                'checkpoints': [entry.to_dict() for entry in self.entries],
                'indices': {
                    'by_sharpe': self.indices['by_sharpe'][:10],  # Top 10 only
                    'by_reward': self.indices['by_reward'][:10],
                    'markets': list(self.indices['by_market'].keys()),
                    'phases': list(self.indices['by_phase'].keys()),
                }
            }

            # Atomic write with temp file
            temp_path = self.registry_path.with_suffix('.json.tmp')
            try:
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2)

                temp_path.replace(self.registry_path)
                self.logger.debug(f"Saved registry with {len(self.entries)} checkpoints")

            except Exception as e:
                self.logger.error(f"Failed to save registry: {e}")
                if temp_path.exists():
                    temp_path.unlink()

    def _rebuild_indices(self) -> None:
        """Rebuild all in-memory indices from entries"""
        with self._lock:
            # Clear indices
            self.indices = {
                'by_sharpe': [],
                'by_reward': [],
                'by_timesteps': [],
                'by_market': {},
                'by_phase': {},
                'by_event': {},
            }

            # Sort by Sharpe ratio (descending)
            sorted_by_sharpe = sorted(
                self.entries,
                key=lambda e: e.sharpe_ratio,
                reverse=True
            )
            self.indices['by_sharpe'] = [e.path for e in sorted_by_sharpe]

            # Sort by reward (descending)
            sorted_by_reward = sorted(
                self.entries,
                key=lambda e: e.val_reward,
                reverse=True
            )
            self.indices['by_reward'] = [e.path for e in sorted_by_reward]

            # Sort by timesteps (ascending)
            sorted_by_timesteps = sorted(
                self.entries,
                key=lambda e: e.timesteps
            )
            self.indices['by_timesteps'] = [e.path for e in sorted_by_timesteps]

            # Group by market
            for entry in self.entries:
                if entry.market not in self.indices['by_market']:
                    self.indices['by_market'][entry.market] = []
                self.indices['by_market'][entry.market].append(entry.path)

            # Group by phase
            for entry in self.entries:
                phase_key = str(entry.phase)
                if phase_key not in self.indices['by_phase']:
                    self.indices['by_phase'][phase_key] = []
                self.indices['by_phase'][phase_key].append(entry.path)

            # Group by event tag
            for entry in self.entries:
                if entry.event_tag not in self.indices['by_event']:
                    self.indices['by_event'][entry.event_tag] = []
                self.indices['by_event'][entry.event_tag].append(entry.path)

            # Update rankings
            self._update_rankings()

    def _update_rankings(self) -> None:
        """Update global, market, and phase rankings for all entries"""
        with self._lock:
            # Global rankings (by Sharpe)
            for rank, path in enumerate(self.indices['by_sharpe'], start=1):
                entry = self._find_entry_by_path(path)
                if entry:
                    entry.rank_global = rank

            # Market rankings
            for market, paths in self.indices['by_market'].items():
                # Sort by Sharpe within market
                market_entries = [self._find_entry_by_path(p) for p in paths]
                market_entries = [e for e in market_entries if e is not None]
                market_entries.sort(key=lambda e: e.sharpe_ratio, reverse=True)

                for rank, entry in enumerate(market_entries, start=1):
                    entry.rank_market = rank

            # Phase rankings
            for phase, paths in self.indices['by_phase'].items():
                # Sort by Sharpe within phase
                phase_entries = [self._find_entry_by_path(p) for p in paths]
                phase_entries = [e for e in phase_entries if e is not None]
                phase_entries.sort(key=lambda e: e.sharpe_ratio, reverse=True)

                for rank, entry in enumerate(phase_entries, start=1):
                    entry.rank_phase = rank

    def _find_entry_by_path(self, path: str) -> Optional[CheckpointEntry]:
        """Find entry by checkpoint path"""
        for entry in self.entries:
            if entry.path == path:
                return entry
        return None

    def register_checkpoint(
        self,
        checkpoint_path: Path,
        metadata: Dict[str, Any]
    ) -> CheckpointEntry:
        """
        Register a new checkpoint in the registry.

        Args:
            checkpoint_path: Path to checkpoint .zip file
            metadata: Checkpoint metadata dict

        Returns:
            Created CheckpointEntry

        Raises:
            ValueError: If required metadata is missing
        """
        with self._lock:
            # Validate required fields
            required = ['phase', 'market', 'timesteps', 'val_reward', 'sharpe_ratio']
            missing = [k for k in required if k not in metadata]
            if missing:
                raise ValueError(f"Missing required metadata: {missing}")

            # Create entry
            entry = CheckpointEntry(
                path=str(checkpoint_path),
                market=metadata['market'],
                phase=metadata['phase'],
                timesteps=metadata['timesteps'],
                event_tag=metadata.get('event_tag', 'unknown'),
                created_at=metadata.get('timestamp', datetime.now().isoformat()),
                val_reward=metadata['val_reward'],
                sharpe_ratio=metadata['sharpe_ratio'],
                win_rate=metadata.get('win_rate', 0.0),
                max_drawdown=metadata.get('max_drawdown', 0.0),
                total_return=metadata.get('total_return', 0.0),
                seed=metadata.get('seed'),
                learning_rate=metadata.get('learning_rate'),
                training_elapsed_seconds=metadata.get('training_elapsed_seconds'),
            )

            # Check for duplicate
            existing = self._find_entry_by_path(str(checkpoint_path))
            if existing:
                self.logger.warning(f"Checkpoint already registered: {checkpoint_path}")
                # Update existing entry
                self.entries.remove(existing)

            # Add to entries
            self.entries.append(entry)

            # Rebuild indices
            self._rebuild_indices()

            # Save to disk
            self._save()

            self.logger.info(
                f"Registered checkpoint: {checkpoint_path.name} "
                f"(Sharpe: {entry.sharpe_ratio:.2f}, Global rank: {entry.rank_global})"
            )

            return entry

    def get_best_checkpoint(
        self,
        metric: str = 'sharpe_ratio',
        market: Optional[str] = None,
        phase: Optional[int] = None,
        min_timesteps: Optional[int] = None,
        max_timesteps: Optional[int] = None
    ) -> Optional[CheckpointEntry]:
        """
        Get the best checkpoint by a given metric.

        Args:
            metric: Metric to optimize ('sharpe_ratio', 'val_reward', etc.)
            market: Filter by market (None = all markets)
            phase: Filter by phase (None = all phases)
            min_timesteps: Minimum timesteps threshold
            max_timesteps: Maximum timesteps threshold

        Returns:
            Best checkpoint entry, or None if no matches
        """
        with self._lock:
            candidates = self.entries.copy()

            # Apply filters
            if market:
                candidates = [e for e in candidates if e.market == market]
            if phase is not None:
                candidates = [e for e in candidates if e.phase == phase]
            if min_timesteps is not None:
                candidates = [e for e in candidates if e.timesteps >= min_timesteps]
            if max_timesteps is not None:
                candidates = [e for e in candidates if e.timesteps <= max_timesteps]

            if not candidates:
                return None

            # Sort by metric (descending)
            candidates.sort(key=lambda e: getattr(e, metric), reverse=True)
            return candidates[0]

    def get_recent_checkpoints(
        self,
        n: int = 10,
        market: Optional[str] = None,
        phase: Optional[int] = None
    ) -> List[CheckpointEntry]:
        """
        Get the N most recent checkpoints.

        Args:
            n: Number of checkpoints to return
            market: Filter by market
            phase: Filter by phase

        Returns:
            List of recent checkpoints (most recent first)
        """
        with self._lock:
            candidates = self.entries.copy()

            # Apply filters
            if market:
                candidates = [e for e in candidates if e.market == market]
            if phase is not None:
                candidates = [e for e in candidates if e.phase == phase]

            # Sort by creation time (most recent first)
            candidates.sort(key=lambda e: e.created_at, reverse=True)
            return candidates[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with statistics about registered checkpoints
        """
        with self._lock:
            if not self.entries:
                return {
                    'total_checkpoints': 0,
                    'markets': [],
                    'phases': [],
                    'best_sharpe': None,
                    'best_reward': None,
                }

            return {
                'total_checkpoints': len(self.entries),
                'markets': list(self.indices['by_market'].keys()),
                'phases': sorted([int(p) for p in self.indices['by_phase'].keys()]),
                'best_sharpe': max(e.sharpe_ratio for e in self.entries),
                'best_reward': max(e.val_reward for e in self.entries),
                'avg_sharpe': sum(e.sharpe_ratio for e in self.entries) / len(self.entries),
                'avg_reward': sum(e.val_reward for e in self.entries) / len(self.entries),
                'checkpoints_by_market': {
                    market: len(paths)
                    for market, paths in self.indices['by_market'].items()
                },
                'checkpoints_by_phase': {
                    phase: len(paths)
                    for phase, paths in self.indices['by_phase'].items()
                },
            }

    def print_summary(self) -> None:
        """Print a human-readable summary of the registry"""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("CHECKPOINT REGISTRY SUMMARY")
        print("="*60)
        print(f"Total Checkpoints: {stats['total_checkpoints']}")
        print(f"Markets: {', '.join(stats.get('markets', []))}")
        print(f"Phases: {', '.join(map(str, stats.get('phases', [])))}")

        if stats['total_checkpoints'] > 0:
            print(f"\nBest Sharpe Ratio: {stats['best_sharpe']:.2f}")
            print(f"Best Val Reward: {stats['best_reward']:.3f}")
            print(f"Average Sharpe: {stats['avg_sharpe']:.2f}")
            print(f"Average Reward: {stats['avg_reward']:.3f}")

            print("\nCheckpoints by Market:")
            for market, count in sorted(stats.get('checkpoints_by_market', {}).items()):
                print(f"  {market}: {count}")

            print("\nCheckpoints by Phase:")
            for phase, count in sorted(stats.get('checkpoints_by_phase', {}).items()):
                print(f"  Phase {phase}: {count}")

        print("="*60 + "\n")
