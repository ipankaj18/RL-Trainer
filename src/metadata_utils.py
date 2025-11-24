"""Utility helpers for persisting and loading training metadata."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def metadata_path(resource_path: str) -> str:
    """Return the metadata path associated with a resource file."""
    return f"{resource_path}.meta.json"


def write_metadata(resource_path: str, metadata: Dict[str, Any]) -> str:
    """Write metadata next to a resource file.

    Args:
        resource_path: Path to the model or artifact the metadata describes.
        metadata: JSON-serializable dictionary with metadata fields.

    Returns:
        The path to the metadata file written.
    """
    meta_path = metadata_path(resource_path)
    payload = dict(metadata) if metadata is not None else {}
    payload.setdefault("resource", os.path.abspath(resource_path))
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())

    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return meta_path


def read_metadata(resource_path: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Read metadata for a resource if it exists."""
    meta_path = metadata_path(resource_path)
    if not os.path.exists(meta_path):
        return default
    with open(meta_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def require_metadata_field(metadata: Dict[str, Any], field: str) -> Any:
    """Return a required metadata field or raise a ValueError."""
    if metadata is None or field not in metadata:
        raise ValueError(f"Missing required metadata field '{field}'")
    return metadata[field]


# Checkpoint-specific metadata functions

def write_checkpoint_metadata(checkpoint_path: str, metadata: Dict[str, Any]) -> str:
    """
    Write checkpoint metadata with standardized fields.

    This is a specialized version of write_metadata() for training checkpoints
    that ensures consistent metadata structure across all checkpoints.

    Args:
        checkpoint_path: Path to checkpoint file (without extension)
        metadata: Dictionary with checkpoint metadata

    Returns:
        Path to metadata file written

    Expected Metadata Fields:
        Standard:
            - phase: Training phase (1, 2, or 3)
            - market: Market symbol (ES, NQ, etc.)
            - timesteps: Training timesteps at checkpoint
            - seed: Random seed
            - event_tag: Event type (periodic, best, phase_end, etc.)
            - timestamp: ISO timestamp

        Metrics:
            - val_reward: Mean validation reward
            - sharpe_ratio: Sharpe ratio from evaluation
            - win_rate: Win rate (% profitable episodes)
            - max_drawdown: Maximum drawdown
            - total_return: Cumulative return

        Runtime:
            - training_elapsed_seconds: Wall-clock training time
            - eval_episodes_run: Total eval episodes
            - learning_rate: Current learning rate
            - n_envs: Number of parallel environments

        Phase 3 LLM (optional):
            - reasoning_usage_rate: % of decisions using LLM
            - llm_confidence_avg: Average LLM confidence
            - rl_llm_agreement_rate: RL-LLM agreement rate
    """
    # Use checkpoint-specific metadata path: {checkpoint_path}_metadata.json
    meta_path = f"{checkpoint_path}_metadata.json"

    # Prepare metadata payload
    payload = dict(metadata) if metadata is not None else {}
    payload.setdefault("checkpoint_path", os.path.abspath(checkpoint_path))
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())

    # Add checkpoint schema version
    payload.setdefault("schema_version", "1.0")

    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    return meta_path


def read_checkpoint_metadata(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Read checkpoint metadata.

    Args:
        checkpoint_path: Path to checkpoint file (with or without extension)

    Returns:
        Metadata dictionary or None if not found
    """
    # Remove extension if present
    base_path = checkpoint_path.replace('.zip', '')

    # Try checkpoint-specific metadata path first
    meta_path = f"{base_path}_metadata.json"

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # Fallback to standard metadata path
    return read_metadata(base_path)


def parse_checkpoint_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse checkpoint filename to extract embedded metrics.

    Format: {market}_ts-{timesteps:07d}_evt-{event}_val-{val_reward:+.3f}_sharpe-{sharpe:+.2f}_seed-{seed}.zip

    Args:
        filename: Checkpoint filename (with or without path/extension)

    Returns:
        Dictionary with parsed fields or None if parsing fails

    Example:
        >>> parse_checkpoint_filename('NQ_ts-0025000_evt-periodic_val-+0.112_sharpe-+1.45_seed-42.zip')
        {
            'market': 'NQ',
            'timesteps': 25000,
            'event_tag': 'periodic',
            'val_reward': 0.112,
            'sharpe_ratio': 1.45,
            'seed': 42
        }
    """
    import re
    from pathlib import Path

    # Remove path and extension
    name = Path(filename).stem

    # Regex pattern for checkpoint filename
    pattern = r'(?P<market>[A-Z0-9]+)_ts-(?P<timesteps>\d+)_evt-(?P<event>[\w\-]+)_val-(?P<val_reward>[+-]?\d+\.\d+)_sharpe-(?P<sharpe>[+-]?\d+\.\d+)_seed-(?P<seed>\d+)'

    match = re.match(pattern, name)
    if not match:
        return None

    return {
        'market': match.group('market'),
        'timesteps': int(match.group('timesteps')),
        'event_tag': match.group('event'),
        'val_reward': float(match.group('val_reward')),
        'sharpe_ratio': float(match.group('sharpe')),
        'seed': int(match.group('seed'))
    }


def format_checkpoint_filename(
    market: str,
    timesteps: int,
    event_tag: str,
    val_reward: float,
    sharpe_ratio: float,
    seed: int,
    extension: str = '.zip'
) -> str:
    """
    Format checkpoint filename with embedded metrics.

    Args:
        market: Market symbol
        timesteps: Training timesteps
        event_tag: Event type
        val_reward: Validation reward
        sharpe_ratio: Sharpe ratio
        seed: Random seed
        extension: File extension (default: .zip)

    Returns:
        Formatted checkpoint filename

    Example:
        >>> format_checkpoint_filename('NQ', 25000, 'periodic', 0.112, 1.45, 42)
        'NQ_ts-0025000_evt-periodic_val-+0.112_sharpe-+1.45_seed-42.zip'
    """
    name = f"{market}_ts-{timesteps:07d}_evt-{event_tag}_val-{val_reward:+.3f}_sharpe-{sharpe_ratio:+.2f}_seed-{seed}"
    return name + extension


def validate_checkpoint_metadata(metadata: Dict[str, Any]) -> bool:
    """
    Validate that checkpoint metadata contains required fields.

    Args:
        metadata: Metadata dictionary to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = ['phase', 'market', 'timesteps', 'seed', 'event_tag']

    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Checkpoint metadata missing required field: {field}")

    # Validate types
    if not isinstance(metadata['phase'], int) or metadata['phase'] not in [1, 2, 3]:
        raise ValueError(f"Invalid phase: {metadata.get('phase')} (must be 1, 2, or 3)")

    if not isinstance(metadata['timesteps'], int) or metadata['timesteps'] < 0:
        raise ValueError(f"Invalid timesteps: {metadata.get('timesteps')} (must be non-negative integer)")

    if not isinstance(metadata['seed'], int):
        raise ValueError(f"Invalid seed: {metadata.get('seed')} (must be integer)")

    return True
