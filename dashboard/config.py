"""Configuration helpers for the dashboard CLI."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml


@dataclass
class DashboardConfig:
    """Runtime knobs for the dashboard."""

    log_patterns: Sequence[str] = field(default_factory=lambda: ("logs/*.log",))
    refresh_interval: float = 2.0
    history_size: int = 200

    @property
    def expanded_logs(self) -> List[Path]:
        """Return concrete paths for every configured glob pattern."""

        paths: List[Path] = []
        for pattern in self.log_patterns:
            paths.extend(Path().glob(pattern))
        return sorted({p.resolve() for p in paths if p.exists()})


def load_dashboard_config(path: Path | None, overrides: dict | None = None) -> DashboardConfig:
    """Load dashboard configuration from YAML if present."""

    config_data: dict = {}
    if path is not None and path.exists():
        config_data = yaml.safe_load(path.read_text()) or {}
    if overrides:
        config_data.update(overrides)
    patterns = config_data.get("log_patterns") or ("logs/*.log",)
    refresh_interval = float(config_data.get("refresh_interval", 2.0))
    history_size = int(config_data.get("history_size", 200))
    return DashboardConfig(
        log_patterns=tuple(str(p) for p in patterns),
        refresh_interval=refresh_interval,
        history_size=history_size,
    )
