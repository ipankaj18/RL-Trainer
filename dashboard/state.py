"""State containers for the dashboard runtime."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, Iterable, List, MutableMapping, Tuple

from .parsers import MetricEvent, PhaseEvent, SectionEvent


@dataclass
class MetricSnapshot:
    """Stores the latest values for a metric section."""

    section: str
    values: Dict[str, float | str] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PhaseStatus:
    """Tracks where each training phase is in the pipeline."""

    name: str
    status: str
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DashboardState:
    """Mutable aggregation of parsed events."""

    def __init__(self, history_size: int = 200) -> None:
        self.history_size = history_size
        self.phases: Dict[str, PhaseStatus] = {}
        self.metric_sections: Dict[str, MetricSnapshot] = {}
        self.history: MutableMapping[str, Deque[Tuple[datetime, float]]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.active_section_by_source: Dict[str, str | None] = {}

    def apply_events(
        self,
        events: Iterable[SectionEvent | PhaseEvent | MetricEvent],
        timestamp: datetime | None = None,
    ) -> None:
        """Update in-memory state with parser events."""

        ts = timestamp or datetime.now(timezone.utc)
        for event in events:
            if isinstance(event, SectionEvent):
                self.active_section_by_source[event.source] = event.section
            elif isinstance(event, PhaseEvent):
                self.phases[event.phase] = PhaseStatus(
                    name=event.phase, status=event.status, updated_at=ts
                )
            elif isinstance(event, MetricEvent):
                section = event.section or self.active_section_by_source.get(event.source) or "general"
                snapshot = self.metric_sections.setdefault(
                    section, MetricSnapshot(section=section)
                )
                snapshot.values[event.key] = event.value
                snapshot.updated_at = ts
                if isinstance(event.value, float):
                    series_key = f"{section}:{event.key}"
                    self.history[series_key].append((ts, event.value))

    def latest_phase_rows(self) -> List[PhaseStatus]:
        """Return phases ordered by phase id when possible."""

        return sorted(self.phases.values(), key=lambda item: item.name)

    def section_snapshots(self) -> List[MetricSnapshot]:
        """Return metric sections ordered by freshness."""

        return sorted(
            self.metric_sections.values(),
            key=lambda snapshot: snapshot.updated_at,
            reverse=True,
        )

    def history_for(self, section: str, key: str) -> Deque[Tuple[datetime, float]]:
        """Return the rolling series for a given metric."""

        return self.history.get(f"{section}:{key}", deque())
