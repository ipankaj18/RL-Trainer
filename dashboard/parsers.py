"""Parsing utilities for dashboard events."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Sequence

SECTION_RE = re.compile(r"^([A-Za-z0-9_\- ]+)/\s*$")
PHASE_RE = re.compile(r"(PHASE|Phase)\s*(?P<phase>\d+)(?::|\s+-)\s*(?P<label>.+)")
COLON_METRIC_RE = re.compile(r"^([A-Za-z0-9_./-]+)\s*[:=]\s*(.+)$")


@dataclass
class ParserContext:
    """Per-file parser state (currently only tracks the active section)."""

    section: str | None = None


@dataclass
class SectionEvent:
    """Signals that following metrics belong to a section."""

    source: str
    section: str


@dataclass
class PhaseEvent:
    """Marks which curriculum phase produced the surrounding logs."""

    source: str
    phase: str
    status: str


@dataclass
class MetricEvent:
    """Represents a scalar metric extracted from the logs."""

    source: str
    section: str | None
    key: str
    value: float | str


ParsedEvent = SectionEvent | PhaseEvent | MetricEvent


def parse_line(line: str, source: str, context: ParserContext) -> List[ParsedEvent]:
    """Convert a single log line into zero or more structured events."""

    stripped = line.strip()
    if not stripped or stripped.startswith("="):
        return []

    events: List[ParsedEvent] = []

    section_match = SECTION_RE.match(stripped)
    if section_match:
        section = section_match.group(1).strip().lower().replace(" ", "_")
        context.section = section
        events.append(SectionEvent(source=source, section=section))
        return events

    phase_match = PHASE_RE.search(line)
    if phase_match:
        phase_id = phase_match.group("phase")
        phase_label = phase_match.group("label").strip()
        events.append(
            PhaseEvent(source=source, phase=f"Phase {phase_id}", status=phase_label)
        )

    metric_event = _parse_metric(stripped, source, context)
    if metric_event:
        events.append(metric_event)

    return events


def _parse_metric(line: str, source: str, context: ParserContext) -> MetricEvent | None:
    """Try to interpret the provided line as a metric."""

    colon_match = COLON_METRIC_RE.match(line)
    if colon_match:
        key = colon_match.group(1).strip()
        value_text = colon_match.group(2).strip()
        return MetricEvent(
            source=source,
            section=context.section,
            key=_sanitize_key(key),
            value=_coerce_value(value_text),
        )

    if context.section is None:
        return None

    tokens = line.split()
    if len(tokens) < 2:
        return None
    key = _sanitize_key(tokens[0])
    value_text = tokens[-1]
    if not _looks_numeric(value_text):
        return None
    return MetricEvent(
        source=source,
        section=context.section,
        key=key,
        value=_coerce_value(value_text),
    )


def _sanitize_key(key: str) -> str:
    return key.strip().lower()


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _coerce_value(value: str) -> float | str:
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return value
