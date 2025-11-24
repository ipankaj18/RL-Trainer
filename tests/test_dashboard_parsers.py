"""Unit tests for the dashboard parsers/state."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.parsers import ParserContext, parse_line
from dashboard.state import DashboardState


def test_parse_line_extracts_phase_and_metrics() -> None:
    context = ParserContext()
    events = parse_line("PHASE 1: Entry Learning (Production Mode)", "sample.log", context)
    assert any(getattr(event, "phase", None) == "Phase 1" for event in events)

    parse_line("rollout/", "sample.log", context)
    metric_events = parse_line("    ep_len_mean       2000", "sample.log", context)
    assert metric_events, "Expected metric event"
    metric = metric_events[-1]
    assert getattr(metric, "key", None) == "ep_len_mean"
    assert getattr(metric, "value", None) == 2000.0


def test_dashboard_state_tracks_sections() -> None:
    state = DashboardState(history_size=5)
    context = ParserContext()
    for line in [
        "PHASE 2: Refinement",
        "train/",
        "    loss              0.1",
        "    loss              0.2",
    ]:
        events = parse_line(line, "sample.log", context)
        state.apply_events(events, timestamp=datetime(2025, 1, 1))
    sections = state.section_snapshots()
    assert sections[0].section == "train"
    assert sections[0].values["loss"] == 0.2
    history = state.history_for("train", "loss")
    assert len(history) == 2
