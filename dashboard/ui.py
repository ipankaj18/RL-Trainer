"""Rich renderables for the dashboard."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Sequence

from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .state import DashboardState, MetricSnapshot, PhaseStatus


class DashboardUI:
    """Transforms dashboard state into Rich layouts."""

    def __init__(self, max_sections: int = 3) -> None:
        self.max_sections = max_sections

    def render(self, state: DashboardState) -> RenderableType:
        """Build the full dashboard layout."""

        layout = Layout()
        layout.split_column(
            Layout(self._render_header(), size=3),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(self._render_phases(state), size=40),
            Layout(self._render_sections(state)),
        )
        return layout

    def _render_header(self) -> RenderableType:
        text = Text("RL Training Dashboard", style="bold cyan")
        text.append("  •  ")
        text.append(
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            style="bold green",
        )
        return Panel(text, title="Session", border_style="cyan")

    def _render_phases(self, state: DashboardState) -> RenderableType:
        table = Table(title="Curriculum Phases", expand=True)
        table.add_column("Phase", justify="left", style="bold")
        table.add_column("Status")
        table.add_column("Updated")
        phases = state.latest_phase_rows()
        if not phases:
            table.add_row("-", "waiting for logs", "-")
            return table
        for phase in phases:
            table.add_row(
                phase.name,
                phase.status,
                phase.updated_at.strftime("%H:%M:%S"),
            )
        return table

    def _render_sections(self, state: DashboardState) -> RenderableType:
        snapshots = state.section_snapshots()[: self.max_sections]
        if not snapshots:
            return Panel("Awaiting metrics…", title="Metrics", border_style="magenta")
        panels = [self._render_snapshot(state, snapshot) for snapshot in snapshots]
        return Group(*panels)

    def _render_snapshot(self, state: DashboardState, snapshot: MetricSnapshot) -> RenderableType:
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        numeric_keys = [k for k, v in snapshot.values.items() if isinstance(v, float)]
        numeric_keys.sort()
        display_items = list(snapshot.values.items())[:4]
        for key, value in display_items:
            table.add_row(key, f"{value}")
        spark = ""
        if numeric_keys:
            series = state.history_for(snapshot.section, numeric_keys[0])
            spark = _sparkline([value for _, value in series])
        caption = f"last update: {snapshot.updated_at.strftime('%H:%M:%S')} {spark}".strip()
        return Panel(table, title=snapshot.section, subtitle=caption, border_style="magenta")


def _sparkline(values: Sequence[float], width: int = 20) -> str:
    """Render a lightweight ASCII sparkline for the provided values."""

    if not values:
        return ""
    trimmed = values[-width:]
    minimum = min(trimmed)
    maximum = max(trimmed)
    if minimum == maximum:
        return "-" * min(len(trimmed), width)
    charset = " .:-=+*#%@"
    scale = (len(charset) - 1) / (maximum - minimum)
    output = []
    for value in trimmed:
        idx = int((value - minimum) * scale)
        output.append(charset[idx])
    return "".join(output)
