"""Textual-powered dashboard UI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Static

if __package__:
    from .config import load_dashboard_config
    from .log_reader import LogTailer, discover_logs
    from .parsers import ParserContext, parse_line
    from .state import DashboardState
else:  # pragma: no cover - CLI entry
    import sys

    PACKAGE_ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = PACKAGE_ROOT.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from dashboard.config import load_dashboard_config
    from dashboard.log_reader import LogTailer, discover_logs
    from dashboard.parsers import ParserContext, parse_line
    from dashboard.state import DashboardState


class TrendConfig:
    def __init__(self) -> None:
        self.metrics = (
            ("eval", "mean_reward", "Eval mean reward"),
            ("rollout", "ep_rew_mean", "Rollout reward"),
            ("train", "loss", "Train loss"),
        )


class DashboardTextualApp(App):
    """Renders the dashboard using a Textual layout."""

    CSS = """
    #body-row {
        height: 1fr;
    }
    #phase-panel {
        min-width: 40;
        width: 44;
    }
    #right-column {
        height: 1fr;
    }
    #metrics-panel {
        height: 1fr;
        min-height: 20;
    }
    #trends-panel {
        height: 11;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_dashboard", "Refresh"),
    ]

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.state = DashboardState(history_size=config.history_size)
        self.trend_config = TrendConfig()
        self.tailers: Dict[Path, LogTailer] = {}
        self.contexts: Dict[Path, ParserContext] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="body"):
            with Horizontal(id="body-row"):
                yield Static(id="phase-panel")
                with Vertical(id="right-column"):
                    yield Static(id="metrics-panel")
                    yield Static(id="trends-panel")
        yield Footer()

    async def on_mount(self) -> None:  # pragma: no cover - UI runtime
        self.phase_widget: Static = self.query_one("#phase-panel")
        self.metrics_widget: Static = self.query_one("#metrics-panel")
        self.trends_widget: Static = self.query_one("#trends-panel")
        await self._refresh_dashboard()
        self.set_interval(self.config.refresh_interval, self._refresh_dashboard)

    async def action_refresh_dashboard(self) -> None:
        await self._refresh_dashboard()

    async def _refresh_dashboard(self) -> None:
        self._ensure_tailers()
        self._consume_logs()
        self.phase_widget.update(_render_phase_table(self.state))
        self.metrics_widget.update(_render_metric_columns(self.state))
        self.trends_widget.update(_render_trend_panel(self.state, self.trend_config.metrics))

    def _ensure_tailers(self) -> None:
        auto_dirs = self.config.log_dirs if self.config.auto_discover else None
        extensions = self.config.extensions if self.config.auto_discover else None
        for path in discover_logs(
            self.config.log_patterns,
            auto_dirs=auto_dirs,
            extensions=extensions,
        ):
            if path not in self.tailers:
                self.tailers[path] = LogTailer(path=path)
                self.contexts[path] = ParserContext()

    def _consume_logs(self) -> None:
        for path, tailer in list(self.tailers.items()):
            lines = tailer.read_new_lines()
            if not lines:
                continue
            context = self.contexts[path]
            for line in lines:
                events = parse_line(line, source=str(path.name), context=context)
                if not events:
                    continue
                self.state.apply_events(events, timestamp=datetime.now(timezone.utc))


def _render_phase_table(state: DashboardState) -> Table:
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


def _render_metric_columns(state: DashboardState) -> Panel:
    snapshots = state.section_snapshots()
    if not snapshots:
        return Panel("Awaiting metrics…", title="Latest Metrics", border_style="magenta")
    cards = []
    for snapshot in snapshots[:6]:
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        items = list(snapshot.values.items())[:4]
        for key, value in items:
            table.add_row(key, f"{value}")
        cards.append(
            Panel(
                table,
                title=snapshot.section,
                subtitle=snapshot.updated_at.strftime("%H:%M:%S"),
                border_style="magenta",
            )
        )
    return Panel(Columns(cards, equal=True, expand=True), title="Latest Metrics", border_style="magenta")


def _render_trend_panel(state: DashboardState, trend_metrics) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column("Metric", style="bold", ratio=1)
    table.add_column("Trend", ratio=3)
    added = False
    for section, key, label in trend_metrics:
        history = state.history_for(section, key)
        if not history:
            continue
        values = [value for _, value in history]
        if not values:
            continue
        spark = _sparkline(values, width=32)
        latest = values[-1]
        table.add_row(label, f"{spark}  {latest:.3f}")
        added = True
    if not added:
        table.add_row("waiting for data", "-")
    return Panel(table, title="Key Trends", border_style="green")


def _sparkline(values, width: int = 20) -> str:
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


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Textual dashboard for RL logs")
    parser.add_argument("--config", help="Optional path to dashboard YAML config")
    parser.add_argument(
        "--log-glob",
        action="append",
        help="Glob pattern for log files (can be provided multiple times)",
    )
    parser.add_argument("--refresh", type=float, help="UI refresh interval in seconds")
    parser.add_argument(
        "--log-dir",
        action="append",
        help="Directory to recursively monitor for new log files (default: logs)",
    )
    parser.add_argument(
        "--extension",
        action="append",
        help="File extension to include during auto-discovery",
    )
    parser.add_argument(
        "--disable-auto-discovery",
        action="store_true",
        help="Disable automatic scanning of log directories",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = Path(args.config) if args.config else None
    overrides = {}
    if args.log_glob:
        overrides["log_patterns"] = args.log_glob
    if args.refresh:
        overrides["refresh_interval"] = args.refresh
    if args.log_dir:
        overrides["log_dirs"] = args.log_dir
    if args.extension:
        overrides["extensions"] = args.extension
    if args.disable_auto_discovery:
        overrides["auto_discover"] = False
    config = load_dashboard_config(config_path, overrides)
    DashboardTextualApp(config).run()


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
