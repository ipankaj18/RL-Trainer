"""Entry point for the CLI dashboard."""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

from rich.live import Live

if __package__:
    from .config import load_dashboard_config
    from .log_reader import LogTailer, discover_logs
    from .parsers import ParserContext, parse_line
    from .state import DashboardState
    from .ui import DashboardUI
else:  # pragma: no cover - script execution fallback
    import sys

    PACKAGE_ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = PACKAGE_ROOT.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from dashboard.config import load_dashboard_config
    from dashboard.log_reader import LogTailer, discover_logs
    from dashboard.parsers import ParserContext, parse_line
    from dashboard.state import DashboardState
    from dashboard.ui import DashboardUI


def main(argv: Iterable[str] | None = None) -> None:
    """Run the dashboard event loop."""

    args = _parse_args(argv)
    config_path = Path(args.config) if args.config else None
    overrides = {
        "log_patterns": args.log_glob or None,
        "refresh_interval": args.refresh if args.refresh else None,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    config = load_dashboard_config(config_path, overrides)
    state = DashboardState(history_size=config.history_size)
    ui = DashboardUI()

    tailers: Dict[Path, LogTailer] = {}
    contexts: Dict[Path, ParserContext] = {}

    refresh_hz = 1.0 / config.refresh_interval if config.refresh_interval > 0 else 1.0

    with Live(ui.render(state), refresh_per_second=max(1, int(refresh_hz))) as live:
        try:
            while True:
                _ensure_tailers(config.log_patterns, tailers, contexts)
                _consume_logs(tailers, contexts, state)
                live.update(ui.render(state))
                time.sleep(config.refresh_interval)
        except KeyboardInterrupt:
            pass


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime CLI dashboard for RL training logs")
    parser.add_argument("--config", help="Optional path to dashboard YAML config")
    parser.add_argument(
        "--log-glob",
        action="append",
        help="Glob pattern for log files (can be provided multiple times)",
    )
    parser.add_argument("--refresh", type=float, help="UI refresh interval in seconds")
    return parser.parse_args(argv)


def _ensure_tailers(
    patterns: Iterable[str],
    tailers: Dict[Path, LogTailer],
    contexts: Dict[Path, ParserContext],
) -> None:
    for path in discover_logs(patterns):
        if path not in tailers:
            tailers[path] = LogTailer(path=path)
            contexts[path] = ParserContext()


def _consume_logs(
    tailers: Dict[Path, LogTailer],
    contexts: Dict[Path, ParserContext],
    state: DashboardState,
) -> None:
    for path, tailer in tailers.items():
        lines = tailer.read_new_lines()
        if not lines:
            continue
        context = contexts[path]
        for line in lines:
            events = parse_line(line, source=str(path.name), context=context)
            if not events:
                continue
            state.apply_events(events, timestamp=datetime.now(timezone.utc))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
