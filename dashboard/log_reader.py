"""Log tailing utilities for the dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence


@dataclass
class LogTailer:
    """Incrementally read new content from a log file."""

    path: Path
    position: int = 0

    def read_new_lines(self) -> List[str]:
        """Return every line appended since the last call."""

        if not self.path.exists():
            return []
        lines: List[str] = []
        with self.path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(self.position)
            chunk = handle.read()
            self.position = handle.tell()
        if chunk:
            lines = chunk.splitlines()
        return lines


def discover_logs(
    patterns: Iterable[str],
    auto_dirs: Sequence[str] | None = None,
    extensions: Sequence[str] | None = None,
) -> List[Path]:
    """Expand every glob pattern plus auto-discovered directories."""

    discovered: List[Path] = []
    for pattern in patterns:
        discovered.extend(Path().glob(pattern))

    if auto_dirs:
        for directory in auto_dirs:
            base = Path(directory)
            if not base.exists():
                continue
            if extensions:
                for ext in extensions:
                    discovered.extend(base.rglob(f"*{ext}"))
            else:
                discovered.extend(base.rglob("*"))

    unique = sorted({path.resolve() for path in discovered if path.exists()})
    return unique
