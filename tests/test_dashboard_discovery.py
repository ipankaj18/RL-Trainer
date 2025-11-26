from __future__ import annotations

from dashboard.log_reader import discover_logs


def test_auto_discovery(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    log_file = logs_dir / "phase1.log"
    log_file.write_text("entry")

    paths = discover_logs([], auto_dirs=[str(logs_dir)], extensions=[".log"])
    assert log_file.resolve() in paths

    # Unknown extension should be ignored
    other = logs_dir / "metrics.bin"
    other.write_text("binary")
    paths = discover_logs([], auto_dirs=[str(logs_dir)], extensions=[".txt"])
    assert log_file.resolve() not in paths
