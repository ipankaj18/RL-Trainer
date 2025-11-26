# CLI Dashboard

The dashboard package provides a light-weight terminal UI that can be launched in a parallel Jupyter/RunPod terminal to follow long running training jobs.

## Launching

```bash
python dashboard/cli.py --refresh 2.0
```

```bash
python dashboard/textual_app.py --refresh 1.0
```

The CLI now auto-discovers every `.log`, `.txt`, or `.out` file under `logs/`, so you can simply start it without specifying glob patterns. The Textual app shares the same options and adds richer multi-panel visuals. Key flags:

- `--config`: optional YAML file containing `log_patterns`, `log_dirs`, `extensions`, `refresh_interval`, and `history_size` overrides.
- `--log-dir`: add additional folders to scan recursively (e.g., `--log-dir tensorboard_logs`).
- `--extension`: append extra extensions (for custom suffixes like `.training`).
- `--log-glob`: still available when you need explicit glob control.
- `--disable-auto-discovery`: turn off the recursive scan if you only want explicit patterns.
- `--refresh`: UI refresh cadence in seconds; defaults to `2.0`.

## Features

- Automatically tails matching log files and groups metrics into sections such as `rollout`, `train`, `time`, and `checkpoint`.
- Detects curriculum phase announcements (e.g., `Phase 1: Entry Learning`) and summarizes their most recent status.
- Displays the freshest sections in dedicated panels together with tiny ASCII sparklines for numeric metrics.
- Adds a "Key Trends" panel that plots reward/loss sparklines (e.g., eval mean reward, rollout reward, train loss) so progress can be gauged at a glance.
- Textual mode arranges those cards into a split dashboard (phase table + metric grid + trend strip) using mouse/keyboard-friendly widgets.
- Maintains a bounded history window (default 200 samples) so that trends are visible without overwhelming terminal output.

## Extending

- Update `dashboard/parsers.py` if log formats change or additional sections need to be tracked.
- Use a config file to change discovery defaults, for example:

```yaml
refresh_interval: 1.0
history_size: 400
log_patterns:
  - logs/pipeline_production_*.log
  - logs/pipeline_test_*.log
log_dirs:
  - logs
  - tensorboard_logs
extensions:
  - .log
  - .txt
  - .metrics
```

- Add richer UI widgets inside `dashboard/ui.py` when more real-time signals become available (GPU stats, evaluator outputs, etc.).
