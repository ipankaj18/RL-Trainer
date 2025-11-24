# CLI Dashboard

The dashboard package provides a light-weight terminal UI that can be launched in a parallel Jupyter/RunPod terminal to follow long running training jobs.

## Launching

```bash
python dashboard/cli.py --log-glob "logs/pipeline*.log" --refresh 2.0
```

Arguments:

- `--config`: optional YAML file containing `log_patterns`, `refresh_interval`, and `history_size` overrides.
- `--log-glob`: may be specified multiple times to point at any log stream that should be monitored.
- `--refresh`: UI refresh cadence in seconds; defaults to `2.0`.

## Features

- Automatically tails matching log files and groups metrics into sections such as `rollout`, `train`, `time`, and `checkpoint`.
- Detects curriculum phase announcements (e.g., `Phase 1: Entry Learning`) and summarizes their most recent status.
- Displays the freshest sections in dedicated panels together with tiny ASCII sparklines for numeric metrics.
- Maintains a bounded history window (default 200 samples) so that trends are visible without overwhelming terminal output.

## Extending

- Update `dashboard/parsers.py` if log formats change or additional sections need to be tracked.
- Increase `history_size` in a custom config file to run longer charts:

```yaml
refresh_interval: 1.0
history_size: 400
log_patterns:
  - logs/pipeline_production_*.log
  - logs/pipeline_test_*.log
```

- Add richer UI widgets inside `dashboard/ui.py` when more real-time signals become available (GPU stats, evaluator outputs, etc.).
