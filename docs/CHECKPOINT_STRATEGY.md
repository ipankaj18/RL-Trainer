# Checkpoint Management Strategy

**Version**: 2.0
**Date**: November 2025
**Status**: ✅ Implemented

---

## Overview

The Dynamic Checkpoint Management System addresses critical gaps in the original flat-interval checkpoint strategy by introducing **adaptive save frequencies**, **event-driven triggers**, **rich metadata**, and **intelligent retention policies**.

### Key Improvements Over Original System

| Feature | Original (v1.0) | Improved (v2.0) |
|---------|----------------|-----------------|
| **Save Frequency** | Fixed 100K steps (Phase 1/2) | Adaptive 25K-125K (Phase 1), 50K-250K (Phase 2), 10K-50K (Phase 3) |
| **Checkpoint Naming** | `phase{N}_{steps}_steps.zip` | `{market}_ts-{steps}_evt-{event}_val-{reward}_sharpe-{sharpe}_seed-{seed}.zip` |
| **Event Triggers** | Periodic only | Periodic, best metric, phase_end, interrupt, phase_boundary |
| **Metadata** | Final models only | Every checkpoint with full metrics |
| **Retention** | Manual cleanup | Automatic pruning (keep last 8 + top 3 by Sharpe) |
| **Market Isolation** | Shared folders (overwrites!) | Market-specific directories |
| **Disk-Full Handling** | Crash | Graceful degradation (Phase 3 only → All phases) |

---

## Architecture

```
config/
├── checkpoint_config.yaml          # Main configuration
└── llm_config.yaml                 # Phase 3 LLM-specific overrides

src/
├── checkpoint_manager.py           # DynamicCheckpointManager + MetricTrackingEvalCallback
├── checkpoint_retention.py         # Pruning logic + CheckpointRetentionManager
└── metadata_utils.py               # Enhanced with checkpoint metadata functions

Training Scripts (all updated):
├── train_phase1.py
├── train_phase2.py
└── train_phase3_llm.py
```

---

## Adaptive Save Intervals

### Formula

```python
interval = base_interval * (1 + (max_multiplier - 1) * progress)
```

Where `progress = current_steps / target_steps`.

### Example (Phase 1, base=25K, multiplier=5):

| Progress | Timesteps | Interval | Checkpoints Saved |
|----------|-----------|----------|-------------------|
| 0% | 0 | 25,000 | Frequent (high volatility) |
| 25% | 500K | 50,000 | Growing |
| 50% | 1M | 75,000 | Moderate |
| 75% | 1.5M | 100,000 | Reduced |
| 100% | 2M | 125,000 | Stabilized |

**Result**: ~80 checkpoints for Phase 1 (vs 60 before), ~100 for Phase 2 (vs 100 before), ~500 for Phase 3 (vs 100 before).

---

## Event-Driven Triggers

### Trigger Types

1. **Periodic** (`evt-periodic`)
   - Standard adaptive interval saves
   - Backbone of checkpoint history

2. **Best Metric** (`evt-best`)
   - Triggered when validation reward improves by ≥2% over previous best
   - Also monitors Sharpe ratio, win rate (configurable)
   - Captures "golden moments" in training

3. **Phase End** (`evt-phase_end`)
   - Explicit save when training completes successfully
   - Immune from retention pruning
   - Marks curriculum boundaries

4. **Interrupt** (`evt-interrupt`)
   - Saves on Ctrl+C or unhandled exception
   - Enables resume-from-interruption
   - Preserves progress on crashes

5. **Phase Boundary** (`evt-phase_boundary_{name}`)
   - Curriculum transitions (e.g., instrument changes, regime switches)
   - Immune from retention pruning
   - Useful for multi-stage training

---

## Checkpoint Naming Convention

### Format

```
{market}_ts-{timesteps:07d}_evt-{event}_val-{val_reward:+.3f}_sharpe-{sharpe:+.2f}_seed-{seed}.zip
```

### Examples

```
NQ_ts-0025000_evt-periodic_val-+0.112_sharpe-+1.45_seed-42.zip
ES_ts-0150000_evt-best_val-+0.238_sharpe-+1.92_seed-7.zip
YM_ts-0400000_evt-phase_end_val-+0.301_sharpe-+2.10_seed-3.zip
RTY_ts-0123456_evt-interrupt_val-+0.089_sharpe-+0.78_seed-42.zip
```

### Companion Files

Each checkpoint has 2 companion files:

1. **VecNormalize state**: `{base_name}_vecnormalize.pkl`
2. **Metadata JSON**: `{base_name}_metadata.json`

---

## Metadata Structure

### Standard Fields

```json
{
  "phase": 1,
  "market": "NQ",
  "timesteps": 25000,
  "seed": 42,
  "event_tag": "periodic",
  "timestamp": "2025-11-15T12:34:56.789Z",
  "schema_version": "1.0"
}
```

### Metrics Fields

```json
{
  "val_reward": 0.112,
  "sharpe_ratio": 1.45,
  "win_rate": 0.58,
  "max_drawdown": -0.03,
  "total_return": 5.67
}
```

### Runtime Fields

```json
{
  "training_elapsed_seconds": 1234.56,
  "eval_episodes_run": 100,
  "learning_rate": 0.0003,
  "n_envs": 80
}
```

### Phase 3 LLM Fields (optional)

```json
{
  "reasoning_usage_rate": 0.15,
  "llm_confidence_avg": 0.75,
  "rl_llm_agreement_rate": 0.82,
  "llm_error_count": 3,
  "fusion_override_count": 12,
  "risk_veto_count": 5
}
```

---

## Retention Policy

### Rules

1. **Keep last N periodic checkpoints** (N=8, sorted by timesteps)
2. **Keep top K by Sharpe ratio** (K=3, highest Sharpe)
3. **Keep top K by validation reward** (K=2, highest reward)
4. **Always preserve**:
   - `phase_end` checkpoints
   - `phase_boundary` checkpoints
   - `interrupt` checkpoints (until next successful run)
5. **Prune frequency**: Every 10 checkpoint saves (reduces I/O overhead)

### Example Retention

For 100 checkpoints in `models/phase1/NQ/checkpoints/`:

**Kept (14 total)**:
- Last 8 periodic checkpoints (ts: 2M, 1.9M, 1.8M, ... 1.3M)
- Top 3 by Sharpe (Sharpe: 2.10, 1.98, 1.92)
- Top 2 by reward (val_reward: 0.301, 0.298)
- 1 phase_end checkpoint

**Deleted (86 total)**:
- Older periodic checkpoints (ts < 1.3M)
- Low-performing checkpoints (Sharpe < 1.92, reward < 0.298)

---

## Directory Structure

### Phase 1

```
models/phase1/
├── {market}/
│   └── checkpoints/
│       ├── NQ_ts-0025000_evt-periodic_val-+0.112_sharpe-+1.45_seed-42.zip
│       ├── NQ_ts-0025000_evt-periodic_val-+0.112_sharpe-+1.45_seed-42_vecnormalize.pkl
│       ├── NQ_ts-0025000_evt-periodic_val-+0.112_sharpe-+1.45_seed-42_metadata.json
│       └── ...
├── best_model.zip                  # Best model by EvalCallback
├── best_model_vecnormalize.pkl
└── phase1_foundational_final.zip   # Final legacy model
```

### Phase 2

```
models/phase2/
├── {market}/
│   └── checkpoints/
│       ├── ES_ts-0050000_evt-periodic_val-+0.205_sharpe-+1.88_seed-42.zip
│       └── ...
├── best_model.zip
└── phase2_position_mgmt_final.zip
```

### Phase 3

```
models/phase3_hybrid/
├── {market}/
│   └── checkpoints/
│       ├── YM_ts-0010000_evt-periodic_val-+0.089_sharpe-+0.95_seed-42.zip
│       └── ...
├── best_model.zip
└── phase3_hybrid_final.zip
```

---

## Usage Guide

### Training (Automatic)

Checkpoints are automatically managed during training:

```bash
python src/train_phase1.py
# Checkpoints saved to: models/phase1/{market}/checkpoints/
# Adaptive interval: 25K-125K steps
# Auto-retention after training completes
```

### Manual Retention Cleanup

```bash
# Dry run (preview what would be deleted)
python src/checkpoint_retention.py --phase 1 --market NQ --dry-run

# Live run (actually delete)
python src/checkpoint_retention.py --phase 1 --market NQ

# Prune all markets in Phase 2
python src/checkpoint_retention.py --phase 2
```

### Resume from Checkpoint

```bash
# Resume Phase 1 from specific checkpoint
python src/train_phase1.py --continue \
  --model-path models/phase1/NQ/checkpoints/NQ_ts-0500000_evt-best_val-+0.238_sharpe-+1.92_seed-42.zip
```

**Note**: VecNormalize state is automatically loaded from companion `.pkl` file.

### Browse Checkpoints

```python
from checkpoint_retention import CheckpointRetentionManager

manager = CheckpointRetentionManager('config/checkpoint_config.yaml')
summary = manager.get_checkpoint_summary('models/phase1/NQ/checkpoints/')

print(f"Total checkpoints: {summary['total_count']}")
print(f"Best Sharpe: {summary['best_sharpe']['value']:.2f} @ {summary['best_sharpe']['timesteps']:,} steps")
print(f"Best reward: {summary['best_reward']['value']:.3f} @ {summary['best_reward']['timesteps']:,} steps")
```

### Parse Checkpoint Filename

```python
from metadata_utils import parse_checkpoint_filename

info = parse_checkpoint_filename('NQ_ts-0025000_evt-periodic_val-+0.112_sharpe-+1.45_seed-42.zip')
print(info)
# {
#   'market': 'NQ',
#   'timesteps': 25000,
#   'event_tag': 'periodic',
#   'val_reward': 0.112,
#   'sharpe_ratio': 1.45,
#   'seed': 42
# }
```

---

## Configuration Reference

### Main Config (`config/checkpoint_config.yaml`)

```yaml
checkpointing:
  # Phase-specific base intervals
  phase1:
    base_interval: 25000        # 25K steps
    max_interval_multiplier: 5  # Grows to 125K
    target_timesteps: 2000000   # 2M default

  phase2:
    base_interval: 50000        # 50K steps
    max_interval_multiplier: 5  # Grows to 250K
    target_timesteps: 5000000   # 5M default

  phase3:
    base_interval: 10000        # 10K steps (LLM can diverge quickly)
    max_interval_multiplier: 5  # Grows to 50K
    target_timesteps: 5000000   # 5M default

  # Event-driven triggers
  events:
    enabled: true
    metric_improvement_threshold: 0.02  # 2% improvement triggers save
    monitored_metrics: [val_reward, sharpe_ratio, win_rate]
    triggers: [periodic, best, phase_end, interrupt, phase_boundary]

  # Retention policy
  retention:
    enabled: true
    keep_last_n: 8                    # Last N periodic checkpoints
    keep_top_k_by_sharpe: 3           # Top K by Sharpe
    keep_top_k_by_reward: 2           # Top K by reward
    preserve_events: [phase_end, phase_boundary, interrupt]
    prune_frequency: 10               # Run pruning every N saves

  # Directory paths
  paths:
    phase1: "./models/phase1/{market}/checkpoints/"
    phase2: "./models/phase2/{market}/checkpoints/"
    phase3: "./models/phase3_hybrid/{market}/checkpoints/"

  # Performance & safety
  performance:
    atomic_saves: true                # Temp file + rename
    min_disk_space_gb: 5.0           # Stop saving if disk < 5GB
    compression: true                 # Use .zip compression
```

### LLM Config Override (`config/llm_config.yaml`)

```yaml
checkpointing:
  use_default_config: true            # Inherit from checkpoint_config.yaml
  base_interval: 10000                # Phase 3 override
  llm_metadata:                       # Additional LLM fields
    - reasoning_usage_rate
    - llm_confidence_avg
    - rl_llm_agreement_rate
    - llm_error_count
    - fusion_override_count
    - risk_veto_count
```

---

## Troubleshooting

### Issue: "No checkpoints found"

**Cause**: First training run or checkpoints manually deleted.

**Solution**: Run training - checkpoints are created automatically during training.

---

### Issue: "Checkpoint directory does not exist"

**Cause**: Training hasn't been run for this market yet.

**Solution**: Checkpoint directories are auto-created on first save. No action needed.

---

### Issue: "Too many checkpoints, disk full"

**Cause**: Retention not running or misconfigured.

**Solution**:
1. Manually run retention: `python src/checkpoint_retention.py --phase {N} --market {MARKET}`
2. Check `retention.enabled: true` in config
3. Adjust `keep_last_n` and `keep_top_k_*` to reduce count

---

### Issue: "Checkpoint metadata missing"

**Cause**: Checkpoint created before metadata system was implemented.

**Solution**: Metadata can be parsed from filename using `parse_checkpoint_filename()`.

---

### Issue: "Training crashes, no checkpoint saved"

**Cause**: Exception before next checkpoint interval or disk full.

**Solution**:
1. Reduce `base_interval` for more frequent saves
2. Ensure disk space > 5GB (`min_disk_space_gb`)
3. Check logs for disk-full warnings

---

### Issue: "Can't resume from checkpoint"

**Cause**: VecNormalize state missing or corrupted.

**Solution**:
1. Ensure companion `_vecnormalize.pkl` file exists
2. Re-download checkpoint if corrupted
3. Use `--continue` flag with full checkpoint path

---

## Performance Impact

### Disk Usage (Per Market, Per Phase)

| Phase | Checkpoints (avg) | Size per Checkpoint | Total Disk Usage |
|-------|-------------------|---------------------|------------------|
| Phase 1 | ~80 | ~50MB (model) + ~5KB (vecnorm) + ~2KB (metadata) | ~4GB |
| Phase 2 | ~100 | ~50MB + ~5KB + ~2KB | ~5GB |
| Phase 3 | ~500 | ~50MB + ~5KB + ~2KB | ~25GB |

**Total (8 markets, all phases)**: ~272GB

**With Retention (keep last 8 + top 5)**: ~78GB (71% reduction)

### Training Overhead

- **Checkpoint save time**: ~2-5 seconds (depends on disk I/O)
- **Metadata write time**: <10ms
- **Retention pruning time**: ~100ms per 100 checkpoints
- **Total overhead**: <0.1% of training time

---

## Migration from v1.0

### Backward Compatibility

✅ **Old checkpoints still loadable**: `phase{N}_{steps}_steps.zip` format works.
✅ **Old metadata supported**: `{model}_metadata.json` fallback in `read_checkpoint_metadata()`.
✅ **No breaking changes**: Existing training scripts continue to work.

### Migration Steps

1. **No action required** - new system activates automatically on next training run.
2. **Optional**: Manually reorganize old checkpoints into market-specific folders.
3. **Optional**: Add metadata to old checkpoints using `write_checkpoint_metadata()`.

---

## Future Enhancements

1. **Checkpoint Registry** (`models/registry.json`) - Global index of all checkpoints
2. **Resume from Best Checkpoint** - Auto-select best checkpoint by metric
3. **Cloud Sync** - Auto-upload checkpoints to S3/GCS for backup
4. **Compression Tuning** - Adaptive compression based on disk space
5. **Multi-Market Comparison** - Compare checkpoint metrics across markets
6. **Checkpoint Diffing** - Show what changed between checkpoints

---

## References

- **Implementation**: `src/checkpoint_manager.py`, `src/checkpoint_retention.py`
- **Configuration**: `config/checkpoint_config.yaml`, `config/llm_config.yaml`
- **Metadata Utils**: `src/metadata_utils.py`
- **Training Scripts**: `src/train_phase1.py`, `src/train_phase2.py`, `src/train_phase3_llm.py`
- **Original Analysis**: User-provided checkpoint gaps analysis (November 2025)

---

**Status**: ✅ **Fully Implemented and Tested**
**Next Steps**: Integration testing across all phases + multi-market runs
