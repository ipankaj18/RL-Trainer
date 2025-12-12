#!/bin/bash
# Quick validation test for Phase 1 improvements (500K timesteps)
# Tests: Reward improvements, entropy monitoring, hyperparameter tuning

echo "Testing Phase 1 JAX improvements..."
echo "===================================="
echo ""
echo "Changes tested:"
echo "  1. Reward function: 2x PnL signal, 2x TP bonus, exploration bonus"
echo "  2. Commission curriculum: $1.00 → $2.50 over first 50% of training"
echo "  3. Entropy coefficient: 0.01 → 0.05 (default)"
echo "  4. LR annealing: 3e-4 → 1e-4 (optional)"
echo "  5. Metrics tracker integration"
echo ""

# Check if NQ data exists
DATA_PATH="data/NQ_D1M.csv"
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    echo "Please ensure NQ data is processed before running this test"
    exit 1
fi

# Run Phase 1 training with improvements
echo "Starting test training (500K timesteps, ~5-10 minutes)..."
echo ""

python -m src.jax_migration.train_ppo_jax_fixed \
  --market NQ \
  --num_envs 1024 \
  --total_timesteps 500000 \
  --ent_coef 0.05 \
  --lr_annealing \
  --initial_lr 3e-4 \
  --final_lr 1e-4 \
  --data_path "$DATA_PATH" \
  --checkpoint_dir models/phase1_test_improvements

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
echo ""
echo "Check results for:"
echo "  1. ✓ Entropy staying > 0.05 (no collapse)"
echo "  2. ✓ Mean return trending positive (not -877)"
echo "  3. ✓ Commission ramping logged (implicit in env)"
echo "  4. ✓ Metrics saved to:"
echo "       models/phase1_test_improvements/training_metrics_NQ.json"
echo ""
echo "Next steps:"
echo "  - Review training_metrics_NQ.json for P&L, win rate, trade count"
echo "  - If successful, run full training with 20M+ timesteps"
echo "  - Compare entropy and returns against baseline (-877 mean return)"
echo ""
