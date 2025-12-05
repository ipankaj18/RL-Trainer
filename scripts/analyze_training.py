"""Simple script to extract summary statistics from tensorboard logs"""
import json
import os
import struct
import tensorflow as tf

def extract_metrics_simple(log_path, phase_name):
    """Extract metrics from tensorboard event file"""
    print(f"\n{'='*60}")
    print(f"{phase_name} TRAINING SUMMARY")
    print(f"{'='*60}\n")
    
    metrics_data = {}
    
    try:
        for event in tf.compat.v1.train.summary_iterator(log_path):
            for value in event.summary.value:
                tag = value.tag
                if tag not in metrics_data:
                    metrics_data[tag] = []
                metrics_data[tag].append((event.step, value.simple_value))
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return
    
    # Print summary for each metric
    for tag, values in sorted(metrics_data.items()):
        if len(values) > 0:
            vals = [v[1] for v in values]
            steps = [v[0] for v in values]
            
            print(f"\n{tag}:")
            print(f"  Total updates: {len(values)}")
            print(f"  Steps: {min(steps)} to {max(steps)}")
            print(f"  First value: {vals[0]:.4f}")
            print(f"  Last value: {vals[-1]:.4f}")
            print(f"  Min: {min(vals):.4f}")
            print(f"  Max: {max(vals):.4f}")
            
            # Calculate trend (improvement)
            if len(vals) >= 10:
                early_avg = sum(vals[:5]) / 5
                late_avg = sum(vals[-5:]) / 5
                change = ((late_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0
                print(f"  Trend: {change:+.2f}% (early avg: {early_avg:.4f}, late avg: {late_avg:.4f})")

def main():
    # Phase 1
    p1_log = "tensorboard_logs/phase1/PPO_1/events.out.tfevents.1764207013.c5bfef8b32ab.1449.0"
    if os.path.exists(p1_log):
        extract_metrics_simple(p1_log, "PHASE 1 (100M timesteps)")
    
    # Phase 2
    p2_log = "tensorboard_logs/phase2/PPO_1/events.out.tfevents.1764212963.c5bfef8b32ab.1629.0"
    if os.path.exists(p2_log):
        extract_metrics_simple(p2_log, "PHASE 2 (476 updates)")
    
    # Phase 2 evaluation
    print(f"\n{'='*60}")
    print("PHASE 2 FINAL EVALUATION")
    print(f"{'='*60}\n")
    
    metrics_path = "src/results/phase2_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print(f"Total Steps: {metrics.get('total_steps', 'N/A')}")
        print(f"Final Equity: ${metrics.get('final_equity', 'N/A'):,.2f}")
        print(f"Total Return: {metrics.get('total_return_pct', 'N/A'):.2f}%")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.4f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 'N/A'):.2f}%")
        print(f"PM Actions Used: {metrics.get('pm_actions_used', 'N/A')}")
        print(f"PM Usage: {metrics.get('pm_usage_pct', 'N/A'):.2f}%")
        print(f"Episode End Reason: {metrics.get('done_reason', 'N/A')}")
        print(f"\nAction Distribution:")
        for action, count in metrics.get('action_distribution', {}).items():
            print(f"  Action {action}: {count} times")

if __name__ == "__main__":
    main()
