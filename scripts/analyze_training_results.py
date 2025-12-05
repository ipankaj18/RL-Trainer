#!/usr/bin/env python3
"""
Training Results Analyzer - Comprehensive profitability assessment
Analyzes JAX training results from RunPod: logs, models, and metrics
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, List
import sys

# Try to import plotting libraries (optional)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Skipping visualizations.")

# Color output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")


def analyze_phase1_metrics(metrics_path: Path) -> Dict:
    """Analyze Phase 1 JSON metrics file"""
    print(f"{Colors.OKCYAN}📊 Analyzing Phase 1 metrics: {metrics_path.name}{Colors.ENDC}")
    
    with open(metrics_path, 'r') as f:
        metrics_list = json.load(f)
    
    if not metrics_list:
        return {'error': 'Empty metrics file'}
    
    # Extract all metrics
    updates = [m['update'] for m in metrics_list]
    timesteps = [m['timesteps'] for m in metrics_list]
    returns = [m['mean_return'] for m in metrics_list]
    policy_losses = [m['policy_loss'] for m in metrics_list]
    value_losses = [m['value_loss'] for m in metrics_list]
    sps = [m['sps'] for m in metrics_list]
    
    results = {
        'total_updates': max(updates),
        'total_timesteps': max(timesteps),
        'avg_return': sum(returns) / len(returns),
        'max_return': max(returns),
        'min_return': min(returns),
        'final_return': returns[-1],
        'return_improvement': returns[-1] - returns[0],
        'avg_policy_loss': sum(policy_losses) / len(policy_losses),
       'avg_value_loss': sum(value_losses) / len(value_losses),
        'final_policy_loss': policy_losses[-1],
        'final_value_loss': value_losses[-1],
        'avg_sps': sum(sps) / len(sps),
        'raw_data': {
            'updates': updates,
            'timesteps': timesteps,
            'returns': returns,
            'policy_losses': policy_losses,
            'value_losses': value_losses,
            'sps': sps
        }
    }
    
    return results


def parse_training_log(log_path: Path) -> Dict:
    """Parse training log file for additional metrics"""
    print(f"{Colors.OKCYAN}📋 Parsing training log: {log_path.name}{Colors.ENDC}")
    
    returns = []
    losses = []
    action_stats = {}
    episode_lengths = []
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
        # Extract action distribution (if present)
        action_match = re.search(r'Action Distribution:.*?HOLD: ([\d.]+)%.*?BUY: ([\d.]+)%.*?SELL: ([\d.]+)%', content, re.DOTALL)
        if action_match:
            action_stats = {
                'hold_pct': float(action_match.group(1)),
                'buy_pct': float(action_match.group(2)),
                'sell_pct': float(action_match.group(3))
            }
        
        # Extract final statistics
        episode_match = re.search(r'Average Episode Length: ([\d.]+)', content)
        if episode_match:
            episode_lengths.append(float(episode_match.group(1)))
    
    return {
        'action_distribution': action_stats,
        'avg_episode_length': episode_lengths[0] if episode_lengths else None,
        'log_size_kb': log_path.stat().st_size / 1024
    }


def check_model_checkpoints(models_dir: Path) -> Dict:
    """Check available model checkpoints"""
    print(f"{Colors.OKCYAN}💾 Checking model checkpoints in: {models_dir.name}{Colors.ENDC}")
    
    checkpoints = []
    total_size_mb = 0
    
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            # Look for JAX checkpoint files
            if any(file.endswith(ext) for ext in ['.pkl', '.msgpack', '.zip']):
                full_path = Path(root) / file
                size_mb = full_path.stat().st_size / (1024 * 1024)
                total_size_mb += size_mb
                checkpoints.append({
                    'name': file,
                    'path': str(full_path.relative_to(models_dir)),
                    'size_mb': round(size_mb, 2)
                })
    
    return {
        'total_checkpoints': len(checkpoints),
        'total_size_mb': round(total_size_mb, 2),
        'checkpoints': checkpoints
    }


def create_visualizations(phase1_data: Dict, output_dir: Path):
    """Create training visualization plots"""
    if not PLOTTING_AVAILABLE:
        return []
    
    print(f"{Colors.OKCYAN}📈 Creating visualizations...{Colors.ENDC}")
    
    raw = phase1_data.get('raw_data', {})
    if not raw:
        return []
    
    output_dir.mkdir(exist_ok=True)
    created_plots = []
    
    # Plot 1: Returns over time
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(raw['updates'], raw['returns'], linewidth=2, color='#2E86AB', label='Mean Episode Return')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax.set_xlabel('Update Number', fontsize=12)
    ax.set_ylabel('Mean Episode Return', fontsize=12)
    ax.set_title('Phase 1 Training: Episode Returns Progression', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot1_path = output_dir / 'phase1_returns.png'
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close()
    created_plots.append(plot1_path)
    
    # Plot 2: Loss curves
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(raw['updates'], raw['policy_losses'], linewidth=2, color='#A23B72', label='Policy Loss')
    ax1.set_xlabel('Update Number', fontsize=12)
    ax1.set_ylabel('Policy Loss', fontsize=12)
    ax1.set_title('Policy Loss Over Training', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(raw['updates'], raw['value_losses'], linewidth=2, color='#F18F01', label='Value Loss')
    ax2.set_xlabel('Update Number', fontsize=12)
    ax2.set_ylabel('Value Loss', fontsize=12)
    ax2.set_title('Value Loss Over Training', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot2_path = output_dir / 'phase1_losses.png'
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close()
    created_plots.append(plot2_path)
    
    # Plot 3: Training speed (SPS)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(raw['updates'], raw['sps'], linewidth=2, color='#06A77D', label='Steps Per Second')
    ax.set_xlabel('Update Number', fontsize=12)
    ax.set_ylabel('Steps Per Second', fontsize=12)
    ax.set_title('Training Speed (JAX Throughput)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot3_path = output_dir / 'training_speed.png'
    plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
    plt.close()
    created_plots.append(plot3_path)
    
    return created_plots


def print_summary_table(phase1_metrics: Dict, phase1_log: Dict, phase2_log: Dict, models: Dict):
    """Print formatted ASCII table summary"""
    
    print_header("TRAINING RESULTS SUMMARY")
    
    # Phase 1 Metrics Table
    print(f"{Colors.BOLD}╔══════════════════════════════════════════════════════════╗{Colors.ENDC}")
    print(f"{Colors.BOLD}║          PHASE 1 TRAINING (ENTRY LEARNING)               ║{Colors.ENDC}")
    print(f"{Colors.BOLD}╠══════════════════════════════════════════════════════════╣{Colors.ENDC}")
    print(f"║ Total Updates:           {phase1_metrics.get('total_updates', 0):>30} ║")
    print(f"║ Total Timesteps:         {phase1_metrics.get('total_timesteps', 0):>30,} ║")
    print(f"║ Mean Episode Return:     {phase1_metrics.get('avg_return', 0):>30.2f} ║")
    print(f"║ Final Episode Return:    {phase1_metrics.get('final_return', 0):>30.2f} ║")
    print(f"║ Return Improvement:      {phase1_metrics.get('return_improvement', 0):>30.2f} ║")
    print(f"║ Max Return:              {phase1_metrics.get('max_return', 0):>30.2f} ║")
    print(f"║ Min Return:              {phase1_metrics.get('min_return', 0):>30.2f} ║")
    print(f"║ Avg Policy Loss:         {phase1_metrics.get('avg_policy_loss', 0):>30.6f} ║")
    print(f"║ Avg Value Loss:          {phase1_metrics.get('avg_value_loss', 0):>30.2f} ║")
    print(f"║ Avg Training Speed:      {phase1_metrics.get('avg_sps', 0):>25,.0f} SPS ║")
    print(f"{Colors.BOLD}╚══════════════════════════════════════════════════════════╝{Colors.ENDC}")
    
    # Phase 2 Table (if available)
    if phase2_log and phase2_log.get('avg_episode_length'):
        print(f"\n{Colors.BOLD}╔══════════════════════════════════════════════════════════╗{Colors.ENDC}")
        print(f"{Colors.BOLD}║      PHASE 2 TRAINING (POSITION MANAGEMENT)              ║{Colors.ENDC}")
        print(f"{Colors.BOLD}╠══════════════════════════════════════════════════════════╣{Colors.ENDC}")
        print(f"║ Avg Episode Length:      {phase2_log.get('avg_episode_length', 0):>30.1f} ║")
        if phase2_log.get('action_distribution'):
            action_dist = phase2_log['action_distribution']
            print(f"║ HOLD actions:            {action_dist.get('hold_pct', 0):>29.2f}% ║")
            print(f"║ BUY actions:             {action_dist.get('buy_pct', 0):>29.2f}% ║")
            print(f"║ SELL actions:            {action_dist.get('sell_pct', 0):>29.2f}% ║")
        print(f"{Colors.BOLD}╚══════════════════════════════════════════════════════════╝{Colors.ENDC}")
    
    # Model Checkpoints Table
    print(f"\n{Colors.BOLD}╔══════════════════════════════════════════════════════════╗{Colors.ENDC}")
    print(f"{Colors.BOLD}║             SAVED MODEL CHECKPOINTS                      ║{Colors.ENDC}")
    print(f"{Colors.BOLD}╠══════════════════════════════════════════════════════════╣{Colors.ENDC}")
    print(f"║ Total Checkpoints:       {models.get('total_checkpoints', 0):>30} ║")
    print(f"║ Total Size:              {models.get('total_size_mb', 0):>25.2f} MB ║")
    if models.get('checkpoints'):
        print(f"║ ────────────────────────────────────────────────────────── ║")
        for ckpt in models['checkpoints'][:5]:  # Show first 5
            name = ckpt['name'][:45]  # Truncate long names
            print(f"║ {name:<45} {ckpt['size_mb']:>6.1f}MB ║")
        if len(models['checkpoints']) > 5:
            print(f"║ ... and {len(models['checkpoints']) - 5} more{' '*35} ║")
    print(f"{Colors.BOLD}╚══════════════════════════════════════════════════════════╝{Colors.ENDC}")


def assess_profitability(phase1_metrics: Dict):
    """Critical assessment of profitability"""
    
    print_header("PROFITABILITY ASSESSMENT")
    
    print(f"{Colors.WARNING}⚠️  CRITICAL: Training Metrics ≠ Trading Profitability{Colors.ENDC}\n")
    
    print("The metrics shown above reflect:")
    print(f"  {Colors.OKCYAN}✓{Colors.ENDC} How well the agent learned to maximize rewards during training")
    print(f"  {Colors.OKCYAN}✓{Colors.ENDC} Policy optimization progress (reducing loss)")
    print(f"  {Colors.OKCYAN}✓{Colors.ENDC} Value function accuracy improvements\n")
    
    print(f"{Colors.FAIL}But they DO NOT show:{Colors.ENDC}")
    print(f"  {Colors.FAIL}✗{Colors.ENDC} Actual P&L in dollars")
    print(f"  {Colors.FAIL}✗{Colors.ENDC} Win rate or profit factor")
    print(f"  {Colors.FAIL}✗{Colors.ENDC} Maximum trailing drawdown (Apex: must be < $2,500)")
    print(f"  {Colors.FAIL}✗{Colors.ENDC} Number of trading days (Apex: need ≥ 7)")
    print(f"  {Colors.FAIL}✗{Colors.ENDC} 4:59 PM close compliance\n")
    
    # Check training quality
    final_return = phase1_metrics.get('final_return', 0)
    improvement = phase1_metrics.get('return_improvement', 0)
    
    print(f"{Colors.BOLD}Training Quality Analysis:{Colors.ENDC}\n")
    
    if final_return < 0:
        status = f"{Colors.FAIL}⚠️  CONCERNING{Colors.ENDC}"
        message = f"Final return is negative ({final_return:.2f}). Model may not have learned effective entries."
    elif improvement > 0:
        status = f"{Colors.OKGREEN}✓ POSITIVE{Colors.ENDC}"
        message = f"Returns improved by {improvement:.2f} during training. Model shows learning progress."
    else:
        status = f"{Colors.WARNING}⚠️  DECLINING{Colors.ENDC}"
        message = f"Returns declined by {abs(improvement):.2f}. Possible overfitting or instability."
    
    print(f"  Status: {status}")
    print(f"  {message}\n")
    
    print(f"{Colors.BOLD}╔══════════════════════════════════════════════════════════════════════════════╗{Colors.ENDC}")
    print(f"{Colors.BOLD}║  TO DETERMINE IF THE MODEL IS ACTUALLY PROFITABLE:                          ║{Colors.ENDC}")
    print(f"{Colors.BOLD}╠══════════════════════════════════════════════════════════════════════════════╣{Colors.ENDC}")
    print(f"║  Run the evaluation script to backtest on UNSEEN data:                      ║")
    print(f"║                                                                              ║")
    print(f"║  {Colors.OKCYAN}python -m src.jax_migration.evaluate_phase2_jax \\{Colors.ENDC}                      ║")
    print(f"║  {Colors.OKCYAN}       --model models/phase2_jax_nq \\{Colors.ENDC}                                  ║")
    print(f"║  {Colors.OKCYAN}       --market NQ{Colors.ENDC}                                                     ║")
    print(f"║                                                                              ║")
    print(f"║  This will generate:                                                         ║")
    print(f"║    • Total P&L and equity curve                                              ║")
    print(f"║    • Max trailing drawdown (Apex limit: $2,500)                              ║")
    print(f"║    • Win rate, profit factor, Sharpe ratio                                   ║")
    print(f"║    • Number of trading days (Apex minimum: 7)                                ║")
    print(f"║    • Trade-by-trade breakdown                                                ║")
    print(f"{Colors.BOLD}╚══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}\n")


def provide_recommendations(phase1_metrics: Dict, models: Dict):
    """Actionable improvement recommendations"""
    
    print_header("IMPROVEMENT RECOMMENDATIONS")
    
    recommendations = []
    
    # Check final performance
    final_return = phase1_metrics.get('final_return', 0)
    improvement = phase1_metrics.get('return_improvement', 0)
    total_timesteps = phase1_metrics.get('total_timesteps', 0)
    
    # Training outcome recommendations
    if final_return < -500:
        recommendations.append({
            'priority': '🔴 CRITICAL',
            'area': 'Negative Returns',
            'finding': f"Final return is {final_return:.0f}, indicating poor entry strategy",
            'action': [
                "Review reward function - may be penalizing valid trades",
                "Check data quality and normalization",
                "Consider Phase 1 hyperparameter tuning (learning rate, entropy coefficient)"
            ]
        })
    
    if improvement < 0:
        recommendations.append({
            'priority': '🟠 HIGH',
            'area': 'Declining Performance',
            'finding': f"Returns declined by {abs(improvement):.0f} during training",
            'action': [
                "Possible overfitting - reduce model capacity or add regularization",
                "Try early stopping based on validation set",
                "Lower learning rate for more stable training"
            ]
        })
    
    if total_timesteps < 50_000_000:
        recommendations.append({
            'priority': '🟡 MEDIUM',
            'area': 'Training Duration',
            'finding': f"Only {total_timesteps:,} timesteps completed (production: 50-100M)",
            'action': [
                "Continue training for 50-100M timesteps for production readiness",
                "Monitor for convergence - returns should stabilize"
            ]
        })
    
    if not models.get('checkpoints'):
        recommendations.append({
            'priority': '🔴 CRITICAL',
            'area': 'Missing Checkpoints',
            'finding': "No model checkpoints found",
            'action': [
                "Verify training completed successfully",
                "Check models directory permissions",
                "Re-run training if checkpoints were not saved"
            ]
        })
    
    # Always include evaluation reminder
    recommendations.append({
        'priority': '⚪ MANDATORY',
        'area': 'Profitability Validation',
        'finding': "Training metrics do not reflect actual trading P&L",
        'action': [
            "Run evaluate_phase2_jax.py to get real backtest results",
            "Verify Apex compliance (drawdown < $2,500, 7+ days)",
            "Compare with benchmark (buy-and-hold, random policy)"
        ]
    })
    
    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{rec['priority']} {rec['area']}")
        print(f"{'─' * 80}")
        print(f"Finding:  {rec['finding']}")
        print(f"Actions:")
        for action in rec['action']:
            print(f"  • {action}")
    
    print()


def main():
    # Define paths
    base_dir = Path(__file__).parent
    logs_dir = base_dir / 'logs'
    models_dir = base_dir / 'models'
    results_dir = base_dir / 'results'
    output_dir = base_dir / 'analysis_output'
    
    print_header("JAX TRAINING RESULTS ANALYZER")
    print(f"Analysis directory: {base_dir}\n")
    
    # Analyze Phase 1 metrics
    phase1_metrics_path = results_dir / 'nq_jax_phase1_metrics.json'
    phase1_metrics = {}
    if phase1_metrics_path.exists():
        phase1_metrics = analyze_phase1_metrics(phase1_metrics_path)
    else:
        print(f"{Colors.WARNING}⚠️  Phase 1 metrics not found{Colors.ENDC}")
        return
    
    # Parse training logs
    phase1_log = parse_training_log(logs_dir / 'jax_phase1_nq.log') if (logs_dir / 'jax_phase1_nq.log').exists() else {}
    phase2_log = parse_training_log(logs_dir / 'jax_phase2_nq.log') if (logs_dir / 'jax_phase2_nq.log').exists() else {}
    
    # Check models
    models = check_model_checkpoints(models_dir)
    
    # Create visualizations
    plots = create_visualizations(phase1_metrics, output_dir)
    if plots:
        print(f"{Colors.OKGREEN}✓ Created {len(plots)} visualization(s) in {output_dir}/{Colors.ENDC}\n")
    
    # Print results
    print_summary_table(phase1_metrics, phase1_log, phase2_log, models)
    assess_profitability(phase1_metrics)
    provide_recommendations(phase1_metrics, models)
    
    print(f"{Colors.OKGREEN}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}Analysis complete! Check analysis_output/ for visualizations.{Colors.ENDC}")
    print(f"{Colors.OKGREEN}{'=' * 80}{Colors.ENDC}\n")


if __name__ == '__main__':
    main()
