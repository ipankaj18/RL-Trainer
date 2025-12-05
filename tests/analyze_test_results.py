#!/usr/bin/env python3
"""
Analyze Phase 1 and Phase 2 test results to determine profitability.
"""
import numpy as np
import os
import sys

def analyze_phase(phase_num, eval_path, output_file):
    """Analyze evaluation results for a phase."""
    output_file.write(f"\n{'='*60}\n")
    output_file.write(f"PHASE {phase_num} ANALYSIS\n")
    output_file.write(f"{'='*60}\n")
    
    if not os.path.exists(eval_path):
        output_file.write(f"No evaluation data found at {eval_path}\n")
        return None
    
    data = np.load(eval_path)
    
    # Extract data
    timesteps = data['timesteps']
    results = data['results']  # Shape: (num_evals, num_episodes_per_eval)
    ep_lengths = data.get('ep_lengths', None)
    
    output_file.write(f"\nTraining Progress:\n")
    output_file.write(f"   Evaluations: {len(timesteps)}\n")
    output_file.write(f"   Timesteps range: {timesteps[0]:,} -> {timesteps[-1]:,}\n")
    output_file.write(f"   Episodes per eval: {results.shape[1]}\n")
    
    output_file.write(f"\nReward Analysis:\n")
    mean_rewards = []
    for i, ts in enumerate(timesteps):
        mean_reward = np.mean(results[i])
        std_reward = np.std(results[i])
        min_reward = np.min(results[i])
        max_reward = np.max(results[i])
        mean_rewards.append(mean_reward)
        
        output_file.write(f"\n   Eval {i+1} @ {ts:,} steps:\n")
        output_file.write(f"      Mean: {mean_reward:>8.2f} +/- {std_reward:.2f}\n")
        output_file.write(f"      Range: [{min_reward:>8.2f}, {max_reward:>8.2f}]\n")
        
        # Profitability indicator
        if mean_reward > 0:
            output_file.write(f"      Status: PROFITABLE (positive mean reward)\n")
        elif mean_reward > -1:
            output_file.write(f"      Status: MARGINAL (near breakeven)\n")
        else:
            output_file.write(f"      Status: UNPROFITABLE (negative mean reward)\n")
    
    # Overall trend
    output_file.write(f"\nLearning Trend:\n")
    
    if len(mean_rewards) >= 2:
        first_half_avg = np.mean(mean_rewards[:len(mean_rewards)//2])
        second_half_avg = np.mean(mean_rewards[len(mean_rewards)//2:])
        improvement = second_half_avg - first_half_avg
        
        output_file.write(f"   Early average: {first_half_avg:.2f}\n")
        output_file.write(f"   Late average: {second_half_avg:.2f}\n")
        output_file.write(f"   Improvement: {improvement:+.2f}\n")
        
        if improvement > 1:
            output_file.write(f"   Trend: IMPROVING\n")
        elif improvement > -1:
            output_file.write(f"   Trend: STABLE\n")
        else:
            output_file.write(f"   Trend: DEGRADING\n")
    else:
        improvement = 0
    
    # Episode lengths
    if ep_lengths is not None:
        output_file.write(f"\nEpisode Length Analysis:\n")
        for i, ts in enumerate(timesteps):
            mean_len = np.mean(ep_lengths[i])
            output_file.write(f"   Eval {i+1}: {mean_len:.0f} steps (avg)\n")
    
    # Final verdict
    final_mean = np.mean(results[-1])
    output_file.write(f"\nFINAL VERDICT (Last Evaluation):\n")
    output_file.write(f"   Mean Reward: {final_mean:.2f}\n")
    
    if final_mean > 5:
        verdict = "STRONGLY PROFITABLE - Model is ready for production"
    elif final_mean > 0:
        verdict = "PROFITABLE - Model shows positive returns"
    elif final_mean > -2:
        verdict = "MARGINAL - Model needs more training or tuning"
    else:
        verdict = "UNPROFITABLE - Model is not ready, needs redesign"
    
    output_file.write(f"   {verdict}\n")
    
    return {
        'final_mean': final_mean,
        'final_std': np.std(results[-1]),
        'improvement': improvement,
        'verdict': verdict,
        'mean_rewards': mean_rewards
    }

def main():
    # Write to file
    with open('test_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("\n" + "="*60 + "\n")
        f.write("PROFITABILITY ANALYSIS - PHASE 1 & 2 TEST RESULTS\n")
        f.write("="*60 + "\n")
        
        # Analyze Phase 1
        phase1_results = analyze_phase(1, 'logs/phase1/evaluations.npz', f)
        
        # Analyze Phase 2
        phase2_results = analyze_phase(2, 'logs/phase2/evaluations.npz', f)
        
        # Summary
        f.write(f"\n\n{'='*60}\n")
        f.write("OVERALL SUMMARY\n")
        f.write(f"{'='*60}\n")
        
        if phase1_results:
            f.write(f"\nPhase 1 (Entry Pattern Learning):\n")
            f.write(f"   Final Performance: {phase1_results['final_mean']:.2f} +/- {phase1_results['final_std']:.2f}\n")
            f.write(f"   {phase1_results['verdict']}\n")
        
        if phase2_results:
            f.write(f"\nPhase 2 (Position Management):\n")
            f.write(f"   Final Performance: {phase2_results['final_mean']:.2f} +/- {phase2_results['final_std']:.2f}\n")
            f.write(f"   {phase2_results['verdict']}\n")
        
        # Combined assessment
        f.write(f"\nCOMBINED ASSESSMENT:\n")
        
        if phase1_results and phase2_results:
            both_profitable = phase1_results['final_mean'] > 0 and phase2_results['final_mean'] > 0
            both_improving = phase1_results['improvement'] > 0 and phase2_results['improvement'] > 0
            
            if both_profitable and both_improving:
                f.write("   Both phases are profitable and improving - EXCELLENT\n")
            elif both_profitable:
                f.write("   Both phases are profitable - GOOD\n")
            elif phase1_results['final_mean'] > 0 or phase2_results['final_mean'] > 0:
                f.write("   Mixed results - one phase profitable, one not\n")
            else:
                f.write("   Both phases unprofitable - needs investigation\n")
        
        f.write("\n" + "="*60 + "\n\n")
    
    print("Analysis complete. Results written to test_analysis_report.txt")

if __name__ == "__main__":
    main()
