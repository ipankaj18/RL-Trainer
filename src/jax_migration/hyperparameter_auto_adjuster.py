"""
Hyperparameter Auto-Adjuster for JAX PPO Training

Applies recommended parameter adjustments based on quality monitor feedback.
Part of Phase C: Adaptive Training System
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class HyperparameterAutoAdjuster:
    """
    Applies hyperparameter adjustments based on quality monitor recommendations.
    
    Adjustment strategies:
    - increase_entry_bonus: +0.1 to entry bonus in environment
    - reduce_hold_penalty: Multiply by 0.5
    - increase_entropy_coef: Multiply by 1.3 (max 0.20)
    - slow_commission_curriculum: Multiply progress by 0.5
    - reduce_learning_rate: Multiply by 0.7
    
    All adjustments are logged to adjustment_history.json.
    """
    
    def __init__(self, log_dir: str = "models/phase1_jax"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.log_dir / "adjustment_history.json"
        self.adjustment_history = []
        
        # Load existing history if available
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self.adjustment_history = json.load(f)
    
    def apply(self, config, env_params, recommendations: List[str], update_num: int) -> Dict[str, Any]:
        """
        Apply recommended adjustments to config and env_params.
        
        Args:
            config: PPOConfig object
            env_params: EnvParams object
            recommendations: List of adjustment strings
            update_num: Current training update number
        
        Returns:
            updated_params: Dict with 'config' and 'env_params' keys
        """
        adjustments_made = []
        
        for rec in recommendations:
            # DISABLED: Simplified reward function doesn't use entry_bonus/hold_penalty
            # if rec == "increase_entry_bonus":
            #     new_entry_bonus = env_params.entry_bonus * 1.2
            #     env_params = env_params._replace(entry_bonus=new_entry_bonus)
            #     adjustments_made.append(f"Increased entry bonus: {env_params.entry_bonus:.2f} → {new_entry_bonus:.2f}")

            # DISABLED: Simplified reward function doesn't use hold_penalty param
            # elif rec == "reduce_hold_penalty":
            #     new_hold_penalty = env_params.hold_penalty * 0.5
            #     env_params = env_params._replace(hold_penalty=new_hold_penalty)
            #     adjustments_made.append(f"Reduced hold penalty: {env_params.hold_penalty:.4f} → {new_hold_penalty:.4f}")

            if rec == "increase_entropy_coef":
                new_ent_coef = min(config.ent_coef * 1.3, 0.20)
                if new_ent_coef != config.ent_coef:
                    config = config._replace(ent_coef=new_ent_coef)
                    adjustments_made.append(f"Increased ent_coef: {config.ent_coef:.4f} → {new_ent_coef:.4f}")
            
            elif rec == "slow_commission_curriculum":
                # Slow down curriculum by reducing effective progress
                # This is handled in training loop by modifying progress calculation
                adjustments_made.append("Slow commission curriculum (apply 0.5x multiplier to progress)")
            
            elif rec == "reduce_learning_rate":
                new_lr = config.learning_rate * 0.7
                config = config._replace(learning_rate=new_lr)
                adjustments_made.append(f"Reduced LR: {config.learning_rate:.6f} → {new_lr:.6f}")
        
        # Log adjustments
        if adjustments_made:
            log_entry = {
                'update': update_num,
                'timestamp': datetime.now().isoformat(),
                'recommendations': recommendations,
                'adjustments': adjustments_made
            }
            self.adjustment_history.append(log_entry)
            self._save_history()
        
        return {
            'config': config,
            'env_params': env_params,
            'adjustments': adjustments_made,
            'curriculum_slowdown': 'slow_commission_curriculum' in recommendations
        }
    
    def _save_history(self):
        """Save adjustment history to JSON file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.adjustment_history, f, indent=2)
    
    def log_recommendations(self, recommendations: List[str], update_num: int):
        """Log recommendations without applying them (for manual review mode)."""
        log_entry = {
            'update': update_num,
            'timestamp': datetime.now().isoformat(),
            'recommendations': recommendations,
            'applied': False
        }
        self.adjustment_history.append(log_entry)
        self._save_history()
    
    def get_summary(self) -> Dict:
        """Get summary of all adjustments made."""
        return {
            'total_adjustments': len([e for e in self.adjustment_history if e.get('applied', True)]),
            'total_recommendations': len(self.adjustment_history),
            'recent_adjustments': self.adjustment_history[-5:] if self.adjustment_history else []
        }
