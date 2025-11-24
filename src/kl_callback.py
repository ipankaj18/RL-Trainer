#!/usr/bin/env python3
"""
KL Divergence Monitoring Callback for PPO
Monitors policy updates to prevent them from being too aggressive
"""

import numpy as np
import torch
from typing import Dict, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback


class KLDivergenceCallback(BaseCallback):
    """
    Monitor KL divergence between old and new policies during PPO training.
    
    This callback:
    1. Calculates KL divergence after each policy update
    2. Implements early stopping if KL exceeds target
    3. Logs KL statistics for monitoring
    4. Helps prevent policy collapse from overly aggressive updates
    
    Based on OpenAI Spinning Up PPO recommendations.
    """
    
    def __init__(
        self,
        target_kl: float = 0.01,
        verbose: int = 0,
        log_freq: int = 100
    ):
        """
        Initialize KL divergence callback.
        
        Args:
            target_kl: Target KL divergence threshold (default: 0.01)
            verbose: Verbosity level
            log_freq: Frequency of logging (every N steps)
        """
        super().__init__(verbose)
        self.target_kl = target_kl
        self.log_freq = log_freq
        
        # Tracking variables
        self.kl_history = []
        self.early_stops = 0
        self.last_log_step = 0
        
    def _on_step(self) -> bool:
        """Called at each step."""
        return True  # Continue training
    
    def _on_rollout_end(self) -> bool:
        """
        Called after each rollout (collection of experiences).
        Calculate KL divergence and decide whether to continue training.
        """
        # Calculate KL divergence between old and new policies
        kl_divergence = self._calculate_kl_divergence()
        
        if kl_divergence is not None:
            self.kl_history.append(kl_divergence)
            
            # Log if verbose
            if self.verbose > 0 and self.num_timesteps - self.last_log_step >= self.log_freq:
                self.logger.record("train/kl_divergence", kl_divergence)
                self.logger.record("train/kl_mean", np.mean(self.kl_history[-100:]) if self.kl_history else 0)
                self.logger.record("train/early_stops", self.early_stops)
                self.last_log_step = self.num_timesteps
                
                if self.verbose > 1:
                    print(f"[KL] Step {self.num_timesteps}: KL = {kl_divergence:.6f} (target: {self.target_kl})")
            
            # Check if KL exceeds target
            if kl_divergence > self.target_kl:
                self.early_stops += 1
                
                if self.verbose > 0:
                    print(f"\n[KL] Early stopping at step {self.num_timesteps}")
                    print(f"[KL] KL divergence ({kl_divergence:.6f}) > target ({self.target_kl})")
                    print(f"[KL] This is early stop #{self.early_stops}")
                
                # Signal to stop current policy update
                return False  # This will stop the current update cycle
        
        return True  # Continue training
    
    def _calculate_kl_divergence(self) -> Optional[float]:
        """
        Calculate KL divergence between old and new policies.
        
        This is a simplified implementation. In practice, you'd need access
        to both the old and new policy distributions over the same states.
        
        Returns:
            KL divergence value or None if cannot calculate
        """
        try:
            # Get rollout data from the model
            if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
                # Get observations from rollout buffer
                observations = self.model.rollout_buffer.observations
                
                # Convert to tensor if needed
                if isinstance(observations, np.ndarray):
                    observations = torch.from_numpy(observations).float()
                
                # Get old action distribution
                with torch.no_grad():
                    old_dist = self.model.policy.get_distribution(observations)
                    
                    # Get new action distribution (current policy)
                    new_dist = self.model.policy.get_distribution(observations)
                
                # Calculate KL divergence
                # KL(old || new) = sum(old * log(old/new))
                kl_divergence = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()
                
                return kl_divergence.item()
            
        except Exception as e:
            if self.verbose > 1:
                # Handle Unicode encoding issues in error messages
                try:
                    print(f"[KL] Warning: Could not calculate KL divergence: {e}")
                except UnicodeEncodeError:
                    print("[KL] Warning: Could not calculate KL divergence (encoding error)")
            return None
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose > 0 and self.kl_history:
            print(f"\n[KL] Training Summary:")
            print(f"[KL] Total KL measurements: {len(self.kl_history)}")
            print(f"[KL] Mean KL: {np.mean(self.kl_history):.6f}")
            print(f"[KL] Max KL: {np.max(self.kl_history):.6f}")
            print(f"[KL] Std KL: {np.std(self.kl_history):.6f}")
            print(f"[KL] Early stops triggered: {self.early_stops}")
            print(f"[KL] Early stop rate: {self.early_stops / len(self.kl_history) * 100:.1f}%")
    
    def get_kl_stats(self) -> Dict[str, float]:
        """Get KL divergence statistics."""
        if not self.kl_history:
            return {}
        
        return {
            'mean_kl': np.mean(self.kl_history),
            'std_kl': np.std(self.kl_history),
            'max_kl': np.max(self.kl_history),
            'min_kl': np.min(self.kl_history),
            'early_stops': self.early_stops,
            'early_stop_rate': self.early_stops / len(self.kl_history)
        }


class AdaptiveKLCallback(KLDivergenceCallback):
    """
    Adaptive KL divergence callback that adjusts the learning rate
    based on KL divergence magnitude.
    
    This implements a simplified version of the adaptive KL penalty
    approach from some PPO implementations.
    """
    
    def __init__(
        self,
        target_kl: float = 0.01,
        verbose: int = 0,
        log_freq: int = 100,
        lr_adjustment_factor: float = 0.5,
        min_lr_fraction: float = 0.1
    ):
        """
        Initialize adaptive KL callback.
        
        Args:
            target_kl: Target KL divergence threshold
            verbose: Verbosity level
            log_freq: Frequency of logging
            lr_adjustment_factor: Factor to reduce LR when KL is high
            min_lr_fraction: Minimum LR as fraction of initial
        """
        super().__init__(target_kl, verbose, log_freq)
        self.lr_adjustment_factor = lr_adjustment_factor
        self.min_lr_fraction = min_lr_fraction
        self.initial_lr = None
        self.lr_adjustments = 0
    
    def _on_step(self) -> bool:
        """Called at each step."""
        return True  # Continue training
    
    def _on_training_start(self) -> None:
        """Store initial learning rate."""
        if hasattr(self.model, 'learning_rate'):
            if callable(self.model.learning_rate):
                # Get initial LR from schedule
                self.initial_lr = self.model.learning_rate(0)
            else:
                self.initial_lr = self.model.learning_rate
            
            if self.verbose > 0:
                print(f"[KL] Initial learning rate: {self.initial_lr}")
    
    def _on_rollout_end(self) -> bool:
        """Check KL and adjust learning rate if needed."""
        continue_training = super()._on_rollout_end()
        
        # Get latest KL value
        if self.kl_history:
            latest_kl = self.kl_history[-1]
            
            # Adjust learning rate if KL is too high
            if latest_kl > self.target_kl and self.initial_lr is not None:
                current_lr = self.model.learning_rate
                if callable(current_lr):
                    # For scheduled LR, we can't easily adjust
                    if self.verbose > 1:
                        print("[KL] Cannot adjust scheduled learning rate")
                else:
                    # Reduce learning rate
                    new_lr = max(
                        current_lr * self.lr_adjustment_factor,
                        self.initial_lr * self.min_lr_fraction
                    )
                    
                    if new_lr != current_lr:
                        self.model.learning_rate = new_lr
                        self.lr_adjustments += 1
                        
                        if self.verbose > 0:
                            print(f"[KL] Adjusted learning rate: {current_lr} -> {new_lr}")
        
        return continue_training
    
    def _on_training_end(self) -> None:
        """Print summary including LR adjustments."""
        super()._on_training_end()
        
        if self.verbose > 0 and self.lr_adjustments > 0:
            print(f"[KL] Learning rate adjustments: {self.lr_adjustments}")
