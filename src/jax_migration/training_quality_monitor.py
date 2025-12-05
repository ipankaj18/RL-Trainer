"""
Training Quality Monitor for JAX PPO Training

Detects degenerate training patterns and recommends interventions.
Part of Phase C: Adaptive Training System
"""

import jax.numpy as jnp
from collections import deque
from typing import Dict, List, Tuple, Optional


class TrainingQualityMonitor:
    """
    Monitors training quality over a sliding window and detects degenerate patterns.
    
    Degenerate patterns detected:
    - Zero trades for N consecutive updates
    - Entropy below threshold for N consecutive updates
    - Win rate < 30% despite many trades
    - Monotonically decreasing returns
    - HOLD-dominated action distribution (>95%)
    
    Returns quality score (0.0-1.0) and recommended parameter adjustments.
    """
    
    def __init__(
        self, 
        window_size: int = 10,
        min_trades_per_update: int = 1,
        min_entropy: float = 0.10,
        min_win_rate: float = 0.30,
        max_hold_ratio: float = 0.95
    ):
        self.window_size = window_size
        self.min_trades_per_update = min_trades_per_update
        self.min_entropy = min_entropy
        self.min_win_rate = min_win_rate
        self.max_hold_ratio = max_hold_ratio
        
        # Sliding windows for metrics
        self.entropy_window = deque(maxlen=window_size)
        self.trades_window = deque(maxlen=window_size)
        self.winrate_window = deque(maxlen=window_size)
        self.returns_window = deque(maxlen=window_size)
        self.hold_ratio_window = deque(maxlen=window_size)
        
        # Counters for consecutive violations
        self.consecutive_zero_trades = 0
        self.consecutive_low_entropy = 0
        self.consecutive_hold_dominated = 0
        
        # Current quality score
        self.quality_score = 1.0
    
    def update(self, metrics: Dict) -> float:
        """
        Update monitor with new metrics and calculate quality score.
        
        Args:
            metrics: Dictionary with keys:
                - entropy: float
                - trades: int (total trades in last batch)
                - win_rate: float (0-1)
                - mean_return: float
                - action_dist: array [hold_ratio, buy_ratio, sell_ratio]
        
        Returns:
            quality_score: 0.0-1.0 (1.0 = perfect, 0.0 = completely degenerate)
        """
        # Extract metrics
        entropy = metrics.get('entropy', 0.15)
        trades = metrics.get('trades', 0)
        win_rate = metrics.get('win_rate', 0.0)
        mean_return = metrics.get('mean_return', 0.0)
        action_dist = metrics.get('action_dist', jnp.array([0.7, 0.15, 0.15]))
        hold_ratio = float(action_dist[0]) if len(action_dist) > 0 else 0.7
        
        # Update windows
        self.entropy_window.append(entropy)
        self.trades_window.append(trades)
        self.winrate_window.append(win_rate)
        self.returns_window.append(mean_return)
        self.hold_ratio_window.append(hold_ratio)
        
        # Update consecutive violation counters
        if trades < self.min_trades_per_update:
            self.consecutive_zero_trades += 1
        else:
            self.consecutive_zero_trades = 0
        
        if entropy < self.min_entropy:
            self.consecutive_low_entropy += 1
        else:
            self.consecutive_low_entropy = 0
        
        if hold_ratio > self.max_hold_ratio:
            self.consecutive_hold_dominated += 1
        else:
            self.consecutive_hold_dominated = 0
        
        # Calculate quality score components (0-1, higher is better)
        entropy_score = min(1.0, max(0.0, (entropy - 0.05) / 0.15))  # 0.05-0.20 range
        trades_score = min(1.0, trades / (self.min_trades_per_update * 10))  # Scale to 10x min
        hold_score = max(0.0, 1.0 - (hold_ratio - 0.60) / 0.35)  # 60-95% range
        
        # Win rate score (only if trading)
        if trades > 10:
            winrate_score = max(0.0, (win_rate - 0.20) / 0.30)  # 20-50% range
        else:
            winrate_score = 0.5  # Neutral if not enough trades
        
        # Returns trend score (based on last 5 vs first 5 in window)
        if len(self.returns_window) >= 10:
            recent_avg = sum(list(self.returns_window)[-5:]) / 5
            older_avg = sum(list(self.returns_window)[:5]) / 5
            if older_avg != 0:
                returns_improvement = (recent_avg - older_avg) / abs(older_avg)
                returns_score = 0.5 + min(0.5, max(-0.5, returns_improvement))
            else:
                returns_score = 0.5
        else:
            returns_score = 0.5  # Neutral if not enough data
        
        # Weighted average of components
        self.quality_score = (
            0.30 * entropy_score +
            0.25 * trades_score +
            0.20 * hold_score +
            0.15 * winrate_score +
            0.10 * returns_score
        )
        
        return self.quality_score
    
    def should_adjust(self) -> bool:
        """
        Returns True if intervention is recommended.
        
        Triggers:
        - Quality score < 0.5
        - 5+ consecutive zero-trade updates
        - 5+ consecutive low-entropy updates
        - 5+ consecutive HOLD-dominated updates
        """
        if self.quality_score < 0.5:
            return True
        if self.consecutive_zero_trades >= 5:
            return True
        if self.consecutive_low_entropy >= 5:
            return True
        if self.consecutive_hold_dominated >= 5:
            return True
        return False
    
    def get_recommendations(self) -> List[str]:
        """
        Returns list of recommended parameter adjustments.
        
        Recommendations based on detected issues:
        - Zero trades → increase entry bonus, reduce hold penalty
        - Low entropy → increase entropy coefficient
        - HOLD-dominated → increase entry bonus
        - Poor win rate → slow commission curriculum
        """
        recommendations = []
        
        # DISABLED: Simplified reward function doesn't use entry_bonus/hold_penalty
        # Zero trades issue
        # if self.consecutive_zero_trades >= 3:
        #     recommendations.append("increase_entry_bonus")
        #     recommendations.append("reduce_hold_penalty")
        
        # Low entropy issue
        if self.consecutive_low_entropy >= 3:
            recommendations.append("increase_entropy_coef")
        
        # DISABLED: Simplified reward function doesn't use entry_bonus/hold_penalty
        # HOLD-dominated issue
        # if self.consecutive_hold_dominated >= 3:
        #     recommendations.append("increase_entry_bonus")
        #     recommendations.append("reduce_hold_penalty")
        
        # Poor win rate (if trading)
        if len(self.trades_window) > 0 and sum(self.trades_window) > 50:
            avg_winrate = sum(self.winrate_window) / len(self.winrate_window)
            if avg_winrate < self.min_win_rate:
                recommendations.append("slow_commission_curriculum")
        
        return list(set(recommendations))  # Remove duplicates
    
    def get_status_summary(self) -> Dict:
        """Returns summary of current monitoring status."""
        return {
            'quality_score': self.quality_score,
            'consecutive_zero_trades': self.consecutive_zero_trades,
            'consecutive_low_entropy': self.consecutive_low_entropy,
            'consecutive_hold_dominated': self.consecutive_hold_dominated,
            'avg_entropy': sum(self.entropy_window) / len(self.entropy_window) if self.entropy_window else 0.0,
            'avg_trades': sum(self.trades_window) / len(self.trades_window) if self.trades_window else 0.0,
            'avg_hold_ratio': sum(self.hold_ratio_window) / len(self.hold_ratio_window) if self.hold_ratio_window else 0.0,
        }
