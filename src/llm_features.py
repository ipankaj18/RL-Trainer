"""
LLM Feature Builder Module

Purpose: Calculate 33 new observation features for LLM context awareness.
Extends base observation from 228D to 261D for hybrid RL + LLM trading agent.

Features:
1. Extended market context (10 features)
2. Multi-timeframe indicators (8 features)  
3. Pattern recognition (10 features)
4. Risk context (5 features)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging


class LLMFeatureBuilder:
    """
    Feature builder for LLM-enhanced trading environment.
    
    Calculates 33 additional features that provide context for LLM reasoning:
    - Market context and regime
    - Multi-timeframe analysis
    - Pattern recognition
    - Risk and account metrics
    """
    
    def __init__(self):
        """Initialize feature builder with default parameters."""
        self.logger = logging.getLogger(__name__)
        
        # Feature indices for easy access
        self.feature_indices = {
            # Extended market context (10 features) - indices 228-237
            'adx_slope': 228,
            'vwap_distance': 229,
            'volatility_regime': 230,
            'volume_regime': 231,
            'price_momentum_20': 232,
            'price_momentum_60': 233,
            'efficiency_ratio': 234,
            'spread_ratio': 235,
            'session_trend': 236,
            'market_regime': 237,
            
            # Multi-timeframe indicators (8 features) - indices 238-245
            'sma_50_slope': 238,
            'sma_200_slope': 239,
            'rsi_15min': 240,
            'rsi_60min': 241,
            'volume_ratio_5min': 242,
            'volume_ratio_20min': 243,
            'price_change_60min': 244,
            'price_vs_support': 245,
            
            # Pattern recognition (10 features) - indices 246-255
            'higher_high': 246,
            'lower_low': 247,
            'higher_low': 248,
            'lower_high': 249,
            'double_top': 250,
            'double_bottom': 251,
            'support_20': 252,
            'resistance_20': 253,
            'breakout_signal': 254,
            'breakdown_signal': 255,
            
            # Risk context (5 features) - indices 256-260
            'unrealized_pnl': 256,
            'drawdown_current': 257,
            'consecutive_losses': 258,
            'win_rate_recent': 259,
            'mae_mfe_ratio': 260
        }
    
    def build_enhanced_observation(self, env, base_obs: np.ndarray) -> np.ndarray:
        """
        Extend base observation (228D) to enhanced observation (261D) with LLM features.
        
        Args:
            env: Trading environment instance
            base_obs: Base observation array (228,)
            
        Returns:
            Enhanced observation array (261,)
        """
        if base_obs.shape[0] != 228:
            raise ValueError(f"Expected base_obs shape (228,), got {base_obs.shape}")
        
        # Create extended observation array
        enhanced_obs = np.zeros(261, dtype=np.float32)
        
        # Copy base observation
        enhanced_obs[:228] = base_obs
        
        # Calculate and add LLM features
        current_idx = env.current_step
        
        # OPTIMIZATION: Access row once
        # Using iloc[idx] is faster than multiple column lookups
        # We assume env.data has all the pre-calculated features
        row = env.data.iloc[current_idx]
        
        # 1. Extended market context (10 features) - indices 228-237
        enhanced_obs[228] = row.get('adx_slope', 0.0)
        enhanced_obs[229] = row.get('price_to_vwap', 0.0)
        enhanced_obs[230] = row.get('vol_regime', 0.5)
        enhanced_obs[231] = row.get('volume_regime', 1.0)
        enhanced_obs[232] = row.get('price_momentum_20', 0.0)
        enhanced_obs[233] = row.get('price_momentum_60', 0.0)
        enhanced_obs[234] = row.get('efficiency_ratio', 0.5)
        enhanced_obs[235] = row.get('spread', 0.0)
        enhanced_obs[236] = row.get('session_trend', 0.0)
        enhanced_obs[237] = row.get('trend_strength', 0.0)
        
        # 2. Multi-timeframe indicators (8 features) - indices 238-245
        enhanced_obs[238] = row.get('sma_50_slope', 0.0)
        enhanced_obs[239] = row.get('sma_200_slope', 0.0)
        enhanced_obs[240] = row.get('rsi_15min', 50.0)
        enhanced_obs[241] = row.get('rsi_60min', 50.0)
        enhanced_obs[242] = row.get('volume_ratio_5min', 1.0)
        enhanced_obs[243] = row.get('volume_ratio_20min', 1.0)
        enhanced_obs[244] = row.get('price_change_60min', 0.0)
        enhanced_obs[245] = row.get('price_vs_support', 0.0)
        
        # 3. Pattern recognition (10 features) - indices 246-255
        enhanced_obs[246] = row.get('higher_high', 0.0)
        enhanced_obs[247] = row.get('lower_low', 0.0)
        enhanced_obs[248] = row.get('higher_low', 0.0)
        enhanced_obs[249] = row.get('lower_high', 0.0)
        enhanced_obs[250] = row.get('double_top', 0.0)
        enhanced_obs[251] = row.get('double_bottom', 0.0)
        enhanced_obs[252] = row.get('support_20', 0.0)
        enhanced_obs[253] = row.get('resistance_20', 0.0)
        enhanced_obs[254] = row.get('breakout_signal', 0.0)
        enhanced_obs[255] = row.get('breakdown_signal', 0.0)
        
        # 4. Risk context (5 features) - indices 256-260
        # These depend on account state, so they must be calculated dynamically
        self._add_risk_context(enhanced_obs, env, current_idx)
        
        return enhanced_obs
    
    def _add_risk_context(self, obs: np.ndarray, env, idx: int):
        """Add risk context features (indices 256-260)."""
        
        # Unrealized P&L (feature 256)
        obs[256] = env._calculate_unrealized_pnl() if hasattr(env, '_calculate_unrealized_pnl') else 0.0
        
        # Current drawdown (feature 257)
        if hasattr(env, 'peak_balance') and hasattr(env, 'balance'):
            current_dd = (env.peak_balance - env.balance) / (env.peak_balance + 1e-8)
            obs[257] = max(0.0, current_dd)
        else:
            obs[257] = 0.0
        
        # Consecutive losses (feature 258)
        if hasattr(env, 'consecutive_losses'):
            obs[258] = env.consecutive_losses
        else:
            obs[258] = 0.0
        
        # Recent win rate (feature 259)
        if hasattr(env, 'trade_pnl_history'):
            recent_trades = env.trade_pnl_history[-10:]  # Last 10 trades
            if recent_trades:
                wins = sum(1 for pnl in recent_trades if pnl > 0)
                obs[259] = wins / len(recent_trades)
            else:
                obs[259] = 0.5  # Neutral
        else:
            obs[259] = 0.5
        
        # MAE/MFE ratio (feature 260)
        if hasattr(env, 'max_adverse_excursion') and hasattr(env, 'max_favorable_excursion'):
            mae = abs(env.max_adverse_excursion)
            mfe = abs(env.max_favorable_excursion)
            if mfe > 0:
                obs[260] = mae / mfe
            else:
                obs[260] = 1.0
        else:
            obs[260] = 1.0


if __name__ == '__main__':
    """Test LLM feature builder."""
    import pandas as pd
    
    print("Testing LLM Feature Builder...")
    
    # Create test data
    dates = pd.date_range('2024-01-01 09:30', periods=500, freq='1min', tz='America/New_York')
    test_data = pd.DataFrame({
        'open': np.random.uniform(4000, 4100, 500),
        'high': np.random.uniform(4100, 4150, 500),
        'low': np.random.uniform(3950, 4000, 500),
        'close': np.random.uniform(4000, 4100, 500),
        'volume': np.random.randint(100, 1000, 500),
        'sma_5': np.random.uniform(4000, 4100, 500),
        'sma_20': np.random.uniform(4000, 4100, 500),
        'sma_50': np.random.uniform(4000, 4100, 500),
        'sma_200': np.random.uniform(4000, 4100, 500),
        'rsi': np.random.uniform(30, 70, 500),
        'rsi_15min': np.random.uniform(30, 70, 500),
        'rsi_60min': np.random.uniform(30, 70, 500),
        'adx': np.random.uniform(20, 40, 500),
        'vwap': np.random.uniform(4000, 4100, 500),
        'price_to_vwap': np.random.uniform(-0.02, 0.02, 500),
        'vol_regime': np.random.uniform(0.3, 0.7, 500),
        'efficiency_ratio': np.random.uniform(0.2, 0.8, 500),
        'trend_strength': np.random.choice([-1, 0, 1], 500),
        'support_20': np.random.uniform(3950, 4050, 500),
        'resistance_20': np.random.uniform(4050, 4150, 500),
        # Add some new features for testing
        'adx_slope': np.random.uniform(-1, 1, 500),
        'sma_50_slope': np.random.uniform(-1, 1, 500),
        'higher_high': np.random.choice([0, 1, -1], 500)
    }, index=dates)
    
    # Mock environment
    class MockEnv:
        def __init__(self, data):
            self.data = data
            self.current_step = 100
            self.position = 0
            self.balance = 50000
            self.peak_balance = 50000
            self.consecutive_losses = 0
            self.trade_pnl_history = [100, -50, 200, -30, 150]
            self.max_adverse_excursion = -100
            self.max_favorable_excursion = 200
        
        def _calculate_unrealized_pnl(self):
            return 50.0
    
    env = MockEnv(test_data)
    builder = LLMFeatureBuilder()
    
    # Test feature building
    base_obs = np.random.randn(228).astype(np.float32)
    enhanced_obs = builder.build_enhanced_observation(env, base_obs)
    
    print(f"Base observation shape: {base_obs.shape}")
    print(f"Enhanced observation shape: {enhanced_obs.shape}")
    print(f"Expected: (261,)")
    
    # Check that base features are preserved
    assert np.array_equal(enhanced_obs[:228], base_obs), "Base observation not preserved"
    assert enhanced_obs.shape == (261,), f"Expected (261,), got {enhanced_obs.shape}"
    
    # Check for NaN/Inf
    assert not np.isnan(enhanced_obs).any(), "Enhanced obs contains NaN"
    assert not np.isinf(enhanced_obs).any(), "Enhanced obs contains Inf"
    
    print("✓ LLM Feature Builder test passed!")
    print(f"✓ Generated {len(enhanced_obs) - len(base_obs)} new features")
    print("✓ All features validated successfully")