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
        data = env.data
        
        # 1. Extended market context (10 features)
        self._add_extended_market_context(enhanced_obs, data, current_idx)
        
        # 2. Multi-timeframe indicators (8 features)
        self._add_multitimeframe_indicators(enhanced_obs, data, current_idx)
        
        # 3. Pattern recognition (10 features)
        self._add_pattern_recognition(enhanced_obs, data, current_idx)
        
        # 4. Risk context (5 features)
        self._add_risk_context(enhanced_obs, env, current_idx)
        
        return enhanced_obs
    
    def _add_extended_market_context(self, obs: np.ndarray, data: pd.DataFrame, idx: int):
        """Add extended market context features (indices 228-237)."""
        
        # ADX slope (feature 228)
        if 'adx' in data.columns and idx >= 5:
            obs[228] = self.calculate_sma_slope(data['adx'].values, idx, 5)
        else:
            obs[228] = 0.0
        
        # VWAP distance (feature 229)
        if 'price_to_vwap' in data.columns:
            obs[229] = data['price_to_vwap'].iloc[idx]
        else:
            obs[229] = 0.0
        
        # Volatility regime (feature 230)
        if 'vol_regime' in data.columns:
            obs[230] = data['vol_regime'].iloc[idx]
        else:
            obs[230] = 0.5
        
        # Volume regime (feature 231)
        if 'volume' in data.columns and idx >= 20:
            current_vol = data['volume'].iloc[idx]
            avg_vol = data['volume'].iloc[idx-20:idx].mean()
            obs[231] = current_vol / (avg_vol + 1e-8)
        else:
            obs[231] = 1.0
        
        # Price momentum 20 (feature 232)
        if 'close' in data.columns and idx >= 20:
            obs[232] = (data['close'].iloc[idx] / data['close'].iloc[idx-20]) - 1
        else:
            obs[232] = 0.0
        
        # Price momentum 60 (feature 233)
        if 'close' in data.columns and idx >= 60:
            obs[233] = (data['close'].iloc[idx] / data['close'].iloc[idx-60]) - 1
        else:
            obs[233] = 0.0
        
        # Efficiency ratio (feature 234)
        if 'efficiency_ratio' in data.columns:
            obs[234] = data['efficiency_ratio'].iloc[idx]
        else:
            obs[234] = 0.5
        
        # Spread ratio (feature 235)
        if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            spread = data['high'].iloc[idx] - data['low'].iloc[idx]
            obs[235] = spread / data['close'].iloc[idx]
        else:
            obs[235] = 0.0
        
        # Session trend (feature 236)
        if 'close' in data.columns and idx >= 30:
            # Compare current price to session open (30 bars ago ~ 30 minutes)
            session_open = data['close'].iloc[max(0, idx-30)]
            obs[236] = (data['close'].iloc[idx] / session_open) - 1
        else:
            obs[236] = 0.0
        
        # Market regime (feature 237)
        if 'trend_strength' in data.columns:
            obs[237] = data['trend_strength'].iloc[idx]
        else:
            obs[237] = 0.0
    
    def _add_multitimeframe_indicators(self, obs: np.ndarray, data: pd.DataFrame, idx: int):
        """Add multi-timeframe indicator features (indices 238-245)."""
        
        # SMA 50 slope (feature 238)
        if 'sma_50' in data.columns and idx >= 10:
            obs[238] = self.calculate_sma_slope(data['sma_50'].values, idx, 10)
        else:
            obs[238] = 0.0
        
        # SMA 200 slope (feature 239)
        if 'sma_200' in data.columns and idx >= 20:
            obs[239] = self.calculate_sma_slope(data['sma_200'].values, idx, 20)
        else:
            obs[239] = 0.0
        
        # RSI 15min (feature 240)
        if 'rsi_15min' in data.columns:
            obs[240] = data['rsi_15min'].iloc[idx]
        elif 'rsi' in data.columns:
            obs[240] = data['rsi'].iloc[idx]
        else:
            obs[240] = 50.0
        
        # RSI 60min (feature 241)
        if 'rsi_60min' in data.columns:
            obs[241] = data['rsi_60min'].iloc[idx]
        elif 'rsi' in data.columns:
            obs[241] = data['rsi'].iloc[idx]
        else:
            obs[241] = 50.0
        
        # Volume ratio 5min (feature 242)
        if 'volume' in data.columns and idx >= 5:
            current_vol = data['volume'].iloc[idx]
            avg_vol_5 = data['volume'].iloc[idx-5:idx].mean()
            obs[242] = current_vol / (avg_vol_5 + 1e-8)
        else:
            obs[242] = 1.0
        
        # Volume ratio 20min (feature 243)
        if 'volume' in data.columns and idx >= 20:
            current_vol = data['volume'].iloc[idx]
            avg_vol_20 = data['volume'].iloc[idx-20:idx].mean()
            obs[243] = current_vol / (avg_vol_20 + 1e-8)
        else:
            obs[243] = 1.0
        
        # Price change 60min (feature 244)
        if 'close' in data.columns and idx >= 60:
            obs[244] = (data['close'].iloc[idx] / data['close'].iloc[idx-60]) - 1
        else:
            obs[244] = 0.0
        
        # Price vs support (feature 245)
        if 'support_20' in data.columns and 'close' in data.columns:
            support = data['support_20'].iloc[idx]
            current_price = data['close'].iloc[idx]
            obs[245] = (current_price - support) / (current_price + 1e-8)
        else:
            obs[245] = 0.0
    
    def _add_pattern_recognition(self, obs: np.ndarray, data: pd.DataFrame, idx: int):
        """Add pattern recognition features (indices 246-255)."""
        
        # Higher high (feature 246)
        obs[246] = self.detect_higher_high(data, idx)
        
        # Lower low (feature 247)
        obs[247] = self.detect_lower_low(data, idx)
        
        # Higher low (feature 248)
        obs[248] = self.detect_higher_low(data, idx)
        
        # Lower high (feature 249)
        obs[249] = self.detect_lower_high(data, idx)
        
        # Double top (feature 250)
        obs[250] = self.detect_double_top(data, idx)
        
        # Double bottom (feature 251)
        obs[251] = self.detect_double_bottom(data, idx)
        
        # Support 20 (feature 252)
        if 'support_20' in data.columns:
            obs[252] = data['support_20'].iloc[idx]
        else:
            obs[252] = 0.0
        
        # Resistance 20 (feature 253)
        if 'resistance_20' in data.columns:
            obs[253] = data['resistance_20'].iloc[idx]
        else:
            obs[253] = 0.0
        
        # Breakout signal (feature 254)
        obs[254] = self.detect_breakout(data, idx)
        
        # Breakdown signal (feature 255)
        obs[255] = self.detect_breakdown(data, idx)
    
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
    
    def calculate_sma_slope(self, data: np.ndarray, idx: int, period: int) -> float:
        """
        Calculate SMA slope for trend detection.
        
        Args:
            data: Array of values
            idx: Current index
            period: Lookback period
            
        Returns:
            Normalized slope value
        """
        if idx < period or period < 2:
            return 0.0
        
        # Calculate linear regression slope on recent values
        recent_values = data[idx-period:idx]
        x = np.arange(period)
        
        # Simple slope calculation
        slope = (recent_values[-1] - recent_values[0]) / period
        
        # Normalize by current value to make it relative
        current_value = data[idx]
        if current_value != 0:
            slope = slope / current_value
        
        return slope
    
    def detect_higher_high(self, data: pd.DataFrame, idx: int, lookback: int = 5) -> float:
        """
        Detect if current high is higher than previous high.
        
        Returns:
            1.0 if higher high, -1.0 if lower high, 0.0 if no pattern
        """
        if 'high' not in data.columns or idx < lookback * 2:
            return 0.0
        
        # Find recent highs
        current_high = data['high'].iloc[idx]
        prev_high = data['high'].iloc[idx-lookback:idx].max()
        
        if current_high > prev_high * 1.001:  # 0.1% buffer
            return 1.0
        elif current_high < prev_high * 0.999:
            return -1.0
        else:
            return 0.0
    
    def detect_lower_low(self, data: pd.DataFrame, idx: int, lookback: int = 5) -> float:
        """
        Detect if current low is lower than previous low.
        
        Returns:
            1.0 if lower low, -1.0 if higher low, 0.0 if no pattern
        """
        if 'low' not in data.columns or idx < lookback * 2:
            return 0.0
        
        current_low = data['low'].iloc[idx]
        prev_low = data['low'].iloc[idx-lookback:idx].min()
        
        if current_low < prev_low * 0.999:
            return 1.0
        elif current_low > prev_low * 1.001:
            return -1.0
        else:
            return 0.0
    
    def detect_higher_low(self, data: pd.DataFrame, idx: int, lookback: int = 5) -> float:
        """Detect higher low pattern."""
        if 'low' not in data.columns or idx < lookback * 2:
            return 0.0
        
        current_low = data['low'].iloc[idx]
        prev_low_1 = data['low'].iloc[idx-lookback:idx].min()
        prev_low_2 = data['low'].iloc[idx-lookback*2:idx-lookback].min()
        
        if current_low > prev_low_1 * 1.001 and prev_low_1 > prev_low_2 * 1.001:
            return 1.0
        else:
            return 0.0
    
    def detect_lower_high(self, data: pd.DataFrame, idx: int, lookback: int = 5) -> float:
        """Detect lower high pattern."""
        if 'high' not in data.columns or idx < lookback * 2:
            return 0.0
        
        current_high = data['high'].iloc[idx]
        prev_high_1 = data['high'].iloc[idx-lookback:idx].max()
        prev_high_2 = data['high'].iloc[idx-lookback*2:idx-lookback].max()
        
        if current_high < prev_high_1 * 0.999 and prev_high_1 < prev_high_2 * 0.999:
            return 1.0
        else:
            return 0.0
    
    def detect_double_top(self, data: pd.DataFrame, idx: int, lookback: int = 20) -> float:
        """
        Detect potential double top pattern.
        
        Returns:
            Confidence score 0-1
        """
        if 'high' not in data.columns or idx < lookback * 2:
            return 0.0
        
        recent_highs = data['high'].iloc[idx-lookback:idx]
        prev_highs = data['high'].iloc[idx-lookback*2:idx-lookback]
        
        # Check if recent highs are similar and lower than previous peak
        recent_max = recent_highs.max()
        prev_max = prev_highs.max()
        
        if recent_max < prev_max * 0.99 and abs(recent_highs.std() / recent_max) < 0.02:
            return 0.7  # Potential double top
        else:
            return 0.0
    
    def detect_double_bottom(self, data: pd.DataFrame, idx: int, lookback: int = 20) -> float:
        """Detect potential double bottom pattern."""
        if 'low' not in data.columns or idx < lookback * 2:
            return 0.0
        
        recent_lows = data['low'].iloc[idx-lookback:idx]
        prev_lows = data['low'].iloc[idx-lookback*2:idx-lookback]
        
        recent_min = recent_lows.min()
        prev_min = prev_lows.min()
        
        if recent_min > prev_min * 1.01 and abs(recent_lows.std() / recent_min) < 0.02:
            return 0.7  # Potential double bottom
        else:
            return 0.0
    
    def detect_breakout(self, data: pd.DataFrame, idx: int, lookback: int = 20) -> float:
        """Detect breakout above resistance."""
        if 'close' not in data.columns or 'high' not in data.columns or idx < lookback:
            return 0.0
        
        current_close = data['close'].iloc[idx]
        recent_highs = data['high'].iloc[idx-lookback:idx]
        resistance = recent_highs.max()
        
        if current_close > resistance * 1.002:
            return 1.0
        else:
            return 0.0
    
    def detect_breakdown(self, data: pd.DataFrame, idx: int, lookback: int = 20) -> float:
        """Detect breakdown below support."""
        if 'close' not in data.columns or 'low' not in data.columns or idx < lookback:
            return 0.0
        
        current_close = data['close'].iloc[idx]
        recent_lows = data['low'].iloc[idx-lookback:idx]
        support = recent_lows.min()
        
        if current_close < support * 0.998:
            return 1.0
        else:
            return 0.0


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
        'resistance_20': np.random.uniform(4050, 4150, 500)
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