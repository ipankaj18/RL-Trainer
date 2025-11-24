"""
Phase 3: LLM-Enhanced Trading Environment

Hybrid RL + LLM Trading Agent

Key changes from Phase 2:
- Observation space: 228     261 dimensions (adds 33 LLM features)
- Adds: Account state, trade memory, risk metrics, pattern recognition
- Compatible with: MaskablePPO (action masking preserved)
- Enables: LLM reasoning and decision fusion

Features:
- Extended market context (ADX slope, VWAP distance, volatility regime)
- Multi-timeframe indicators (SMA-50/200 slopes, multi-timeframe RSI)
- Pattern recognition (higher/lower highs/lows, support/resistance)
- Risk context (unrealized P&L, drawdown tracking, consecutive losses)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
from environment_phase2 import TradingEnvironmentPhase2
from llm_features import LLMFeatureBuilder


class TradingEnvironmentPhase3LLM(TradingEnvironmentPhase2):
    """
    Phase 3: LLM-Enhanced Trading Environment

    Extends Phase 2 with 33 additional features for LLM context awareness.
    Maintains backward compatibility by allowing use_llm_features=False.

    Note: Action masking is inherited from Phase 2 via the action_masks() method.
    This method is called by ActionMasker wrapper during training to prevent invalid actions.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        use_llm_features: bool = True,
        initial_balance: float = 50000,
        window_size: int = 20,
        second_data: pd.DataFrame = None,
        # Market specifications
        market_spec=None,
        commission_override: float = None,
        # Position management
        initial_sl_multiplier: float = 1.5,
        initial_tp_ratio: float = 3.0,
        position_size_contracts: float = 1.0,
        trailing_drawdown_limit: float = 2500,
        tighten_sl_step: float = 0.5,
        extend_tp_step: float = 1.0,
        trailing_activation_profit: float = 1.0,
        # PHASE 1 & 2: Hybrid agent for outcome tracking
        hybrid_agent=None,
        # Episode start management
        start_index: Optional[int] = None,
        randomize_start_offsets: bool = True,
        min_episode_bars: int = 1500
    ):
        """
        Initialize Phase 3 environment with LLM features.
        
        Args:
            data: OHLCV + indicators DataFrame (must include LLM features)
            use_llm_features: Whether to use 261D observations (True) or 228D (False)
            initial_balance: Starting capital
            window_size: Lookback window
            second_data: Optional second-level data
            market_spec: Market specification object
            commission_override: Override default commission
            initial_sl_multiplier: Initial SL distance
            initial_tp_ratio: Initial TP ratio
            position_size_contracts: Position size in contracts
            trailing_drawdown_limit: Apex trailing drawdown limit
            tighten_sl_step: Amount to tighten SL (in ATR)
            extend_tp_step: Amount to extend TP (in ATR)
            trailing_activation_profit: Min profit to enable trailing
        """
        # Phase 3: LLM feature configuration - MUST be set before super().__init__
        # because parent calls reset() which calls _get_observation()
        self.use_llm_features = use_llm_features
        
        # Initialize feature builder BEFORE parent initialization
        # This is needed because parent __init__ calls reset() which calls _get_observation()
        self.feature_builder = LLMFeatureBuilder()
        
        # Initialize parent Phase 2 environment
        super().__init__(
            data, initial_balance, window_size, second_data,
            market_spec, commission_override,
            initial_sl_multiplier, initial_tp_ratio, position_size_contracts,
            trailing_drawdown_limit, tighten_sl_step, extend_tp_step,
            trailing_activation_profit,
            start_index=start_index,
            randomize_start_offsets=randomize_start_offsets,
            min_episode_bars=min_episode_bars
        )
        
        # Track additional state for LLM features
        self.trade_pnl_history = []
        self.max_adverse_excursion = 0.0
        self.max_favorable_excursion = 0.0
        self.consecutive_losses = 0
        self.last_trade_result = 0.0
        
        # Track peak balance for drawdown calculation
        self.peak_balance = initial_balance
        
        # PHASE 1 & 2: Store hybrid agent reference for outcome tracking
        self.hybrid_agent = hybrid_agent
        
        # Update observation space if using LLM features
        if use_llm_features:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(261,),  # Extended from 228 to 261
                dtype=np.float32
            )
            print(f"[ENV] Phase 3 LLM environment initialized with {261}D observations")
        else:
            print(f"[ENV] Phase 3 environment initialized with {228}D observations (LLM features disabled)")
        
        # Validate data has required LLM features
        if use_llm_features:
            self._validate_llm_features()
    
    def _validate_llm_features(self):
        """Validate that data contains required LLM features."""
        required_features = [
            'sma_50', 'sma_200', 'rsi_15min', 'rsi_60min',
            'volume_ratio_5min', 'volume_ratio_20min',
            'support_20', 'resistance_20', 'price_change_60min'
        ]
        
        missing = [feat for feat in required_features if feat not in self.data.columns]
        if missing:
            print(f"[WARNING] Missing LLM features: {missing}")
            print("[WARNING] Some LLM features will be set to default values")
    
    def _get_observation(self) -> np.ndarray:
        """
        Get enhanced observation with LLM features.

        HIGH PRIORITY FIX #2: Added runtime validation of observation dimensions.

        Returns:
            Observation array (261D if use_llm_features=True, else 228D)
        """
        # Get base observation from Phase 2 (228D)
        base_obs = super()._get_observation()

        # HIGH PRIORITY FIX #2: Validate base observation shape
        if base_obs.shape[0] != 228:
            raise ValueError(
                f"Phase 3 expects 228D base observation from Phase 2, but got {base_obs.shape[0]}D. "
                f"This indicates Phase 2 environment changed. Expected: "
                f"220 market features + 5 position features + 3 validity features = 228D"
            )

        if not self.use_llm_features:
            return base_obs

        # Extend to 261D with LLM features
        try:
            enhanced_obs = self.feature_builder.build_enhanced_observation(
                self, base_obs
            )

            # HIGH PRIORITY FIX #2: Validate final observation shape
            if enhanced_obs.shape[0] != 261:
                raise ValueError(
                    f"Phase 3 expects 261D enhanced observation, but got {enhanced_obs.shape[0]}D. "
                    f"Expected: 228D (base) + 33D (LLM features) = 261D. "
                    f"Check LLMFeatureBuilder.build_enhanced_observation()"
                )

            return enhanced_obs
        except Exception as e:
            # Only catch exceptions during feature building, not assertion errors
            if isinstance(e, (ValueError, AssertionError)):
                raise  # Re-raise dimension validation errors
            print(f"[WARNING] Error building enhanced observation: {e}")
            print("[WARNING] Falling back to base observation (this will cause training issues!)")
            return base_obs
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment and LLM-specific tracking variables.
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Observation and info dictionary
        """
        # Reset parent environment
        obs, info = super().reset(seed, options)
        
        # Reset LLM-specific tracking
        self.trade_pnl_history = []
        self.max_adverse_excursion = 0.0
        self.max_favorable_excursion = 0.0
        self.consecutive_losses = 0
        self.last_trade_result = 0.0
        self.peak_balance = self.initial_balance

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action with LLM state tracking.
        
        Args:
            action: Action to take (0-5)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Execute step in parent environment
        obs, reward, terminated, truncated, info = super().step(action)
        
        # CRITICAL: Update position tracking for hybrid agent
        self._update_position_tracking(action)
        
        # Track trade results for LLM context
        if 'trade_pnl' in info and info['trade_pnl'] != 0:
            trade_pnl = info['trade_pnl']
            self.trade_pnl_history.append(trade_pnl)
            self.last_trade_result = trade_pnl
            
            # Update consecutive losses counter
            if trade_pnl > 0:
                self.consecutive_losses = 0  # Reset on win
            else:
                self.consecutive_losses += 1  # Increment on loss
            
            # PHASE 1: Update fusion outcome (for neural fusion training)
            if hasattr(self, 'hybrid_agent') and hasattr(self.hybrid_agent, 'update_fusion_outcome'):
                # Convert P&L to reward signal (scale to reasonable range)
                reward_signal = np.clip(trade_pnl / 100.0, -10, 10)
                self.hybrid_agent.update_fusion_outcome(reward_signal)
            
            # PHASE 2: Update LLM outcome (for LoRA fine-tuning)
            # We need to track the last LLM query ID. This should be passed from the agent.
            # For now, we'll add a placeholder that can be called by the agent.
        
        # Track MAE/MFE for risk features
        if self.position != 0:
            unrealized = self._calculate_unrealized_pnl()
            
            if unrealized < self.max_adverse_excursion:
                self.max_adverse_excursion = unrealized
            if unrealized > self.max_favorable_excursion:
                self.max_favorable_excursion = unrealized
        else:
            # Reset when flat
            self.max_adverse_excursion = 0.0
            self.max_favorable_excursion = 0.0
        
        # Update peak balance for drawdown tracking
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        # Add LLM-specific info
        info.update({
            'consecutive_losses': self.consecutive_losses,
            'max_adverse_excursion': self.max_adverse_excursion,
            'max_favorable_excursion': self.max_favorable_excursion,
            'peak_balance': self.peak_balance,
            'trade_pnl_history': self.trade_pnl_history[-10:],  # Last 10 trades
        })
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_unrealized_pnl(self, current_price=None) -> float:
        """
        Calculate current unrealized P&L.
        
        Returns:
            Unrealized P&L in dollars
        """
        if self.position == 0:
            return 0.0
        
        if current_price is None:
            current_price = self.data['close'].iloc[self.current_step]
        
        # Get contract multiplier from parent class (inherited from Phase1)
        point_value = getattr(self, 'contract_size', 20.0)  # Default to 20.0 for NQ/ES if not set
        
        if self.position > 0:  # Long position
            unrealized = (current_price - self.entry_price) * self.position_size * point_value
        else:  # Short position
            unrealized = (self.entry_price - current_price) * abs(self.position_size) * point_value
        
        return unrealized
    
    def get_llm_context(self) -> Dict:
        """
        Get market context for LLM reasoning.
        
        Returns:
            Dictionary with market context information
        """
        current_idx = self.current_step
        
        # Extract relevant features from current data
        context = {
            'market_name': self.market_name,
            'current_time': self.data.index[current_idx].strftime('%H:%M') if isinstance(self.data.index, pd.DatetimeIndex) else 'Unknown',
            'current_price': self.data['close'].iloc[current_idx],
            'trend_strength': 'Strong' if self.data.get('trend_strength', pd.Series([0])).iloc[current_idx] > 0 else 'Weak',
            'adx': self.data.get('adx', pd.Series([25])).iloc[current_idx],
            'vwap_distance': self.data.get('price_to_vwap', pd.Series([0])).iloc[current_idx],
            'rsi': self.data.get('rsi', pd.Series([50])).iloc[current_idx],
            'position_status': 'FLAT' if self.position == 0 else f'{"LONG" if self.position > 0 else "SHORT"}',
            'balance': self.balance,
            'win_rate': self._calculate_win_rate(),
            'consecutive_losses': self.consecutive_losses,
            'unrealized_pnl': self._calculate_unrealized_pnl() if self.position != 0 else 0.0,
        }
        
        return context
    
    def _calculate_win_rate(self) -> float:
        """
        Calculate recent win rate.
        
        Returns:
            Win rate as percentage (0-1)
        """
        if not self.trade_pnl_history:
            return 0.5
        
        recent_trades = self.trade_pnl_history[-10:]  # Last 10 trades
        wins = sum(1 for pnl in recent_trades if pnl > 0)
        
        return wins / len(recent_trades)
    
    def predict(self, observation: np.ndarray, action_masks: Optional[np.ndarray] = None, 
                deterministic: bool = False) -> Tuple[int, Optional[Dict]]:
        """
        CRITICAL FIX: Prediction method called by SB3 during training.
        Routes through hybrid agent to enable LLM integration.
        
        This method is called by the HybridAgentPolicy during training.
        It enables the hybrid agent (and LLM) to participate in the training loop.
        
        Args:
            observation: Observation array (261D if use_llm_features=True)
            action_masks: Action mask array (6D)
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (action, info_dict)
        """
        # CRITICAL: Route through hybrid agent if available
        if hasattr(self, 'hybrid_agent') and self.hybrid_agent is not None:
            # Build comprehensive position state for hybrid agent
            position_state = {
                'position': self.position,
                'balance': self.balance,
                'win_rate': self._calculate_win_rate(),
                'consecutive_losses': self.consecutive_losses,
                'dd_buffer_ratio': self._calculate_dd_buffer(),
                'time_in_position': self._get_time_in_position(),
                'unrealized_pnl': self._calculate_unrealized_pnl(),
                'max_adverse_excursion': self.max_adverse_excursion,
                'max_favorable_excursion': self.max_favorable_excursion,
                'entry_price': self.entry_price if self.position != 0 else 0.0,
                'timestamp': self.current_step
            }
            
            # Get market context for LLM
            market_context = self.get_llm_context()
            
            # Route through hybrid agent (activates LLM!)
            # This is where the magic happens - LLM is queried during training
            action, meta = self.hybrid_agent.predict(
                observation, action_masks, position_state, market_context,
                env_id=getattr(self, 'env_id', 0)
            )
            
            # Log that LLM was used during training
            if self.current_step % 1000 == 0:
                print(f"[ENV] Hybrid agent used for prediction (step {self.current_step})")
                if meta.get('llm_queried', False):
                    print(f"[ENV] LLM query executed: confidence={meta.get('llm_confidence', 0):.2f}")
            
            return action, meta
        
        # Fallback to parent class (RL-only) if hybrid agent not available
        # This maintains backward compatibility
        print("[ENV] WARNING: Hybrid agent not available, using RL-only prediction")
        return super().predict(observation, action_masks, deterministic)
    
    def _calculate_dd_buffer(self) -> float:
        """
        Calculate drawdown buffer ratio.
        
        Returns:
            Ratio of current balance to peak balance (1.0 = no drawdown)
        """
        if self.peak_balance <= 0:
            return 1.0
        return self.balance / self.peak_balance
    
    def _get_time_in_position(self) -> int:
        """
        Get time (in steps) spent in current position.
        
        Returns:
            Time in position (0 if flat)
        """
        if self.position == 0 or not hasattr(self, 'entry_step'):
            return 0
        return self.current_step - self.entry_step
    
    def _update_position_tracking(self, action: int):
        """
        Update position tracking variables.
        
        Args:
            action: Action taken (for tracking entry/exit)
        """
        if action in [1, 2] and self.position == 0:  # Entry action
            self.entry_step = self.current_step
        elif action in [0] and self.position != 0:  # Exit/HOLD action
            # Position will be closed by parent class
            pass


if __name__ == '__main__':
    """Test Phase 3 environment."""
    import pandas as pd
    
    print("Testing Phase 3 LLM Environment...")
    
    # Create test data with LLM features
    dates = pd.date_range('2024-01-01 09:30', periods=500, freq='1min', tz='America/New_York')
    test_data = pd.DataFrame({
        'open': np.random.uniform(4000, 4100, 500),
        'high': np.random.uniform(4100, 4150, 500),
        'low': np.random.uniform(3950, 4000, 500),
        'close': np.random.uniform(4000, 4100, 500),
        'volume': np.random.randint(100, 1000, 500),
        'atr': np.random.uniform(10, 30, 500),
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
        'trend_strength': np.random.choice([-1, 0, 1], 500),
        'support_20': np.random.uniform(3950, 4050, 500),
        'resistance_20': np.random.uniform(4050, 4150, 500),
        'volume_ratio_5min': np.random.uniform(0.5, 2.0, 500),
        'volume_ratio_20min': np.random.uniform(0.5, 2.0, 500),
        'price_change_60min': np.random.uniform(-0.01, 0.01, 500),
        'efficiency_ratio': np.random.uniform(0.2, 0.8, 500)
    }, index=dates)
    
    # Test environment with LLM features
    print("\n1. Testing with LLM features enabled...")
    env = TradingEnvironmentPhase3LLM(test_data, use_llm_features=True)
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Expected: (261,)")
    assert obs.shape == (261,), f"Expected (261,), got {obs.shape}"
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (261,), f"Step {i}: Expected (261,), got {obs.shape}"
    
    print("    LLM features test passed")
    
    # Test environment without LLM features (backward compatibility)
    print("\n2. Testing with LLM features disabled...")
    env_no_llm = TradingEnvironmentPhase3LLM(test_data, use_llm_features=False)
    obs, info = env_no_llm.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Expected: (228,)")
    assert obs.shape == (228,), f"Expected (228,), got {obs.shape}"
    
    print("    Backward compatibility test passed")
    
    # Test LLM context
    print("\n3. Testing LLM context generation...")
    context = env.get_llm_context()
    print(f"Context keys: {list(context.keys())}")
    assert 'market_name' in context
    assert 'current_price' in context
    assert 'position_status' in context
    print("    LLM context test passed")
    
    print("\n    All Phase 3 environment tests passed!")
