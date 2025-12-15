"""
Phase 1 Trading Environment - FIXED VERSION

CRITICAL FIXES:
1. Simplified reward function focusing ONLY on entry quality
2. Relaxed constraints to allow full episode learning
3. Removed portfolio management complexity
4. Focused on pattern recognition and entry timing

Changes from original:
- Uses simplified reward function (see environment_phase1_simplified.py)
- Relaxed trailing drawdown (much looser for Phase 1)
- Removed daily loss limit
- Removed profit target constraints
- Focuses purely on entry signal quality
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
from market_specs import MarketSpecification, ES_SPEC
from market_specs import MarketSpecification, ES_SPEC


def calculate_phase1_reward_simplified(env, exit_reason: str, trade_pnl: float, portfolio_value: float) -> float:
    """
    Simplified reward function for Phase 1 focusing on entry quality.

    Args:
        env: The trading environment instance
        exit_reason: Reason for position exit ('tp', 'sl', 'apex', 'eod', etc.)
        trade_pnl: P&L from the trade
        portfolio_value: Current portfolio value

    Returns:
        float: Reward value
    """
    reward = 0.0

    # If in position, small holding penalty to encourage action
    if env.position != 0:
        reward -= 0.01  # Small penalty for holding

    # If trade closed, reward based on P&L
    if exit_reason in ['tp', 'sl', 'eod']:
        if trade_pnl > 0:
            # Reward winning trades
            reward += trade_pnl / 100.0  # Scale down for stability
        else:
            # Penalty for losing trades
            reward += trade_pnl / 100.0  # Negative value

    # Heavy penalty for Apex violations (trailing drawdown)
    if exit_reason == 'apex':
        reward -= 10.0  # Strong penalty to avoid violations

    # Small reward for staying flat (encouraging selectivity)
    if env.position == 0 and exit_reason == '':
        reward += 0.001  # Tiny reward for patience

    return reward


class TradingEnvironmentPhase1(gym.Env):
    """
    Phase 1: Foundational Trading Patterns

    Focuses on learning quality entry signals with:
    - Simplified reward focusing on entry quality
    - Relaxed constraints for better learning
    - Removed unnecessary complexity
    """

    metadata = {"render_modes": []}

    # Action space constants
    ACTION_HOLD = 0
    ACTION_BUY = 1
    ACTION_SELL = 2

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 50000,
        window_size: int = 20,
        second_data: pd.DataFrame = None,
        # Market specifications
        market_spec: MarketSpecification = None,
        commission_override: float = None,
        # Phase 1 specific parameters (FIXED SL/TP)
        initial_sl_multiplier: float = 1.5,
        initial_tp_ratio: float = 3.0,
        position_size_contracts: float = 1.0,
        trailing_drawdown_limit: float = 15000,  # RELAXED for Phase 1 (was 5000)
        # NEW: Enable/disable constraints
        enable_daily_loss_limit: bool = False,  # DISABLED for Phase 1
        enable_profit_target: bool = False,     # DISABLED for Phase 1
        enable_4pm_rule: bool = True,           # Keep this for safety
        # Episode randomization parameters (FIXED BUG)
        start_index: Optional[int] = None,
        randomize_start_offsets: bool = True,
        min_episode_bars: int = 1500,
    ):
        """
        Initialize Phase 1 trading environment - FIXED.

        Args:
            data: OHLCV + indicators DataFrame (minute bars)
            initial_balance: Starting capital ($50,000 for Apex)
            window_size: Lookback window for observations (20 bars)
            second_data: Optional second-level data for precise drawdown
            market_spec: Market specification object (defaults to ES if None)
            commission_override: Override default commission (None = use market default)
            initial_sl_multiplier: SL distance in ATR (1.5 = 1.5× ATR) - FIXED
            initial_tp_ratio: TP as multiple of SL (3.0 = 3:1 R:R) - FIXED
            position_size_contracts: Contract quantity (minimum 1 contract)
            trailing_drawdown_limit: Max drawdown allowed (RELAXED to $15K)
            enable_daily_loss_limit: Enable daily loss limit (FALSE for Phase 1)
            enable_profit_target: Enable profit target (FALSE for Phase 1)
            enable_4pm_rule: Enable 4:59 PM close rule (TRUE for safety)
            start_index: Static episode start index (None = use randomization)
            randomize_start_offsets: Enable random episode start points (TRUE for training)
            min_episode_bars: Minimum bars remaining after episode start (1500 default)
        """
        super().__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.second_data = second_data

        # PHASE 1 CONSTRAINT: Fixed SL/TP (not adaptive)
        self.sl_atr_mult = initial_sl_multiplier
        self.tp_sl_ratio = initial_tp_ratio
        self.position_size = self._normalize_position_size(position_size_contracts)
        self.trailing_dd_limit = trailing_drawdown_limit
        
        # NEW: Constraint flags
        self.enable_daily_loss_limit = enable_daily_loss_limit
        self.enable_profit_target = enable_profit_target
        self.enable_4pm_rule = enable_4pm_rule

        # APEX PROFIT TARGET (RITHMIC rule) - DISABLED for Phase 1
        if self.enable_profit_target:
            self.profit_target = initial_balance + 3000  # $53,000 for $50K account
            self.profit_target_reached = False
            self.profit_target_reached_step = None
            self.trailing_stopped = False
        else:
            self.profit_target = None
            self.profit_target_reached = False
            self.profit_target_reached_step = None
            self.trailing_stopped = False

        # Market specifications (DYNAMIC - supports all futures markets)
        if market_spec is None:
            market_spec = ES_SPEC  # Default to ES for backward compatibility

        self.market_symbol = market_spec.symbol
        self.market_name = market_spec.name
        self.contract_size = market_spec.contract_multiplier
        self.tick_size = market_spec.tick_size
        self.tick_value = market_spec.tick_value

        # Commission (user can override, otherwise use market default)
        self.commission_per_side = commission_override if commission_override is not None else market_spec.commission

        # Slippage (market-dependent: liquid markets = 1 tick, less liquid = 2 ticks)
        self.slippage_points = market_spec.tick_size * market_spec.slippage_ticks

        # Gymnasium spaces
        self.action_space = spaces.Discrete(3)
        
        # Episode randomness
        self.randomize_start_offsets = randomize_start_offsets
        self.static_start_index = start_index
        self.min_episode_bars = max(min_episode_bars, window_size + 10)
        self._episode_start_index: Optional[int] = None

        # SIMPLIFIED observation space - reduced features
        # Previous: (window_size * 11 + 5,) = (225,)
        # New: (window_size * 11 + 5,) = (225,) - Kept core features, removed some complex ones
        # Actually using 11 features per timestep (8 market + 3 time)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * 11 + 5,),  # 225 dimensions total
            dtype=np.float32
        )

        # State tracking
        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        episode_start = self._determine_episode_start(seed)
        self._episode_start_index = episode_start
        self.current_step = max(self.window_size, min(episode_start, len(self.data) - 1))
        self.balance = self.initial_balance
        self.position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = 0
        self.sl_price = 0  # Fixed on entry
        self.tp_price = 0  # Fixed on entry

        # Tracking
        self.trade_history = []
        self.portfolio_values = [self.initial_balance]
        self.trailing_dd_levels = []  # Track trailing DD history for compliance
        self.num_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # APEX rules - RELAXED for Phase 1
        self.highest_balance = self.initial_balance
        self.trailing_dd_level = self.initial_balance - self.trailing_dd_limit

        # Reset profit target tracking (if enabled)
        if self.enable_profit_target:
            self.profit_target_reached = False
            self.profit_target_reached_step = None
            self.trailing_stopped = False

        # NEW: Apex compliance tracking
        self.apex_violations = []
        self.allow_new_trades = True
        self.daily_loss_limit = 1000  # $1,000 daily loss limit
        self.daily_pnl = 0
        self.current_date = None
        self.position_entry_step = 0
        self.max_hold_time = 390  # Maximum 1 day (390 minutes)
        self.violation_occurred = False  # NEW: Track violations

        # Action diversity tracking (fix sell bias)
        self.buy_count = 0
        self.sell_count = 0

        # DIAGNOSTIC: Track termination reason for analysis
        self.done_reason = None
        self.max_drawdown_reached = self.initial_balance

        info = {
            'episode_start_index': self._episode_start_index,
            'episode_start_timestamp': str(self.data.index[min(self.current_step, len(self.data) - 1)])
        }

        return self._get_observation(), info

    def _determine_episode_start(self, seed: Optional[int] = None) -> int:
        """Compute start index ensuring sufficient remaining bars."""
        min_start = self.window_size
        max_start = len(self.data) - max(self.min_episode_bars, 10)
        if max_start <= min_start:
            return min_start

        if self.randomize_start_offsets:
            rng = getattr(self, 'np_random', None)
            if rng is not None:
                return int(rng.integers(min_start, max_start + 1))
            generator = np.random.default_rng(seed)
            return int(generator.integers(min_start, max_start + 1))

        if self.static_start_index is not None:
            return max(min_start, min(self.static_start_index, max_start))

        return min_start

    def _normalize_position_size(self, raw_size: float) -> float:
        """Ensure futures contract counts remain integer and >= 1."""
        size = float(raw_size)
        if size < 1.0:
            raise ValueError("Position size must be at least 1 futures contract.")
        if not size.is_integer():
            raise ValueError("Fractional futures contracts are not supported.")
        return size

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation window with position-aware features.
        
        SIMPLIFIED: Removed complex regime features for Phase 1 focus.
        
        Returns:
            np.ndarray: Flattened shape (window_size*8 + 5,) = (165,)
                Market features (per timestep × 20 bars):
                    [close, volume, sma_5, sma_20, rsi, macd, momentum, atr]
                Position features (global):
                    [position, entry_price_ratio, sl_distance_atr, tp_distance_atr, time_in_position]
        """
        if self.current_step < self.window_size:
            # Return zero observation for early steps
            # 11 features per timestep (8 market + 3 time) × window + 5 position = 225 total
            return np.zeros((self.window_size * 11 + 5,), dtype=np.float32)

        start = self.current_step - self.window_size
        end = self.current_step

        # SIMPLIFIED: Core price features only (removed complex indicators)
        feature_cols = ['close', 'volume', 'sma_5', 'sma_20', 'rsi',
                       'macd', 'momentum', 'atr']
        obs = self.data[feature_cols].iloc[start:end].values

        # Time features (for RTH awareness)
        timestamps = self.data.index[start:end]
        time_features = np.zeros((len(timestamps), 3))

        for i, ts in enumerate(timestamps):
            hour_decimal = ts.hour + ts.minute / 60.0
            time_features[i, 0] = (hour_decimal - 9.5) / (16.98 - 9.5)

            market_open = ts.replace(hour=9, minute=30, second=0)
            min_open = (ts - market_open).total_seconds() / 60.0
            time_features[i, 1] = min_open / 449.0

            market_close = ts.replace(hour=16, minute=59, second=0)
            min_close = (market_close - ts).total_seconds() / 60.0
            time_features[i, 2] = min_close / 449.0

        # Combine market features (8 features per timestep)
        market_obs = np.concatenate([obs, time_features], axis=1)

        # Let VecNormalize handle normalization (don't double-normalize)
        market_obs_flat = market_obs.flatten()

        # Position-aware features (5 features)
        current_price = self.data['close'].iloc[self.current_step]
        if not np.isfinite(current_price) or current_price <= 0:
            # Fall back to last valid close to avoid NaNs propagating into the policy
            fallback_window = self.data['close'].iloc[max(self.current_step - self.window_size, 0):self.current_step]
            valid_prices = fallback_window.replace([np.inf, -np.inf], np.nan).dropna()
            current_price = float(valid_prices.iloc[-1]) if len(valid_prices) else 1.0

        current_atr = self.data['atr'].iloc[self.current_step]

        # Handle invalid ATR
        if current_atr <= 0 or np.isnan(current_atr):
            current_atr = current_price * 0.01  # Fallback to 1% of price

        if self.position != 0:
            # Agent has a position - provide detailed position information
            position_features = np.array([
                float(self.position),  # -1 (short), 0 (flat), +1 (long)
                self.entry_price / current_price,  # Entry price ratio
                abs(self.sl_price - current_price) / current_atr,  # SL distance in ATR
                abs(self.tp_price - current_price) / current_atr,  # TP distance in ATR
                (self.current_step - self.position_entry_step) / 390.0  # Time in position
            ], dtype=np.float32)
        else:
            # Agent is flat - zero out position-specific features
            position_features = np.array([
                0.0,  # No position
                1.0,  # Entry price ratio = 1 (neutral)
                0.0,  # No SL
                0.0,  # No TP
                0.0   # No time in position
            ], dtype=np.float32)

        # Combine market features + position features
        full_observation = np.concatenate([market_obs_flat, position_features])
        full_observation = np.nan_to_num(full_observation, nan=0.0, posinf=1e6, neginf=-1e6)

        return full_observation.astype(np.float32)

    def _check_second_level_drawdown(self, current_bar_time) -> Tuple[bool, float]:
        """
        Check if trailing drawdown was hit at second-level granularity.
        RELAXED for Phase 1 - less frequent checks.
        """
        if self.second_data is None or self.position == 0:
            return False, self.balance

        # Don't check second-level drawdown on entry step
        if self.position_entry_step == self.current_step:
            return False, self.balance

        # Only check every 5 minutes (not every minute) for Phase 1
        if self.current_step % 5 != 0:
            return False, self.balance

        # Get all second-level bars for this minute
        start_time = current_bar_time
        end_time = current_bar_time + pd.Timedelta(minutes=1)

        # Normalize timezones for comparison
        if start_time.tzinfo is None:
            start_time = start_time.tz_localize('UTC').tz_convert('America/New_York')
            end_time = end_time.tz_localize('UTC').tz_convert('America/New_York')
        else:
            start_time = start_time.tz_convert('America/New_York')
            end_time = end_time.tz_convert('America/New_York')

        # Filter second data for this minute bar
        try:
            mask = (self.second_data.index >= start_time) & (self.second_data.index < end_time)
            second_bars = self.second_data[mask]
        except Exception as e:
            return False, self.balance

        if len(second_bars) == 0:
            return False, self.balance

        # Track minimum equity reached during this minute bar
        min_equity = float('inf')

        for _, bar in second_bars.iterrows():
            high = bar['high']
            low = bar['low']

            # Calculate unrealized PnL at second-level precision
            if self.position == 1:  # Long
                unrealized_pnl = (low - self.entry_price) * self.contract_size * self.position_size
            else:  # Short
                unrealized_pnl = (self.entry_price - high) * self.contract_size * self.position_size

            current_equity = self.balance + unrealized_pnl

            # Check if drawdown violated at this second
            if current_equity < self.trailing_dd_level:
                return True, current_equity

            # Track minimum equity
            if current_equity < min_equity:
                min_equity = current_equity

        return False, min_equity

    def _calculate_fixed_sl_tp(self, entry_price: float, position_type: int) -> Tuple[float, float]:
        """Calculate FIXED stop loss and take profit."""
        atr = self.data['atr'].iloc[self.current_step]
        if atr <= 0 or np.isnan(atr):
            atr = entry_price * 0.01  # Fallback to 1% of price

        sl_distance = atr * self.sl_atr_mult
        tp_distance = sl_distance * self.tp_sl_ratio

        if position_type == 1:  # Long
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:  # Short
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance

        return sl, tp

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0.0, False, True, {}

        # Current market state
        current_price = self.data['close'].iloc[self.current_step]
        high = self.data['high'].iloc[self.current_step]
        low = self.data['low'].iloc[self.current_step]

        reward = 0.0
        terminated = False
        truncated = False
        position_changed = False
        trade_pnl = 0.0
        exit_reason = ""  # CRITICAL FIX: Empty string (not None) for patience reward
        self.violation_occurred = False  # Reset violation flag

        # 1. CHECK SL/TP ON EXISTING POSITION
        if self.position != 0:
            sl_hit, tp_hit, exit_price = self._check_sl_tp_hit(high, low)

            if sl_hit or tp_hit:
                # Close position at SL or TP
                if self.position == 1:
                    trade_pnl = (exit_price - self.entry_price) * self.contract_size * self.position_size
                else:
                    trade_pnl = (self.entry_price - exit_price) * self.contract_size * self.position_size

                # Deduct commissions
                trade_pnl -= (self.commission_per_side * 2 * self.position_size)

                self.balance += trade_pnl
                exit_reason = "take_profit" if tp_hit else "stop_loss"

                if trade_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

                self.trade_history.append({
                    'entry_price': self.entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl': trade_pnl,
                    'position_type': 'long' if self.position == 1 else 'short',
                    'step': self.current_step
                })

                # Reset position
                self.position = 0
                self.entry_price = 0
                self.sl_price = 0
                self.tp_price = 0
                position_changed = True

        # 2. CHECK APEX TIME RULES (RELAXED for Phase 1)
        current_time = self.data.index[self.current_step]
        if self._check_apex_time_rules(current_time):
            terminated = True
            self.violation_occurred = True
            self.done_reason = "apex_time_violation"  # DIAGNOSTIC
            reward = -0.05  # Reduced penalty for Phase 1
            # return self._get_observation(), reward, terminated, False, {
            #     'portfolio_value': self.balance,
            #     'position': self.position,
            #     'apex_violation': True,
            #     'done_reason': self.done_reason  # DIAGNOSTIC
            # }
        
        # 3. EXECUTE NEW ACTION (if flat)
        # RTH gating: allow new entries only between 09:30 and 16:59 ET
        if current_time.tzinfo is None:
            ts = current_time.tz_localize('UTC').tz_convert('America/New_York')
        else:
            ts = current_time.tz_convert('America/New_York')

        rth_open = ts.replace(hour=9, minute=30, second=0)
        rth_close = ts.replace(hour=16, minute=59, second=0)
        allowed_to_open = (ts >= rth_open) and (ts <= rth_close) and self.allow_new_trades

        if self.position == 0 and allowed_to_open:
            if action == self.ACTION_BUY:
                # Open long
                self.position = 1
                self.entry_price = current_price + self.slippage_points
                self.sl_price, self.tp_price = self._calculate_fixed_sl_tp(
                    self.entry_price, 1
                )

                # Deduct entry commission
                self.balance -= (self.commission_per_side * self.position_size)
                self.num_trades += 1
                self.buy_count += 1
                self.position_entry_step = self.current_step
                position_changed = True

            elif action == self.ACTION_SELL:
                # Open short
                self.position = -1
                self.entry_price = current_price - self.slippage_points
                self.sl_price, self.tp_price = self._calculate_fixed_sl_tp(
                    self.entry_price, -1
                )

                # Deduct entry commission
                self.balance -= (self.commission_per_side * self.position_size)
                self.num_trades += 1
                self.sell_count += 1
                self.position_entry_step = self.current_step
                position_changed = True

        # 4. UPDATE PORTFOLIO VALUE & TRAILING DRAWDOWN (RELAXED)
        unrealized_pnl = 0.0
        if self.position != 0:
            if self.position == 1:
                unrealized_pnl = (current_price - self.entry_price) * self.contract_size * self.position_size
            else:
                unrealized_pnl = (self.entry_price - current_price) * self.contract_size * self.position_size

        portfolio_value = self.balance + unrealized_pnl
        self.portfolio_values.append(portfolio_value)

        # Profit target logic (if enabled)
        if self.enable_profit_target and portfolio_value >= self.profit_target and not self.profit_target_reached:
            self.profit_target_reached = True
            self.profit_target_reached_step = self.current_step
            self.trailing_stopped = True

        # Update trailing drawdown (RELAXED - much looser)
        if portfolio_value > self.highest_balance:
            self.highest_balance = portfolio_value
            
            if not self.trailing_stopped:
                # Normal trailing behavior (before profit target)
                self.trailing_dd_level = self.highest_balance - self.trailing_dd_limit
            # else: trailing_dd_level remains frozen at profit target level

        # Track trailing DD level history
        self.trailing_dd_levels.append(self.trailing_dd_level)

        # Check violation at minute-level (RELAXED - less frequent)
        if self.current_step % 10 == 0 and portfolio_value < self.trailing_dd_level:
            terminated = True
            self.violation_occurred = True
            self.done_reason = "trailing_drawdown_minute"  # DIAGNOSTIC
            self.max_drawdown_reached = self.highest_balance - portfolio_value  # DIAGNOSTIC
            reward = -0.05  # Reduced penalty

        # Check violation at second-level (RELAXED - less frequent)
        if not terminated and self.second_data is not None and self.current_step % 5 == 0:
            current_bar_time = self.data.index[self.current_step]
            drawdown_hit, _ = self._check_second_level_drawdown(current_bar_time)
            if drawdown_hit:
                terminated = True
                self.violation_occurred = True
                self.done_reason = "trailing_drawdown_second"  # DIAGNOSTIC
                self.max_drawdown_reached = self.highest_balance - portfolio_value  # DIAGNOSTIC
                reward = -0.05  # Reduced penalty

        # Daily loss limit (DISABLED for Phase 1)
        if self.enable_daily_loss_limit and trade_pnl != 0:
            self._update_daily_pnl(trade_pnl)

            if self._check_daily_loss_limit():
                terminated = True
                self.violation_occurred = True
                self.done_reason = "daily_loss_limit"  # DIAGNOSTIC
                reward = -0.05  # Reduced penalty
                return self._get_observation(), reward, terminated, False, {
                    'portfolio_value': self.balance,
                    'position': self.position,
                    'daily_loss_violation': True,
                    'done_reason': self.done_reason  # DIAGNOSTIC
                }
        
        # Track position hold time
        if self.position != 0:
            if self.position_entry_step == 0:
                self.position_entry_step = self.current_step
        else:
            self.position_entry_step = 0

        # 5. CALCULATE REWARD (SIMPLIFIED for Phase 1)
        if not terminated:
            # Use simplified reward function focusing on entry quality
            reward = calculate_phase1_reward_simplified(
                self, exit_reason, trade_pnl, portfolio_value
            )

        # 6. ADVANCE TIME
        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            truncated = True
            if self.done_reason is None:  # DIAGNOSTIC
                self.done_reason = "end_of_data"

        obs = self._get_observation()

        # DIAGNOSTIC: Enhanced info dict with compliance details
        info = {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'num_trades': self.num_trades,
            'balance': self.balance,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / max(self.num_trades, 1),
            'violation_occurred': self.violation_occurred,
            'done_reason': self.done_reason,  # DIAGNOSTIC
            'max_drawdown': self.max_drawdown_reached,  # DIAGNOSTIC
            'episode_bars': self.current_step - self._episode_start_index,  # DIAGNOSTIC
            'trailing_dd_level': self.trailing_dd_level,  # DIAGNOSTIC
        }
        self.current_step += 1

        return obs, reward, terminated, truncated, info

    def _check_apex_time_rules(self, current_time) -> bool:
        """
        Enforce Apex trading time rules - RELAXED for Phase 1.
        Only critical violations terminate.
        """
        if not self.enable_4pm_rule:
            return False  # Rule disabled
            
        # Convert to America/New_York if needed
        if current_time.tzinfo is None:
            ts = current_time.tz_localize('UTC').tz_convert('America/New_York')
        else:
            ts = current_time.tz_convert('America/New_York')

        # Only terminate if still holding at 4:59 PM (critical violation)
        if ts.hour == 16 and ts.minute >= 59:
            if self.position != 0:
                # VIOLATION: Still holding position at 4:59 PM
                self._force_close_position(reason="VIOLATION: 4:59 PM AUTO-CLOSE")
                self._add_apex_violation("Late position closure - held past 4:59 PM ET")
                self._cancel_all_pending_orders()
                return True  # terminated
            else:
                # COMPLIANT: Agent already closed position
                self._cancel_all_pending_orders()
                return False  # not terminated (compliant)

        # No new trades after 4:00 PM (give agent time to close)
        if ts.hour >= 16:
            self.allow_new_trades = False

        return False  # not terminated

    def _check_sl_tp_hit(self, high: float, low: float) -> Tuple[bool, bool, float]:
        """Check if stop loss or take profit was hit."""
        if self.position == 0:
            return False, False, 0.0

        sl_hit = False
        tp_hit = False
        exit_price = 0.0

        if self.position == 1:  # Long
            # Check if low went below SL (stop hit)
            if low <= self.sl_price:
                sl_hit = True
                exit_price = self.sl_price
            # Check if high went above TP (target hit)
            elif high >= self.tp_price:
                tp_hit = True
                exit_price = self.tp_price
            else:
                # Neither hit - stay in position
                return False, False, 0.0
        else:  # Short
            # Check if high went above SL (stop hit)
            if high >= self.sl_price:
                sl_hit = True
                exit_price = self.sl_price
            # Check if low went below TP (target hit)
            elif low <= self.tp_price:
                tp_hit = True
                exit_price = self.tp_price
            else:
                # Neither hit - stay in position
                return False, False, 0.0

        return sl_hit, tp_hit, exit_price

    def _force_close_position(self, reason="Auto-close"):
        """Force close position immediately at market price."""
        if self.position == 0:
            return
        
        current_price = self.data['close'].iloc[self.current_step]
        
        if self.position == 1:  # Long
            exit_price = current_price - self.slippage_points
            trade_pnl = (exit_price - self.entry_price) * self.contract_size * self.position_size
        else:  # Short
            exit_price = current_price + self.slippage_points
            trade_pnl = (self.entry_price - exit_price) * self.contract_size * self.position_size
        
        # Deduct exit commission
        trade_pnl -= (self.commission_per_side * self.position_size)
        self.balance += trade_pnl
        
        # Record trade
        self.trade_history.append({
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'exit_reason': reason,
            'pnl': trade_pnl,
            'position_type': 'long' if self.position == 1 else 'short',
            'step': self.current_step,
            'forced': True
        })
        
        # Update trade counts
        if trade_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Reset position
        self.position = 0
        self.entry_price = 0
        self.sl_price = 0
        self.tp_price = 0

    def _add_apex_violation(self, violation_type, description=""):
        """Record Apex rule violation."""
        violation = {
            'step': self.current_step,
            'timestamp': self.data.index[self.current_step],
            'type': violation_type,
            'description': description,
            'portfolio_value': self.balance + self._calculate_unrealized_pnl()
        }
        self.apex_violations.append(violation)

    def _cancel_all_pending_orders(self):
        """Cancel all pending orders (placeholder)."""
        pass

    def _calculate_unrealized_pnl(self):
        """Calculate current unrealized P&L."""
        if self.position == 0:
            return 0.0
        
        current_price = self.data['close'].iloc[self.current_step]
        
        if self.position == 1:  # Long
            return (current_price - self.entry_price) * self.contract_size * self.position_size
        else:  # Short
            return (self.entry_price - current_price) * self.contract_size * self.position_size

    def _update_daily_pnl(self, trade_pnl: float):
        """Update daily P&L tracking."""
        current_time = self.data.index[self.current_step]
        
        # Reset daily PnL on new day
        if self.current_date is None or self.current_date != current_time.date():
            self.daily_pnl = 0
            self.current_date = current_time.date()
        
        self.daily_pnl += trade_pnl

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit exceeded."""
        if not self.enable_daily_loss_limit:
            return False  # Disabled

        return self.daily_pnl < -self.daily_loss_limit

    def action_masks(self) -> np.ndarray:
        """
        Return mask of valid actions for MaskablePPO.

        Phase 1 Policy:
        - When FLAT (position=0): Can HOLD, BUY, or SELL
        - When IN POSITION (position!=0): Can ONLY HOLD

        This forces the agent to wait until position closes (TP/SL)
        before taking any new action.

        Returns:
            np.ndarray: Boolean mask [HOLD, BUY, SELL]
                       True = action allowed, False = action blocked
        """
        if self.position == 0:
            # Flat: All actions valid
            return np.array([True, True, True], dtype=np.bool_)
        else:
            # In position: Only HOLD valid
            return np.array([True, False, False], dtype=np.bool_)

    def get_action_mask(self) -> np.ndarray:
        """
        Gymnasium-friendly accessor for the current action mask.
        Returns a copy so downstream wrappers can't mutate internal state.
        """
        mask = self.action_masks()
        return mask.copy() if isinstance(mask, np.ndarray) else np.array(mask, dtype=bool)
