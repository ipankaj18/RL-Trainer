"""
Phase 2: Advanced Position Management Environment

Goal: Learn HOW to dynamically manage risk and optimize positions
- Actions: 8 discrete (Hold, Buy, Sell, Close, Tighten SL, Move to BE, Extend TP, Toggle Trail)
- SL/TP: DYNAMIC (learned & adaptive)
- Focus: Risk management, position optimization, Sharpe ratio
- Reward: Risk-adjusted returns, max drawdown minimization, consistency

Based on: OpenAI Spinning Up PPO + Transfer Learning + Advanced RL
"""

import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
from environment_phase1 import TradingEnvironmentPhase1
from market_specs import MarketSpecification


class TradingEnvironmentPhase2(TradingEnvironmentPhase1):
    """
    Phase 2: Advanced Position Management

    RL FIX #10: Simplified action space from 9 to 6 actions

    Observation Space: (window_size*11 + 5,) - Market features + position state
    Action Space: Discrete(6) - Streamlined position management
    Reward: Risk-adjusted returns, Sharpe ratio, consistency
    """

    # Simplified action space
    # RL FIX #10: Reduced from 9 to 6 actions for better sample efficiency
    ACTION_HOLD = 0
    ACTION_BUY = 1
    ACTION_SELL = 2
    ACTION_MOVE_SL_TO_BE = 3  # Was ACTION 5
    ACTION_ENABLE_TRAIL = 4   # Was ACTION 7
    ACTION_DISABLE_TRAIL = 5  # Was ACTION 8

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 50000,
        window_size: int = 20,
        second_data: pd.DataFrame = None,
        # Market specifications
        market_spec: MarketSpecification = None,
        commission_override: float = None,
        # Phase 2 specific
        initial_sl_multiplier: float = 1.5,
        initial_tp_ratio: float = 3.0,
        position_size_contracts: float = 1.0,  # Full size in Phase 2
        trailing_drawdown_limit: float = 2500,  # Strict Apex rules
        tighten_sl_step: float = 0.5,  # How much to tighten SL (in ATR)
        extend_tp_step: float = 1.0,   # How much to extend TP (in ATR)
        trailing_activation_profit: float = 1.0,  # Activate trail after 1R profit
        start_index: Optional[int] = None,
        randomize_start_offsets: bool = True,
        min_episode_bars: int = 1500,
    ):
        """
        Initialize Phase 2 environment with position management.

        Args:
            data: OHLCV + indicators DataFrame
            initial_balance: Starting capital
            window_size: Lookback window
            second_data: Optional second-level data
            market_spec: Market specification object (defaults to ES if None)
            commission_override: Override default commission
            initial_sl_multiplier: Initial SL distance (can be adjusted)
            initial_tp_ratio: Initial TP ratio (can be adjusted)
            position_size_contracts: Full position size (1.0 for Apex)
            trailing_drawdown_limit: Strict Apex limit ($2,500)
            tighten_sl_step: Amount to tighten SL by (in ATR units)
            extend_tp_step: Amount to extend TP by (in ATR units)
            trailing_activation_profit: Min profit (in R) to enable trailing
        """
        # Initialize before parent reset so _determine_episode_start can reference it
        self._rth_start_indices: list = []

        # PERFORMANCE FIX: Cache timezone conversions (major bottleneck with SubprocVecEnv)
        # CRITICAL: Must initialize BEFORE super().__init__() because parent's init calls reset()
        self._timezone_cache = {}
        self._rth_cache = {}

        # Initialize parent class (Phase 1) with market specs
        super().__init__(
            data, initial_balance, window_size, second_data,
            market_spec, commission_override,  # Pass market specs
            initial_sl_multiplier, initial_tp_ratio, position_size_contracts,
            trailing_drawdown_limit,
            start_index=start_index,
            randomize_start_offsets=randomize_start_offsets,
            min_episode_bars=min_episode_bars
        )

        # Phase 2: Simplified action space (3 -> 6 actions)
        # RL FIX #10: Reduced from 9 to 6 for improved sample efficiency
        self.action_space = spaces.Discrete(6)

        # ACTION MASKING FIX: Override observation space to include validity features
        # Original: (window_size * 11 + 5,) = (225,)
        # Updated: (window_size * 11 + 5 + 3,) = (228,)
        # Added features: [can_enter_trade, can_manage_position, has_position]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * 11 + 8,),  # 220 market + 5 position + 3 validity = 228
            dtype=np.float32
        )

        # Phase 2 position management parameters
        self.tighten_step_atr = tighten_sl_step
        self.extend_step_atr = extend_tp_step
        self.trail_activation_r = trailing_activation_profit

        # NEW: Dynamic position sizing parameters
        self.max_position_size = int(self.position_size)
        self.position_scaling_enabled = True
        self.volatility_adjustment = True

        # Position management state
        self.trailing_stop_active = False
        self.highest_profit_point = 0
        self.be_move_count = 0

        # Compute RTH start indices (cache already initialized above)
        self._rth_start_indices = self._compute_rth_start_indices()

        # Action diversity tracking (fix sell bias)
        self.buy_count = 0
        self.sell_count = 0
        self.action_diversity_window = 100  # Track last 100 actions

        # Diagnostics
        self.last_action_mask = None
        self.last_done_reason = None

        # Phase 2: Disable profit target (not used in Phase 2)
        self.enable_profit_target = False
        self.profit_target = None
        self.profit_target_reached = False
        self.profit_target_reached_step = None

        # Track realized equity high-water mark to avoid trailing DD spikes from unrealized PnL
        self.realized_high_balance = self.initial_balance


    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset with Phase 2 tracking."""
        obs, info = super().reset(seed=seed, options=options)

        # Reset position management state
        self.trailing_stop_active = False
        self.highest_profit_point = 0
        self.be_move_count = 0

        # Action diversity tracking (fix sell bias)
        self.buy_count = 0
        self.sell_count = 0
        self.action_diversity_window = 100  # Track last 100 actions

        # Diagnostics
        self.last_action_mask = None
        self.last_done_reason = None

        # Reset realized equity high-water
        self.realized_high_balance = self.initial_balance

        # Recompute RTH start candidates in case data length changes between resets
        self._rth_start_indices = self._compute_rth_start_indices()

        if info is None:
            info = {}

        info.update({
            'data_length': len(self.data),
            'window_size': self.window_size,
            'market_symbol': getattr(self, 'market_symbol', None),
        })

        # Prime initial action mask so wrappers receive a valid mask immediately
        self.action_masks()
        self._attach_action_mask(info)

        # Phase 2 specific position management parameters are already set in __init__
        # No need to re-assign them here

        return obs, info

    def _get_et_time(self, timestamp):
        """
        Convert timestamp to Eastern Time with caching.

        PERFORMANCE FIX: Timezone conversions are expensive (~1-2ms each).
        With 64 environments and frequent calls, this adds up to 100+ FPS impact.
        Caching reduces per-step overhead by 60-80%.
        """
        # Use timestamp as cache key (works with pd.Timestamp)
        cache_key = timestamp

        if cache_key in self._timezone_cache:
            return self._timezone_cache[cache_key]

        # Perform conversion
        if timestamp.tzinfo is None:
            ts = timestamp.tz_localize('UTC').tz_convert('America/New_York')
        else:
            ts = timestamp.tz_convert('America/New_York')

        # Cache result
        self._timezone_cache[cache_key] = ts
        return ts

    def _is_rth_bar(self, timestamp) -> bool:
        """
        Return True if timestamp falls inside regular trading hours (09:30-16:59 ET).

        PERFORMANCE FIX: Uses cached timezone conversion.
        """
        # Check cache first
        if timestamp in self._rth_cache:
            return self._rth_cache[timestamp]

        # Convert using cached method
        ts = self._get_et_time(timestamp)

        rth_open = ts.replace(hour=9, minute=30, second=0, microsecond=0)
        rth_close = ts.replace(hour=16, minute=59, second=0, microsecond=0)
        result = rth_open <= ts <= rth_close

        # Cache the result
        self._rth_cache[timestamp] = result
        return result

    def _compute_rth_start_indices(self) -> list:
        """
        Pre-compute eligible start indices that land inside RTH.

        This avoids episodes that begin pre-market or after hours where the action
        mask will block BUY/SELL for dozens of steps, wasting experience and
        making action masking look broken.
        """
        max_start = len(self.data) - max(self.min_episode_bars, 10)
        if max_start <= self.window_size:
            return []

        indices = []
        for idx, ts in enumerate(self.data.index):
            if idx < self.window_size or idx > max_start:
                continue
            try:
                if self._is_rth_bar(ts):
                    indices.append(idx)
            except Exception:
                # If we cannot parse the timestamp, skip and let base logic handle it
                continue
        return indices

    def _determine_episode_start(self, seed: Optional[int] = None) -> int:
        """
        Choose an RTH-aligned start index so BUY/SELL are valid immediately.
        Falls back to parent behavior if no RTH bars meet the length constraints.
        """
        if self._rth_start_indices:
            # Respect static start requests by snapping to the next available RTH bar
            if not self.randomize_start_offsets and self.static_start_index is not None:
                for idx in self._rth_start_indices:
                    if idx >= self.static_start_index:
                        return idx
                return self._rth_start_indices[-1]

            rng = getattr(self, 'np_random', None)
            if rng is not None:
                return int(rng.choice(self._rth_start_indices))
            generator = np.random.default_rng(seed)
            return int(generator.choice(self._rth_start_indices))

        # Fallback: use parent sampling if no valid RTH starts found
        return super()._determine_episode_start(seed)

    def _get_observation(self) -> np.ndarray:
        """
        Override parent's _get_observation() to add 3 explicit validity features.

        ACTION MASKING FIX: Added explicit validity features to help model learn
        when actions are valid. This makes the action masking rules obvious to the network.

        Returns:
            np.ndarray: Shape (228,) = 220 market + 5 position + 3 validity
                Market features (220): Same as Phase 1
                Position features (5): [position, entry_price_ratio, sl_distance_atr, tp_distance_atr, time_in_position]
                Validity features (3): [can_enter_trade, can_manage_position, has_position]
        """
        # Get parent's observation (225 features)
        parent_obs = super()._get_observation()

        # Calculate validity features
        # PERFORMANCE FIX: Use cached timezone conversion
        current_time = self.data.index[self.current_step]
        ts = self._get_et_time(current_time)

        rth_open = ts.replace(hour=9, minute=30, second=0)
        rth_close = ts.replace(hour=16, minute=59, second=0)
        in_rth = (ts >= rth_open) and (ts <= rth_close) and self.allow_new_trades

        # Validity features make action masking rules explicit to the network
        can_enter_trade = float(self.position == 0 and in_rth)  # Can use BUY/SELL
        can_manage_position = float(self.position != 0)         # Can use PM actions
        has_position = float(self.position != 0)                 # Has open position

        validity_features = np.array([
            can_enter_trade,
            can_manage_position,
            has_position
        ], dtype=np.float32)

        # Combine parent observation + validity features
        full_observation = np.concatenate([parent_obs, validity_features])

        return full_observation.astype(np.float32)

    def _calculate_apex_reward(self, position_changed, exit_reason, trade_pnl, portfolio_value, pm_action_taken):
        """
        Calculate Apex-optimized reward with outcome-based PM feedback.
        
        REWARD FIX v2:
        1. Removed free points for PM actions (prevent reward hacking)
        2. Stronger win/loss signals (2.0/-1.0)
        3. Much stronger portfolio value signal (20x on percentage)
        4. Outcome-based PM action feedback (reward good timing, penalize bad)
        5. Bonus for successful trailing stop usage
        """
        reward = 0.0
        
        # Position management outcome-based rewards
        # REWARD FIX: Provide feedback on PM action quality, not just execution
        if pm_action_taken:
            if 'move_to_be' in pm_action_taken:
                # Neutral - outcome will show in trade_pnl
                # If position exits at BE shortly after, that's captured in trade result
                reward += 0.0
            
            elif 'enable_trail' in pm_action_taken or 'trail_enabled' in pm_action_taken:
                # Check if trailing was enabled at a good time
                # Good: Position has significant profit (>1% of balance)
                # Bad: Position barely profitable, trail likely to cut it short
                unrealized_pnl = portfolio_value - self.balance
                profit_threshold = self.initial_balance * 0.01  # 1% of initial balance
                
                if unrealized_pnl > profit_threshold:
                    reward += 0.1  # Good timing - enough profit to trail
                else:
                    reward -= 0.1  # Too early - likely to cut profits short
            
            elif 'trail_disabled' in pm_action_taken or 'disable_trail' in pm_action_taken:
                # Disabling trail is usually to let profits run
                # Neutral for now, outcome will show in trade_pnl
                reward += 0.0
        
        # Trade completion reward
        # REWARD FIX: Stronger signals for wins/losses
        if exit_reason:
            if trade_pnl > 0:
                reward += 2.0  # Strong win signal
                
                # BONUS: If trailing was used and trade won, extra reward
                # This encourages learning to use trailing effectively
                if self.trailing_stop_active or (pm_action_taken and 'trail' in pm_action_taken):
                    reward += 0.5  # Bonus for successful trailing usage
            else:
                reward -= 1.0  # Strong loss penalty
        
        # Portfolio value reward (shaped and MUCH stronger)
        # REWARD FIX: Use percentage-based scaling for meaningful continuous feedback
        # 2% gain = +0.4 reward (competitive with discrete signals)
        pnl_pct = (portfolio_value - self.initial_balance) / self.initial_balance
        reward += pnl_pct * 20.0  # Was /1000.0 - now 20x stronger
        
        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action with Phase 2 position management.

        RL FIX #10: Simplified action space (9 -> 6) for better sample efficiency

        Actions:
            0: Hold
            1: Buy (open long)
            2: Sell (open short)
            3: Move SL to break-even (was 5)
            4: Enable trailing stop (was 7)
            5: Disable trailing stop (was 8)
        """
        if self.current_step >= len(self.data) - 1:
            obs = self._get_observation()
            info = {
                'portfolio_value': self.balance,
                'position': self.position,
                'num_trades': self.num_trades,
                'balance': self.balance,
                'done_reason': 'data_exhausted',
                'observation_quality': self._observation_quality(obs),
                'step_index': self.current_step,
                'remaining_bars': 0,
            }
            self._attach_action_mask(info)
            self.last_done_reason = 'data_exhausted'
            return obs, 0.0, False, True, info

        current_price = self.data['close'].iloc[self.current_step]
        high = self.data['high'].iloc[self.current_step]
        low = self.data['low'].iloc[self.current_step]
        atr = self.data['atr'].iloc[self.current_step]

        reward = 0.0
        terminated = False
        truncated = False
        done_reason = None
        position_changed = False
        trade_pnl = 0.0
        exit_reason = None
        pm_action_taken = None  # Position management action

        # ============================================================
        # 0. EARLY INVALID ACTION CHECK (ACTION MASKING FIX)
        # ============================================================
        # Check if action is invalid BEFORE processing
        # This provides strong immediate feedback and prevents wasted computation
        action_mask = self.action_masks()
        if not action_mask[action]:
            # Invalid action detected - apply penalty and return immediately
            # CRITICAL: Advance time even for invalid actions (prevents exploitation)
            self.current_step += 1

            reward = -1.0  # REWARD FIX: Increased from -0.1 to strongly discourage invalid actions
            obs = self._get_observation()

            # Check if episode ended after time advance
            truncated = self.current_step >= len(self.data) - 1

            info = {
                'portfolio_value': self.balance,
                'position': self.position,
                'invalid_action': True,
                'action': action,
                'reason': f"Action {action} invalid for position={self.position}",
                'balance': self.balance,
                'step_index': self.current_step,
            }
            self._attach_action_mask(info)
            # DO NOT execute invalid action - return current state (but time advanced)
            return obs, reward, False, truncated, info

        # ============================================================
        # 1. UPDATE TRAILING STOP (if active)
        # ============================================================
        if self.trailing_stop_active and self.position != 0:
            self._update_trailing_stop(current_price, atr)

        # ============================================================
        # 2. CHECK SL/TP (now potentially dynamic)
        # ============================================================
        if self.position != 0:
            sl_hit, tp_hit, exit_price = self._check_sl_tp_hit(high, low)

            if sl_hit or tp_hit:
                # Close position
                if self.position == 1:
                    trade_pnl = (exit_price - self.entry_price) * self.contract_size * self.position_size
                else:
                    trade_pnl = (self.entry_price - exit_price) * self.contract_size * self.position_size

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
                    'be_moves': self.be_move_count,
                    'trailing_used': self.trailing_stop_active,
                    'step': self.current_step
                })

                # Reset
                self._reset_position_state()
                position_changed = True

        # ============================================================
        # 3. EXECUTE ACTION
        # ============================================================
        current_time = self.data.index[self.current_step]

        # RTH gating
        if current_time.tzinfo is None:
            ts = current_time.tz_localize('UTC').tz_convert('America/New_York')
        else:
            ts = current_time.tz_convert('America/New_York')

        rth_open = ts.replace(hour=9, minute=30, second=0)
        rth_close = ts.replace(hour=16, minute=59, second=0)
        allowed_to_open = (ts >= rth_open) and (ts <= rth_close)

        # PHASE 1 ACTIONS (Hold, Buy, Sell)
        if action == self.ACTION_BUY and self.position == 0 and allowed_to_open:
            # Open long with dynamic position sizing
            self.position = 1
            self.entry_price = current_price + self.slippage_points
            self.sl_price, self.tp_price = self._calculate_fixed_sl_tp(self.entry_price, 1)

            # NEW: Calculate dynamic position size based on volatility
            self.position_size = self._calculate_position_size()

            self.balance -= (self.commission_per_side * self.position_size)
            self.num_trades += 1
            self.buy_count += 1  # Track for diversity
            position_changed = True

        elif action == self.ACTION_SELL and self.position == 0 and allowed_to_open:
            # Open short with dynamic position sizing
            self.position = -1
            self.entry_price = current_price - self.slippage_points
            self.sl_price, self.tp_price = self._calculate_fixed_sl_tp(self.entry_price, -1)

            # NEW: Calculate dynamic position size based on volatility
            self.position_size = self._calculate_position_size()

            self.balance -= (self.commission_per_side * self.position_size)
            self.num_trades += 1
            self.sell_count += 1  # Track for diversity
            # ❌ REMOVED BUG: self.buy_count += 1  # This was a copy-paste error
            position_changed = True

        # PHASE 2 ACTIONS (Position Management)
        elif action == self.ACTION_MOVE_SL_TO_BE and self.position != 0:
            is_valid, reason = self._validate_position_management_action(
                action, current_price, atr
            )
            if is_valid:
                success = self._move_sl_to_breakeven(current_price)
                if success:
                    pm_action_taken = "move_to_be"
                    self.be_move_count += 1
            else:
                # TUNED: Reduced penalty from -1.0 to -0.1 for gentler learning
                reward -= 0.1
                pm_action_taken = f"invalid_move_to_be: {reason}"

        elif action == self.ACTION_ENABLE_TRAIL and self.position != 0:
            is_valid, reason = self._validate_position_management_action(
                action, current_price, atr
            )
            if is_valid:
                if not self.trailing_stop_active:
                    self.trailing_stop_active = True
                    pm_action_taken = "trail_enabled"
                else:
                    # Already enabled - no-op but not an error
                    pm_action_taken = "trail_already_on"
            else:
                # TUNED: Reduced penalty from -1.0 to -0.1 for gentler learning
                reward -= 0.1
                pm_action_taken = f"invalid_enable_trail: {reason}"

        elif action == self.ACTION_DISABLE_TRAIL and self.position != 0:
            is_valid, reason = self._validate_position_management_action(
                action, current_price, atr
            )
            if is_valid:
                if self.trailing_stop_active:
                    self.trailing_stop_active = False
                    pm_action_taken = "trail_disabled"
                else:
                    # Already disabled - no-op but not an error
                    pm_action_taken = "trail_already_off"
            else:
                # TUNED: Reduced penalty from -1.0 to -0.1 for gentler learning
                reward -= 0.1
                pm_action_taken = f"invalid_disable_trail: {reason}"

        # ============================================================
        # 4. UPDATE PORTFOLIO & DRAWDOWN
        # ============================================================
        unrealized_pnl = 0.0
        if self.position != 0:
            if self.position == 1:
                unrealized_pnl = (current_price - self.entry_price) * self.contract_size * self.position_size
            else:
                unrealized_pnl = (self.entry_price - current_price) * self.contract_size * self.position_size

            # Track highest profit for trailing
            if unrealized_pnl > self.highest_profit_point:
                self.highest_profit_point = unrealized_pnl

        portfolio_value = self.balance + unrealized_pnl
        self.portfolio_values.append(portfolio_value)

        # APEX RITHMIC RULE: Stop trailing when profit target reached
        if self.profit_target is not None and portfolio_value >= self.profit_target and not self.profit_target_reached:
            self.profit_target_reached = True
            self.profit_target_reached_step = self.current_step
            self.trailing_stopped = True

        # Update trailing drawdown (Apex rules)
        if portfolio_value > self.highest_balance:
            self.highest_balance = portfolio_value

            if not self.trailing_stopped:
                # Normal trailing behavior (before profit target)
                self.trailing_dd_level = self.highest_balance - self.trailing_dd_limit
            # else: trailing_dd_level remains frozen at profit target level
        if not terminated and self.second_data is not None:
            current_bar_time = self.data.index[self.current_step]
            drawdown_hit, _ = self._check_second_level_drawdown(current_bar_time)
            if drawdown_hit:
                terminated = True
                reward = -10.0  # REWARD FIX: Massive penalty for Apex violation (was -0.1)
                done_reason = 'second_level_trailing_drawdown'
                self.done_reason = done_reason  # DIAGNOSTIC
                self.max_drawdown_reached = self.highest_balance - portfolio_value  # DIAGNOSTIC

        # ============================================================
        # 5. CALCULATE REWARD (Apex-Optimized)
        # ============================================================
        if not terminated:
            reward = self._calculate_apex_reward(
                position_changed, exit_reason, trade_pnl,
                portfolio_value, pm_action_taken
            )

        # ============================================================
        # 6. ADVANCE TIME
        # ============================================================
        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            truncated = True
            if done_reason is None:
                done_reason = 'end_of_data'
                self.done_reason = done_reason  # DIAGNOSTIC

        obs = self._get_observation()

        # DIAGNOSTIC: Enhanced info dict with compliance details
        info = {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'pm_action': pm_action_taken,
            'be_moves': self.be_move_count,
            'num_trades': self.num_trades,
            'balance': self.balance,
            'unrealized_pnl': unrealized_pnl,
            'trade_pnl': trade_pnl,
            'exit_reason': exit_reason,
            'observation_quality': self._observation_quality(obs),
            'step_index': self.current_step,
            'prev_step_index': self.current_step - 1,
            'remaining_bars': max(len(self.data) - self.current_step, 0),
            'bar_timestamp': str(self.data.index[min(self.current_step - 1, len(self.data) - 1)]),
            'done_reason': done_reason,
            'max_drawdown': getattr(self, 'max_drawdown_reached', 0),  # DIAGNOSTIC
            'episode_bars': self.current_step - self._episode_start_index,  # DIAGNOSTIC
            'trailing_dd_level': self.trailing_dd_level,  # DIAGNOSTIC
        }
        self._attach_action_mask(info)

        if done_reason:
            self.last_done_reason = done_reason
            self.done_reason = done_reason  # DIAGNOSTIC: Ensure instance variable set

        return obs, reward, terminated, truncated, info

    def _last_action_mask_as_list(self):
        if self.last_action_mask is None:
            return None
        return self.last_action_mask.astype(bool).tolist()

    def _attach_action_mask(self, info: Dict) -> None:
        mask_list = self._last_action_mask_as_list()
        if mask_list is not None:
            info['action_mask'] = mask_list

    def _observation_quality(self, obs: np.ndarray) -> Dict[str, float]:
        if obs is None:
            return {
                'has_nan': False,
                'has_inf': False,
                'min': float('nan'),
                'max': float('nan'),
            }

        obs = np.asarray(obs)
        has_nan = bool(np.isnan(obs).any())
        has_inf = bool(np.isinf(obs).any())
        finite_mask = np.isfinite(obs)
        if finite_mask.any():
            finite_vals = obs[finite_mask]
            min_val = float(finite_vals.min())
            max_val = float(finite_vals.max())
        else:
            min_val = float('nan')
            max_val = float('nan')

        return {
            'has_nan': has_nan,
            'has_inf': has_inf,
            'min': min_val,
            'max': max_val,
        }

    def _move_sl_to_breakeven(self, current_price: float) -> bool:
        """
        Move stop loss to entry price (break-even).
        Only allowed if profitable.

        Returns:
            True if SL was moved to BE
        """
        if self.position == 0:
            return False

        unrealized = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)

        # Only move to BE if profitable
        if unrealized <= 0:
            return False

        # Check if already at or past break-even
        if self.position == 1 and self.sl_price >= self.entry_price:
            return False
        if self.position == -1 and self.sl_price <= self.entry_price:
            return False

        # Add small buffer (commission coverage)
        buffer = 0.25

        if self.position == 1:
            self.sl_price = self.entry_price + buffer
        else:
            self.sl_price = self.entry_price - buffer

        return True

    def _update_trailing_stop(self, current_price: float, atr: float):
        """
        Update trailing stop if profit increases.
        Trails by 1× ATR behind current price.
        """
        if self.position == 0 or not self.trailing_stop_active or atr <= 0:
            return

        unrealized = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)
        unrealized_pnl = unrealized * self.contract_size * self.position_size

        if unrealized_pnl > self.highest_profit_point:
            # New high - move SL up
            trail_distance = atr * 1.0  # Trail 1 ATR behind

            if self.position == 1:
                new_sl = current_price - trail_distance
                self.sl_price = max(self.sl_price, new_sl)
            else:
                new_sl = current_price + trail_distance
                self.sl_price = min(self.sl_price, new_sl)

    def _calculate_position_size(self):
        """
        Calculate dynamic position size based on volatility.

        Reduces position size in high volatility environments to manage risk.
        This implements volatility-adjusted position sizing from the improvement plan.

        Returns:
            int: Adjusted position size (integer contracts, min 1)
        """
        if not self.position_scaling_enabled:
            return self.max_position_size

        # Get current ATR and average ATR
        current_atr = self.data['atr'].iloc[self.current_step]
        avg_atr = self.data['atr'].rolling(50, min_periods=10).mean().iloc[self.current_step]

        # Handle edge cases
        if np.isnan(current_atr) or np.isnan(avg_atr) or avg_atr <= 0:
            return self.max_position_size

        if self.volatility_adjustment:
            # Calculate volatility ratio
            vol_ratio = current_atr / avg_atr

            # Reduce size in high volatility (vol_ratio > 1)
            # When volatility is 2x normal, the multiplier halves
            # Formula: size = max_size / (1 + vol_ratio)
            size_multiplier = 1.0 / (1.0 + max(0, vol_ratio - 1.0))

            # Ensure multiplier never collapses completely
            size_multiplier = max(0.5, size_multiplier)
        else:
            size_multiplier = 1.0

        target_size = self.max_position_size * size_multiplier
        adjusted_size = max(1, math.floor(target_size))
        return adjusted_size

    def _reset_position_state(self):
        """Reset position and management state after exit."""
        self.position = 0
        self.entry_price = 0
        self.sl_price = 0
        self.tp_price = 0
        self.trailing_stop_active = False
        self.highest_profit_point = 0
        # Note: Don't reset counters (tracked across episode)

    def _validate_position_management_action(self, action, current_price, atr):
        """
        Validate position management actions before execution.

        CRITICAL: Prevent invalid actions that could violate trading rules
        """
        if atr <= 0 or np.isnan(atr):
            return False, "Invalid ATR"

        if action == self.ACTION_MOVE_SL_TO_BE:
            # Only allow if currently profitable
            unrealized = self._calculate_unrealized_pnl(current_price)
            if unrealized <= 0:
                return False, "Cannot move to BE when losing"
            
            # Check if already at or past break-even
            if self.position == 1 and self.sl_price >= self.entry_price:
                return False, "SL already at or past BE"
            if self.position == -1 and self.sl_price <= self.entry_price:
                return False, "SL already at or past BE"

        elif action == self.ACTION_ENABLE_TRAIL:
            # Only allow enabling trailing if profitable
            unrealized = self._calculate_unrealized_pnl(current_price)
            if unrealized <= 0:
                return False, "Cannot enable trailing when losing"

        elif action == self.ACTION_DISABLE_TRAIL:
            # Disabling trailing is always valid (but may be no-op)
            pass  # Always valid

        return True, "Valid"

    def _calculate_unrealized_pnl(self, current_price):
        """Calculate unrealized P&L at current price."""
        if self.position == 0:
            return 0.0

        if self.position == 1:  # Long
            return (current_price - self.entry_price) * self.contract_size * self.position_size
        else:  # Short
            return (self.entry_price - current_price) * self.contract_size * self.position_size

    def action_masks(self) -> np.ndarray:
        """
        Get action mask for current state.

        RL FIX #4: Action masking prevents wasted exploration on invalid actions.
        RL FIX #10: Updated for simplified 6-action space.
        RL FIX #11: Enhanced position management dependencies - ENABLE_TRAIL requires BE first.

        Returns:
            np.ndarray: Boolean mask of shape (6,) where True = valid action

        Called by MaskablePPO before each action selection.
        """
        # Start with all actions valid
        mask = np.ones(6, dtype=bool)

        current_price = self.data['close'].iloc[self.current_step]
        current_atr = self.data['atr'].iloc[self.current_step]

        # Handle invalid ATR
        if current_atr <= 0 or np.isnan(current_atr):
            current_atr = current_price * 0.01

        # Get current time for RTH gating
        # PERFORMANCE FIX: Use cached timezone conversion
        current_time = self.data.index[self.current_step]
        ts = self._get_et_time(current_time)

        rth_open = ts.replace(hour=9, minute=30, second=0)
        rth_close = ts.replace(hour=16, minute=59, second=0)
        in_rth = (ts >= rth_open) and (ts <= rth_close) and self.allow_new_trades

        if self.position == 0:
            # Agent is FLAT - only entry actions valid
            mask[0] = True  # Hold always valid
            mask[1] = in_rth  # Buy only in RTH
            mask[2] = in_rth  # Sell only in RTH
            # Disable all position management actions
            mask[3:6] = False  # Move BE, Enable Trail, Disable Trail

        else:
            # Agent HAS POSITION - validate each position management action
            mask[0] = True  # Hold always valid
            mask[1] = False  # Can't open new long when in position
            mask[2] = False  # Can't open new short when in position

            unrealized_pnl = self._calculate_unrealized_pnl(current_price)

            # RL FIX #11: Check if SL is at or past break-even
            # This enforces logical sequence: Move to BE → THEN enable trailing
            sl_at_or_past_be = False
            if self.position == 1:
                sl_at_or_past_be = (self.sl_price >= self.entry_price)
            else:
                sl_at_or_past_be = (self.sl_price <= self.entry_price)

            # Move to BE: only if profitable and NOT already at BE
            if unrealized_pnl > 0:
                mask[3] = not sl_at_or_past_be  # Can move if NOT at BE yet
            else:
                mask[3] = False

            # RL FIX #11: Enable trailing requires BE protection first
            # Prevents suboptimal sequence: Trail → BE (should be BE → Trail)
            mask[4] = (
                (unrealized_pnl > 0) and
                (not self.trailing_stop_active) and
                sl_at_or_past_be  # NEW: Must secure BE before trailing
            )

            # Disable trailing: only if currently enabled
            mask[5] = self.trailing_stop_active

        mask = mask.astype(bool, copy=True)
        self.last_action_mask = mask.copy()
        return self.last_action_mask.copy()

    # Phase 2 now uses unified _calculate_apex_reward() inherited from Phase 1
    # No separate reward function needed - pm_action parameter handles Phase 2 bonuses
