"""
JAX Trading Environment - Phase 2: Advanced Position Management

Pure JAX implementation extending Phase 1 with position management actions.
All operations are JIT-compilable with no Python control flow in the hot path.

Actions:
- 0: HOLD - Do nothing
- 1: BUY - Enter long position (flat only)
- 2: SELL - Enter short position (flat only)
- 3: MOVE_SL_TO_BE - Move stop loss to break-even (in profit only)
- 4: ENABLE_TRAIL - Enable trailing stop
- 5: DISABLE_TRAIL - Disable trailing stop
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, NamedTuple
from functools import partial
import chex

from .data_loader import MarketData


# =============================================================================
# Environment State & Parameters (Extended for Phase 2)
# =============================================================================

class EnvStatePhase2(NamedTuple):
    """Immutable environment state for Phase 2 JAX training."""
    # Position tracking (inherited from Phase 1)
    step_idx: jnp.ndarray
    position: jnp.ndarray
    entry_price: jnp.ndarray
    sl_price: jnp.ndarray
    tp_price: jnp.ndarray
    position_entry_step: jnp.ndarray
    
    # Portfolio tracking
    balance: jnp.ndarray
    highest_balance: jnp.ndarray
    trailing_dd_level: jnp.ndarray
    
    # Trade statistics
    num_trades: jnp.ndarray
    winning_trades: jnp.ndarray
    losing_trades: jnp.ndarray
    total_pnl: jnp.ndarray
    
    # Episode tracking
    episode_start_idx: jnp.ndarray
    
    # Phase 2 additions: Position management state
    trailing_stop_active: jnp.ndarray      # bool - is trailing stop enabled
    highest_profit_point: jnp.ndarray      # float32 - highest unrealized profit
    be_move_count: jnp.ndarray             # int32 - number of BE moves made
    original_sl_price: jnp.ndarray         # float32 - original SL for BE calculation
    trail_activation_price: jnp.ndarray    # float32 - price at which trailing activated


@chex.dataclass(frozen=True)
class EnvParamsPhase2:
    """Environment parameters for Phase 2."""
    # Market specifications
    contract_size: float = 50.0
    tick_size: float = 0.25
    tick_value: float = 12.50
    commission: float = 2.50
    slippage_ticks: int = 1
    
    # Position management
    initial_balance: float = 50000.0
    sl_atr_mult: float = 1.5
    tp_sl_ratio: float = 3.0
    position_size: float = 1.0  # Default fixed size
    trailing_dd_limit: float = 2500.0  # Stricter for Phase 2
    
    # Dynamic position sizing (NEW for parity)
    risk_per_trade: float = 0.01  # 1% risk per trade
    max_position_size: float = 3.0  # Max contracts
    contract_value: float = 50.0  # Value per contract point (same as contract_size for ES)
    
    # Observation parameters
    window_size: int = 20
    num_features: int = 11
    
    # Time rules
    rth_open: float = 9.5
    rth_close: float = 16.983
    
    # Episode parameters
    min_episode_bars: int = 1500
    
    # Phase 2 specific
    trail_activation_mult: float = 1.0    # Activate trail after 1R profit
    trail_distance_atr: float = 1.0       # Trail distance in ATR multiples
    be_min_profit_atr: float = 0.5        # Min profit to move SL to BE


# =============================================================================
# Action Constants
# =============================================================================

ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
ACTION_MOVE_SL_TO_BE = 3
ACTION_ENABLE_TRAIL = 4
ACTION_DISABLE_TRAIL = 5


# =============================================================================
# Core Environment Functions
# =============================================================================

def get_observation_phase2(
    state: EnvStatePhase2,
    data: MarketData,
    params: EnvParamsPhase2
) -> jnp.ndarray:
    """
    Get current observation window for Phase 2.
    
    Returns shape (window_size * num_features + 11,) = (231,)
    Additional features for position management state + validity features.
    """
    step = state.step_idx
    window = params.window_size
    
    # Compute start index
    start_idx = jnp.maximum(0, step - window)
    
    # Market features
    market_obs = lax.dynamic_slice(data.features, (start_idx, 0), (window, 8))
    time_obs = lax.dynamic_slice(data.time_features, (start_idx, 0), (window, 3))
    combined = jnp.concatenate([market_obs, time_obs], axis=1)
    flat_obs = combined.flatten()
    
    # Current market data
    current_price = data.prices[step, 3]
    current_atr = data.atr[step]
    safe_atr = jnp.where(current_atr > 0, current_atr, current_price * 0.01)
    safe_price = jnp.where(current_price > 0, current_price, 1.0)
    
    # Position features (5 dims) - same as Phase 1
    has_position = state.position != 0
    position_features = jnp.array([
        state.position.astype(jnp.float32),
        jnp.where(has_position, state.entry_price / safe_price, 1.0),
        jnp.where(has_position, jnp.abs(state.sl_price - safe_price) / safe_atr, 0.0),
        jnp.where(has_position, jnp.abs(state.tp_price - safe_price) / safe_atr, 0.0),
        jnp.where(has_position, (step - state.position_entry_step).astype(jnp.float32) / 390.0, 0.0),
    ], dtype=jnp.float32)
    
    # Phase 2 additional features (3 dims)
    unrealized_pnl = calculate_unrealized_pnl(state, current_price, params)
    phase2_features = jnp.array([
        state.trailing_stop_active.astype(jnp.float32),           # Trail active
        jnp.where(has_position, unrealized_pnl / 1000.0, 0.0),    # Normalized unrealized PnL
        state.be_move_count.astype(jnp.float32) / 3.0,            # BE move count normalized
    ], dtype=jnp.float32)
    
    # Validity features (3 dims) - NEW for parity with PyTorch
    current_hour = data.timestamps_hour[step]
    in_rth = (current_hour >= params.rth_open) & (current_hour < params.rth_close)
    can_enter = (state.position == 0) & in_rth
    can_manage = has_position
    
    validity_features = jnp.array([
        can_enter.astype(jnp.float32),      # Can enter new trade
        can_manage.astype(jnp.float32),     # Can manage position
        has_position.astype(jnp.float32),   # Has active position
    ], dtype=jnp.float32)
    
    # Combine all features: 220 + 5 + 3 + 3 = 231
    obs = jnp.concatenate([flat_obs, position_features, phase2_features, validity_features])
    obs = jnp.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return obs.astype(jnp.float32)


def calculate_unrealized_pnl(
    state: EnvStatePhase2,
    current_price: jnp.ndarray,
    params: EnvParamsPhase2
) -> jnp.ndarray:
    """Calculate current unrealized P&L."""
    pnl_long = (current_price - state.entry_price) * params.contract_size * params.position_size
    pnl_short = (state.entry_price - current_price) * params.contract_size * params.position_size
    
    pnl = jnp.where(
        state.position == 1, pnl_long,
        jnp.where(state.position == -1, pnl_short, 0.0)
    )
    return pnl


def action_masks_phase2(state: EnvStatePhase2, data: MarketData, params: EnvParamsPhase2) -> jnp.ndarray:
    """
    Return valid action mask for Phase 2 (6 actions).
    
    Returns: Boolean array shape (6,)
    """
    is_flat = state.position == 0
    has_position = state.position != 0
    
    # Check RTH
    current_hour = data.timestamps_hour[state.step_idx]
    within_rth = (current_hour >= params.rth_open) & (current_hour < params.rth_close)
    
    # Check profitability for BE move
    current_price = data.prices[state.step_idx, 3]
    unrealized_pnl = calculate_unrealized_pnl(state, current_price, params)
    current_atr = data.atr[state.step_idx]
    min_profit = current_atr * params.be_min_profit_atr * params.contract_size * params.position_size
    is_profitable = unrealized_pnl > min_profit
    
    # Check if BE not already moved (SL still at original position)
    be_not_moved = state.be_move_count == 0
    
    mask = jnp.array([
        True,                                           # 0: HOLD - always valid
        is_flat & within_rth,                           # 1: BUY
        is_flat & within_rth,                           # 2: SELL
        has_position & is_profitable & be_not_moved,    # 3: MOVE_SL_TO_BE
        has_position & ~state.trailing_stop_active,     # 4: ENABLE_TRAIL
        has_position & state.trailing_stop_active,      # 5: DISABLE_TRAIL
    ], dtype=jnp.bool_)
    
    return mask


def calculate_sl_tp_phase2(
    entry_price: jnp.ndarray,
    position_type: jnp.ndarray,
    atr: jnp.ndarray,
    params: EnvParamsPhase2
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate initial stop loss and take profit prices."""
    safe_atr = jnp.where(atr > 0, atr, entry_price * 0.01)
    
    sl_distance = safe_atr * params.sl_atr_mult
    tp_distance = sl_distance * params.tp_sl_ratio
    
    sl_long = entry_price - sl_distance
    tp_long = entry_price + tp_distance
    sl_short = entry_price + sl_distance
    tp_short = entry_price - tp_distance
    
    sl_price = jnp.where(position_type == 1, sl_long, sl_short)
    tp_price = jnp.where(position_type == 1, tp_long, tp_short)
    
    return sl_price, tp_price


def update_trailing_stop(
    state: EnvStatePhase2,
    current_price: jnp.ndarray,
    current_atr: jnp.ndarray,
    params: EnvParamsPhase2
) -> jnp.ndarray:
    """
    Update trailing stop price if active and in profit.
    Returns new SL price.
    """
    trail_distance = current_atr * params.trail_distance_atr
    
    # Long: trail below price
    new_sl_long = current_price - trail_distance
    should_update_long = (state.position == 1) & state.trailing_stop_active & (new_sl_long > state.sl_price)
    
    # Short: trail above price
    new_sl_short = current_price + trail_distance
    should_update_short = (state.position == -1) & state.trailing_stop_active & (new_sl_short < state.sl_price)
    
    new_sl = jnp.where(
        should_update_long, new_sl_long,
        jnp.where(should_update_short, new_sl_short, state.sl_price)
    )
    
    return new_sl


def calculate_position_size(
    atr: jnp.ndarray,
    balance: jnp.ndarray,
    params: EnvParamsPhase2
) -> jnp.ndarray:
    """
    Calculate volatility-based position size (NEW for parity with PyTorch).
    
    Size based on risk per trade and ATR-based SL distance.
    """
    risk_amount = balance * params.risk_per_trade
    sl_distance = atr * params.sl_atr_mult
    size = risk_amount / (sl_distance * params.contract_value + 1e-8)
    
    # Clamp to [1.0, max_position_size]
    size = jnp.clip(size, 1.0, params.max_position_size)
    
    return size


def validate_pm_action(
    action: jnp.ndarray,
    state: EnvStatePhase2,
    current_price: jnp.ndarray,
    current_atr: jnp.ndarray,
    params: EnvParamsPhase2
) -> jnp.ndarray:
    """
    Validate position management actions (NEW for parity with PyTorch).
    
    Returns: Boolean indicating if action is valid
    """
    # Calculate unrealized PnL
    pnl = jnp.where(
        state.position == 1,
        (current_price - state.entry_price) * params.contract_size * params.position_size,
        jnp.where(
            state.position == -1,
            (state.entry_price - current_price) * params.contract_size * params.position_size,
            0.0
        )
    )
    
    is_profitable = pnl > 0
    has_position = state.position != 0
    
    # Validation logic for each PM action
    valid = jnp.where(
        action == ACTION_MOVE_SL_TO_BE,
        # Valid if profitable, has position, and SL not already at BE
        is_profitable & has_position & (state.be_move_count == 0),
        jnp.where(
            action == ACTION_ENABLE_TRAIL,
            # Valid if profitable and has position
            is_profitable & has_position & ~state.trailing_stop_active,
            jnp.where(
                action == ACTION_DISABLE_TRAIL,
                # Valid if has position and trailing is active
                has_position & state.trailing_stop_active,
                True  # Non-PM actions are valid by default (validated elsewhere)
            )
        )
    )
    
    return valid


def reset_phase2(
    key: jax.random.PRNGKey,
    params: EnvParamsPhase2,
    data: MarketData
) -> Tuple[jnp.ndarray, EnvStatePhase2]:
    """Reset Phase 2 environment to initial state with RTH-aligned start."""
    # Sample from pre-computed RTH indices instead of full range
    # This ensures episodes start during regular trading hours (9:30 AM - 4:00 PM ET)
    num_rth_indices = data.rth_indices.shape[0]
    idx_choice = jax.random.randint(key, shape=(), minval=0, maxval=num_rth_indices)
    episode_start = data.rth_indices[idx_choice]
    
    state = EnvStatePhase2(
        step_idx=episode_start,
        position=jnp.array(0, dtype=jnp.int32),
        entry_price=jnp.array(0.0, dtype=jnp.float32),
        sl_price=jnp.array(0.0, dtype=jnp.float32),
        tp_price=jnp.array(0.0, dtype=jnp.float32),
        position_entry_step=jnp.array(0, dtype=jnp.int32),
        balance=jnp.array(params.initial_balance, dtype=jnp.float32),
        highest_balance=jnp.array(params.initial_balance, dtype=jnp.float32),
        trailing_dd_level=jnp.array(params.initial_balance - params.trailing_dd_limit, dtype=jnp.float32),
        num_trades=jnp.array(0, dtype=jnp.int32),
        winning_trades=jnp.array(0, dtype=jnp.int32),
        losing_trades=jnp.array(0, dtype=jnp.int32),
        total_pnl=jnp.array(0.0, dtype=jnp.float32),
        episode_start_idx=episode_start,
        # Phase 2 additions
        trailing_stop_active=jnp.array(False, dtype=jnp.bool_),
        highest_profit_point=jnp.array(0.0, dtype=jnp.float32),
        be_move_count=jnp.array(0, dtype=jnp.int32),
        original_sl_price=jnp.array(0.0, dtype=jnp.float32),
        trail_activation_price=jnp.array(0.0, dtype=jnp.float32),
    )
    
    obs = get_observation_phase2(state, data, params)
    return obs, state


def step_phase2(
    key: jax.random.PRNGKey,
    state: EnvStatePhase2,
    action: jnp.ndarray,
    params: EnvParamsPhase2,
    data: MarketData
) -> Tuple[jnp.ndarray, EnvStatePhase2, jnp.ndarray, jnp.ndarray, dict]:
    """
    Execute one Phase 2 environment step.
    
    Handles position management actions in addition to entry actions.
    """
    step_idx = state.step_idx
    data_length = data.features.shape[0]
    
    current_price = data.prices[step_idx, 3]
    high = data.prices[step_idx, 1]
    low = data.prices[step_idx, 2]
    current_atr = data.atr[step_idx]
    current_hour = data.timestamps_hour[step_idx]
    
    slippage = params.tick_size * params.slippage_ticks
    
    # Initialize tracking
    trade_pnl = jnp.array(0.0)
    exit_type = jnp.array(0)  # 0=none, 1=SL, 2=TP
    
    # =========================================================================
    # 1. UPDATE TRAILING STOP IF ACTIVE
    # =========================================================================
    new_sl_price = update_trailing_stop(state, current_price, current_atr, params)
    state = state._replace(sl_price=new_sl_price)
    
    # =========================================================================
    # 2. CHECK SL/TP ON EXISTING POSITION
    # =========================================================================
    # Long SL/TP check
    sl_hit_long = (state.position == 1) & (low <= state.sl_price)
    tp_hit_long = (state.position == 1) & (high >= state.tp_price)
    
    # Short SL/TP check
    sl_hit_short = (state.position == -1) & (high >= state.sl_price)
    tp_hit_short = (state.position == -1) & (low <= state.tp_price)
    
    sl_hit = sl_hit_long | sl_hit_short
    tp_hit = tp_hit_long | tp_hit_short
    position_closed = (sl_hit | tp_hit) & (state.position != 0)
    
    exit_price = jnp.where(sl_hit, state.sl_price, jnp.where(tp_hit, state.tp_price, 0.0))
    
    # Calculate PnL
    pnl_long = (exit_price - state.entry_price) * params.contract_size * params.position_size
    pnl_short = (state.entry_price - exit_price) * params.contract_size * params.position_size
    closed_pnl = jnp.where(state.position == 1, pnl_long, pnl_short) - params.commission * 2 * params.position_size
    
    trade_pnl = jnp.where(position_closed, closed_pnl, 0.0)
    exit_type = jnp.where(tp_hit & position_closed, 2, jnp.where(sl_hit & position_closed, 1, 0))
    
    # Update state after close
    balance_after = state.balance + trade_pnl
    winning_after = state.winning_trades + jnp.where(position_closed & (closed_pnl > 0), 1, 0)
    losing_after = state.losing_trades + jnp.where(position_closed & (closed_pnl <= 0), 1, 0)
    
    # Reset position state on close
    position_after = jnp.where(position_closed, 0, state.position)
    entry_after = jnp.where(position_closed, 0.0, state.entry_price)
    sl_after = jnp.where(position_closed, 0.0, state.sl_price)
    tp_after = jnp.where(position_closed, 0.0, state.tp_price)
    trailing_after = jnp.where(position_closed, False, state.trailing_stop_active)
    be_count_after = jnp.where(position_closed, 0, state.be_move_count)
    
    # =========================================================================
    # 3. CHECK TIME RULES
    # =========================================================================
    past_close = current_hour >= params.rth_close
    within_rth = (current_hour >= params.rth_open) & (current_hour < params.rth_close)
    
    forced_close = past_close & (position_after != 0)
    forced_exit_price = jnp.where(
        position_after == 1,
        current_price - slippage,
        current_price + slippage
    )
    forced_pnl_long = (forced_exit_price - entry_after) * params.contract_size * params.position_size
    forced_pnl_short = (entry_after - forced_exit_price) * params.contract_size * params.position_size
    forced_pnl = jnp.where(position_after == 1, forced_pnl_long, forced_pnl_short) - params.commission * 2 * params.position_size
    
    trade_pnl = jnp.where(forced_close, forced_pnl, trade_pnl)
    balance_after = balance_after + jnp.where(forced_close, forced_pnl, 0.0)
    position_after = jnp.where(forced_close, 0, position_after)
    
    # =========================================================================
    # 4. VALIDATE AND EXECUTE POSITION MANAGEMENT ACTIONS
    # =========================================================================
    is_flat = position_after == 0
    has_position = position_after != 0
    can_open = is_flat & within_rth
    
    # Validate PM action first (NEW for parity)
    action_is_valid = validate_pm_action(action, state, current_price, current_atr, params)
    effective_action = jnp.where(action_is_valid, action, ACTION_HOLD)
    
    # Calculate dynamic position size for new trades (NEW for parity)
    dynamic_size = calculate_position_size(current_atr, balance_after, params)
    
    # --- Action 1: BUY ---
    opening_long = can_open & (effective_action == ACTION_BUY)
    buy_entry = current_price + slippage
    buy_sl, buy_tp = calculate_sl_tp_phase2(buy_entry, jnp.array(1), current_atr, params)
    
    # --- Action 2: SELL ---
    opening_short = can_open & (effective_action == ACTION_SELL)
    sell_entry = current_price - slippage
    sell_sl, sell_tp = calculate_sl_tp_phase2(sell_entry, jnp.array(-1), current_atr, params)
    
    opening_any = opening_long | opening_short
    
    # --- Action 3: MOVE SL TO BE (validated) ---
    unrealized_pnl = calculate_unrealized_pnl(state, current_price, params)
    min_profit_for_be = current_atr * params.be_min_profit_atr * params.contract_size * dynamic_size
    can_move_be = has_position & (unrealized_pnl > min_profit_for_be) & (be_count_after == 0)
    moving_to_be = can_move_be & (effective_action == ACTION_MOVE_SL_TO_BE)
    be_sl = state.entry_price  # Move SL to entry price
    
    # --- Action 4: ENABLE TRAILING (validated) ---
    enabling_trail = has_position & ~trailing_after & (effective_action == ACTION_ENABLE_TRAIL)
    
    # --- Action 5: DISABLE TRAILING (validated) ---
    disabling_trail = has_position & trailing_after & (effective_action == ACTION_DISABLE_TRAIL)
    
    # Store the position size used for this trade
    new_position_size = jnp.where(opening_any, dynamic_size, params.position_size)
    
    # Apply all updates
    new_position = jnp.where(opening_long, 1, jnp.where(opening_short, -1, position_after))
    new_entry_price = jnp.where(opening_long, buy_entry, jnp.where(opening_short, sell_entry, entry_after))
    new_sl_price = jnp.where(opening_long, buy_sl, jnp.where(opening_short, sell_sl, sl_after))
    new_sl_price = jnp.where(moving_to_be, be_sl, new_sl_price)
    new_tp_price = jnp.where(opening_long, buy_tp, jnp.where(opening_short, sell_tp, tp_after))
    new_balance = balance_after - jnp.where(opening_any, params.commission * params.position_size, 0.0)
    new_trailing_active = jnp.where(enabling_trail, True, jnp.where(disabling_trail, False, trailing_after))
    new_be_count = be_count_after + jnp.where(moving_to_be, 1, 0)
    new_num_trades = state.num_trades + jnp.where(opening_any, 1, 0)
    new_position_entry_step = jnp.where(opening_any, step_idx, state.position_entry_step)
    new_original_sl = jnp.where(opening_any, new_sl_price, state.original_sl_price)
    
    # =========================================================================
    # 5. UPDATE PORTFOLIO & TRAILING DRAWDOWN
    # =========================================================================
    unrealized_pnl = jnp.where(
        new_position == 1,
        (current_price - new_entry_price) * params.contract_size * params.position_size,
        jnp.where(
            new_position == -1,
            (new_entry_price - current_price) * params.contract_size * params.position_size,
            0.0
        )
    )
    
    portfolio_value = new_balance + unrealized_pnl
    new_highest = jnp.maximum(state.highest_balance, portfolio_value)
    new_trailing_dd = new_highest - params.trailing_dd_limit
    
    # Update highest profit point
    new_highest_profit = jnp.where(
        new_position != 0,
        jnp.maximum(state.highest_profit_point, unrealized_pnl),
        0.0
    )
    
    # =========================================================================
    # 6. CHECK TERMINATION
    # =========================================================================
    dd_violation = portfolio_value < new_trailing_dd
    end_of_data = step_idx >= data_length - 2
    time_violation = forced_close
    
    done = dd_violation | end_of_data | time_violation
    
    # =========================================================================
    # 7. CALCULATE REWARD (Apex-Optimized Parity)
    # =========================================================================
    # Ported from environment_phase2.py _calculate_apex_reward
    
    # 1. Position Management Outcome-based Feedback
    # ---------------------------------------------------------
    pm_reward = 0.0
    
    # Enable Trail Feedback
    # Good: Unrealized PnL > 1% of balance (good timing)
    # Bad: Unrealized PnL <= 1% (too early)
    profit_threshold = params.initial_balance * 0.01
    is_good_trail = unrealized_pnl > profit_threshold
    
    pm_reward += jnp.where(enabling_trail & is_good_trail, 0.1, 0.0)
    pm_reward += jnp.where(enabling_trail & ~is_good_trail, -0.1, 0.0)
    
    # Disable Trail Feedback (Neutral)
    # Move to BE Feedback (Neutral/Slight Positive handled in action mask/bonus)
    
    # 2. Trade Completion Reward
    # ---------------------------------------------------------
    # Stronger signals: Win=+2.0, Loss=-1.0
    trade_reward = 0.0
    trade_reward += jnp.where(position_closed & (closed_pnl > 0), 2.0, 0.0)
    trade_reward += jnp.where(position_closed & (closed_pnl <= 0), -1.0, 0.0)
    
    # Bonus: Successful trailing usage
    # If trade won AND trailing was active (or just activated), give bonus
    had_trailing = state.trailing_stop_active | enabling_trail
    trade_reward += jnp.where(position_closed & (closed_pnl > 0) & had_trailing, 0.5, 0.0)
    
    # 3. Portfolio Value Signal (Scaled)
    # ---------------------------------------------------------
    # 20x scaling on percentage return
    pnl_pct = (portfolio_value - params.initial_balance) / params.initial_balance
    portfolio_reward = pnl_pct * 20.0
    
    # 4. Penalties
    # ---------------------------------------------------------
    # Invalid action penalty (handled in step function return if invalid, but here for valid actions that are bad)
    # We already have action masks, so we don't need to penalize invalid actions here again
    # But we do penalize violations
    violation_penalty = jnp.where(dd_violation | time_violation, -10.0, 0.0) # Apex violation is severe
    
    # 5. BE Move Bonus (Neutral/Slight Positive handled in action mask/bonus)
    # PyTorch implementation sets this to 0.0
    be_move_bonus = 0.0
    
    # Combine all components
    reward = pm_reward + trade_reward + portfolio_reward + violation_penalty + be_move_bonus
    
    # Clip for stability (though portfolio reward might exceed this, so maybe relax clip or clip only components)
    # PyTorch implementation doesn't explicitly clip total reward, but PPO usually likes clipped rewards.
    # However, portfolio signal is continuous. Let's clip to [-10, 10] to be safe but allow strong signals.
    reward = jnp.clip(reward, -10.0, 10.0)
    
    # =========================================================================
    # 8. CREATE NEW STATE
    # =========================================================================
    new_state = EnvStatePhase2(
        step_idx=step_idx + 1,
        position=new_position.astype(jnp.int32),
        entry_price=new_entry_price,
        sl_price=new_sl_price,
        tp_price=new_tp_price,
        position_entry_step=new_position_entry_step.astype(jnp.int32),
        balance=new_balance,
        highest_balance=new_highest,
        trailing_dd_level=new_trailing_dd,
        num_trades=new_num_trades.astype(jnp.int32),
        winning_trades=winning_after.astype(jnp.int32),
        losing_trades=losing_after.astype(jnp.int32),
        total_pnl=state.total_pnl + trade_pnl,
        episode_start_idx=state.episode_start_idx,
        trailing_stop_active=new_trailing_active,
        highest_profit_point=new_highest_profit,
        be_move_count=new_be_count.astype(jnp.int32),
        original_sl_price=new_original_sl,
        trail_activation_price=state.trail_activation_price,
    )
    
    obs = get_observation_phase2(new_state, data, params)
    
    info = {
        'portfolio_value': portfolio_value,
        'num_trades': new_num_trades,
        'win_rate': winning_after / jnp.maximum(new_num_trades, 1),
        'total_pnl': state.total_pnl + trade_pnl,
        'trailing_active': new_trailing_active,
        'be_moved': new_be_count > 0,
        'trade_pnl': trade_pnl,
        'position_closed': position_closed,
        # NEW: Additional fields for training metrics tracker
        'final_balance': new_balance,
        'trailing_dd': new_highest - new_balance,
        'forced_close': forced_close,
        'episode_return': new_balance - params.initial_balance,
    }
    
    return obs, new_state, reward, done, info


# =============================================================================
# Vectorized Wrappers
# =============================================================================

@partial(jax.jit, static_argnums=(1, 2))
def batch_reset_phase2(
    keys: jax.random.PRNGKey,
    params: EnvParamsPhase2,
    num_envs: int,
    data: MarketData
) -> Tuple[jnp.ndarray, EnvStatePhase2]:
    """Reset multiple Phase 2 environments in parallel."""
    keys = jax.random.split(keys, num_envs)
    return jax.vmap(reset_phase2, in_axes=(0, None, None))(keys, params, data)


@partial(jax.jit, static_argnums=(3,))
def batch_step_phase2(
    keys: jax.random.PRNGKey,
    states: EnvStatePhase2,
    actions: jnp.ndarray,
    params: EnvParamsPhase2,
    data: MarketData
) -> Tuple[jnp.ndarray, EnvStatePhase2, jnp.ndarray, jnp.ndarray, dict]:
    """Step multiple Phase 2 environments in parallel."""
    return jax.vmap(step_phase2, in_axes=(0, 0, 0, None, None))(
        keys, states, actions, params, data
    )


@partial(jax.jit, static_argnums=(2,))
def batch_action_masks_phase2(
    states: EnvStatePhase2,
    data: MarketData,
    params: EnvParamsPhase2
) -> jnp.ndarray:
    """Get action masks for multiple Phase 2 environments."""
    return jax.vmap(action_masks_phase2, in_axes=(0, None, None))(states, data, params)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Phase 2 JAX Trading Environment...")
    print(f"JAX devices: {jax.devices()}")
    
    # Create dummy data
    num_timesteps = 10000
    key = jax.random.key(0)
    
    # Create dummy RTH indices (sample indices within RTH hours)
    rth_indices = jnp.arange(100, 8000, 10)  # Sample RTH indices
    
    dummy_data = MarketData(
        features=jax.random.normal(key, (num_timesteps, 8)),
        prices=jnp.abs(jax.random.normal(key, (num_timesteps, 4))) * 100 + 5000,
        atr=jnp.abs(jax.random.normal(key, (num_timesteps,))) * 10 + 5,
        time_features=jax.random.uniform(key, (num_timesteps, 3)),
        trading_mask=jnp.ones(num_timesteps),
        timestamps_hour=jnp.linspace(9.5, 16.9, num_timesteps),
        rth_indices=rth_indices,
        low_s=jnp.abs(jax.random.normal(key, (num_timesteps,))) * 100 + 4990,
        high_s=jnp.abs(jax.random.normal(key, (num_timesteps,))) * 100 + 5010,
    )
    
    params = EnvParamsPhase2()
    
    # Test single reset
    obs, state = reset_phase2(key, params, dummy_data)
    print(f"\nSingle reset:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Expected: (231,)")
    
    # Test action mask
    mask = action_masks_phase2(state, dummy_data, params)
    print(f"\nAction mask (flat): {mask}")
    
    # Test BUY action
    key, step_key = jax.random.split(key)
    obs, state, reward, done, info = step_phase2(step_key, state, jnp.array(1), params, dummy_data)
    print(f"\nAfter BUY:")
    print(f"  Position: {state.position}")
    print(f"  Entry: {state.entry_price:.2f}")
    print(f"  SL: {state.sl_price:.2f}")
    print(f"  TP: {state.tp_price:.2f}")
    
    mask = action_masks_phase2(state, dummy_data, params)
    print(f"  Action mask (in position): {mask}")
    
    # Test vectorized
    num_envs = 1000
    obs_batch, state_batch = batch_reset_phase2(key, params, num_envs, dummy_data)
    print(f"\nVectorized reset ({num_envs} envs):")
    print(f"  Observation batch shape: {obs_batch.shape}")
    
    # Benchmark
    import time
    actions = jax.random.randint(key, (num_envs,), 0, 6)
    keys = jax.random.split(key, num_envs)
    
    # Warm-up
    _ = batch_step_phase2(keys, state_batch, actions, params, dummy_data)
    
    num_iterations = 100
    start = time.time()
    for _ in range(num_iterations):
        obs_batch, state_batch, rewards, dones, _ = batch_step_phase2(
            keys, state_batch, actions, params, dummy_data
        )
        jax.block_until_ready(obs_batch)
    elapsed = time.time() - start
    
    steps_per_sec = (num_envs * num_iterations) / elapsed
    print(f"\nBenchmark ({num_envs} envs × {num_iterations} steps):")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {steps_per_sec:,.0f} env steps/sec")
