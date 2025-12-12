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
    
    # NEW: Dynamic position sizing (2025-12-12)
    position_size: jnp.ndarray             # float32 - actual contracts for current trade


@chex.dataclass(frozen=True)
class EnvParamsPhase2:
    """Environment parameters for Phase 2.
    
    ⚠️ DEFAULTS ARE ES CONTRACT SPECS - Override for other markets!
    Use market_specs.py to get correct values for NQ, YM, RTY, etc.
    """
    # Market specifications (ES defaults - override via market_specs.py for other markets)
    contract_size: float = 50.0      # ES = $50/point, NQ = $20/point
    tick_size: float = 0.25          # ES/NQ tick size
    tick_value: float = 12.50        # ES = $12.50 per tick (50 * 0.25)
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
    min_episode_bars: int = 400  # Reduced from 1500 for faster learning cycles
    
    # Phase 2 specific
    trail_activation_mult: float = 1.0    # Activate trail after 1R profit
    trail_distance_atr: float = 1.0       # Trail distance in ATR multiples
    be_min_profit_atr: float = 0.5        # Min profit to move SL to BE
    
    # Commission curriculum (ported from Phase 1 - solves HOLD trap)
    initial_commission: float = 1.0       # Start with reduced commission
    final_commission: float = 2.5         # End with realistic commission
    commission_curriculum: bool = True    # Enable commission ramping
    training_progress: float = 0.0        # 0.0 to 1.0, updated each rollout
    
    # Exploration bonus curriculum (ported from Phase 1 - critical for trades)
    current_global_timestep: float = 0.0          # Global timestep counter
    total_training_timesteps: float = 100_000_000 # Total timesteps for exploration decay
    
    # Forced position curriculum (NEW - solves chicken-and-egg problem)
    # Phase 2A: Start with forced positions to guarantee PM experiences
    forced_position_ratio: float = 0.0     # 0.0 = no forced, 0.5 = 50% start with position
    forced_position_profit_range: float = 1.0  # Random unrealized P/L range in ATR multiples
    
    # PM exploration bonus (INCREASED to $200 - PM actions need stronger signal)
    # PM actions are completely novel - agent never experienced them in Phase 1
    pm_action_exploration_bonus: float = 200.0   # Base bonus for MOVE_SL, ENABLE_TRAIL
    entry_action_exploration_bonus: float = 100.0  # Base bonus for BUY/SELL


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
    
    FIX (2025-12-10): Added drawdown features for Apex compliance awareness.
    Returns shape (window_size * num_features + 13,) = (233,)
    Additional features for position management state + validity features + drawdown.
    """
    step = state.step_idx
    # CRITICAL FIX (2025-12-08): window_size MUST be a Python int for lax.dynamic_slice
    # When env_params is traced (not static_argnums), params.window_size returns a traced value
    # but lax.dynamic_slice requires compile-time static slice sizes.
    # Since window_size NEVER changes during training, we hardcode it here.
    window = 20  # Hardcoded: matches EnvParamsPhase2.window_size default
    
    # Compute start index
    start_idx = jnp.maximum(0, step - window)
    
    # Market features - use hardcoded window size for slice shape
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
    # SYNCHRONIZED WITH Agent_temp (2025-12-07): time_in_position is RAW bars, not normalized
    has_position = state.position != 0
    position_features = jnp.array([
        state.position.astype(jnp.float32),
        jnp.where(has_position, state.entry_price / safe_price, 1.0),
        jnp.where(has_position, jnp.abs(state.sl_price - safe_price) / safe_atr, 0.0),
        jnp.where(has_position, jnp.abs(state.tp_price - safe_price) / safe_atr, 0.0),
        jnp.where(has_position, (step - state.position_entry_step).astype(jnp.float32), 0.0),  # RAW bars (match Agent_temp)
    ], dtype=jnp.float32)
    
    # Calculate unrealized PnL and portfolio value for drawdown calculation
    unrealized_pnl = calculate_unrealized_pnl(state, current_price, params)
    portfolio_value = state.balance + unrealized_pnl
    
    # FIX (2025-12-10): Drawdown features for Apex compliance awareness
    current_drawdown = state.highest_balance - portfolio_value
    drawdown_ratio = current_drawdown / params.trailing_dd_limit  # 0.0 = no DD, 1.0 = at limit
    drawdown_room = (params.trailing_dd_limit - current_drawdown) / 1000.0  # Room left (in $1000s)
    
    # Phase 2 additional features (5 dims, was 3 - added 2 for drawdown)
    phase2_features = jnp.array([
        state.trailing_stop_active.astype(jnp.float32),           # Trail active
        jnp.where(has_position, unrealized_pnl / 1000.0, 0.0),    # Normalized unrealized PnL
        state.be_move_count.astype(jnp.float32) / 3.0,            # BE move count normalized
        jnp.clip(drawdown_ratio, 0.0, 2.0),                       # NEW: Drawdown ratio (0-1, can exceed)
        jnp.clip(drawdown_room, -2.5, 2.5),                       # NEW: Room left to limit ($1000s)
    ], dtype=jnp.float32)
    
    # Validity features (3 dims)
    # SYNCHRONIZED WITH Agent_temp (2025-12-07): can_enter only checks position, not RTH
    # RTH check is done separately in action mask (matches Agent_temp behavior)
    can_enter = state.position == 0
    can_manage = has_position
    
    validity_features = jnp.array([
        can_enter.astype(jnp.float32),      # Can enter new trade (position check only)
        can_manage.astype(jnp.float32),     # Can manage position
        has_position.astype(jnp.float32),   # Has active position
    ], dtype=jnp.float32)
    
    # Combine all features: 220 + 5 + 5 + 3 = 233 (was 231)
    obs = jnp.concatenate([flat_obs, position_features, phase2_features, validity_features])
    obs = jnp.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return obs.astype(jnp.float32)


def calculate_unrealized_pnl(
    state: EnvStatePhase2,
    current_price: jnp.ndarray,
    params: EnvParamsPhase2
) -> jnp.ndarray:
    """Calculate current unrealized P&L using the position's actual size."""
    # FIX (2025-12-12): Use state.position_size for dynamic sizing
    pnl_long = (current_price - state.entry_price) * params.contract_size * state.position_size
    pnl_short = (state.entry_price - current_price) * params.contract_size * state.position_size
    
    pnl = jnp.where(
        state.position == 1, pnl_long,
        jnp.where(state.position == -1, pnl_short, 0.0)
    )
    return pnl


def action_masks_phase2(state: EnvStatePhase2, data: MarketData, params: EnvParamsPhase2) -> jnp.ndarray:
    """
    Return valid action mask for Phase 2 (6 actions).
    
    FIX D (2025-12-10 v2): Emergency HOLD when DD > 80%
    
    Returns: Boolean array shape (6,)
    """
    is_flat = state.position == 0
    has_position = state.position != 0
    
    # Check RTH
    current_hour = data.timestamps_hour[state.step_idx]
    within_rth = (current_hour >= params.rth_open) & (current_hour < params.rth_close)
    
    # FIX D (2025-12-10 v2): Emergency HOLD when DD > 80%
    # Prevent opening new positions when too close to Apex limit
    current_price = data.prices[state.step_idx, 3]
    unrealized_pnl = calculate_unrealized_pnl(state, current_price, params)
    portfolio_value = state.balance + unrealized_pnl
    current_dd = state.highest_balance - portfolio_value
    dd_ratio = current_dd / params.trailing_dd_limit
    
    # Allow new entries only if DD < 75% (was 80%)
    # 80% ($2000) left $500 room, which is exactly the max loss per trade.
    # Commissions/slippage pushed it over ($502+). 75% ($1875) leaves $625 room.
    safe_to_enter = dd_ratio < 0.75
    
    # Check profitability for BE move
    current_atr = data.atr[state.step_idx]
    # FIX (2025-12-12): Use state.position_size for dynamic sizing
    min_profit = current_atr * params.be_min_profit_atr * params.contract_size * state.position_size
    is_profitable = unrealized_pnl > min_profit
    
    # Check if BE not already moved (SL still at original position)
    be_not_moved = state.be_move_count == 0
    
    mask = jnp.array([
        True,                                                   # 0: HOLD - always valid
        is_flat & within_rth & safe_to_enter,                   # 1: BUY (+ DD check)
        is_flat & within_rth & safe_to_enter,                   # 2: SELL (+ DD check)
        has_position & is_profitable & be_not_moved,            # 3: MOVE_SL_TO_BE
        has_position & ~state.trailing_stop_active,             # 4: ENABLE_TRAIL
        has_position & state.trailing_stop_active,              # 5: DISABLE_TRAIL
    ], dtype=jnp.bool_)
    
    return mask


def calculate_sl_tp_phase2(
    entry_price: jnp.ndarray,
    position_type: jnp.ndarray,
    atr: jnp.ndarray,
    params: EnvParamsPhase2
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate initial stop loss and take profit prices.
    
    FIX (2025-12-10): Cap SL distance so max loss with 1 contract ≤ $500.
    This ensures Apex DD compliance - allows 5 consecutive losses before
    hitting the $2,500 trailing drawdown limit.
    
    Example for NQ ($20/point): max_sl = 500/20 = 25 points
    Example for ES ($50/point): max_sl = 500/50 = 10 points
    """
    safe_atr = jnp.where(atr > 0, atr, entry_price * 0.01)
    
    # Calculate raw SL distance based on ATR
    raw_sl_distance = safe_atr * params.sl_atr_mult
    
    # FIX: Cap SL distance so max loss with 1 contract ≤ $500
    # This prevents single trades from exceeding the Apex DD limit
    max_sl_distance_points = 500.0 / params.contract_size  # NQ: 25pts, ES: 10pts
    sl_distance = jnp.minimum(raw_sl_distance, max_sl_distance_points)
    
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
    params: EnvParamsPhase2,
    portfolio_value: jnp.ndarray = None,
    highest_balance: jnp.ndarray = None
) -> jnp.ndarray:
    """
    Calculate volatility-based position size with Apex safety features.
    
    FIX (2025-12-10 v2): Enhanced with:
    - Fix B: Max SL loss reduced to $500 (was $1000)
    - Fix C: DD-based position scaling (reduce size as DD increases)
    """
    # FIX (2025-12-10 v3): Harden position sizing
    # 1. Use EFFECTIVE (capped) SL distance for calculation
    max_sl_points = 500.0 / params.contract_size
    raw_sl_distance = atr * params.sl_atr_mult
    effective_sl_distance = jnp.minimum(raw_sl_distance, max_sl_points)
    
    # 2. Calculate max risk allowed based on REMAINING drawdown
    # Leave 10% buffer or min $100
    if portfolio_value is not None and highest_balance is not None:
        current_dd = highest_balance - portfolio_value
        remaining_dd = params.trailing_dd_limit - current_dd
        # Allow max risk = remaining_dd * 0.9 (keep 10% buffer)
        max_risk_allowed = jnp.maximum(0.0, remaining_dd * 0.9)
        # Cap at global max per trade ($500)
        max_risk_allowed = jnp.minimum(max_risk_allowed, 500.0)
    else:
        max_risk_allowed = 500.0

    # 3. Calculate safe size
    sl_distance_dollars = effective_sl_distance * params.contract_size
    apex_safe_size = max_risk_allowed / (sl_distance_dollars + 1e-8)
    
    # 4. Determine final size
    risk_based_size = (balance * params.risk_per_trade) / (sl_distance_dollars + 1e-8)
    size = jnp.minimum(risk_based_size, apex_safe_size)
    
    # FIX C (2025-12-10 v2): Extra DD scaling (keep this for additional safety)
    if portfolio_value is not None and highest_balance is not None:
        current_dd = highest_balance - portfolio_value
        dd_ratio = current_dd / params.trailing_dd_limit
        dd_ratio = jnp.clip(dd_ratio, 0.0, 1.0)
        
        # Scale: 1.0 at 0% DD, linearly down to 0.2 at 80%+ DD
        dd_scale = jnp.where(
            dd_ratio >= 0.8,
            0.2,  # Minimum 20% position when near limit
            1.0 - dd_ratio * 0.9
        )
        size = size * dd_scale
    
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
    # FIX (2025-12-12): Use state.position_size for dynamic sizing
    pnl = jnp.where(
        state.position == 1,
        (current_price - state.entry_price) * params.contract_size * state.position_size,
        jnp.where(
            state.position == -1,
            (state.entry_price - current_price) * params.contract_size * state.position_size,
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


def get_curriculum_commission_phase2(params: EnvParamsPhase2) -> jnp.ndarray:
    """
    Commission curriculum for Phase 2 (ported from Phase 1).
    
    Linearly interpolates commission over training:
    - First 25%: Ultra-low commission ($0.25) for easy learning
    - Remaining 75%: Ramps from $1.00 to $2.50
    
    Returns:
        Commission value based on training progress
    """
    progress = params.training_progress
    
    # Case 1: First 25% - ultra-low commission
    ultra_low_commission = params.initial_commission * 0.25  # $0.25 per side
    
    # Case 2: Remaining 75% - ramping commission
    adjusted_progress = (progress - 0.25) / 0.75
    ramp_progress = jnp.minimum(1.0, adjusted_progress * 2.0)
    ramped_commission = params.initial_commission + \
                       (params.final_commission - params.initial_commission) * ramp_progress
    
    # Select based on progress using jnp.where (JAX-compatible)
    curriculum_commission = jnp.where(
        progress < 0.25,
        ultra_low_commission,
        ramped_commission
    )
    
    # Select based on curriculum flag
    return jnp.where(
        params.commission_curriculum,
        curriculum_commission,
        params.commission
    )


def calculate_reward_phase2(
    trade_pnl: jnp.ndarray,
    portfolio_value: jnp.ndarray,
    position_closed: jnp.ndarray,
    closed_pnl: jnp.ndarray,
    opening_any: jnp.ndarray,
    moving_to_be: jnp.ndarray,
    enabling_trail: jnp.ndarray,
    disabling_trail: jnp.ndarray,
    unrealized_pnl: jnp.ndarray,
    dd_violation: jnp.ndarray,
    time_violation: jnp.ndarray,
    had_trailing: jnp.ndarray,
    params: EnvParamsPhase2
) -> jnp.ndarray:
    """
    Phase 2 reward function with exploration bonus curriculum.
    
    Ported from Phase 1 to solve HOLD trap issue.
    
    Exploration Bonus (for trading):
    - Entry actions (BUY/SELL): $300 bonus, decaying over first 30% of training
    - PM actions (MOVE_SL_TO_BE, ENABLE_TRAIL): $200 bonus
    - This encourages the agent to try trading early in training
    
    Regular Rewards:
    - Trade completion: +2.0 win, -1.0 loss
    - Trailing usage bonus: +0.5 if won with trailing
    - Portfolio signal: 20x scaled percentage return
    - Violation penalty: -10.0 for Apex violations
    """
    # ===== EXPLORATION BONUS CURRICULUM (CRITICAL FIX) =====
    # This solves the HOLD trap by incentivizing trades throughout training
    # 
    # FIX (2025-12-10): DD-Aware exploration bonus
    # Problem: If model hits DD often, it learns HOLD is safest → new HOLD trap
    # Solution: Keep bonus active (20% floor) + boost when in drawdown
    exploration_horizon = params.total_training_timesteps * 1.0  # 100% of training
    
    # Handle None case for timestep (JAX-compatible)
    current_ts = jnp.array(params.current_global_timestep)
    exploration_progress = current_ts / exploration_horizon
    
    # Standard decay based on training progress
    base_decay = jnp.maximum(0.0, 1.0 - exploration_progress)
    
    # DD-aware boost: increase bonus when near DD limit
    # Encourages trading out of difficult situations instead of HOLDing
    safe_initial = jnp.maximum(params.initial_balance, 1.0)
    dd_from_initial = (safe_initial - portfolio_value) / params.trailing_dd_limit
    dd_ratio = jnp.clip(dd_from_initial, 0.0, 1.5)  # 0 = no DD, 1 = at limit
    
    # DD boost kicks in at 30% drawdown, scales with severity
    # At 50% DD: boost = 0.25, at 100% DD: boost = 0.5
    dd_boost = jnp.where(dd_ratio > 0.3, (dd_ratio - 0.3) * 0.7, 0.0)
    dd_boost = jnp.clip(dd_boost, 0.0, 0.5)
    
    # Final multiplier: at least 20% floor + DD boost
    # This ensures exploration never fully stops, even late in training
    exploration_mult = jnp.maximum(base_decay, 0.2) + dd_boost
    exploration_mult = jnp.clip(exploration_mult, 0.2, 1.0)  # Cap at 100%
    
    # Entry bonus: configurable (default $300), with DD-aware multiplier
    base_entry_bonus = params.entry_action_exploration_bonus
    entry_bonus = base_entry_bonus * exploration_mult
    scaled_entry_bonus = entry_bonus / 100.0  # Scale for reward
    
    # PM action bonus: configurable (default $400 - HIGHER than entries!)
    # PM actions are completely novel - agent never experienced them in Phase 1
    base_pm_bonus = params.pm_action_exploration_bonus
    pm_bonus = base_pm_bonus * exploration_mult
    scaled_pm_bonus = pm_bonus / 100.0
    
    # Apply exploration bonuses
    exploration_reward = jnp.where(opening_any, scaled_entry_bonus, 0.0)
    exploration_reward = exploration_reward + jnp.where(moving_to_be, scaled_pm_bonus, 0.0)
    exploration_reward = exploration_reward + jnp.where(enabling_trail, scaled_pm_bonus, 0.0)
    
    # ===== POSITION MANAGEMENT OUTCOME FEEDBACK =====
    pm_reward = jnp.array(0.0)
    
    # Enable Trail Feedback
    profit_threshold = params.initial_balance * 0.01
    is_good_trail = unrealized_pnl > profit_threshold
    pm_reward = pm_reward + jnp.where(enabling_trail & is_good_trail, 0.1, 0.0)
    pm_reward = pm_reward + jnp.where(enabling_trail & ~is_good_trail, -0.1, 0.0)
    
    # ===== TRADE COMPLETION REWARD =====
    trade_reward = jnp.array(0.0)
    trade_reward = trade_reward + jnp.where(position_closed & (closed_pnl > 0), 2.0, 0.0)
    trade_reward = trade_reward + jnp.where(position_closed & (closed_pnl <= 0), -1.0, 0.0)
    
    # Bonus: Successful trailing usage
    trade_reward = trade_reward + jnp.where(position_closed & (closed_pnl > 0) & had_trailing, 0.5, 0.0)
    
    # ===== PORTFOLIO VALUE SIGNAL =====
    pnl_pct = (portfolio_value - params.initial_balance) / params.initial_balance
    portfolio_reward = pnl_pct * 20.0
    
    # ===== PENALTIES =====
    # FIX (2025-12-10 v2): STRONGER graduated drawdown penalty for Apex compliance
    # Start penalizing when drawdown exceeds 30% of limit, increasing to -15 at 100%
    # This is 3x stronger than the previous -5 penalty
    safe_balance = jnp.maximum(params.initial_balance, 1.0)
    dd_from_start = (safe_balance - portfolio_value) / params.trailing_dd_limit
    dd_ratio = jnp.clip(dd_from_start, 0.0, 1.5)  # Ratio of drawdown to limit
    
    # Graduated penalty: 0 below 30%, scales to -15 at 100%
    penalty_threshold = 0.3  # Start penalty earlier (was 0.5)
    graduated_penalty = jnp.where(
        dd_ratio > penalty_threshold,
        -15.0 * (dd_ratio - penalty_threshold) / (1.0 - penalty_threshold),  # -15 max (was -5)
        0.0
    )
    graduated_penalty = jnp.clip(graduated_penalty, -15.0, 0.0)
    
    # Add cliff penalty on actual violation (still -10 for hard violation)
    violation_penalty = jnp.where(dd_violation | time_violation, -10.0, graduated_penalty)
    
    # ===== COMBINE ALL COMPONENTS =====
    reward = exploration_reward + pm_reward + trade_reward + portfolio_reward + violation_penalty
    
    # ===== REWARD NORMALIZATION (2025-12-12) =====
    # Normalize rewards: 12 micros @ 1/12 each = 1.0 total (same as 1 emini)
    # This ensures consistent learning signal regardless of position sizing limits
    normalization_factor = 1.0 / params.max_position_size
    reward = reward * normalization_factor
    
    # Clip for stability (adjusted for normalization)
    reward = jnp.clip(reward, -15.0 * normalization_factor, 10.0 * normalization_factor)
    
    return reward


def reset_phase2(
    key: jax.random.PRNGKey,
    params: EnvParamsPhase2,
    data: MarketData
) -> Tuple[jnp.ndarray, EnvStatePhase2]:
    """
    Reset Phase 2 environment to initial state with RTH-aligned start.
    
    NEW: Supports forced position initialization for Phase 2A curriculum.
    When forced_position_ratio > 0, some episodes start with a synthetic
    pre-existing position to guarantee position management experiences.
    """
    # Split keys for different random operations
    key, key_start, key_force, key_direction, key_profit = jax.random.split(key, 5)
    
    # Sample episode start from pre-computed RTH indices
    num_rth_indices = data.rth_indices.shape[0]
    idx_choice = jax.random.randint(key_start, shape=(), minval=0, maxval=num_rth_indices)
    episode_start = jnp.take(data.rth_indices, idx_choice)
    
    # Determine if this episode starts with a forced position
    should_force_position = jax.random.uniform(key_force) < params.forced_position_ratio
    
    # Get current market data for forced position
    current_price = data.prices[episode_start, 3]
    current_atr = data.atr[episode_start]
    safe_atr = jnp.where(current_atr > 0, current_atr, current_price * 0.01)
    
    # Random direction: 1 (long) or -1 (short)
    direction = jnp.where(
        jax.random.uniform(key_direction) > 0.5,
        jnp.array(1, dtype=jnp.int32),
        jnp.array(-1, dtype=jnp.int32)
    )
    
    # Create synthetic entry price (simulate entering 5-20 bars ago)
    # Entry is slightly offset from current price to create unrealized P/L
    profit_offset = (jax.random.uniform(key_profit) * 2 - 1) * safe_atr * params.forced_position_profit_range
    # For long: if profitable, entry was below current price
    # For short: if profitable, entry was above current price
    forced_entry = jnp.where(
        direction == 1,
        current_price - profit_offset,  # Long: entry below current = profit
        current_price + profit_offset   # Short: entry above current = profit
    )
    
    # Calculate SL/TP for forced position
    # FIX (2025-12-10): Cap SL distance for Apex compliance (same as calculate_sl_tp_phase2)
    raw_sl_distance = safe_atr * params.sl_atr_mult
    max_sl_distance_points = 500.0 / params.contract_size  # NQ: 25pts, ES: 10pts
    sl_distance = jnp.minimum(raw_sl_distance, max_sl_distance_points)
    tp_distance = sl_distance * params.tp_sl_ratio
    
    forced_sl = jnp.where(
        direction == 1,
        forced_entry - sl_distance,  # Long: SL below entry
        forced_entry + sl_distance   # Short: SL above entry
    )
    forced_tp = jnp.where(
        direction == 1,
        forced_entry + tp_distance,  # Long: TP above entry
        forced_entry - tp_distance   # Short: TP below entry
    )
    
    # Final position values (use forced or flat based on should_force_position)
    final_position = jnp.where(should_force_position, direction, jnp.array(0, dtype=jnp.int32))
    final_entry = jnp.where(should_force_position, forced_entry, 0.0)
    final_sl = jnp.where(should_force_position, forced_sl, 0.0)
    final_tp = jnp.where(should_force_position, forced_tp, 0.0)
    final_entry_step = jnp.where(
        should_force_position, 
        episode_start - 10,  # Simulate entered 10 bars ago
        jnp.array(0, dtype=jnp.int32)
    )
    
    state = EnvStatePhase2(
        step_idx=episode_start,
        position=final_position,
        entry_price=jnp.array(final_entry, dtype=jnp.float32),
        sl_price=jnp.array(final_sl, dtype=jnp.float32),
        tp_price=jnp.array(final_tp, dtype=jnp.float32),
        position_entry_step=jnp.array(final_entry_step, dtype=jnp.int32),
        balance=jnp.array(params.initial_balance, dtype=jnp.float32),
        highest_balance=jnp.array(params.initial_balance, dtype=jnp.float32),
        trailing_dd_level=jnp.array(params.initial_balance - params.trailing_dd_limit, dtype=jnp.float32),
        num_trades=jnp.where(should_force_position, jnp.array(1, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)),
        winning_trades=jnp.array(0, dtype=jnp.int32),
        losing_trades=jnp.array(0, dtype=jnp.int32),
        total_pnl=jnp.array(0.0, dtype=jnp.float32),
        episode_start_idx=episode_start,
        # Phase 2 additions
        trailing_stop_active=jnp.array(False, dtype=jnp.bool_),
        highest_profit_point=jnp.array(0.0, dtype=jnp.float32),
        be_move_count=jnp.array(0, dtype=jnp.int32),
        original_sl_price=jnp.array(final_sl, dtype=jnp.float32),
        trail_activation_price=jnp.array(0.0, dtype=jnp.float32),
        # NEW (2025-12-12): Dynamic position sizing - initialize with default
        position_size=jnp.array(params.position_size, dtype=jnp.float32),
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
    
    # Calculate PnL with curriculum commission
    # FIX (2025-12-12): Use state.position_size for dynamic sizing
    curriculum_comm = get_curriculum_commission_phase2(params)
    pnl_long = (exit_price - state.entry_price) * params.contract_size * state.position_size
    pnl_short = (state.entry_price - exit_price) * params.contract_size * state.position_size
    closed_pnl = jnp.where(state.position == 1, pnl_long, pnl_short) - curriculum_comm * 2 * state.position_size
    
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
    # FIX (2025-12-12): Use state.position_size for dynamic sizing
    forced_pnl_long = (forced_exit_price - entry_after) * params.contract_size * state.position_size
    forced_pnl_short = (entry_after - forced_exit_price) * params.contract_size * state.position_size
    forced_pnl = jnp.where(position_after == 1, forced_pnl_long, forced_pnl_short) - curriculum_comm * 2 * state.position_size
    
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
    
    # Calculate dynamic position size for new trades (with DD-based scaling)
    # Pass portfolio_value and highest_balance for Fix C (DD-based position scaling)
    current_portfolio = state.balance + calculate_unrealized_pnl(state, current_price, params)
    dynamic_size = calculate_position_size(
        current_atr, balance_after, params,
        portfolio_value=current_portfolio,
        highest_balance=state.highest_balance
    )
    
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
    # FIX (2025-12-12): Use state.position_size for existing positions
    min_profit_for_be = current_atr * params.be_min_profit_atr * params.contract_size * state.position_size
    can_move_be = has_position & (unrealized_pnl > min_profit_for_be) & (be_count_after == 0)
    moving_to_be = can_move_be & (effective_action == ACTION_MOVE_SL_TO_BE)
    be_sl = state.entry_price  # Move SL to entry price
    
    # --- Action 4: ENABLE TRAILING (validated) ---
    enabling_trail = has_position & ~trailing_after & (effective_action == ACTION_ENABLE_TRAIL)
    
    # --- Action 5: DISABLE TRAILING (validated) ---
    disabling_trail = has_position & trailing_after & (effective_action == ACTION_DISABLE_TRAIL)
    
    # Store the position size used for this trade
    # FIX (2025-12-12): Use dynamic_size for new trades, preserve state.position_size for existing
    new_position_size = jnp.where(opening_any, dynamic_size, state.position_size)
    
    # Apply all updates
    new_position = jnp.where(opening_long, 1, jnp.where(opening_short, -1, position_after))
    new_entry_price = jnp.where(opening_long, buy_entry, jnp.where(opening_short, sell_entry, entry_after))
    new_sl_price = jnp.where(opening_long, buy_sl, jnp.where(opening_short, sell_sl, sl_after))
    new_sl_price = jnp.where(moving_to_be, be_sl, new_sl_price)
    new_tp_price = jnp.where(opening_long, buy_tp, jnp.where(opening_short, sell_tp, tp_after))
    # FIX (2025-12-12): Use new_position_size for entry commission
    new_balance = balance_after - jnp.where(opening_any, params.commission * new_position_size, 0.0)
    new_trailing_active = jnp.where(enabling_trail, True, jnp.where(disabling_trail, False, trailing_after))
    new_be_count = be_count_after + jnp.where(moving_to_be, 1, 0)
    new_num_trades = state.num_trades + jnp.where(opening_any, 1, 0)
    new_position_entry_step = jnp.where(opening_any, step_idx, state.position_entry_step)
    new_original_sl = jnp.where(opening_any, new_sl_price, state.original_sl_price)
    
    # =========================================================================
    # 5. UPDATE PORTFOLIO & TRAILING DRAWDOWN
    # =========================================================================
    # FIX (2025-12-12): Use new_position_size for unrealized PnL after entry
    unrealized_pnl = jnp.where(
        new_position == 1,
        (current_price - new_entry_price) * params.contract_size * new_position_size,
        jnp.where(
            new_position == -1,
            (new_entry_price - current_price) * params.contract_size * new_position_size,
            0.0
        )
    )
    
    portfolio_value = new_balance + unrealized_pnl
    
    # FIX (2025-12-12): Apex Trailing Rule - Cap at Initial + $100
    # The trailing limit stops rising once the THRESHOLD reaches Initial + $100.
    # Threshold = High Water Mark - 2500.
    # So if HWM > Initial + 2600, limit is capped?
    # Actually, Apex rule: "The Trailing Threshold stops at Initial + 100."
    # So if Initial=50k, Threshold stops at 50,100.
    # Our logic uses `trailing_dd_limit` as the DISTANCE ($2500).
    # So `new_trailing_dd` here represents the THRESHOLD PRICE.
    
    uncapped_threshold = jnp.maximum(state.highest_balance, portfolio_value) - params.trailing_dd_limit
    cap_threshold = params.initial_balance + 100.0
    new_trailing_dd = jnp.minimum(uncapped_threshold, cap_threshold)
    
    # We must also update `new_highest` to be consistent with the threshold for next step?
    # Ideally `highest_balance` determines the threshold.
    # If we cap the threshold, `highest_balance` effectively stops mattering for the threshold.
    # But we keep tracking `new_highest` for metrics.
    new_highest = jnp.maximum(state.highest_balance, portfolio_value)
    
    # =========================================================================
    # 5a. AUTO-LIQUIDATION (Safety Guardrail)
    # =========================================================================
    # If we are dangerously close to the limit (e.g. > 90% utilized), FORCE CLOSE.
    # Drawdown Amount = Highest - Portfolio (Wait, this is wrong if capped).
    # Drawdown Proximity = Portfolio - Threshold.
    # If Portfolio < Threshold + $250, we are in danger area.
    
    # If we are dangerously close to the limit (e.g. > 90% utilized), FORCE CLOSE.
    # Drawdown Amount = Highest - Portfolio (Wait, this is wrong if capped).
    # Drawdown Proximity = Portfolio - Threshold.
    # If Portfolio < Threshold + $1000, we are in danger area.
    # NQ Volatility: 50 points = $1000. A single bar can move 50 points.
    
    dist_to_fail = portfolio_value - new_trailing_dd
    auto_liquidate = (new_position != 0) & (dist_to_fail < 1000.0)  # Aggressive $1000 buffer
    
    # Apply Auto-Liquidation (Force Close at current prices)
    # ... (rest of logic same) ...
    
    # ...
    
    # ...
    
    # Apply Auto-Liquidation (Force Close at current prices)
    # Similar to forced_close RTH logic
    liq_exit_price = jnp.where(
        new_position == 1,
        current_price - slippage, # Market sell
        current_price + slippage  # Market buy
    )
    liq_pnl_long = (liq_exit_price - new_entry_price) * params.contract_size * new_position_size
    liq_pnl_short = (new_entry_price - liq_exit_price) * params.contract_size * new_position_size
    liq_pnl = jnp.where(new_position == 1, liq_pnl_long, liq_pnl_short) - curriculum_comm * 2 * new_position_size
    
    # Update state if liquidated
    trade_pnl = trade_pnl + jnp.where(auto_liquidate, liq_pnl, 0.0)
    new_balance = new_balance + jnp.where(auto_liquidate, liq_pnl, 0.0)
    new_position = jnp.where(auto_liquidate, 0, new_position)
    portfolio_value = jnp.where(auto_liquidate, new_balance, portfolio_value) # Recalc portfolio
    
    # Re-calculate Threshold after liquidation (balance changed)
    # Actually threshold depends on HWM which doesn't shrink. So threshold stays same.
    
    # Update highest profit point
    new_highest_profit = jnp.where(
        new_position != 0,
        jnp.maximum(state.highest_profit_point, unrealized_pnl),
        0.0
    )
    
    # =========================================================================
    # 6. CHECK TERMINATION
    # =========================================================================
    # FIX (2025-12-10 v2): Intra-bar drawdown check using worst-case price
    # CAPPED AT SL LEVEL - we can't lose more than SL distance!
    # For long: worst case is MAX(bar_low, sl_price) - SL exits before going lower
    # For short: worst case is MIN(bar_high, sl_price) - SL exits before going higher
    intra_low = data.low_s[step_idx]   # Intra-bar low
    intra_high = data.high_s[step_idx]  # Intra-bar high
    
    # FIX: Cap worst price at SL level - position would exit at SL, not raw bar price
    # For LONG: if low goes below SL, we'd exit at SL not at low
    worst_price_long = jnp.maximum(intra_low, new_sl_price)
    # For SHORT: if high goes above SL, we'd exit at SL not at high  
    worst_price_short = jnp.minimum(intra_high, new_sl_price)
    
    # Calculate worst-case PnL using SL-capped prices
    # FIX (2025-12-12): Use new_position_size for dynamic sizing
    worst_pnl_long = (worst_price_long - new_entry_price) * params.contract_size * new_position_size
    worst_pnl_short = (new_entry_price - worst_price_short) * params.contract_size * new_position_size
    worst_unrealized = jnp.where(
        new_position == 1, worst_pnl_long,
        jnp.where(new_position == -1, worst_pnl_short, 0.0)
    )
    worst_case_equity = new_balance + worst_unrealized
    
    # Intra-bar DD violation: would hit limit at some point during this bar
    intra_bar_dd_violation = (new_position != 0) & (worst_case_equity < new_trailing_dd)
    
    
    # Standard end-of-bar check
    dd_violation = (portfolio_value < new_trailing_dd) | intra_bar_dd_violation
    end_of_data = step_idx >= data_length - 2
    time_violation = forced_close
    
    done = dd_violation | end_of_data | time_violation
    
    # =========================================================================
    # 7. CALCULATE REWARD (with Exploration Bonus Curriculum)
    # =========================================================================
    # Use new reward function with exploration bonus to solve HOLD trap
    had_trailing = state.trailing_stop_active | enabling_trail
    
    reward = calculate_reward_phase2(
        trade_pnl=trade_pnl,
        portfolio_value=portfolio_value,
        position_closed=position_closed,
        closed_pnl=closed_pnl,
        opening_any=opening_any,
        moving_to_be=moving_to_be,
        enabling_trail=enabling_trail,
        disabling_trail=disabling_trail,
        unrealized_pnl=unrealized_pnl,
        dd_violation=dd_violation,
        time_violation=time_violation,
        had_trailing=had_trailing,
        params=params
    )
    
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
        # NEW (2025-12-12): Dynamic position sizing
        position_size=new_position_size,
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
        'apex_margin': dist_to_fail,  # NEW: Distance to Trailing Floor
        # NEW (2025-12-10): Termination reason flags for debugging
        'step_idx': step_idx,
        'dd_violation': dd_violation,
        'intra_bar_dd_violation': intra_bar_dd_violation,
        'end_of_data': end_of_data,
        'time_violation': time_violation,
        'worst_case_equity': worst_case_equity,
        'trailing_dd_level': new_trailing_dd,
        'current_price': current_price,
        'position': new_position,
        'entry_price': new_entry_price,
    }
    
    return obs, new_state, reward, done, info


# =============================================================================
# Vectorized Wrappers
# =============================================================================

def batch_reset_phase2(
    keys: jax.random.PRNGKey,
    params: EnvParamsPhase2,
    num_envs: int,
    data: MarketData
) -> Tuple[jnp.ndarray, EnvStatePhase2]:
    """Reset multiple Phase 2 environments in parallel."""
    return jax.vmap(reset_phase2, in_axes=(0, None, None))(keys, params, data)


# MEMORY FIX (2025-12-08): Removed static_argnums - params changes each update
@jax.jit
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


# MEMORY FIX (2025-12-08): Removed static_argnums - params changes each update
@jax.jit
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
    obs_batch, state_batch = batch_reset_phase2(
        jax.random.split(key, num_envs),
        params,
        num_envs,
        dummy_data
    )
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
