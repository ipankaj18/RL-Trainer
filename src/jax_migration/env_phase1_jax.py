"""
JAX Trading Environment - Phase 1

Pure JAX implementation following Gymnax patterns for GPU-resident training.
All operations are JIT-compilable with no Python control flow in the hot path.

Key Design Principles:
1. Immutable state (EnvState dataclass)
2. Pure functions (step/reset take and return state)
3. All control flow via jnp.where()
4. Pre-computed data arrays (no Pandas)
5. vmap-ready for 10k+ parallel environments
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, NamedTuple, Any
from functools import partial
import chex

from .data_loader import MarketData


# =============================================================================
# Environment State & Parameters (Immutable Dataclasses)
# =============================================================================

class EnvState(NamedTuple):
    """Immutable environment state for JAX training."""
    # Position tracking
    step_idx: jnp.ndarray        # Current timestep in data (scalar int32)
    position: jnp.ndarray        # -1=short, 0=flat, 1=long (scalar int32)
    entry_price: jnp.ndarray     # Entry price (scalar float32)
    sl_price: jnp.ndarray        # Stop loss price (scalar float32)
    tp_price: jnp.ndarray        # Take profit price (scalar float32)
    position_entry_step: jnp.ndarray  # Step when position was opened (scalar int32)
    
    # Portfolio tracking
    balance: jnp.ndarray         # Current cash balance (scalar float32)
    highest_balance: jnp.ndarray # High-water mark for trailing DD (scalar float32)
    trailing_dd_level: jnp.ndarray  # Current drawdown limit (scalar float32)
    
    # Trade statistics (for reward computation)
    num_trades: jnp.ndarray      # Total trades (scalar int32)
    winning_trades: jnp.ndarray  # Winning trades (scalar int32)
    losing_trades: jnp.ndarray   # Losing trades (scalar int32)
    total_pnl: jnp.ndarray       # Cumulative PnL (scalar float32)
    
    # Episode tracking
    episode_start_idx: jnp.ndarray  # Start index for this episode (scalar int32)


class EnvParams(NamedTuple):
    """Environment parameters (can vary per environment instance)."""
    # Market specifications
    contract_size: float = 50.0      # ES = $50 per point
    tick_size: float = 0.25          # ES tick size
    tick_value: float = 12.50        # ES tick value
    commission: float = 2.50         # Per side commission
    slippage_ticks: int = 1          # Slippage in ticks

    # Commission curriculum (NEW - Phase 1A improvement)
    initial_commission: float = 1.0   # Start with reduced commission
    final_commission: float = 2.5     # End with realistic commission
    commission_curriculum: bool = True  # Enable commission ramping

    # Position management
    initial_balance: float = 50000.0
    sl_atr_mult: float = 1.5         # SL distance in ATR multiples
    tp_sl_ratio: float = 3.0         # TP as multiple of SL distance
    position_size: float = 1.0       # Number of contracts
    trailing_dd_limit: float = 15000.0  # Phase 1: relaxed ($15k)

    # Observation parameters
    window_size: int = 20            # Lookback window for observations
    num_features: int = 11           # Features per timestep (8 market + 3 time)

    # Time rules (hours as decimals, ET)
    rth_open: float = 9.5            # 9:30 AM
    rth_close: float = 16.983        # 4:59 PM
    rth_start_count: int = 0         # Number of valid RTH start indices (static, set by caller)

    # Episode parameters
    min_episode_bars: int = 1500

    # Training curriculum (NEW - Phase A1)
    training_progress: float = 0.0  # 0.0 to 1.0, updated each rollout for commission curriculum

    # Exploration bonus curriculum (NEW - 2025-12-03: Phase 1B)
    # Solves HOLD trap by incentivizing early trade exploration
    current_global_timestep: float = 0.0    # Global timestep counter (updated by training script)
    total_training_timesteps: float = 20_000_000  # Total timesteps for exploration bonus decay

    # Reward function parameters (SIMPLIFIED - aligned with PyTorch)
    # NOTE: Simplified reward function only uses these basic values
    # Removed: entry_bonus, readiness_bonus, tp_bonus, sl_penalty, exit_bonus
    hold_penalty: float = -0.01        # Matches PyTorch (was -0.02)
    pnl_divisor: float = 100.0         # Matches PyTorch (was 50.0)



# =============================================================================
# Action Constants
# =============================================================================

ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2


# =============================================================================
# Core Environment Functions (Pure JAX, JIT-compilable)
# =============================================================================

def get_observation(
    state: EnvState,
    data: MarketData,
    params: EnvParams
) -> jnp.ndarray:
    """
    Get current observation window.
    
    Returns shape (window_size * num_features + 5,) = (225,)
    All operations are pure tensor ops, no Python indexing.
    """
    step = jnp.asarray(state.step_idx, dtype=jnp.int32)
    
    # CRITICAL: window MUST be Python int for lax.dynamic_slice_in_dim
    # JAX requires slice sizes to be static (Python ints), not traced values
    window = int(params.window_size)  # Python int, NOT jnp.asarray!
    
    data_len = data.features.shape[0]  # Python int from shape tuple

    # Clamp step to data bounds to avoid traced boolean checks during slicing
    step = jnp.minimum(step, data_len - 1)

    # Compute start index (clamp to valid range) and ensure scalar
    start_idx = jnp.clip(step - window, 0, jnp.maximum(data_len - window, 0))
    start_idx = jnp.squeeze(start_idx)

    # Get market features: (window, 8)
    market_obs = lax.dynamic_slice_in_dim(
        data.features,
        start_idx,
        window,  # Python int
        axis=0
    )
    
    # Get time features: (window, 3)
    time_obs = lax.dynamic_slice_in_dim(
        data.time_features,
        start_idx,
        window,  # Python int
        axis=0
    )
    
    # Combine and flatten: (window * 11,)
    combined = jnp.concatenate([market_obs, time_obs], axis=1)
    flat_obs = combined.flatten()
    
    # Get current price and ATR for position features
    current_price = data.prices[step, 3]  # Close price
    current_atr = data.atr[step]
    
    # Handle edge cases for ATR
    safe_atr = jnp.where(current_atr > 0, current_atr, current_price * 0.01)
    safe_price = jnp.where(current_price > 0, current_price, 1.0)
    
    # Position-aware features (5 dims)
    has_position = state.position != 0
    
    position_features = jnp.array([
        state.position.astype(jnp.float32),  # Position type
        jnp.where(has_position, state.entry_price / safe_price, 1.0),  # Entry ratio
        jnp.where(has_position, jnp.abs(state.sl_price - safe_price) / safe_atr, 0.0),  # SL dist
        jnp.where(has_position, jnp.abs(state.tp_price - safe_price) / safe_atr, 0.0),  # TP dist
        jnp.where(
            has_position, 
            (step - state.position_entry_step).astype(jnp.float32) / 390.0, 
            0.0
        ),  # Time in position
    ], dtype=jnp.float32)
    
    # Combine all features
    obs = jnp.concatenate([flat_obs, position_features])
    
    # Handle NaN/Inf
    obs = jnp.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return obs.astype(jnp.float32)


def calculate_sl_tp(
    entry_price: jnp.ndarray,
    position_type: jnp.ndarray,
    atr: jnp.ndarray,
    params: EnvParams
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate stop loss and take profit prices.
    Pure tensor ops, no branching.
    """
    # Handle invalid ATR
    safe_atr = jnp.where(atr > 0, atr, entry_price * 0.01)
    
    sl_distance = safe_atr * params.sl_atr_mult
    tp_distance = sl_distance * params.tp_sl_ratio
    
    # Long position: SL below, TP above
    sl_long = entry_price - sl_distance
    tp_long = entry_price + tp_distance
    
    # Short position: SL above, TP below
    sl_short = entry_price + sl_distance
    tp_short = entry_price - tp_distance
    
    # Select based on position type
    sl_price = jnp.where(position_type == 1, sl_long, sl_short)
    tp_price = jnp.where(position_type == 1, tp_long, tp_short)
    
    return sl_price, tp_price


def check_sl_tp_hit(
    state: EnvState,
    high: jnp.ndarray,
    low: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Check if stop loss or take profit was hit.
    Returns (sl_hit, tp_hit, exit_price) as boolean/float arrays.
    """
    has_position = state.position != 0
    is_long = state.position == 1
    
    # Long: SL hit if low <= sl_price, TP hit if high >= tp_price
    sl_hit_long = is_long & (low <= state.sl_price)
    tp_hit_long = is_long & (high >= state.tp_price)
    
    # Short: SL hit if high >= sl_price, TP hit if low <= tp_price
    sl_hit_short = (~is_long) & has_position & (high >= state.sl_price)
    tp_hit_short = (~is_long) & has_position & (low <= state.tp_price)
    
    sl_hit = sl_hit_long | sl_hit_short
    tp_hit = tp_hit_long | tp_hit_short
    
    # Determine exit price (SL price if SL hit, TP price if TP hit)
    exit_price = jnp.where(
        sl_hit, 
        state.sl_price,
        jnp.where(tp_hit, state.tp_price, 0.0)
    )
    
    return sl_hit, tp_hit, exit_price


def get_curriculum_commission(params: EnvParams, progress: float) -> float:
    """
    Linearly interpolate commission from initial to final over training.

    Args:
        params: Environment parameters
        progress: Training progress from 0.0 (start) to 1.0 (end)

    Returns:
        Commission value ramped over training

    NOTE: First 25% uses ultra-low commission ($0.25), then ramps $1.00 → $2.50.
    This gives the agent very easy initial conditions to learn profitable patterns.
    
    CRITICAL FIX (2025-12-02): Added ultra-low commission period
    CRITICAL FIX (2025-12-03): Converted to JAX-compatible functional code (no Python if statements)
    """
    # Compute all possible commission values
    base_commission = params.commission
    
    # Case 1: First 25% - ultra-low commission
    ultra_low_commission = params.initial_commission * 0.25  # $0.25 per side
    
    # Case 2: Remaining 75% - ramping commission  
    adjusted_progress = (progress - 0.25) / 0.75  # Scale to 0-1 range
    ramp_progress = jnp.minimum(1.0, adjusted_progress * 2.0)  # Ramp over first half of this period
    ramped_commission = params.initial_commission + \
                       (params.final_commission - params.initial_commission) * ramp_progress
    
    # Select based on progress using jnp.where (JAX-compatible)
    curriculum_commission = jnp.where(
        progress < 0.25,
        ultra_low_commission,
        ramped_commission
    )
    
    # Select based on curriculum flag using jnp.where (JAX-compatible)
    return jnp.where(
        params.commission_curriculum,
        curriculum_commission,
        base_commission
    )


def calculate_pnl(
    entry_price: jnp.ndarray,
    exit_price: jnp.ndarray,
    position_type: jnp.ndarray,
    params: EnvParams,
    training_progress: float = 1.0  # NEW parameter for curriculum
) -> jnp.ndarray:
    """
    Calculate PnL for a closed position with commission curriculum support.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        position_type: 1 for long, -1 for short
        params: Environment parameters
        training_progress: 0.0 to 1.0, for commission ramping (default 1.0 = full commission)

    Returns:
        Net PnL after commissions
    """
    # Long: (exit - entry) * contract_size * position_size
    pnl_long = (exit_price - entry_price) * params.contract_size * params.position_size

    # Short: (entry - exit) * contract_size * position_size
    pnl_short = (entry_price - exit_price) * params.contract_size * params.position_size

    pnl = jnp.where(position_type == 1, pnl_long, pnl_short)

    # Use curriculum commission (ramps from $1.00 to $2.50 over first 50% of training)
    commission = get_curriculum_commission(params, training_progress)
    commission_cost = commission * 2 * params.position_size

    return pnl - commission_cost


def calculate_reward(
    trade_pnl: jnp.ndarray,
    exit_type: jnp.ndarray,  # 0=none, 1=SL, 2=TP
    current_position: jnp.ndarray,
    opened_new_position: jnp.ndarray = None,  # NEW: True if BUY/SELL action just taken
    current_timestep: jnp.ndarray = None,     # NEW: Global timestep counter
    total_timesteps: float = 20_000_000       # NEW: Total training timesteps
) -> jnp.ndarray:
    """
    Phase 1 reward with exploration bonus curriculum.
    
    ALIGNED WITH PyTorch environment_phase1.py (lines 28-64)
    
    Reward components:
    - Hold penalty: -0.01 when in position (discourages holding too long)
    - Patience reward: +0.001 when flat AFTER exploration (encourages selectivity)
    - Trade PnL: scaled by /100 when trade closes
    - Exploration bonus: $400 → $0 over first 40% of training (PHASE A - 2025-12-03)

    Exit types: 0=no exit, 1=stop_loss, 2=take_profit

    PHASE A HOLD TRAP SOLUTION (2025-12-03):
    - Exploration bonus: $75 → $400 (5.3x increase) to make trading more attractive
    - Exploration period: 25% → 40% (8M steps out of 20M) for longer learning
    - Patience reward: DISABLED during exploration phase (0.001 → 0.0) to remove HOLD incentive
    - Linearly decays bonus over first 8M steps (40% of 20M default)
    - Automatically transitions to pure PnL optimization after exploration phase
    
    Returns:
        Reward value (no clipping - matches PyTorch)
    """
    # Start with zero reward
    reward = jnp.zeros_like(trade_pnl)
    
    # Hold penalty: -0.01 when in position and not exiting
    in_position = current_position != 0
    no_exit = exit_type == 0
    reward = jnp.where(in_position & no_exit, -0.01, reward)
    
    # PHASE A: Conditional patience reward based on exploration progress
    # During exploration phase (first 40%), NO patience reward (prevents HOLD trap)
    # After exploration, restore patience reward (+0.001) for selectivity
    is_flat = current_position == 0

    # JAX-compatible: Use default value approach to avoid if/else with tracers
    timestep_for_calc = jnp.where(
        current_timestep is None,
        total_timesteps * 0.50,  # Default: past exploration (50% > 40%)
        current_timestep
    )
    exploration_progress = timestep_for_calc / (total_timesteps * 0.40)
    in_exploration = exploration_progress < 1.0
    patience_reward = jnp.where(in_exploration, 0.0, 0.001)
    reward = jnp.where(is_flat & no_exit, patience_reward, reward)
    
    # Trade PnL reward: scaled by 100 when trade completes
    trade_completed = exit_type != 0
    pnl_reward = trade_pnl / 100.0
    reward = jnp.where(trade_completed, pnl_reward, reward)
    
    # ===== NEW: EXPLORATION BONUS CURRICULUM =====
    # JAX-compatible: Use default values to avoid if/else with tracers
    # PHASE A: Extended to 40% of training (8M steps out of 20M default)
    exploration_horizon = total_timesteps * 0.40

    # Handle None case for timestep (default to end of training = no bonus)
    timestep_for_bonus = jnp.where(
        current_timestep is None,
        total_timesteps,  # Default: end of training (bonus = 0)
        current_timestep
    )
    exploration_progress = timestep_for_bonus / exploration_horizon

    # PHASE A: Increased from $75 to $400 (5.3x increase)
    base_bonus = 400.0
    exploration_bonus = base_bonus * jnp.maximum(0.0, 1.0 - exploration_progress)
    scaled_bonus = exploration_bonus / 100.0

    # Handle None case for opened_new_position (default to False = no bonus)
    position_opened = jnp.where(
        opened_new_position is None,
        False,  # Default: no position opened
        opened_new_position
    )
    reward = jnp.where(position_opened, reward + scaled_bonus, reward)
    
    # No clipping (matches PyTorch)
    return reward


def action_masks(state: EnvState) -> jnp.ndarray:
    """
    Return valid action mask.
    
    Phase 1 Policy:
    - When FLAT: Can HOLD, BUY, or SELL
    - When IN POSITION: Can only HOLD
    
    Returns: Boolean array shape (3,)
    """
    is_flat = state.position == 0
    
    mask = jnp.array([
        True,        # HOLD always valid
        is_flat,     # BUY only when flat
        is_flat,     # SELL only when flat
    ], dtype=jnp.bool_)
    
    return mask


def reset(
    key: jax.random.PRNGKey,
    params: EnvParams,
    data: MarketData
) -> Tuple[jnp.ndarray, EnvState]:
    """
    Reset environment to RTH-aligned initial state.
    
    Samples episode start from pre-computed RTH indices (9:30 AM - 4:00 PM ET).
    This ensures BUY/SELL actions are valid from step 1, not masked for 100+ steps.
    
    NEW: Phase A2 - RTH-aligned episode starts
    """
    # Sample from pre-computed RTH indices instead of full data range
    num_rth_starts = int(params.rth_start_count)
    if num_rth_starts <= 0:
        raise ValueError("EnvParams.rth_start_count must be > 0 (set from data.rth_indices.shape[0])")
    rth_idx = jax.random.randint(key, shape=(), minval=0, maxval=num_rth_starts)
    
    # CRITICAL FIX: Use jnp.take() instead of Python [] indexing
    # Python's [] calls __index__() on JAX tracers, which fails during JIT
    episode_start = jnp.take(data.rth_indices, rth_idx)
    
    # Initialize state
    state = EnvState(
        step_idx=episode_start,
        position=jnp.array(0, dtype=jnp.int32),
        entry_price=jnp.array(0.0, dtype=jnp.float32),
        sl_price=jnp.array(0.0, dtype=jnp.float32),
        tp_price=jnp.array(0.0, dtype=jnp.float32),
        position_entry_step=jnp.array(0, dtype=jnp.int32),
        balance=jnp.array(params.initial_balance, dtype=jnp.float32),
        highest_balance=jnp.array(params.initial_balance, dtype=jnp.float32),
        trailing_dd_level=jnp.array(
            params.initial_balance - params.trailing_dd_limit, 
            dtype=jnp.float32
        ),
        num_trades=jnp.array(0, dtype=jnp.int32),
        winning_trades=jnp.array(0, dtype=jnp.int32),
        losing_trades=jnp.array(0, dtype=jnp.int32),
        total_pnl=jnp.array(0.0, dtype=jnp.float32),
        episode_start_idx=episode_start,
    )
    
    obs = get_observation(state, data, params)
    
    return obs, state


def step(
    key: jax.random.PRNGKey,
    state: EnvState,
    action: jnp.ndarray,
    params: EnvParams,
    data: MarketData
) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, dict]:
    """
    Execute one environment step.
    
    Returns: (observation, new_state, reward, done, info)
    
    All operations are pure tensor ops using jnp.where for control flow.
    """
    step_idx = state.step_idx
    data_length = data.features.shape[0]
    
    # Get current market data
    current_price = data.prices[step_idx, 3]  # Close
    high = data.prices[step_idx, 1]
    low = data.prices[step_idx, 2]
    # Use second-level data for precise drawdown checks
    low_s = data.low_s[step_idx]
    high_s = data.high_s[step_idx]
    
    current_atr = data.atr[step_idx]
    current_hour = data.timestamps_hour[step_idx]
    
    # Slippage in points
    slippage = params.tick_size * params.slippage_ticks
    
    # Initialize tracking variables
    trade_pnl = jnp.array(0.0)
    exit_type = jnp.array(0)  # 0=none, 1=SL, 2=TP
    position_closed = jnp.array(False)
    
    # =========================================================================
    # 0. CHECK INTRA-BAR DRAWDOWN (CRITICAL)
    # =========================================================================
    # Calculate worst-case equity during the bar
    # For Long: worst case is low_s
    # For Short: worst case is high_s
    
    worst_case_pnl = jnp.where(
        state.position == 1,
        (low_s - state.entry_price) * params.contract_size * params.position_size,
        jnp.where(
            state.position == -1,
            (state.entry_price - high_s) * params.contract_size * params.position_size,
            0.0
        )
    )
    
    worst_case_equity = state.balance + worst_case_pnl
    
    # Check violation
    intra_bar_dd_violation = (state.position != 0) & (worst_case_equity < state.trailing_dd_level)
    
    # =========================================================================
    # 1. CHECK SL/TP ON EXISTING POSITION
    # =========================================================================
    sl_hit, tp_hit, exit_price = check_sl_tp_hit(state, high, low)
    
    # Calculate PnL if position closed (with commission curriculum)
    closed_pnl = calculate_pnl(
        state.entry_price, exit_price, state.position, params,
        training_progress=params.training_progress  # NEW: Phase A1
    )
    
    # Determine if position was closed by SL/TP
    position_closed = (sl_hit | tp_hit) & (state.position != 0)
    trade_pnl = jnp.where(position_closed, closed_pnl, 0.0)
    exit_type = jnp.where(tp_hit & position_closed, 2, jnp.where(sl_hit & position_closed, 1, 0))
    
    # Update state after potential close
    balance_after_close = state.balance + trade_pnl
    position_after_close = jnp.where(position_closed, 0, state.position)
    entry_after_close = jnp.where(position_closed, 0.0, state.entry_price)
    sl_after_close = jnp.where(position_closed, 0.0, state.sl_price)
    tp_after_close = jnp.where(position_closed, 0.0, state.tp_price)
    
    # Update trade counts
    is_winner = trade_pnl > 0
    winning_after = state.winning_trades + jnp.where(position_closed & is_winner, 1, 0)
    losing_after = state.losing_trades + jnp.where(position_closed & ~is_winner, 1, 0)
    
    # =========================================================================
    # 2. CHECK TIME RULES
    # =========================================================================
    past_close = current_hour >= params.rth_close
    within_rth = (current_hour >= params.rth_open) & (current_hour < params.rth_close)
    
    # Force close if past 4:59 PM and still holding
    forced_close = past_close & (position_after_close != 0)
    
    # Calculate forced close PnL
    forced_exit_price = jnp.where(
        position_after_close == 1,
        current_price - slippage,  # Long: sell at bid
        current_price + slippage   # Short: cover at ask
    )
    forced_pnl = calculate_pnl(
        entry_after_close, forced_exit_price, position_after_close, params,
        training_progress=params.training_progress  # NEW: Phase A1
    )
    
    # Apply forced close
    trade_pnl = jnp.where(forced_close, forced_pnl, trade_pnl)
    balance_after_time = balance_after_close + jnp.where(forced_close, forced_pnl, 0.0)
    position_after_time = jnp.where(forced_close, 0, position_after_close)
    
    # =========================================================================
    # 3. EXECUTE NEW ACTION (if flat and within RTH)
    # =========================================================================
    is_flat = position_after_time == 0
    can_open = is_flat & within_rth
    
    # Entry prices with slippage
    buy_entry = current_price + slippage
    sell_entry = current_price - slippage
    
    # Calculate SL/TP for new positions
    buy_sl, buy_tp = calculate_sl_tp(buy_entry, jnp.array(1), current_atr, params)
    sell_sl, sell_tp = calculate_sl_tp(sell_entry, jnp.array(-1), current_atr, params)
    
    # Commission cost for entry
    entry_commission = params.commission * params.position_size
    
    # Update state based on action
    opening_long = can_open & (action == ACTION_BUY)
    opening_short = can_open & (action == ACTION_SELL)
    opening_any = opening_long | opening_short
    
    new_position = jnp.where(opening_long, 1, jnp.where(opening_short, -1, position_after_time))
    new_entry_price = jnp.where(opening_long, buy_entry, jnp.where(opening_short, sell_entry, entry_after_close))
    new_sl_price = jnp.where(opening_long, buy_sl, jnp.where(opening_short, sell_sl, sl_after_close))
    new_tp_price = jnp.where(opening_long, buy_tp, jnp.where(opening_short, sell_tp, tp_after_close))
    new_balance = balance_after_time - jnp.where(opening_any, entry_commission, 0.0)
    new_position_entry_step = jnp.where(opening_any, step_idx, state.position_entry_step)
    new_num_trades = state.num_trades + jnp.where(opening_any, 1, 0)
    
    # =========================================================================
    # 4. UPDATE PORTFOLIO & TRAILING DRAWDOWN
    # =========================================================================
    # Calculate unrealized PnL
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
    
    # Update high-water mark and trailing DD
    new_highest = jnp.maximum(state.highest_balance, portfolio_value)
    new_trailing_dd = new_highest - params.trailing_dd_limit
    
    # =========================================================================
    # 5. CHECK TERMINATION CONDITIONS
    # =========================================================================
    # Drawdown violation (either intra-bar or end-of-step)
    dd_violation = (portfolio_value < new_trailing_dd) | intra_bar_dd_violation
    
    # End of data
    end_of_data = step_idx >= data_length - 2
    
    # Time violation (held past close)
    time_violation = forced_close
    
    done = dd_violation | end_of_data | time_violation
    
    # =========================================================================
    # 6. CALCULATE REWARD
    # =========================================================================
    time_in_position = (step_idx - state.position_entry_step).astype(jnp.float32)
    
    # Pass exploration bonus parameters to calculate_reward
    reward = calculate_reward(
        trade_pnl,
        exit_type,
        current_position=state.position,
        opened_new_position=opening_any,  # NEW: Track if BUY/SELL action was taken
        current_timestep=jnp.array(params.current_global_timestep),  # NEW: Global timestep
        total_timesteps=params.total_training_timesteps  # NEW: Total timesteps for decay
    )

    
    # Penalty for violations (ONLY when episode terminates)
    # FIXED: Apply -10.0 only on termination step, not every step (matches PyTorch)
    violation_penalty = jnp.where(done & (dd_violation | time_violation), -10.0, 0.0)
    reward = reward + violation_penalty
    
    # =========================================================================
    # 7. CREATE NEW STATE
    # =========================================================================
    new_state = EnvState(
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
    )
    
    # Get observation
    obs = get_observation(new_state, data, params)
    
    # Info dict (expanded for TrainingMetricsTracker compatibility)
    info = {
        'portfolio_value': portfolio_value,
        'final_balance': portfolio_value,  # Alias for tracker compatibility
        'num_trades': new_num_trades,
        'win_rate': winning_after / jnp.maximum(new_num_trades, 1),
        'total_pnl': state.total_pnl + trade_pnl,
        'episode_return': state.total_pnl + trade_pnl,  # Cumulative PnL
        'position_closed': position_closed,  # Boolean flag for trade completion
        'trade_pnl': trade_pnl,  # PnL from this specific trade (0 if no close)
        'forced_close': forced_close,  # Whether position was force-closed at 4:59 PM
    }
    
    return obs, new_state, reward, done, info


# =============================================================================
# Vectorized Environment Wrappers
# =============================================================================

def batch_reset(
    keys: jnp.ndarray,
    params: EnvParams,
    data: MarketData
) -> Tuple[jnp.ndarray, EnvState]:
    """
    Reset multiple environments in parallel.

    Args:
        keys: Array of shape (num_envs, 2) with per-env PRNG keys (already split)
    """
    return jax.vmap(reset, in_axes=(0, None, None))(keys, params, data)


@partial(jax.jit, static_argnums=(3,))
def batch_step(
    keys: jax.random.PRNGKey,
    states: EnvState,
    actions: jnp.ndarray,
    params: EnvParams,
    data: MarketData
) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, dict]:
    """Step multiple environments in parallel."""
    return jax.vmap(step, in_axes=(0, 0, 0, None, None))(
        keys, states, actions, params, data
    )


@jax.jit
def batch_action_masks(states: EnvState) -> jnp.ndarray:
    """Get action masks for multiple environments."""
    return jax.vmap(action_masks)(states)


# =============================================================================
# Rollout Function for Training
# =============================================================================

def rollout(
    key: jax.random.PRNGKey,
    policy_fn,  # Function: (params, obs) -> action
    policy_params,
    env_params: EnvParams,
    data: MarketData,
    num_steps: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Rollout a single episode using jax.lax.scan.
    
    Returns: (observations, actions, rewards, dones, masks)
    """
    key_reset, key_episode = jax.random.split(key)
    obs, state = reset(key_reset, env_params, data)
    
    def policy_step(carry, _):
        obs, state, key = carry
        key, key_step, key_action = jax.random.split(key, 3)
        
        # Get action from policy
        action = policy_fn(policy_params, obs)
        mask = action_masks(state)
        
        # Step environment
        next_obs, next_state, reward, done, info = step(
            key_step, state, action, env_params, data
        )
        
        # Auto-reset on done
        key_reset = jax.random.fold_in(key, state.step_idx)
        reset_obs, reset_state = reset(key_reset, env_params, data)
        
        next_obs = jnp.where(done, reset_obs, next_obs)
        next_state = jax.tree.map(
            lambda x, y: jnp.where(done, x, y),
            reset_state, next_state
        )
        
        carry = (next_obs, next_state, key)
        return carry, (obs, action, reward, done, mask)
    
    _, scan_out = jax.lax.scan(
        policy_step,
        (obs, state, key_episode),
        None,
        num_steps
    )
    
    return scan_out


# =============================================================================
# Test Functions
# =============================================================================

if __name__ == "__main__":
    print("Testing JAX Trading Environment...")
    print(f"JAX devices: {jax.devices()}")
    
    # Create dummy data for testing
    num_timesteps = 10000
    num_features = 8
    
    dummy_data = MarketData(
        features=jax.random.normal(jax.random.key(0), (num_timesteps, num_features)),
        prices=jnp.abs(jax.random.normal(jax.random.key(1), (num_timesteps, 4))) * 6000 + 100,
        atr=jnp.abs(jax.random.normal(jax.random.key(2), (num_timesteps,))) * 10 + 1,
        time_features=jax.random.uniform(jax.random.key(3), (num_timesteps, 3)),
        trading_mask=jnp.ones(num_timesteps),
        timestamps_hour=jnp.linspace(9.5, 16.9, num_timesteps),
        rth_indices=jnp.arange(60, num_timesteps - 100),  # Valid RTH start indices
        low_s=jnp.abs(jax.random.normal(jax.random.key(4), (num_timesteps,))) * 6000 + 90,
        high_s=jnp.abs(jax.random.normal(jax.random.key(5), (num_timesteps,))) * 6000 + 110,
    )
    
    params = EnvParams()
    
    # Test single reset
    key = jax.random.key(42)
    obs, state = reset(key, params, dummy_data)
    print(f"\nSingle reset:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Step index: {state.step_idx}")
    print(f"  Balance: {state.balance}")
    
    # Test single step
    action = jnp.array(1)  # BUY
    key, key_step = jax.random.split(key)
    obs, state, reward, done, info = step(key_step, state, action, params, dummy_data)
    print(f"\nAfter BUY action:")
    print(f"  Position: {state.position}")
    print(f"  Entry price: {state.entry_price}")
    print(f"  Reward: {reward}")
    
    # Test vectorized reset
    num_envs = 1000
    key = jax.random.key(0)
    obs_batch, state_batch = batch_reset(
        jax.random.split(key, num_envs),
        params,
        dummy_data
    )
    print(f"\nVectorized reset ({num_envs} envs):")
    print(f"  Observation batch shape: {obs_batch.shape}")
    print(f"  Step indices range: {state_batch.step_idx.min()} - {state_batch.step_idx.max()}")
    
    # Benchmark vectorized step
    import time
    
    actions = jax.random.randint(jax.random.key(1), (num_envs,), 0, 3)
    keys = jax.random.split(key, num_envs)
    
    # Warm-up
    _ = batch_step(keys, state_batch, actions, params, dummy_data)
    
    # Benchmark
    num_iterations = 100
    start = time.time()
    for _ in range(num_iterations):
        obs_batch, state_batch, rewards, dones, _ = batch_step(
            keys, state_batch, actions, params, dummy_data
        )
        jax.block_until_ready(obs_batch)
    elapsed = time.time() - start
    
    steps_per_sec = (num_envs * num_iterations) / elapsed
    print(f"\nBenchmark ({num_envs} envs × {num_iterations} steps):")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {steps_per_sec:,.0f} env steps/sec")
