# JAX Training Contract Sizing Analysis

**Generated**: 2025-12-12
**Analysis Focus**: Contract quantities used during JAX training (Phase 1 & Phase 2)

---

## Executive Summary

The JAX training environments currently use **1 contract per trade** for ALL markets (emini and micro). This does NOT match the target requirements where micro markets should use up to 12 contracts.

### Current State vs Target Requirements

| Market Type | Current Training | Target Requirement | Gap |
|-------------|------------------|-------------------|-----|
| **E-mini** (NQ, ES, YM, RTY) | **1 contract** | **1 contract** | ✅ CORRECT |
| **Micro** (MNQ, MES, MYM, M2K) | **1 contract** | **Max 12 contracts** | ❌ INCORRECT |

**Impact**: Micro markets are being trained with 1/12th of their potential position size, which will cause suboptimal performance when deployed with correct sizing.

---

## Detailed Analysis

### 1. Contract Size Definition (Market Specifications)

**File**: `/src/market_specs.py:43-131`

Contract multipliers are correctly defined:

```python
# E-MINI FUTURES
ES_SPEC = MarketSpecification(
    symbol='ES', contract_multiplier=50.0, ...  # $50/point
)
NQ_SPEC = MarketSpecification(
    symbol='NQ', contract_multiplier=20.0, ...  # $20/point
)
YM_SPEC = MarketSpecification(
    symbol='YM', contract_multiplier=5.0, ...   # $5/point
)
RTY_SPEC = MarketSpecification(
    symbol='RTY', contract_multiplier=50.0, ... # $50/point
)

# MICRO E-MINI FUTURES (1/10th size)
MNQ_SPEC = MarketSpecification(
    symbol='MNQ', contract_multiplier=2.0, ...  # $2/point (1/10th of NQ)
)
MES_SPEC = MarketSpecification(
    symbol='MES', contract_multiplier=5.0, ...  # $5/point (1/10th of ES)
)
M2K_SPEC = MarketSpecification(
    symbol='M2K', contract_multiplier=5.0, ...  # $5/point (1/10th of RTY)
)
MYM_SPEC = MarketSpecification(
    symbol='MYM', contract_multiplier=0.50, ... # $0.50/point (1/10th of YM)
)
```

✅ **Status**: Market multipliers are correctly defined. The issue is in position_size, not contract_size.

---

### 2. Phase 1 JAX Environment

**File**: `/src/jax_migration/env_phase1_jax.py`

#### Position Size Configuration

**Lines 54-76** - EnvParams Definition:
```python
class EnvParams(NamedTuple):
    """
    ⚠️ DEFAULTS ARE ES CONTRACT SPECS - Override for other markets!
    Use market_specs.py to get correct values for NQ, YM, RTY, etc.
    """
    # Market specifications (ES defaults)
    contract_size: float = 50.0      # ES = $50/point, NQ = $20/point
    tick_size: float = 0.25
    tick_value: float = 12.50
    commission: float = 2.50
    slippage_ticks: int = 1

    # Position management
    initial_balance: float = 50000.0
    position_size: float = 1.0       # ⚠️ Number of contracts - FIXED AT 1
    # ... other params
```

**Key Observation**: `position_size` is hardcoded to `1.0` and never adjusted based on market type.

#### PnL Calculation Usage

**Lines 341-353** - PnL with position_size:
```python
def calculate_pnl(
    entry_price, exit_price, position_type, params,
    training_progress = 1.0
):
    # Long: (exit - entry) * contract_size * position_size
    pnl_long = (exit_price - entry_price) * params.contract_size * params.position_size

    # Short: (entry - exit) * contract_size * position_size
    pnl_short = (entry_price - exit_price) * params.contract_size * params.position_size

    pnl = jnp.where(position_type == 1, pnl_long, pnl_short)

    # Commission
    commission_cost = commission * 2 * params.position_size
    return pnl - commission_cost
```

**Lines 341-353**: Used in `calculate_pnl()` - PnL calculation
**Lines 563-570**: Used in intra-bar drawdown calculation
**Lines 644**: Used in entry commission calculation
**Lines 665-670**: Used in unrealized PnL calculation

#### Training Script Initialization

**File**: `/src/jax_migration/train_ppo_jax_fixed.py:1070-1085`

```python
market_spec = get_market_spec(args.market)
if market_spec:
    print(f"Using {args.market} contract specs: ${market_spec.contract_multiplier}/point")
    env_params = EnvParams(
        contract_size=market_spec.contract_multiplier,  # ✅ Market-specific
        tick_size=market_spec.tick_size,                # ✅ Market-specific
        tick_value=market_spec.tick_value,              # ✅ Market-specific
        commission=market_spec.commission,              # ✅ Market-specific
        slippage_ticks=market_spec.slippage_ticks,      # ✅ Market-specific
        rth_start_count=int(data.rth_indices.shape[0])
        # ❌ position_size NOT SET - defaults to 1.0 for ALL markets
    )
```

**Result**: Phase 1 trains with 1 contract for all markets (emini and micro).

---

### 3. Phase 2 JAX Environment

**File**: `/src/jax_migration/env_phase2_jax.py`

#### Enhanced Position Size Configuration

**Lines 62-86** - EnvParamsPhase2 Definition:
```python
@chex.dataclass(frozen=True)
class EnvParamsPhase2:
    """
    ⚠️ DEFAULTS ARE ES CONTRACT SPECS - Override for other markets!
    """
    # Market specifications
    contract_size: float = 50.0      # ES = $50/point, NQ = $20/point
    tick_size: float = 0.25
    tick_value: float = 12.50
    commission: float = 2.50
    slippage_ticks: int = 1

    # Position management
    initial_balance: float = 50000.0
    position_size: float = 1.0       # ⚠️ Default FIXED size
    trailing_dd_limit: float = 2500.0

    # Dynamic position sizing (NEW for Phase 2)
    risk_per_trade: float = 0.01     # 1% risk per trade
    max_position_size: float = 3.0   # ⚠️ Max contracts LIMITED TO 3
    contract_value: float = 50.0     # Same as contract_size for ES
```

**Key Observations**:
- Phase 2 has dynamic position sizing capability
- `max_position_size` is hardcoded to `3.0` contracts
- Still insufficient for micro markets (need up to 12)

#### Dynamic Position Sizing Function

**Lines 351-408** - calculate_position_size():
```python
def calculate_position_size(
    atr: jnp.ndarray,
    balance: jnp.ndarray,
    params: EnvParamsPhase2,
    portfolio_value: jnp.ndarray = None,
    highest_balance: jnp.ndarray = None
) -> jnp.ndarray:
    """
    Calculate volatility-based position size with Apex safety features.
    """
    # 1. Cap SL distance to limit max loss
    max_sl_points = 500.0 / params.contract_size  # NQ: 25pts, ES: 10pts
    raw_sl_distance = atr * params.sl_atr_mult
    effective_sl_distance = jnp.minimum(raw_sl_distance, max_sl_points)

    # 2. Calculate max risk based on remaining drawdown
    if portfolio_value is not None and highest_balance is not None:
        current_dd = highest_balance - portfolio_value
        remaining_dd = params.trailing_dd_limit - current_dd
        max_risk_allowed = jnp.maximum(0.0, remaining_dd * 0.9)
        max_risk_allowed = jnp.minimum(max_risk_allowed, 500.0)  # Global max
    else:
        max_risk_allowed = 500.0

    # 3. Calculate safe size
    sl_distance_dollars = effective_sl_distance * params.contract_size
    apex_safe_size = max_risk_allowed / (sl_distance_dollars + 1e-8)

    # 4. Risk-based size
    risk_based_size = (balance * params.risk_per_trade) / (sl_distance_dollars + 1e-8)
    size = jnp.minimum(risk_based_size, apex_safe_size)

    # 5. Drawdown scaling (reduce size as DD increases)
    if portfolio_value is not None and highest_balance is not None:
        # ... DD-based scaling logic ...
        size = size * dd_scale

    # Clamp to [1.0, max_position_size]
    size = jnp.clip(size, 1.0, params.max_position_size)  # ⚠️ Capped at 3.0

    return size
```

**Analysis**:
- Function dynamically calculates position size based on risk
- Final result clamped to `[1.0, max_position_size]` (default 3.0)
- For micro markets, this caps at 3 contracts instead of 12

#### Position Size Application

**Lines 828-863** - Action execution:
```python
# Calculate dynamic position size for new trades
dynamic_size = calculate_position_size(
    current_atr, balance_after, params,
    portfolio_value=current_portfolio,
    highest_balance=state.highest_balance
)

# Opening position actions
opening_long = can_open & (effective_action == ACTION_BUY)
opening_short = can_open & (effective_action == ACTION_SELL)
opening_any = opening_long | opening_short

# Store the position size used for this trade
new_position_size = jnp.where(opening_any, dynamic_size, params.position_size)
```

**Lines 777-779, 809-811** - PnL calculation:
```python
# Closed position PnL
pnl_long = (exit_price - state.entry_price) * params.contract_size * params.position_size
pnl_short = (state.entry_price - exit_price) * params.contract_size * params.position_size

# Forced close PnL
forced_pnl_long = (forced_exit_price - entry_after) * params.contract_size * params.position_size
forced_pnl_short = (entry_after - forced_exit_price) * params.contract_size * params.position_size
```

**Lines 883-889** - Unrealized PnL:
```python
unrealized_pnl = jnp.where(
    new_position == 1,
    (current_price - new_entry_price) * params.contract_size * params.position_size,
    jnp.where(
        new_position == -1,
        (new_entry_price - current_price) * params.contract_size * params.position_size,
        0.0
    )
)
```

**Key Issue**: While `dynamic_size` is calculated, the actual PnL calculations throughout the code use `params.position_size` (which defaults to 1.0), not the stored `new_position_size`. This appears to be a bug.

#### Training Script Initialization

**File**: `/src/jax_migration/train_phase2_jax.py:906-916`

```python
market_spec = get_market_spec(args.market)
if market_spec:
    print(f"{Colors.GREEN}Using {args.market} contract specs: ${market_spec.contract_multiplier}/point{Colors.RESET}")
    env_params = EnvParamsPhase2(
        contract_size=market_spec.contract_multiplier,  # ✅ Market-specific
        contract_value=market_spec.contract_multiplier, # ✅ Market-specific
        tick_size=market_spec.tick_size,                # ✅ Market-specific
        tick_value=market_spec.tick_value,              # ✅ Market-specific
        commission=market_spec.commission,              # ✅ Market-specific
        slippage_ticks=market_spec.slippage_ticks,      # ✅ Market-specific
        # ❌ position_size NOT SET - defaults to 1.0
        # ❌ max_position_size NOT SET - defaults to 3.0
    )
```

**Result**: Phase 2 trains with dynamic sizing capped at 3 contracts max, but actual calculations may still use 1 contract due to implementation inconsistency.

---

## Market-Specific Position Size Requirements

### Target Configuration

Based on target requirements:

| Market | Type | Contract Multiplier | Target Contracts | Max Risk/Contract* |
|--------|------|-------------------|------------------|-------------------|
| **ES** | E-mini | $50/point | 1 | $500 |
| **NQ** | E-mini | $20/point | 1 | $500 |
| **YM** | E-mini | $5/point | 1 | $500 |
| **RTY** | E-mini | $50/point | 1 | $500 |
| **MES** | Micro | $5/point | **Max 12** | $500 total |
| **MNQ** | Micro | $2/point | **Max 12** | $500 total |
| **MYM** | Micro | $0.50/point | **Max 12** | $500 total |
| **M2K** | Micro | $5/point | **Max 12** | $500 total |

*Assuming 1.5 ATR stop loss with capped max loss per contract at $500 (as implemented in code)

### Implementation Gap

**Current Implementation**:
```python
# Phase 1
position_size: float = 1.0  # Fixed

# Phase 2
position_size: float = 1.0       # Default fixed
max_position_size: float = 3.0   # Maximum allowed
```

**Required Implementation**:
```python
# Market-specific max sizing
if market in ['MES', 'MNQ', 'MYM', 'M2K']:
    max_position_size = 12.0
else:
    max_position_size = 1.0  # E-mini markets
```

---

## Contract Quantity Flow Through Training

### Phase 1 Training Flow

```
1. train_ppo_jax_fixed.py initializes EnvParams
   ├─ contract_size = market_spec.contract_multiplier ✅
   ├─ position_size = 1.0 (default) ❌
   └─ NO differentiation between emini/micro

2. Environment uses position_size in calculations
   ├─ PnL = (price_diff) * contract_size * position_size
   ├─ Commission = commission * 2 * position_size
   ├─ Unrealized PnL = (price_diff) * contract_size * position_size
   └─ All calculations use fixed 1.0 contract

3. Training proceeds with 1 contract for ALL markets
   └─ Micro markets trained at 1/12th target size
```

### Phase 2 Training Flow

```
1. train_phase2_jax.py initializes EnvParamsPhase2
   ├─ contract_size = market_spec.contract_multiplier ✅
   ├─ position_size = 1.0 (default) ❌
   ├─ max_position_size = 3.0 (default) ❌
   └─ NO differentiation between emini/micro

2. Environment calculates dynamic_size
   ├─ calculate_position_size() returns value in [1.0, 3.0]
   ├─ Stored in new_position_size ✅
   └─ BUT: PnL calculations use params.position_size (1.0) ⚠️

3. Potential bug: Dynamic size calculated but not used
   ├─ new_position_size stored but not referenced in EnvState
   ├─ All PnL calculations use params.position_size
   └─ Effective training: 1 contract regardless of calculation

4. Training proceeds with max 3 contracts (or 1 if bug confirmed)
   └─ Micro markets trained at 1/12th or 3/12th target size
```

---

## Recommendations

### 1. Add Market-Specific Max Position Size

**File**: `src/jax_migration/env_phase1_jax.py` and `env_phase2_jax.py`

Update EnvParams to accept market-specific max sizing:

```python
class EnvParams(NamedTuple):
    # ... existing params ...
    position_size: float = 1.0
    max_position_size: float = 1.0  # NEW: Market-specific max
```

### 2. Update Training Script Initialization

**Files**:
- `src/jax_migration/train_ppo_jax_fixed.py:1070-1085`
- `src/jax_migration/train_phase2_jax.py:906-916`

```python
market_spec = get_market_spec(args.market)
if market_spec:
    # Determine max position size based on market type
    is_micro = args.market in ['MES', 'MNQ', 'MYM', 'M2K']
    max_contracts = 12.0 if is_micro else 1.0

    env_params = EnvParams(  # or EnvParamsPhase2
        contract_size=market_spec.contract_multiplier,
        tick_size=market_spec.tick_size,
        tick_value=market_spec.tick_value,
        commission=market_spec.commission,
        slippage_ticks=market_spec.slippage_ticks,
        position_size=1.0,  # Start with 1
        max_position_size=max_contracts,  # NEW
        rth_start_count=int(data.rth_indices.shape[0])
    )
```

### 3. Fix Phase 2 Position Size Bug (If Confirmed)

**File**: `src/jax_migration/env_phase2_jax.py`

Current code stores `new_position_size` but doesn't add it to `EnvStatePhase2`. Need to:

1. Add `position_size_used` to EnvStatePhase2:
```python
class EnvStatePhase2(NamedTuple):
    # ... existing fields ...
    position_size_used: jnp.ndarray  # NEW: Track actual contracts used
```

2. Update PnL calculations to use stored size:
```python
# Instead of:
pnl = (price_diff) * params.contract_size * params.position_size

# Use:
pnl = (price_diff) * params.contract_size * state.position_size_used
```

### 4. Add Market Type to MarketSpecification

**File**: `src/market_specs.py`

Add market type classification:

```python
@dataclass
class MarketSpecification:
    symbol: str
    name: str
    contract_multiplier: float
    tick_size: float
    commission: float
    slippage_ticks: int
    market_type: str = "emini"  # NEW: "emini" or "micro"
    max_position_size: int = 1  # NEW: Market-specific max contracts

# Then update specs:
MNQ_SPEC = MarketSpecification(
    symbol='MNQ',
    name='Micro E-mini Nasdaq-100',
    contract_multiplier=2.0,
    tick_size=0.25,
    commission=0.60,
    slippage_ticks=1,
    market_type="micro",       # NEW
    max_position_size=12       # NEW
)
```

### 5. Validate Risk Consistency

Ensure max loss per trade remains consistent:

**E-mini (1 contract)**:
- Max loss = $500/contract × 1 = $500

**Micro (12 contracts)**:
- Max loss should still = $500 total (not $500 × 12)
- Individual contract SL should be tighter to maintain $500 total risk
- Or: Reduce position size based on volatility

Current implementation caps SL at $500 loss per position, which is correct. Dynamic sizing in Phase 2 already handles this via `apex_safe_size` calculation.

---

## Impact Analysis

### Training Performance

**Micro Markets Trained with 1-3 Contracts (Current)**:
- Agent learns position management for 1-3 contracts
- Risk perception skewed (1 contract = full position)
- Reward scaling misaligned with deployment reality

**Micro Markets Deployed with 12 Contracts (Target)**:
- Agent suddenly managing 4-12x larger positions
- Untrained on proper scaling of:
  - Risk perception
  - Position management actions
  - Commission impact (12x entry/exit costs)
  - Drawdown dynamics

### Recommended Approach

**Option A: Train Micro Markets with Full 12 Contracts**
- Pros: Accurate training environment
- Cons: Higher computational cost, slower convergence
- Risk: May violate Apex rules during training if not careful

**Option B: Train with Graduated Position Sizing**
- Start Phase 1 with 1 contract (learn entries)
- Phase 2 with dynamic 1-6 contracts (learn management)
- Phase 3 with full 1-12 contracts (finalize)
- Pros: Curriculum learning approach, safer
- Cons: More complex implementation

**Option C: Scale Rewards/Losses Instead**
- Train with 1 contract but scale rewards by 12x for micro markets
- Pros: Simpler, no code changes to position sizing
- Cons: May not capture commission/slippage scaling accurately

---

## Files Requiring Changes

### Core Files
1. `/src/market_specs.py` - Add market_type and max_position_size fields
2. `/src/jax_migration/env_phase1_jax.py` - Add max_position_size parameter
3. `/src/jax_migration/env_phase2_jax.py` - Fix position_size_used tracking bug
4. `/src/jax_migration/train_ppo_jax_fixed.py` - Market-specific initialization
5. `/src/jax_migration/train_phase2_jax.py` - Market-specific initialization

### Supporting Files
6. `/src/jax_migration/evaluate_phase2_jax.py` - Use correct position sizing for eval
7. `/src/jax_migration/debug_evaluation.py` - Use correct position sizing

---

## Conclusion

**Current State**:
- Phase 1: Fixed 1 contract for all markets
- Phase 2: Dynamic 1-3 contracts (may be 1 due to bug)
- No differentiation between emini (1 contract) and micro (12 contracts)

**Gap**:
- Micro markets trained at 1/12th (Phase 1) or 1/4th (Phase 2) of target size
- Training environment doesn't match deployment reality
- Position management actions learned on wrong scale

**Priority**: HIGH - Significant impact on micro market trading performance

**Next Steps**:
1. Confirm Phase 2 position_size bug by examining EnvStatePhase2 usage
2. Decide on training approach (Option A/B/C)
3. Implement market-specific max_position_size
4. Retrain micro market models with correct sizing
5. Validate performance on historical data

---

**Analysis Complete**: 2025-12-12
