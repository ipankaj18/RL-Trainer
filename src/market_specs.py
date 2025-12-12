#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Specifications for Futures Trading
Defines contract specifications for all supported futures markets.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MarketSpecification:
    """
    Contract specifications for a futures market.

    Attributes:
        symbol: Market symbol (e.g., 'ES', 'NQ')
        name: Full market name
        contract_multiplier: Dollar value per point (e.g., ES = $50)
        tick_size: Minimum price increment in points (e.g., 0.25)
        commission: Default commission per side in dollars
        slippage_ticks: Number of ticks for slippage modeling (1 for liquid, 2 for less liquid)
        tick_value: Dollar value per tick (calculated as multiplier * tick_size)
        market_type: "emini" or "micro" - used for position sizing limits
        max_position_size: Maximum contracts allowed (3 for emini, 12 for micro)
    """
    symbol: str
    name: str
    contract_multiplier: float
    tick_size: float
    commission: float
    slippage_ticks: int
    market_type: str           # NEW: "emini" or "micro"
    max_position_size: int     # NEW: 3 for emini, 12 for micro

    @property
    def tick_value(self) -> float:
        """Calculate tick value (multiplier * tick_size)"""
        return self.contract_multiplier * self.tick_size


# ============================================================================
# E-MINI FUTURES (Standard Size)
# ============================================================================

ES_SPEC = MarketSpecification(
    symbol='ES',
    name='E-mini S&P 500',
    contract_multiplier=50.0,
    tick_size=0.25,
    commission=0.50,           # UPDATED: $0.50/side for all markets
    slippage_ticks=1,          # Highly liquid
    market_type="emini",       # NEW
    max_position_size=1        # UPDATED: 1 contract max for emini (User Request)
)

NQ_SPEC = MarketSpecification(
    symbol='NQ',
    name='E-mini Nasdaq-100',
    contract_multiplier=20.0,
    tick_size=0.25,
    commission=0.50,           # UPDATED: $0.50/side for all markets
    slippage_ticks=1,          # Highly liquid
    market_type="emini",       # NEW
    max_position_size=1        # UPDATED: 1 contract max for emini (User Request)
)

YM_SPEC = MarketSpecification(
    symbol='YM',
    name='E-mini Dow Jones',
    contract_multiplier=5.0,
    tick_size=1.0,
    commission=0.50,           # UPDATED: $0.50/side for all markets
    slippage_ticks=2,          # Less liquid than ES/NQ
    market_type="emini",       # NEW
    max_position_size=1        # UPDATED: 1 contract max for emini (User Request)
)

RTY_SPEC = MarketSpecification(
    symbol='RTY',
    name='E-mini Russell 2000',
    contract_multiplier=50.0,
    tick_size=0.10,
    commission=0.50,           # UPDATED: $0.50/side for all markets
    slippage_ticks=2,          # Less liquid
    market_type="emini",       # NEW
    max_position_size=1        # UPDATED: 1 contract max for emini (User Request)
)

# ============================================================================
# COMMODITIES (Standard Size)
# ============================================================================

GC_SPEC = MarketSpecification(
    symbol='GC',
    name='Gold',
    contract_multiplier=100.0,
    tick_size=0.10,
    commission=0.50,           # Standardized
    slippage_ticks=1,
    market_type="emini",       # Treated as standard/emini for sizing
    max_position_size=1        # User Request: 1 contract max
)

CL_SPEC = MarketSpecification(
    symbol='CL',
    name='Crude Oil',
    contract_multiplier=1000.0,
    tick_size=0.01,
    commission=0.50,           # Standardized
    slippage_ticks=1,
    market_type="emini",       # Treated as standard/emini for sizing
    max_position_size=1        # User Request: 1 contract max
)

# ============================================================================
# MICRO E-MINI FUTURES (1/10th Size)
# ============================================================================

MNQ_SPEC = MarketSpecification(
    symbol='MNQ',
    name='Micro E-mini Nasdaq-100',
    contract_multiplier=2.0,
    tick_size=0.25,
    commission=0.50,           # UPDATED: $0.50/side for all markets
    slippage_ticks=1,          # Liquid
    market_type="micro",       # NEW
    max_position_size=12       # NEW: 1-12 contracts for micro
)

# ... (Previous micros like MES, M2K, MYM are here, ensuring I don't overwrite them incorrectly or I need to explicitly include them if I am replacing a block)
# The user wants MGC and MCL too. I should append them after other micros or in the micro section.
# Since replace_file_content replaces a block, I need to be careful.
# I will insert GC/CL before micros, and append MGC/MCL after MYM_SPEC.

# Better strategy: 
# 1. Add GC/CL before MNQ_SPEC (as done above)
# 2. Add MGC/MCL after MYM_SPEC separately to avoid matching issues with large blocks.


MES_SPEC = MarketSpecification(
    symbol='MES',
    name='Micro E-mini S&P 500',
    contract_multiplier=5.0,
    tick_size=0.25,
    commission=0.50,           # UPDATED: $0.50/side for all markets
    slippage_ticks=1,          # Liquid
    market_type="micro",       # NEW
    max_position_size=12       # NEW: 1-12 contracts for micro
)

M2K_SPEC = MarketSpecification(
    symbol='M2K',
    name='Micro E-mini Russell 2000',
    contract_multiplier=5.0,
    tick_size=0.10,
    commission=0.50,           # UPDATED: $0.50/side for all markets
    slippage_ticks=2,          # Less liquid
    market_type="micro",       # NEW
    max_position_size=12       # NEW: 1-12 contracts for micro
)

MYM_SPEC = MarketSpecification(
    symbol='MYM',
    name='Micro E-mini Dow Jones',
    contract_multiplier=0.50,
    tick_size=1.0,
    commission=0.50,           # UPDATED: $0.50/side for all markets
    slippage_ticks=2,          # Less liquid
    market_type="micro",       # NEW
    max_position_size=12       # NEW: 1-12 contracts for micro
)

MGC_SPEC = MarketSpecification(
    symbol='MGC',
    name='Micro Gold',
    contract_multiplier=10.0,
    tick_size=0.10,
    commission=0.50,           # Standardized
    slippage_ticks=2,
    market_type="micro",
    max_position_size=12       # User Request: 1-12 micros
)

MCL_SPEC = MarketSpecification(
    symbol='MCL',
    name='Micro Crude Oil',
    contract_multiplier=100.0,
    tick_size=0.01,
    commission=0.50,           # Standardized
    slippage_ticks=2,
    market_type="micro",
    max_position_size=12       # User Request: 1-12 micros
)

# ============================================================================
# MARKET REGISTRY
# ============================================================================

MARKET_SPECS: Dict[str, MarketSpecification] = {
    'ES': ES_SPEC,
    'NQ': NQ_SPEC,
    'YM': YM_SPEC,
    'RTY': RTY_SPEC,
    'GC': GC_SPEC,   # NEW
    'CL': CL_SPEC,   # NEW
    'MNQ': MNQ_SPEC,
    'MES': MES_SPEC,
    'M2K': M2K_SPEC,
    'MYM': MYM_SPEC,
    'MGC': MGC_SPEC, # NEW
    'MCL': MCL_SPEC, # NEW
    # Support 'GENERIC' fallback
    'GENERIC': ES_SPEC,  # Default to ES specs
}


def get_market_spec(symbol: str) -> Optional[MarketSpecification]:
    """
    Get market specification by symbol.

    Args:
        symbol: Market symbol (e.g., 'ES', 'NQ')

    Returns:
        MarketSpecification object, or None if not found
    """
    return MARKET_SPECS.get(symbol.upper())


def list_supported_markets() -> list:
    """Return list of all supported market symbols."""
    return [k for k in MARKET_SPECS.keys() if k != 'GENERIC']


def display_market_specs():
    """Display all market specifications in a formatted table."""
    print("\n" + "=" * 120)
    print("SUPPORTED FUTURES MARKETS")
    print("=" * 120)
    print(f"{'Symbol':<8} {'Name':<30} {'Multiplier':<12} {'Tick':<8} {'Tick Value':<12} {'Commission':<12} {'Type':<8} {'Max Pos':<8}")
    print("-" * 120)

    for symbol in list_supported_markets():
        spec = MARKET_SPECS[symbol]
        print(f"{spec.symbol:<8} {spec.name:<30} ${spec.contract_multiplier:<11.2f} {spec.tick_size:<8} "
              f"${spec.tick_value:<11.2f} ${spec.commission:<11.2f} {spec.market_type:<8} {spec.max_position_size:<8}")

    print("=" * 120)

