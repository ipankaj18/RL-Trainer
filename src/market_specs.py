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
    """
    symbol: str
    name: str
    contract_multiplier: float
    tick_size: float
    commission: float
    slippage_ticks: int

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
    commission=2.50,
    slippage_ticks=1  # Highly liquid
)

NQ_SPEC = MarketSpecification(
    symbol='NQ',
    name='E-mini Nasdaq-100',
    contract_multiplier=20.0,
    tick_size=0.25,
    commission=2.50,
    slippage_ticks=1  # Highly liquid
)

YM_SPEC = MarketSpecification(
    symbol='YM',
    name='E-mini Dow Jones',
    contract_multiplier=5.0,
    tick_size=1.0,
    commission=2.50,
    slippage_ticks=2  # Less liquid than ES/NQ
)

RTY_SPEC = MarketSpecification(
    symbol='RTY',
    name='E-mini Russell 2000',
    contract_multiplier=50.0,
    tick_size=0.10,
    commission=2.50,
    slippage_ticks=2  # Less liquid
)

# ============================================================================
# MICRO E-MINI FUTURES (1/10th Size)
# ============================================================================

MNQ_SPEC = MarketSpecification(
    symbol='MNQ',
    name='Micro E-mini Nasdaq-100',
    contract_multiplier=2.0,
    tick_size=0.25,
    commission=0.60,  # Lower commission for micros
    slippage_ticks=1  # Liquid
)

MES_SPEC = MarketSpecification(
    symbol='MES',
    name='Micro E-mini S&P 500',
    contract_multiplier=5.0,
    tick_size=0.25,
    commission=0.60,  # Lower commission for micros
    slippage_ticks=1  # Liquid
)

M2K_SPEC = MarketSpecification(
    symbol='M2K',
    name='Micro E-mini Russell 2000',
    contract_multiplier=5.0,
    tick_size=0.10,
    commission=0.60,  # Lower commission for micros
    slippage_ticks=2  # Less liquid
)

MYM_SPEC = MarketSpecification(
    symbol='MYM',
    name='Micro E-mini Dow Jones',
    contract_multiplier=0.50,
    tick_size=1.0,
    commission=0.60,  # Lower commission for micros
    slippage_ticks=2  # Less liquid
)

# ============================================================================
# MARKET REGISTRY
# ============================================================================

MARKET_SPECS: Dict[str, MarketSpecification] = {
    'ES': ES_SPEC,
    'NQ': NQ_SPEC,
    'YM': YM_SPEC,
    'RTY': RTY_SPEC,
    'MNQ': MNQ_SPEC,
    'MES': MES_SPEC,
    'M2K': M2K_SPEC,
    'MYM': MYM_SPEC,
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
    print("\n" + "=" * 100)
    print("SUPPORTED FUTURES MARKETS")
    print("=" * 100)
    print(f"{'Symbol':<8} {'Name':<30} {'Multiplier':<12} {'Tick':<8} {'Tick Value':<12} {'Commission':<12}")
    print("-" * 100)

    for symbol in list_supported_markets():
        spec = MARKET_SPECS[symbol]
        print(f"{spec.symbol:<8} {spec.name:<30} ${spec.contract_multiplier:<11.2f} {spec.tick_size:<8} "
              f"${spec.tick_value:<11.2f} ${spec.commission:<11.2f}")

    print("=" * 100)
