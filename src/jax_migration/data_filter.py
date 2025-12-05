"""
Data filtering utilities for curriculum learning.
Filters training data by volatility and market regime.

Part of Phase 1 improvements to fix unprofitable training.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional


def filter_high_volatility_periods(
    data: pd.DataFrame,
    atr_column: str = 'ATR_14',
    percentile: float = 75.0
) -> pd.DataFrame:
    """
    Filter data to include only high-volatility periods.

    Args:
        data: DataFrame with OHLCV + indicators
        atr_column: Column name for ATR indicator
        percentile: ATR percentile threshold (default: 75th percentile)

    Returns:
        Filtered DataFrame with high-volatility periods only
    """
    if atr_column not in data.columns:
        print(f"Warning: {atr_column} not found, returning original data")
        return data

    atr_threshold = np.percentile(data[atr_column].dropna(), percentile)
    mask = data[atr_column] > atr_threshold

    filtered = data[mask].copy()
    print(f"Filtered to {len(filtered):,} bars ({len(filtered)/len(data)*100:.1f}%) " +
          f"with ATR > {atr_threshold:.2f}")

    return filtered


def classify_market_regime(
    data: pd.DataFrame,
    adx_column: str = 'ADX',
    trending_threshold: float = 25.0,
    ranging_threshold: float = 20.0
) -> Dict[str, pd.DataFrame]:
    """
    Classify market data by regime (trending vs ranging).

    Args:
        data: DataFrame with indicators
        adx_column: Column name for ADX indicator
        trending_threshold: ADX threshold for trending (default: 25)
        ranging_threshold: ADX threshold for ranging (default: 20)

    Returns:
        Dictionary with 'trending', 'ranging', 'mixed' DataFrames
    """
    if adx_column not in data.columns:
        print(f"Warning: {adx_column} not found")
        return {'all': data}

    trending_mask = data[adx_column] > trending_threshold
    ranging_mask = data[adx_column] < ranging_threshold

    regimes = {
        'trending': data[trending_mask].copy(),
        'ranging': data[ranging_mask].copy(),
        'mixed': data[~(trending_mask | ranging_mask)].copy()
    }

    print(f"Market regime breakdown:")
    for regime, df in regimes.items():
        pct = len(df) / len(data) * 100
        print(f"  {regime}: {len(df):,} bars ({pct:.1f}%)")

    return regimes


def load_filtered_data(
    data_path: str,
    filter_type: Optional[str] = None,
    **filter_kwargs
) -> pd.DataFrame:
    """
    Load and optionally filter training data.

    Args:
        data_path: Path to CSV file
        filter_type: 'high_volatility', 'trending', 'ranging', or None
        **filter_kwargs: Additional arguments for filtering functions

    Returns:
        Loaded (and optionally filtered) DataFrame
    """
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    print(f"  Loaded {len(data):,} bars")

    if filter_type is None:
        return data

    if filter_type == 'high_volatility':
        return filter_high_volatility_periods(data, **filter_kwargs)
    elif filter_type == 'trending':
        regimes = classify_market_regime(data, **filter_kwargs)
        return regimes['trending']
    elif filter_type == 'ranging':
        regimes = classify_market_regime(data, **filter_kwargs)
        return regimes['ranging']
    else:
        print(f"Warning: Unknown filter_type '{filter_type}', returning original data")
        return data


if __name__ == '__main__':
    """Test filtering on market data"""
    import argparse

    parser = argparse.ArgumentParser(description='Test data filtering')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--filter', type=str, choices=['high_volatility', 'trending', 'ranging'],
                       help='Filter type to apply')
    args = parser.parse_args()

    filtered = load_filtered_data(args.data_path, filter_type=args.filter)
    print(f"\nFiltered data shape: {filtered.shape}")
    if len(filtered) > 0:
        print(f"Date range: {filtered.iloc[0]['timestamp'] if 'timestamp' in filtered.columns else 'N/A'} to " +
              f"{filtered.iloc[-1]['timestamp'] if 'timestamp' in filtered.columns else 'N/A'}")
