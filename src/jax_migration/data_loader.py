"""
JAX Data Loader - Pre-compute all features as JAX arrays

This module converts Pandas DataFrames to GPU-resident JAX arrays
with all features pre-computed for zero-copy access during training.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Dict, Tuple, NamedTuple
from pathlib import Path


class MarketData(NamedTuple):
    """Immutable market data structure for JAX training."""
    # Shape: (num_timesteps, num_features)
    features: jnp.ndarray      # OHLCV + technical indicators
    prices: jnp.ndarray        # (num_timesteps, 4) - open, high, low, close
    atr: jnp.ndarray           # (num_timesteps,) - for SL/TP calculation
    time_features: jnp.ndarray # (num_timesteps, 3) - hour_decimal, min_open, min_close
    trading_mask: jnp.ndarray  # (num_timesteps,) - 1.0 if within RTH, 0.0 otherwise
    timestamps_hour: jnp.ndarray  # (num_timesteps,) - hour as float for time rules
    rth_indices: jnp.ndarray   # (num_rth_indices,) - valid RTH start indices for Phase 2
    # Second-level data features for intra-bar drawdown checks
    low_s: jnp.ndarray         # (num_timesteps,) - min low of the minute
    high_s: jnp.ndarray        # (num_timesteps,) - max high of the minute


def precompute_time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Pre-compute all time-related features as numpy arrays.
    
    SYNCHRONIZED WITH Agent_temp/observation_builder.py (2025-12-07)
    Uses SAME calculation as real-time NinjaTrader bridge for training parity.
    
    Returns shape (num_timesteps, 3):
        - hour_norm: hour / 24.0 (matches Agent_temp)
        - min_from_open: RAW minutes since 9:30 AM (0 to ~449)
        - min_to_close: RAW minutes until 4:59 PM (0 to ~449)
    """
    # Convert to Eastern Time if needed
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize('UTC').tz_convert('America/New_York')
    else:
        timestamps = timestamps.tz_convert('America/New_York')
    
    # Hour normalized to 0-1 (matches Agent_temp: hour / 24.0)
    # Agent_temp uses: hour_norm = hour / 24.0
    hour_norm = timestamps.hour / 24.0
    
    # RAW minutes from open (9:30 AM) - matches Agent_temp
    # Agent_temp uses: min_from_open = current_minutes - open_minutes
    open_minutes = 9 * 60 + 30  # 9:30 AM = 570 minutes from midnight
    current_minutes = timestamps.hour * 60 + timestamps.minute
    min_from_open = (current_minutes - open_minutes).astype(np.float32)
    
    # RAW minutes to close (4:59 PM) - matches Agent_temp
    # Agent_temp uses: min_to_close = close_minutes - current_minutes
    close_minutes = 16 * 60 + 59  # 4:59 PM = 1019 minutes from midnight
    min_to_close = (close_minutes - current_minutes).astype(np.float32)
    
    return np.stack([hour_norm, min_from_open, min_to_close], axis=1).astype(np.float32)


def compute_trading_mask(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Compute mask for valid trading hours (RTH: 9:30 AM - 4:59 PM ET).
    
    Returns shape (num_timesteps,) with 1.0 for valid bars, 0.0 otherwise.
    """
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize('UTC').tz_convert('America/New_York')
    else:
        timestamps = timestamps.tz_convert('America/New_York')
    
    hours = timestamps.hour + timestamps.minute / 60.0
    
    # RTH: 9:30 AM (9.5) to 4:59 PM (16.983)
    mask = (hours >= 9.5) & (hours <= 16.983)
    return mask.astype(np.float32)


def precompute_rth_indices(timestamps: pd.DatetimeIndex, window_size: int = 60) -> np.ndarray:
    """
    Pre-compute valid RTH start indices for JAX Phase 2.
    
    RTH hours: 9:30 AM - 4:00 PM ET (allow entries until 4:00 PM)
    
    FIX (2025-12-10): Increased minimum remaining bars from 100 to 400
    to prevent episodes terminating early due to end of data.
    
    Args:
        timestamps: DatetimeIndex of all bars
        window_size: Minimum lookback window needed
        
    Returns:
        Array of valid start indices that fall within RTH
    """
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize('UTC').tz_convert('America/New_York')
    else:
        timestamps = timestamps.tz_convert('America/New_York')
    
    rth_indices = []
    # FIX (2025-12-10): Need at least 400 bars for proper episode length
    # This prevents "End of data" termination after only a few steps
    min_remaining_bars = 400
    max_start = len(timestamps) - min_remaining_bars
    
    for idx in range(window_size, max_start):
        ts = timestamps[idx]
        hour = ts.hour
        minute = ts.minute
        
        # RTH: 9:30 AM to 4:00 PM ET (16:00 = 4:00 PM)
        # Allow entries from 9:30 to 4:00 (not 4:59, to give time to exit)
        if (hour == 9 and minute >= 30) or (10 <= hour < 16):
            rth_indices.append(idx)
    
    return np.array(rth_indices, dtype=np.int32)


def load_market_data(
    csv_path: str,
    feature_columns: list = None,
    second_data_path: str = None
) -> MarketData:
    """
    Load market data from CSV and convert to JAX arrays.
    
    Args:
        csv_path: Path to processed CSV file (e.g., ES_D1M.csv)
        feature_columns: List of feature columns to include
        second_data_path: Optional path to second-level data (e.g., ES_D1S.csv)
        
    Returns:
        MarketData namedtuple with all arrays on GPU
    """
    if feature_columns is None:
        feature_columns = [
            'close', 'volume', 'sma_5', 'sma_20', 'rsi',
            'macd', 'momentum', 'atr'
        ]
    
    print(f"Loading minute data from {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    
    # Extract price data
    prices = df[['open', 'high', 'low', 'close']].values.astype(np.float32)
    
    # Extract ATR for SL/TP calculations
    atr = df['atr'].values.astype(np.float32)
    
    # Extract features
    features = df[feature_columns].values.astype(np.float32)
    
    # Pre-compute time features
    time_features = precompute_time_features(df.index)
    
    # Compute trading mask
    trading_mask = compute_trading_mask(df.index)
    
    # Extract hour for time rule checks
    if df.index.tz is None:
        ts = df.index.tz_localize('UTC').tz_convert('America/New_York')
    else:
        ts = df.index.tz_convert('America/New_York')
    timestamps_hour = (ts.hour + ts.minute / 60.0).values.astype(np.float32)
    
    # Process second-level data if available
    low_s = prices[:, 2]  # Default to minute low
    high_s = prices[:, 1] # Default to minute high
    
    if second_data_path and Path(second_data_path).exists():
        print(f"Loading second-level data from {second_data_path}...")
        try:
            df_s = pd.read_csv(second_data_path, index_col=0, parse_dates=True)
            if not isinstance(df_s.index, pd.DatetimeIndex):
                df_s.index = pd.to_datetime(df_s.index, utc=True)
                
            # Resample to minute level, taking min(low) and max(high)
            # We need to align this exactly with the minute bars
            # 1. Resample to 1T (minute)
            # 2. Reindex to match the minute dataframe exactly
            
            resampled = df_s.resample('1min').agg({
                'low': 'min',
                'high': 'max'
            })
            
            # Align with minute data index
            # Forward fill missing values (if any seconds missing within minute)
            # Fill remaining NaNs with minute data (fallback)
            aligned = resampled.reindex(df.index)
            
            # Use minute data where second data is missing
            low_s_aligned = aligned['low'].fillna(df['low']).values.astype(np.float32)
            high_s_aligned = aligned['high'].fillna(df['high']).values.astype(np.float32)
            
            # Ensure consistency: low_s <= low, high_s >= high
            low_s = np.minimum(low_s_aligned, prices[:, 2])
            high_s = np.maximum(high_s_aligned, prices[:, 1])
            
            print("  Second-level data integrated successfully.")
            
        except Exception as e:
            print(f"  Error processing second-level data: {e}")
            print("  Falling back to minute-level high/low.")
    else:
        print("  No second-level data found. Using minute-level high/low (less precise).")
    
    # Pre-compute RTH indices for Phase 2 episode starts
    rth_indices = precompute_rth_indices(df.index, window_size=60)
    print(f"  RTH indices computed: {len(rth_indices)} valid start points")
    
    # Convert to JAX arrays and move to GPU
    return MarketData(
        features=jnp.array(features),
        prices=jnp.array(prices),
        atr=jnp.array(atr),
        time_features=jnp.array(time_features),
        trading_mask=jnp.array(trading_mask),
        timestamps_hour=jnp.array(timestamps_hour),
        rth_indices=jnp.array(rth_indices),
        low_s=jnp.array(low_s),
        high_s=jnp.array(high_s)
    )


def load_all_markets(data_dir: str, markets: list = None) -> Dict[str, MarketData]:
    """
    Load all market data files into GPU memory.
    
    Args:
        data_dir: Directory containing processed CSV files
        markets: List of market symbols (default: all 8 futures)
        
    Returns:
        Dictionary mapping market symbol to MarketData
    """
    if markets is None:
        markets = ['ES', 'NQ', 'YM', 'RTY', 'MNQ', 'MES', 'M2K', 'MYM']
    
    data_dir = Path(data_dir)
    market_data = {}
    
    for symbol in markets:
        csv_path = data_dir / f"{symbol}_D1M.csv"
        second_path = data_dir / f"{symbol}_D1S.csv"
        
        if csv_path.exists():
            print(f"Loading {symbol}...")
            market_data[symbol] = load_market_data(
                str(csv_path), 
                second_data_path=str(second_path) if second_path.exists() else None
            )
            print(f"  Shape: {market_data[symbol].features.shape}")
        else:
            print(f"  Warning: {csv_path} not found, skipping {symbol}")
    
    return market_data


def create_batched_data(
    market_data: Dict[str, MarketData],
    max_timesteps: int = None
) -> Tuple[MarketData, jnp.ndarray]:
    """
    Stack multiple markets into batched arrays for multi-market training.
    
    Args:
        market_data: Dictionary of MarketData per market
        max_timesteps: Truncate/pad to this length (None = use min length)
        
    Returns:
        Tuple of (batched_data, market_indices)
        batched_data has shape (num_markets, num_timesteps, ...)
    """
    markets = list(market_data.keys())
    
    if max_timesteps is None:
        max_timesteps = min(d.features.shape[0] for d in market_data.values())
    
    # Stack all markets
    batched_features = jnp.stack([
        market_data[m].features[:max_timesteps] for m in markets
    ])
    batched_prices = jnp.stack([
        market_data[m].prices[:max_timesteps] for m in markets
    ])
    batched_atr = jnp.stack([
        market_data[m].atr[:max_timesteps] for m in markets
    ])
    batched_time = jnp.stack([
        market_data[m].time_features[:max_timesteps] for m in markets
    ])
    batched_mask = jnp.stack([
        market_data[m].trading_mask[:max_timesteps] for m in markets
    ])
    batched_hour = jnp.stack([
        market_data[m].timestamps_hour[:max_timesteps] for m in markets
    ])
    batched_low_s = jnp.stack([
        market_data[m].low_s[:max_timesteps] for m in markets
    ])
    batched_high_s = jnp.stack([
        market_data[m].high_s[:max_timesteps] for m in markets
    ])
    
    # RTH indices don't get stacked - each market keeps its own
    # For batched training, we'll just use the first market's indices
    # (in practice, RTH is the same for all US futures)
    batched_rth_indices = market_data[markets[0]].rth_indices
    
    batched_data = MarketData(
        features=batched_features,
        prices=batched_prices,
        atr=batched_atr,
        time_features=batched_time,
        trading_mask=batched_mask,
        timestamps_hour=batched_hour,
        rth_indices=batched_rth_indices,
        low_s=batched_low_s,
        high_s=batched_high_s
    )
    
    market_indices = {m: i for i, m in enumerate(markets)}
    
    return batched_data, market_indices


if __name__ == "__main__":
    # Test data loading
    import sys
    
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    print("Testing JAX data loader...")
    print(f"JAX devices: {jax.devices()}")
    
    # Load single market
    es_path = data_dir / "ES_D1M.csv"
    if es_path.exists():
        es_data = load_market_data(str(es_path))
        print(f"\nES Data loaded:")
        print(f"  Features shape: {es_data.features.shape}")
        print(f"  Prices shape: {es_data.prices.shape}")
        print(f"  ATR shape: {es_data.atr.shape}")
        print(f"  Time features shape: {es_data.time_features.shape}")
        print(f"  Low_s shape: {es_data.low_s.shape}")
        print(f"  High_s shape: {es_data.high_s.shape}")
        print(f"  Device: {es_data.features.device()}")
    else:
        print(f"ES data not found at {es_path}")
