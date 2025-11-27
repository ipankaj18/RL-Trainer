#!/usr/bin/env python3
"""
Feature Engineering Module for Trading RL

Adds advanced market regime features to improve RL agent state representation.
Based on COMPREHENSIVE_IMPROVEMENT_PLAN.md Phase 2.1.2

Features:
- ADX (Average Directional Index) for trend strength
- Volatility regime detection
- VWAP and volume profile
- Session-based features (morning/midday/afternoon)
"""

import numpy as np
import pandas as pd
from typing import Optional


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced market regime features to dataframe.

    This function is called during data loading to enrich the feature set
    with market microstructure and regime information.

    Args:
        df: DataFrame with OHLCV and basic indicators

    Returns:
        DataFrame with additional regime features
    """
    print("[FEATURES] Adding market regime features...")

    # Make a copy to avoid modifying original
    df = df.copy()

    # 1. Volatility regime detection
    if 'atr' in df.columns:
        df['vol_regime'] = df['atr'].rolling(20).std()
        df['vol_percentile'] = df['atr'].rolling(252*390, min_periods=100).rank(pct=True)
        print("  [OK] Volatility regime features added")
    else:
        print("  [!] Warning: ATR not found, skipping volatility regime")

    # 2. Trend strength (ADX)
    df = calculate_adx(df, period=14)
    if 'adx' in df.columns:
        df['trend_strength'] = np.where(df['adx'] > 25, 1,  # Strong trend
                                       np.where(df['adx'] < 20, -1,  # Weak trend
                                                0))  # Neutral
        print("  [OK] ADX and trend strength added")

    # 3. Volume profile (VWAP)
    df = calculate_vwap(df)
    if 'vwap' in df.columns:
        df['price_to_vwap'] = (df['close'] / df['vwap']) - 1
        print("  [OK] VWAP features added")

    # 4. Market microstructure
    if 'high' in df.columns and 'low' in df.columns:
        df['spread'] = (df['high'] - df['low']) / df['close']

        # Efficiency ratio (price change / path traveled)
        price_change = abs(df['close'] - df['close'].shift(20))
        path_traveled = df['atr'].rolling(20).sum()
        df['efficiency_ratio'] = price_change / (path_traveled + 1e-8)
        print("  [OK] Microstructure features added")

    # 5. Session features (morning/midday/afternoon)
    df = add_session_features(df)
    if 'session_morning' in df.columns:
        print("  [OK] Session features added")

    # 6. LLM features for Phase 3
    df = add_llm_features(df)
    print("  [OK] LLM features added")

    # 7. Fill NaN values (from rolling windows)
    # Forward fill then backward fill to handle initial NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)

    print(f"[FEATURES] Total features: {len(df.columns)}")
    return df


def add_llm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add multi-timeframe and pattern features for LLM-enhanced trading.
    
    These features provide additional context for the LLM advisor:
    - Multi-timeframe SMAs and RSI
    - Volume analysis across timeframes
    - Support/resistance levels
    - Price change metrics
    
    Args:
        df: DataFrame with OHLCV and basic indicators
        
    Returns:
        DataFrame with additional LLM features
    """
    print("  [LLM] Adding multi-timeframe and pattern features...")
    
    # Multi-timeframe SMAs
    if 'close' in df.columns:
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        
        # Vectorized Slope Calculation
        # Slope = (Current - Lagged) / Period / Current
        df['sma_50_slope'] = (df['sma_50'] - df['sma_50'].shift(10)) / 10 / (df['sma_50'] + 1e-8)
        df['sma_200_slope'] = (df['sma_200'] - df['sma_200'].shift(20)) / 20 / (df['sma_200'] + 1e-8)
        
        # ADX Slope (if ADX exists)
        if 'adx' in df.columns:
            df['adx_slope'] = (df['adx'] - df['adx'].shift(5)) / 5 / (df['adx'] + 1e-8)
        
        print("    [OK] Multi-timeframe SMAs and slopes added")
    
    # Multi-timeframe RSI (using different periods)
    if 'close' in df.columns:
        # RSI with 15-period (faster)
        df['rsi_15min'] = calculate_rsi(df['close'], period=15)
        # RSI with 60-period (slower)
        df['rsi_60min'] = calculate_rsi(df['close'], period=60)
        print("    [OK] Multi-timeframe RSI added")
    
    # Volume analysis across timeframes
    if 'volume' in df.columns:
        df['volume_ratio_5min'] = df['volume'] / (df['volume'].rolling(5).mean() + 1e-8)
        df['volume_ratio_20min'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
        
        # Volume regime (current / 20-period avg)
        df['volume_regime'] = df['volume_ratio_20min']
        print("    [OK] Volume ratio features added")
    
    # Support and Resistance levels
    if 'high' in df.columns and 'low' in df.columns:
        df['support_20'] = df['low'].rolling(20).min()
        df['resistance_20'] = df['high'].rolling(20).max()
        
        # Price vs Support/Resistance
        if 'close' in df.columns:
            df['price_vs_support'] = (df['close'] - df['support_20']) / (df['close'] + 1e-8)
        
        print("    [OK] Support/Resistance levels added")
    
    # Price change metrics
    if 'close' in df.columns:
        df['price_change_60min'] = df['close'].pct_change(60)
        
        # Momentum features
        df['price_momentum_20'] = df['close'].pct_change(20)
        df['price_momentum_60'] = df['close'].pct_change(60)
        
        # Session trend (approximate 30-bar lookback)
        df['session_trend'] = df['close'].pct_change(30)
        
        print("    [OK] Price change metrics added")
        
    # Vectorized Pattern Recognition
    if all(col in df.columns for col in ['high', 'low', 'close']):
        df = add_pattern_features(df)
        print("    [OK] Pattern recognition features added")
    
    return df


def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add vectorized pattern recognition features.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with pattern features added
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Lookback windows
    lookback = 5
    
    # Previous highs/lows (shifted to not include current bar)
    prev_high_max = high.shift(1).rolling(lookback).max()
    prev_low_min = low.shift(1).rolling(lookback).min()
    
    # Higher High: Current high > previous lookback high
    df['higher_high'] = np.where(high > prev_high_max * 1.001, 1.0, 
                                np.where(high < prev_high_max * 0.999, -1.0, 0.0))
    
    # Lower Low: Current low < previous lookback low
    df['lower_low'] = np.where(low < prev_low_min * 0.999, 1.0,
                              np.where(low > prev_low_min * 1.001, -1.0, 0.0))
    
    # Higher Low: Current low > previous low AND previous low > low before that
    # This is complex to vectorize perfectly matching the iterative logic, 
    # but we can approximate or use rolling windows.
    # Iterative logic: current_low > prev_low_1 * 1.001 and prev_low_1 > prev_low_2 * 1.001
    # Let's simplify to: Low > Low(t-1) > Low(t-2)
    # Or strictly following the logic:
    prev_low_1 = low.shift(1).rolling(lookback).min()
    prev_low_2 = low.shift(lookback+1).rolling(lookback).min()
    
    df['higher_low'] = np.where((low > prev_low_1 * 1.001) & (prev_low_1 > prev_low_2 * 1.001), 1.0, 0.0)
    
    # Lower High
    prev_high_1 = high.shift(1).rolling(lookback).max()
    prev_high_2 = high.shift(lookback+1).rolling(lookback).max()
    
    df['lower_high'] = np.where((high < prev_high_1 * 0.999) & (prev_high_1 < prev_high_2 * 0.999), 1.0, 0.0)
    
    # Double Top
    # Recent max < Prev max * 0.99 (lower peak) is NOT a double top usually.
    # Standard double top: Two peaks at similar level.
    # The iterative logic was: recent_max < prev_max * 0.99 ... wait, that checks for a LOWER peak.
    # Let's stick to the iterative logic's intent but vectorize.
    # Iterative: recent_max < prev_max * 0.99 AND low std dev.
    
    lookback_20 = 20
    recent_highs = high.rolling(lookback_20)
    prev_highs = high.shift(lookback_20).rolling(lookback_20)
    
    recent_max = recent_highs.max()
    prev_max = prev_highs.max()
    recent_std = recent_highs.std()
    
    # Logic: Peak is slightly lower or equal, and low volatility
    is_double_top = (recent_max < prev_max * 0.99) & ((recent_std / (recent_max + 1e-8)) < 0.02)
    df['double_top'] = np.where(is_double_top, 0.7, 0.0)
    
    # Double Bottom
    recent_lows = low.rolling(lookback_20)
    prev_lows = low.shift(lookback_20).rolling(lookback_20)
    
    recent_min = recent_lows.min()
    prev_min = prev_lows.min()
    recent_low_std = recent_lows.std()
    
    is_double_bottom = (recent_min > prev_min * 1.01) & ((recent_low_std / (recent_min + 1e-8)) < 0.02)
    df['double_bottom'] = np.where(is_double_bottom, 0.7, 0.0)
    
    # Breakout (Close > Resistance)
    # Resistance is max of previous 20 bars
    resistance = high.shift(1).rolling(20).max()
    df['breakout_signal'] = np.where(close > resistance * 1.002, 1.0, 0.0)
    
    # Breakdown (Close < Support)
    support = low.shift(1).rolling(20).min()
    df['breakdown_signal'] = np.where(close < support * 0.998, 1.0, 0.0)
    
    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Price series
        period: RSI period
        
    Returns:
        RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX) indicator.

    ADX measures trend strength (not direction):
    - ADX > 25: Strong trend
    - ADX < 20: Weak trend / ranging market

    Args:
        df: DataFrame with OHLC data
        period: Lookback period (default 14)

    Returns:
        DataFrame with 'adx', 'plus_di', 'minus_di' columns added
    """
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        print("  [!] Warning: OHLC data not found, skipping ADX")
        return df

    high = df['high']
    low = df['low']
    close = df['close']

    # Calculate +DM and -DM (directional movement)
    plus_dm = high.diff()
    minus_dm = -low.diff()

    # Only keep positive values
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # True Range (handles gaps)
    true_range = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)

    # Smoothed ATR
    atr = true_range.rolling(period).mean()

    # Directional Indicators (+DI and -DI)
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-8))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-8))

    # Directional Index (DX)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)

    # Average Directional Index (ADX) - smoothed DX
    adx = dx.rolling(period).mean()

    df['adx'] = adx
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    return df


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Volume Weighted Average Price (VWAP).

    VWAP is used to identify:
    - Fair value (institutional benchmark)
    - Support/resistance levels
    - Price efficiency

    Args:
        df: DataFrame with close and volume data

    Returns:
        DataFrame with 'vwap' column added
    """
    if not all(col in df.columns for col in ['close', 'volume']):
        print("  [!] Warning: Close/volume not found, skipping VWAP")
        return df

    # Reset VWAP at start of each day
    if isinstance(df.index, pd.DatetimeIndex):
        # Daily cumulative sums
        df['date'] = df.index.date

        # Calculate VWAP per day
        df['pv'] = df['close'] * df['volume']
        df['vwap'] = df.groupby('date')['pv'].cumsum() / df.groupby('date')['volume'].cumsum()

        # Clean up temporary columns
        df.drop(['date', 'pv'], axis=1, inplace=True)
    else:
        # Simple VWAP if no datetime index
        pv = df['close'] * df['volume']
        df['vwap'] = pv.cumsum() / df['volume'].cumsum()

    # Handle division by zero
    df['vwap'] = df['vwap'].replace([np.inf, -np.inf], np.nan)
    df['vwap'] = df['vwap'].fillna(df['close'])

    return df


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add session-based features (morning/midday/afternoon).

    Trading sessions have different characteristics:
    - Morning (9:30-11:00): High volatility, trend establishment
    - Midday (11:00-14:00): Lower volatility, consolidation
    - Afternoon (14:00-16:00): Volatility pickup, trend continuation

    Args:
        df: DataFrame with datetime index

    Returns:
        DataFrame with one-hot encoded session features
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        print("  [!] Warning: No datetime index, skipping session features")
        return df

    # Determine session based on hour
    # Ensure timezone-aware (America/New_York = Eastern Time)
    if df.index.tz is None:
        print("  [!] Warning: Index not timezone-aware, assuming America/New_York (ET)")
        idx = df.index.tz_localize('UTC').tz_convert('America/New_York')
    elif 'America/New_York' not in str(df.index.tz) and 'US/Eastern' not in str(df.index.tz):
        # Convert to ET if not already
        idx = df.index.tz_convert('America/New_York')
    else:
        idx = df.index

    # Create session labels
    session = []
    for ts in idx:
        hour = ts.hour
        minute = ts.minute

        if (hour == 9 and minute >= 30) or (hour == 10):
            session.append('morning')
        elif hour >= 11 and hour < 14:
            session.append('midday')
        elif hour >= 14 and hour <= 16:
            session.append('afternoon')
        else:
            session.append('closed')  # Outside RTH

    df['session'] = session

    # One-hot encode
    session_dummies = pd.get_dummies(df['session'], prefix='session')
    df = pd.concat([df, session_dummies], axis=1)

    # Drop original session column
    df.drop('session', axis=1, inplace=True)

    # Ensure we have all expected columns (in case some sessions missing)
    for sess in ['session_morning', 'session_midday', 'session_afternoon', 'session_closed']:
        if sess not in df.columns:
            df[sess] = 0

    return df


def validate_features(df: pd.DataFrame) -> bool:
    """
    Validate that all expected features are present and valid.

    Args:
        df: DataFrame with features

    Returns:
        True if all features valid, False otherwise
    """
    expected_features = [
        'adx', 'vol_regime', 'vol_percentile', 'vwap', 'price_to_vwap',
        'spread', 'efficiency_ratio', 'trend_strength',
        'session_morning', 'session_midday', 'session_afternoon'
    ]

    missing = []
    for feat in expected_features:
        if feat not in df.columns:
            missing.append(feat)

    if missing:
        print(f"[FEATURES] Warning: Missing features: {missing}")
        return False

    # Check for excessive NaNs
    nan_counts = df[expected_features].isna().sum()
    excessive_nans = nan_counts[nan_counts > len(df) * 0.1]  # >10% NaN

    if not excessive_nans.empty:
        print(f"[FEATURES] Warning: Excessive NaNs in: {excessive_nans.to_dict()}")
        return False

    print("[FEATURES] [OK] All features validated")
    return True


if __name__ == '__main__':
    # Test feature engineering on sample data
    print("Testing feature engineering module...")

    # Create sample data
    dates = pd.date_range('2024-01-01 09:30', periods=500, freq='1min', tz='America/New_York')
    test_df = pd.DataFrame({
        'open': np.random.uniform(4000, 4100, 500),
        'high': np.random.uniform(4100, 4150, 500),
        'low': np.random.uniform(3950, 4000, 500),
        'close': np.random.uniform(4000, 4100, 500),
        'volume': np.random.randint(100, 1000, 500),
        'atr': np.random.uniform(10, 30, 500)
    }, index=dates)

    print(f"\nOriginal features: {list(test_df.columns)}")
    print(f"Original shape: {test_df.shape}")

    # Add features
    enhanced_df = add_market_regime_features(test_df)

    print(f"\nEnhanced features: {list(enhanced_df.columns)}")
    print(f"Enhanced shape: {enhanced_df.shape}")

    # Validate
    validate_features(enhanced_df)

    # Show sample
    print("\nSample enhanced data:")
    print(enhanced_df[['close', 'adx', 'vwap', 'vol_percentile', 'session_morning']].head(10))

    print("\n[OK] Feature engineering test complete!")
