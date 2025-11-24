#!/usr/bin/env python3
"""
Technical Indicators Module

Provides reusable technical indicator calculations for ES futures data.
Used by automated weekly data update pipeline.

Indicators:
- SMA (Simple Moving Average)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- ATR (Average True Range)
- Bollinger Bands
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)
- ADX (Average Directional Index)
- ROC (Rate of Change)
- MFI (Money Flow Index)
"""

import pandas as pd
import numpy as np
from typing import Optional


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to OHLCV dataframe.

    Args:
        df: DataFrame with columns: open, high, low, close, volume

    Returns:
        DataFrame with all technical indicators added
    """
    print("[INDICATORS] Adding technical indicators...")

    df = df.copy()

    # Moving Averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    print("  [OK] Moving averages (SMA-5, SMA-20)")

    # RSI
    df['rsi'] = calculate_rsi(df['close'], period=14)
    print("  [OK] RSI")

    # MACD
    df['macd'] = calculate_macd(df['close'])
    print("  [OK] MACD")

    # Momentum
    df['momentum'] = calculate_momentum(df['close'], period=10)
    print("  [OK] Momentum")

    # ATR (Average True Range)
    df['atr'] = calculate_atr(df, period=14)
    print("  [OK] ATR")

    # Volatility (std of close)
    df['volatility'] = df['close'].rolling(20).std()
    print("  [OK] Volatility")

    # Bollinger Bands Width
    df['bb_width'] = calculate_bollinger_width(df['close'], period=20)
    print("  [OK] Bollinger Bands Width")

    # Stochastic Oscillator
    stoch_k, stoch_d = calculate_stochastic(df, period=14)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    print("  [OK] Stochastic Oscillator")

    # Williams %R
    df['williams_r'] = calculate_williams_r(df, period=14)
    print("  [OK] Williams %R")

    # CCI (Commodity Channel Index)
    df['cci'] = calculate_cci(df, period=20)
    print("  [OK] CCI")

    # ADX (Average Directional Index)
    df['adx'] = calculate_adx(df, period=14)
    print("  [OK] ADX")

    # ROC (Rate of Change)
    df['roc'] = calculate_roc(df['close'], period=10)
    print("  [OK] ROC")

    # MFI (Money Flow Index)
    df['mfi'] = calculate_mfi(df, period=14)
    print("  [OK] MFI")

    # Fill NaN values (from rolling windows)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)

    print(f"[INDICATORS] Total columns: {len(df.columns)}")
    return df


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    MACD = EMA(12) - EMA(26)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow

    return macd


def calculate_momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Momentum (rate of price change).

    Momentum = Close / Close[n periods ago]
    """
    momentum = series / series.shift(period)
    return momentum


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    True Range = max(high - low, |high - close_prev|, |low - close_prev|)
    ATR = SMA(True Range, period)
    """
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    # Use DataFrame constructor instead of concat to handle duplicate indices
    true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = true_range.rolling(period).mean()

    return atr


def calculate_bollinger_width(series: pd.Series, period: int = 20, num_std: int = 2) -> pd.Series:
    """
    Calculate Bollinger Bands Width.

    BB Width = (Upper Band - Lower Band) / Middle Band
    where Upper = SMA + (num_std * std)
          Lower = SMA - (num_std * std)
    """
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()

    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)

    bb_width = (upper_band - lower_band) / (sma + 1e-8)

    return bb_width


def calculate_stochastic(df: pd.DataFrame, period: int = 14) -> tuple:
    """
    Calculate Stochastic Oscillator (%K and %D).

    %K = 100 * (Close - Low[period]) / (High[period] - Low[period])
    %D = SMA(%K, 3)
    """
    high = df['high']
    low = df['low']
    close = df['close']

    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()

    stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-8))
    stoch_d = stoch_k.rolling(3).mean()

    return stoch_k, stoch_d


def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Williams %R.

    %R = -100 * (High[period] - Close) / (High[period] - Low[period])
    """
    high = df['high']
    low = df['low']
    close = df['close']

    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()

    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-8))

    return williams_r


def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI).

    CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
    where Typical Price = (High + Low + Close) / 3
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma = typical_price.rolling(period).mean()
    mean_dev = (typical_price - sma).abs().rolling(period).mean()

    cci = (typical_price - sma) / (0.015 * mean_dev + 1e-8)

    return cci


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).

    Measures trend strength (not direction).
    ADX > 25: Strong trend
    ADX < 20: Weak trend
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    # Use DataFrame constructor instead of concat to handle duplicate indices
    true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

    # ATR
    atr = true_range.rolling(period).mean()

    # Directional Indicators
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-8))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-8))

    # Directional Index
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)

    # ADX (smoothed DX)
    adx = dx.rolling(period).mean()

    return adx


def calculate_roc(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Rate of Change (ROC).

    ROC = (Close - Close[n]) / Close[n]
    """
    roc = (series - series.shift(period)) / (series.shift(period) + 1e-8)
    return roc


def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Money Flow Index (MFI).

    Volume-weighted RSI.
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    # Positive and negative money flow
    delta = typical_price.diff()
    positive_flow = money_flow.where(delta > 0, 0).rolling(period).sum()
    negative_flow = money_flow.where(delta < 0, 0).rolling(period).sum()

    # Money Flow Ratio
    mfr = positive_flow / (negative_flow + 1e-8)

    # MFI
    mfi = 100 - (100 / (1 + mfr))

    return mfi


if __name__ == "__main__":
    # Test with sample data
    import pandas as pd

    # Create sample OHLCV data
    dates = pd.date_range('2025-01-01', periods=100, freq='1min')
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 6300,
        'high': np.random.randn(100).cumsum() + 6305,
        'low': np.random.randn(100).cumsum() + 6295,
        'close': np.random.randn(100).cumsum() + 6300,
        'volume': np.random.randint(100, 10000, 100)
    }, index=dates)

    # Add indicators
    df = add_all_indicators(df)

    print("\nSample output:")
    print(df.tail())
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
