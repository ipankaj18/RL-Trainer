#!/usr/bin/env python3
"""
Centralized Data Validation Module

Reusable validation functions for futures market data processing.
Handles common data quality issues across all markets (ES, NQ, YM, RTY, etc.)

Key Features:
- Price format detection and correction (divide-by-100 errors)
- Market-agnostic statistical validation using IQR
- OHLC consistency checks
- Volume validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


# Expected median prices for each market (used for format detection)
# Updated Nov 2025 to current market levels
EXPECTED_MEDIAN_PRICES = {
    'ES': 6400,      # E-mini S&P 500: updated from 5500 (Nov 2025 levels: 6200-6500)
    'NQ': 24000,     # E-mini Nasdaq-100: updated from 18000 (Nov 2025 levels: 23000-25000)
    'YM': 44000,     # E-mini Dow: updated from 35000 (Nov 2025 levels: 42000-46000)
    'RTY': 2200,     # E-mini Russell 2000: updated from 2100 (Nov 2025 levels: 2100-2300)
    'MES': 6400,     # Micro E-mini S&P 500: same as ES
    'MNQ': 24000,    # Micro E-mini Nasdaq-100: same as NQ
    'M2K': 2200,     # Micro E-mini Russell 2000: same as RTY
    'MYM': 44000,    # Micro E-mini Dow: same as YM
}


def detect_and_fix_price_format(df: pd.DataFrame, market: str, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect and fix price format inconsistencies from vendor data.

    Some vendors provide mixed formats within the same dataset:
    - Normal format: 22655.00 (dollars)
    - Corrupted format: 226.55 (appears to be cents divided by 100)

    This function detects the divide-by-100 error by comparing actual median
    price to expected median, then multiplies corrupted bars by 100.

    Args:
        df: DataFrame with OHLC price columns
        market: Market symbol (ES, NQ, YM, etc.)
        verbose: If True, prints detailed diagnostic info

    Returns:
        Tuple of (corrected_df, stats_dict)

    Example:
        df, stats = detect_and_fix_price_format(df, 'NQ')
        if stats['fixed_count'] > 0:
            print(f"Fixed {stats['fixed_count']} corrupted bars")
    """
    if market not in EXPECTED_MEDIAN_PRICES:
        raise ValueError(f"Unknown market: {market}. Valid markets: {list(EXPECTED_MEDIAN_PRICES.keys())}")

    df = df.copy()
    stats = {
        'total_bars': len(df),
        'fixed_count': 0,
        'fixed_pct': 0.0,
        'median_before': 0.0,
        'median_after': 0.0,
        'format_ok': True
    }

    if len(df) == 0:
        return df, stats

    # Get expected median and calculate percentiles
    expected_median = EXPECTED_MEDIAN_PRICES[market]
    actual_median = df['close'].median()
    p1 = df['close'].quantile(0.01)   # 1st percentile
    p5 = df['close'].quantile(0.05)   # 5th percentile
    p95 = df['close'].quantile(0.95)  # 95th percentile

    stats['median_before'] = actual_median

    if verbose:
        print(f"\n  [PRICE FORMAT CHECK]")
        print(f"    Market: {market}")
        print(f"    Expected median: ${expected_median:,.2f}")
        print(f"    Actual median: ${actual_median:,.2f}")
        print(f"    P1: ${p1:,.2f} | P5: ${p5:,.2f} | P95: ${p95:,.2f}")

    # NEW LOGIC: Check for BIMODAL distribution (minority corruption)
    # Check P1 and P5 to catch even tiny amounts of corruption
    threshold_low = expected_median * 0.50  # 50% of expected median

    # Check if P1 OR P5 is suspiciously low (catches corruption in bottom 1-5%)
    if p1 < threshold_low or p5 < threshold_low:
        # Found suspiciously low prices in bottom 5%
        # Check if this looks like divide-by-100 error
        corrupted_mask = df['close'] < threshold_low
        corrupted_count = corrupted_mask.sum()
        corrupted_pct = (corrupted_count / len(df)) * 100

        if corrupted_count > 0:
            corrupted_median = df.loc[corrupted_mask, 'close'].median()
            ratio = expected_median / corrupted_median if corrupted_median > 0 else 0

            if verbose:
                print(f"    ⚠️  BIMODAL CORRUPTION DETECTED!")
                print(f"    Corrupted rows: {corrupted_count:,} ({corrupted_pct:.1f}%)")
                print(f"    Corrupted median: ${corrupted_median:,.2f}")
                print(f"    Ratio: {ratio:.1f}x too low")

            # If ratio is close to 100, it's the divide-by-100 error
            if 50 < ratio < 150:
                if verbose:
                    print(f"    → Applying 100x correction to corrupted rows...")

                # Fix all OHLC columns for corrupted bars
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        df.loc[corrupted_mask, col] = df.loc[corrupted_mask, col] * 100

                # Update stats
                stats['fixed_count'] = corrupted_count
                stats['fixed_pct'] = corrupted_pct
                stats['median_after'] = df['close'].median()
                stats['format_ok'] = False

                # Verify fix worked
                new_median = df['close'].median()
                new_p5 = df['close'].quantile(0.05)

                if verbose:
                    print(f"    New median: ${new_median:,.2f}")
                    print(f"    New P5: ${new_p5:,.2f}")

                # Check if P5 is now in reasonable range (> 80% of expected)
                if new_p5 > expected_median * 0.80:
                    if verbose:
                        print(f"    ✅ Fix successful! All prices in normal range")
                else:
                    if verbose:
                        print(f"    ⚠️  Fix may be incomplete. P5 still low: ${new_p5:,.2f}")
            else:
                if verbose:
                    print(f"    ⚠️  Ratio {ratio:.1f}x doesn't match divide-by-100 pattern")
                    print(f"    Manual inspection recommended")
    else:
        # No corruption detected - all percentiles in reasonable range
        if verbose:
            print(f"    ✅ Price format OK - No corruption detected")
        stats['median_after'] = actual_median

    return df, stats


def validate_prices_statistical(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Market-agnostic price validation using statistical IQR method.

    Instead of hardcoded min/max prices (which don't work across markets),
    this uses the Interquartile Range (IQR) method to detect outliers.

    Valid range: [Q1 - 3*IQR, Q3 + 3*IQR]
    This captures ~99.7% of normal data and rejects extreme outliers.

    Args:
        df: DataFrame with OHLC price columns
        verbose: If True, prints validation statistics

    Returns:
        Tuple of (validated_df, stats_dict)

    Example:
        df, stats = validate_prices_statistical(df)
        print(f"Rejected {stats['rejected']} outliers")
    """
    df = df.copy()
    rows_before = len(df)

    stats = {
        'rows_before': rows_before,
        'rows_after': 0,
        'rejected': 0,
        'rejection_pct': 0.0,
        'median': 0.0,
        'min_price': 0.0,
        'max_price': 0.0
    }

    if len(df) == 0:
        return df, stats

    # Calculate statistical bounds using IQR
    median_price = df['close'].median()
    q1 = df['close'].quantile(0.25)
    q3 = df['close'].quantile(0.75)
    iqr = q3 - q1

    # Set bounds at 3 IQR from quartiles (captures 99.7% of normal data)
    min_price = max(0, q1 - (3 * iqr))  # Can't be negative
    max_price = q3 + (3 * iqr)

    stats['median'] = median_price
    stats['min_price'] = min_price
    stats['max_price'] = max_price

    if verbose:
        print(f"\n  [STATISTICAL VALIDATION]")
        print(f"    Median price: ${median_price:,.2f}")
        print(f"    Valid range: ${min_price:,.2f} - ${max_price:,.2f}")

    # Check OHLC in valid range
    valid_prices = (
        (df['open'] >= min_price) & (df['open'] <= max_price) &
        (df['high'] >= min_price) & (df['high'] <= max_price) &
        (df['low'] >= min_price) & (df['low'] <= max_price) &
        (df['close'] >= min_price) & (df['close'] <= max_price)
    )

    # Check high >= low
    valid_ohlc = (df['high'] >= df['low'])

    # Check volume is positive
    valid_volume = (df['volume'] > 0) if 'volume' in df.columns else True

    # Combine all checks
    valid_rows = valid_prices & valid_ohlc & valid_volume

    # Report violations
    price_violations = (~valid_prices).sum()
    ohlc_violations = (~valid_ohlc).sum()
    volume_violations = (~valid_volume).sum() if 'volume' in df.columns else 0

    if verbose and (price_violations > 0 or ohlc_violations > 0 or volume_violations > 0):
        print(f"    Violations found:")
        if price_violations > 0:
            print(f"      - Price range: {price_violations:,}")
        if ohlc_violations > 0:
            print(f"      - High < Low: {ohlc_violations:,}")
        if volume_violations > 0:
            print(f"      - Invalid volume: {volume_violations:,}")

    df = df[valid_rows].copy()

    rows_after = len(df)
    rejected = rows_before - rows_after

    stats['rows_after'] = rows_after
    stats['rejected'] = rejected
    stats['rejection_pct'] = (rejected / rows_before * 100) if rows_before > 0 else 0

    if verbose:
        print(f"    Rejected: {rejected:,} rows ({stats['rejection_pct']:.2f}%)")

    return df, stats


def detect_price_jumps(df: pd.DataFrame, threshold_pct: float = 5.0, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect and remove anomalous price jumps.

    Large sudden price changes (>5% by default) are likely data errors.
    Real futures markets have circuit breakers that prevent such jumps.

    Args:
        df: DataFrame with close prices
        threshold_pct: Maximum allowed price change percentage
        verbose: If True, prints jump statistics

    Returns:
        Tuple of (cleaned_df, stats_dict)

    Example:
        df, stats = detect_price_jumps(df, threshold_pct=5.0)
        if stats['jumps_found'] > 0:
            print(f"Removed {stats['jumps_found']} anomalous price jumps")
    """
    df = df.copy()
    rows_before = len(df)

    stats = {
        'rows_before': rows_before,
        'rows_after': 0,
        'jumps_found': 0,
        'max_jump_pct': 0.0
    }

    if len(df) <= 1:
        stats['rows_after'] = len(df)
        return df, stats

    # Calculate price changes
    price_change_pct = np.abs(df['close'].pct_change() * 100)

    # Find jumps exceeding threshold
    jumps = price_change_pct > threshold_pct

    # Don't flag first row (can't have change)
    jumps.iloc[0] = False

    n_jumps = jumps.sum()
    max_jump = price_change_pct.max()

    stats['jumps_found'] = n_jumps
    stats['max_jump_pct'] = max_jump

    if verbose:
        print(f"\n  [PRICE JUMP DETECTION]")
        print(f"    Threshold: {threshold_pct}%")
        if n_jumps > 0:
            print(f"    Found: {n_jumps:,} anomalous jumps")
            print(f"    Max jump: {max_jump:.2f}%")
        else:
            print(f"    No anomalous jumps found ✅")

    # Remove jumped rows
    if n_jumps > 0:
        df = df[~jumps].copy()

    stats['rows_after'] = len(df)

    return df, stats


def remove_duplicates(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove duplicate timestamps.

    Args:
        df: DataFrame with datetime index
        verbose: If True, prints duplicate statistics

    Returns:
        Tuple of (deduplicated_df, stats_dict)
    """
    rows_before = len(df)

    # Remove duplicate timestamps (keep first)
    df = df[~df.index.duplicated(keep='first')].copy()

    rows_after = len(df)
    removed = rows_before - rows_after

    stats = {
        'rows_before': rows_before,
        'rows_after': rows_after,
        'duplicates_removed': removed,
        'removal_pct': (removed / rows_before * 100) if rows_before > 0 else 0
    }

    if verbose:
        print(f"\n  [DUPLICATE REMOVAL]")
        print(f"    Removed: {removed:,} duplicates ({stats['removal_pct']:.2f}%)")

    return df, stats


def remove_nan_values(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove rows with NaN values.

    Args:
        df: DataFrame
        verbose: If True, prints NaN statistics

    Returns:
        Tuple of (cleaned_df, stats_dict)
    """
    rows_before = len(df)

    # Check for NaN in any column
    df = df.dropna().copy()

    rows_after = len(df)
    removed = rows_before - rows_after

    stats = {
        'rows_before': rows_before,
        'rows_after': rows_after,
        'nan_rows_removed': removed,
        'removal_pct': (removed / rows_before * 100) if rows_before > 0 else 0
    }

    if verbose:
        print(f"\n  [NAN REMOVAL]")
        print(f"    Removed: {removed:,} rows ({stats['removal_pct']:.2f}%)")

    return df, stats


def full_validation_pipeline(df: pd.DataFrame, market: str, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete validation pipeline combining all checks.

    Recommended order:
    1. Remove NaN values
    2. Remove duplicates
    3. Fix price format issues
    4. Validate prices statistically
    5. Detect price jumps

    Args:
        df: Raw DataFrame
        market: Market symbol (ES, NQ, etc.)
        verbose: If True, prints detailed progress

    Returns:
        Tuple of (validated_df, combined_stats_dict)

    Example:
        df_clean, stats = full_validation_pipeline(df_raw, 'NQ')
        print(f"Original: {stats['total_rows_input']}")
        print(f"Clean: {stats['total_rows_output']}")
        print(f"Rejected: {stats['total_rows_rejected']} ({stats['rejection_pct']:.2f}%)")
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"FULL VALIDATION PIPELINE - {market}")
        print(f"{'='*70}")

    initial_rows = len(df)
    combined_stats = {
        'market': market,
        'total_rows_input': initial_rows,
        'total_rows_output': 0,
        'total_rows_rejected': 0,
        'rejection_pct': 0.0,
    }

    # Step 1: Remove NaN
    df, nan_stats = remove_nan_values(df, verbose=verbose)
    combined_stats['nan_removal'] = nan_stats

    # Step 2: Remove duplicates
    df, dup_stats = remove_duplicates(df, verbose=verbose)
    combined_stats['duplicate_removal'] = dup_stats

    # Step 3: Fix price format
    df, format_stats = detect_and_fix_price_format(df, market, verbose=verbose)
    combined_stats['format_fix'] = format_stats

    # Step 4: Statistical validation
    df, price_stats = validate_prices_statistical(df, verbose=verbose)
    combined_stats['price_validation'] = price_stats

    # Step 5: Price jump detection
    df, jump_stats = detect_price_jumps(df, threshold_pct=5.0, verbose=verbose)
    combined_stats['jump_detection'] = jump_stats

    # Calculate totals
    final_rows = len(df)
    combined_stats['total_rows_output'] = final_rows
    combined_stats['total_rows_rejected'] = initial_rows - final_rows
    combined_stats['rejection_pct'] = ((initial_rows - final_rows) / initial_rows * 100) if initial_rows > 0 else 0

    if verbose:
        print(f"\n{'='*70}")
        print(f"VALIDATION COMPLETE")
        print(f"{'='*70}")
        print(f"  Input rows: {initial_rows:,}")
        print(f"  Output rows: {final_rows:,}")
        print(f"  Rejected: {initial_rows - final_rows:,} ({combined_stats['rejection_pct']:.2f}%)")
        print(f"{'='*70}\n")

    return df, combined_stats


if __name__ == "__main__":
    print("Data Validator Module")
    print("=" * 70)
    print("This module provides reusable validation functions.")
    print("\nAvailable functions:")
    print("  - detect_and_fix_price_format()")
    print("  - validate_prices_statistical()")
    print("  - detect_price_jumps()")
    print("  - remove_duplicates()")
    print("  - remove_nan_values()")
    print("  - full_validation_pipeline()")
    print("\nImport this module in your data processing scripts:")
    print("  from data_validator import full_validation_pipeline")
    print("  df_clean, stats = full_validation_pipeline(df_raw, 'NQ')")
