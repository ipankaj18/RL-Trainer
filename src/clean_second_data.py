#!/usr/bin/env python3
"""
Clean 1-second futures data for trailing drawdown calculation

Market-agnostic cleaning for all futures markets (ES, NQ, YM, RTY, etc.)

Validates and removes:
- NaN values
- Duplicates
- Anomalous prices (using statistical IQR method)
- Invalid volumes
- Corrupted rows
"""

import pandas as pd
import numpy as np
import os
import sys

# Import centralized validation module
try:
    from data_validator import (
        remove_nan_values as validator_remove_nan,
        remove_duplicates as validator_remove_duplicates,
        validate_prices_statistical as validator_validate_prices,
        detect_price_jumps as validator_detect_jumps
    )
except ImportError:
    # Fallback if running from different directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_validator import (
        remove_nan_values as validator_remove_nan,
        remove_duplicates as validator_remove_duplicates,
        validate_prices_statistical as validator_validate_prices,
        detect_price_jumps as validator_detect_jumps
    )


def load_processed_data(file_path):
    """Load processed but uncleaned 1-second data"""
    print(f"[1/5] Loading processed data...")

    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"      Loaded {len(df):,} rows")
        return df
    except Exception as e:
        print(f"      ERROR: {e}")
        return None


def remove_nan_values(df):
    """Remove rows with NaN values (using centralized validator)"""
    print(f"\n[2/5] Removing NaN values...")
    df, stats = validator_remove_nan(df, verbose=False)
    print(f"      Before: {stats['rows_before']:,} rows")
    print(f"      After: {stats['rows_after']:,} rows")
    print(f"      Removed: {stats['nan_rows_removed']:,} rows ({stats['removal_pct']:.2f}%)")
    return df


def remove_duplicates(df):
    """Remove duplicate timestamps (using centralized validator)"""
    print(f"\n[3/5] Removing duplicates...")
    df, stats = validator_remove_duplicates(df, verbose=False)
    print(f"      Before: {stats['rows_before']:,} rows")
    print(f"      After: {stats['rows_after']:,} rows")
    print(f"      Removed: {stats['duplicates_removed']:,} duplicates ({stats['removal_pct']:.2f}%)")
    return df


def validate_prices(df):
    """Validate price values using market-agnostic statistical validation (using centralized validator)"""
    print(f"\n[4/5] Validating prices...")
    df, stats = validator_validate_prices(df, verbose=False)
    print(f"      Median price: ${stats['median']:,.2f}")
    print(f"      Valid range: ${stats['min_price']:,.2f} - ${stats['max_price']:,.2f}")
    if stats['rejected'] > 0:
        print(f"      Violations detected: {stats['rejected']:,} rows")
    print(f"      Before: {stats['rows_before']:,} rows")
    print(f"      After: {stats['rows_after']:,} rows")
    print(f"      Removed: {stats['rejected']:,} rows ({stats['rejection_pct']:.2f}%)")
    return df


def detect_price_jumps(df, threshold_pct=5.0):
    """Detect anomalous price jumps (using centralized validator)"""
    print(f"\n[5/5] Detecting anomalous price jumps (>{threshold_pct}%)...")
    df, stats = validator_detect_jumps(df, threshold_pct=threshold_pct, verbose=False)
    if stats['jumps_found'] > 0:
        print(f"      Found {stats['jumps_found']:,} potential anomalies")
        print(f"      Max jump: {stats['max_jump_pct']:.2f}%")
    print(f"      Before: {stats['rows_before']:,} rows")
    print(f"      After: {stats['rows_after']:,} rows")
    removed = stats['rows_before'] - stats['rows_after']
    print(f"      Removed: {removed:,} rows ({(removed/stats['rows_before']*100):.2f}%)")
    return df


def save_cleaned_data(df, output_path):
    """Save cleaned data"""
    print(f"\nSaving cleaned data...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

    print(f"      Saved to: {output_path}")
    print(f"      Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"      Rows: {len(df):,}")
    print(f"      Time range: {df.index.min()} to {df.index.max()}")

    return output_path


def main(input_path=None, output_path=None):
    """
    Main cleaning function

    Args:
        input_path: Path to input raw CSV file (optional)
        output_path: Path to output cleaned CSV file (optional)
    """
    print("=" * 80)
    print("FUTURES 1-SECOND DATA CLEANING (MARKET-AGNOSTIC)")
    print("=" * 80)

    # Default paths (for backward compatibility)
    if input_path is None:
        input_path = "data/es_second_level_data_raw.csv"
    if output_path is None:
        output_path = "data/es_second_level_data.csv"

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        print(f"Run process_second_data.py first")
        return False

    try:
        # Step 1: Load data
        df = load_processed_data(input_path)
        if df is None:
            return False

        # Step 2: Remove NaN
        df = remove_nan_values(df)

        # Step 3: Remove duplicates
        df = remove_duplicates(df)

        # Step 4: Validate prices
        df = validate_prices(df)

        # Step 5: Detect price jumps
        df = detect_price_jumps(df, threshold_pct=5.0)

        # Save cleaned data
        save_cleaned_data(df, output_path)

        print("\n" + "=" * 80)
        print("CLEANING COMPLETE")
        print("=" * 80)
        print(f"Cleaned output: {output_path}")
        print(f"Data ready for trailing drawdown calculation")

        # Delete raw intermediate file
        if os.path.exists(input_path):
            os.remove(input_path)
            print(f"Removed intermediate file: {input_path}")

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Clean 1-second futures data (market-agnostic)')
    parser.add_argument('--input', type=str, help='Input raw CSV file path')
    parser.add_argument('--output', type=str, help='Output cleaned CSV file path')
    args = parser.parse_args()

    success = main(input_path=args.input, output_path=args.output)
    sys.exit(0 if success else 1)
