#!/usr/bin/env python3
"""
Process 1-second futures data for real-time trailing drawdown calculation

Market-agnostic processing for all futures markets (ES, NQ, YM, RTY, etc.)

Input: Raw 1-second OHLCV data from GLBX
Output: Clean 1-second data compatible with training environments
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys
from zoneinfo import ZoneInfo

# Import centralized validation module
try:
    from data_validator import detect_and_fix_price_format
except ImportError:
    # Fallback if running from different directory
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_validator import detect_and_fix_price_format


def load_raw_second_data(file_path):
    """Load raw 1-second OHLCV data"""
    print(f"[1/7] Loading raw 1-second data from {file_path}...")

    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"      Loaded {len(df):,} rows")
        print(f"      Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"      ERROR: {e}")
        return None


def filter_es_contracts(df, market='ES'):
    """Filter for futures contracts based on market prefix (ES, NQ, YM, etc.)"""
    print(f"\n[2/7] Filtering for {market} futures contracts...")

    # Get all unique symbols that start with the market prefix
    market_symbols = [sym for sym in df['symbol'].unique() if str(sym).startswith(market)]
    df_filtered = df[df['symbol'].isin(market_symbols)].copy()

    # Display counts for each contract found
    for symbol in sorted(market_symbols):
        count = len(df_filtered[df_filtered['symbol'] == symbol])
        print(f"      {symbol} records: {count:,}")

    print(f"      Total {market} records: {len(df_filtered):,}")

    return df_filtered


def create_continuous_contract(df_filtered):
    """Create continuous contract data by handling contract rollovers"""
    print(f"\n[3/7] Creating continuous contract...")

    # Sort by timestamp
    df_filtered = df_filtered.sort_values('ts_event')

    # Get all unique contracts
    unique_contracts = sorted(df_filtered['symbol'].unique())

    if len(unique_contracts) == 0:
        print("      WARNING: No contracts found")
        return df_filtered

    # If multiple contracts exist, show rollover information
    if len(unique_contracts) > 1:
        print(f"      Found {len(unique_contracts)} contracts: {', '.join(unique_contracts)}")
        contract_data = []
        for contract in unique_contracts:
            data = df_filtered[df_filtered['symbol'] == contract]
            if len(data) > 0:
                start = data['ts_event'].min()
                end = data['ts_event'].max()
                print(f"      {contract}: {start} to {end}")
                contract_data.append(data)

        # Combine all contracts chronologically
        df_continuous = pd.concat(contract_data)
        df_continuous = df_continuous.sort_values('ts_event')
    else:
        print(f"      Single contract: {unique_contracts[0]}")
        df_continuous = df_filtered

    print(f"      Continuous data: {len(df_continuous):,} records")

    return df_continuous


def convert_timestamp_and_timezone(df):
    """Convert timestamp to US/Eastern timezone"""
    print(f"\n[4/7] Converting timezone...")

    # Parse timestamp (utc=True handles the Z suffix and UTC timezone)
    df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True)

    # Convert UTC to US/Eastern using zoneinfo
    eastern = ZoneInfo("America/New_York")
    df['ts_event'] = df['ts_event'].dt.tz_convert(eastern)
    df = df.set_index('ts_event')

    print(f"      Time range: {df.index.min()} to {df.index.max()}")
    print(f"      Timezone: {df.index.tz}")

    return df


def filter_trading_hours(df):
    """Filter for Apex trading hours (8:30 AM - 4:59 PM ET)"""
    print(f"\n[5/7] Filtering for Apex trading hours (8:30 AM - 4:59 PM ET)...")

    # Extract hour and minute
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute

    # Apex Trading Hours: 8:30 AM - 4:59 PM ET (08:30-16:59)
    mask_rth = (
        ((df['hour'] == 8) & (df['minute'] >= 30)) |
        ((df['hour'] >= 9) & (df['hour'] < 16)) |
        ((df['hour'] == 16) & (df['minute'] <= 59))
    )

    df_rth = df[mask_rth].copy()
    df_rth = df_rth.drop(['hour', 'minute'], axis=1)

    rows_before = len(df)
    rows_after = len(df_rth)

    print(f"      Before filtering: {rows_before:,} records")
    print(f"      After filtering: {rows_after:,} records")
    print(f"      Removed: {rows_before - rows_after:,} records ({((rows_before-rows_after)/rows_before*100):.1f}%)")

    return df_rth


def extract_ohlcv_only(df):
    """Extract only OHLCV columns"""
    print(f"\n[6/7] Extracting OHLCV data...")

    # Keep only OHLCV
    required_cols = ['open', 'high', 'low', 'close', 'volume']

    df_ohlcv = df[required_cols].copy()

    # Convert to numeric
    for col in required_cols:
        df_ohlcv[col] = pd.to_numeric(df_ohlcv[col], errors='coerce')

    print(f"      Final columns: {list(df_ohlcv.columns)}")
    print(f"      Records: {len(df_ohlcv):,}")

    return df_ohlcv


def save_processed_data(df, output_path):
    """Save processed data"""
    print(f"\nSaving processed data...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

    print(f"      Saved to: {output_path}")
    print(f"      Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"      Rows: {len(df):,}")

    return output_path


def main(input_file=None, output_path=None, market='ES'):
    """
    Main processing function

    Args:
        input_file: Path to input 1s OHLCV CSV file (optional)
        output_path: Path to output processed CSV file (optional)
        market: Market prefix to filter (ES, NQ, YM, etc.)
    """
    print("=" * 80)
    print(f"{market} FUTURES 1-SECOND DATA PROCESSING")
    print("=" * 80)

    # Default paths (for backward compatibility)
    if input_file is None:
        input_dir = "data/GLBX-20251015-FHDV4GDHRX"
        input_file = os.path.join(input_dir, "glbx-mdp3-20250713-20251011.ohlcv-1s.csv")

    if output_path is None:
        output_path = "data/es_second_level_data_raw.csv"

    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return False

    try:
        # Step 1: Load raw data
        df = load_raw_second_data(input_file)
        if df is None:
            return False

        # Step 2: Filter market contracts
        df = filter_es_contracts(df, market=market)
        if len(df) == 0:
            print(f"ERROR: No {market} contracts found")
            return False

        # Step 3: Create continuous contract
        df = create_continuous_contract(df)

        # Step 4: Convert timezone
        df = convert_timestamp_and_timezone(df)

        # Step 5: Filter trading hours
        df = filter_trading_hours(df)
        if len(df) == 0:
            print("ERROR: No data after filtering trading hours")
            return False

        # Step 6: Extract OHLCV only
        df = extract_ohlcv_only(df)

        # Step 7: Validate and fix price format (CRITICAL: detect divide-by-100 errors)
        print(f"\n[7/7] Validating and fixing price format...")
        df, format_stats = detect_and_fix_price_format(df, market, verbose=True)

        if format_stats['fixed_count'] > 0:
            print(f"      ✅ Corrected {format_stats['fixed_count']:,} corrupted bars ({format_stats['fixed_pct']:.1f}%)")
        else:
            print(f"      ✅ No price format issues detected")

        # Save
        save_processed_data(df, output_path)

        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Output file: {output_path}")
        print(f"Next step: python3 clean_second_data.py")

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process 1-second futures data (market-agnostic)')
    parser.add_argument('--input', type=str, help='Input 1s OHLCV CSV file path')
    parser.add_argument('--output', type=str, help='Output processed CSV file path')
    parser.add_argument('--market', type=str, default='ES', help='Market to process (ES, NQ, YM, etc.)')
    args = parser.parse_args()

    success = main(input_file=args.input, output_path=args.output, market=args.market)
    sys.exit(0 if success else 1)
