#!/usr/bin/env python3
"""
Data Quality Validation Script

Checks training data files for corruption and provides detailed statistics.
Use this to verify data quality after reprocessing.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path


# Expected median prices (Nov 2025 levels)
EXPECTED_MEDIANS = {
    'ES': 6400,
    'NQ': 24000,
    'YM': 44000,
    'RTY': 2200,
    'MES': 6400,
    'MNQ': 24000,
    'M2K': 2200,
    'MYM': 44000,
}


def check_file_exists(market: str, data_type: str = 'D1M') -> tuple:
    """Check if data file exists and return path and size"""
    data_dir = Path(__file__).parent.parent / "data"
    file_path = data_dir / f"{market}_{data_type}.csv"

    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        return True, file_path, size_mb
    else:
        return False, None, 0


def analyze_data_quality(market: str, file_path: Path) -> dict:
    """Analyze data file for corruption"""
    print(f"\n{'='*70}")
    print(f"ANALYZING {market} DATA QUALITY")
    print(f"{'='*70}")

    try:
        # Load data
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        total_rows = len(df)

        print(f"File: {file_path.name}")
        print(f"Total rows: {total_rows:,}")

        # Get expected median
        expected_median = EXPECTED_MEDIANS.get(market, 5000)

        # Calculate statistics
        actual_median = df['close'].median()
        p1 = df['close'].quantile(0.01)
        p5 = df['close'].quantile(0.05)
        p25 = df['close'].quantile(0.25)
        p75 = df['close'].quantile(0.75)
        p95 = df['close'].quantile(0.95)
        p99 = df['close'].quantile(0.99)
        min_price = df['close'].min()
        max_price = df['close'].max()

        print(f"\nPrice Statistics:")
        print(f"  Expected median: ${expected_median:,.2f}")
        print(f"  Actual median:   ${actual_median:,.2f}")
        print(f"  Min:             ${min_price:,.2f}")
        print(f"  P1:              ${p1:,.2f}")
        print(f"  P5:              ${p5:,.2f}")
        print(f"  P25:             ${p25:,.2f}")
        print(f"  P75:             ${p75:,.2f}")
        print(f"  P95:             ${p95:,.2f}")
        print(f"  P99:             ${p99:,.2f}")
        print(f"  Max:             ${max_price:,.2f}")

        # Check for corruption (prices below 50% of expected)
        threshold_low = expected_median * 0.50
        corrupted_mask = df['close'] < threshold_low
        corrupted_count = corrupted_mask.sum()
        corrupted_pct = (corrupted_count / total_rows) * 100

        # Check for suspiciously high prices (above 150% of expected)
        threshold_high = expected_median * 1.50
        high_mask = df['close'] > threshold_high
        high_count = high_mask.sum()
        high_pct = (high_count / total_rows) * 100

        print(f"\nCorruption Analysis:")
        print(f"  Threshold (low):  ${threshold_low:,.2f} (50% of expected)")
        print(f"  Corrupted rows:   {corrupted_count:,} ({corrupted_pct:.2f}%)")

        if corrupted_count > 0:
            corrupted_median = df.loc[corrupted_mask, 'close'].median()
            corrupted_min = df.loc[corrupted_mask, 'close'].min()
            corrupted_max = df.loc[corrupted_mask, 'close'].max()
            print(f"  Corrupted median: ${corrupted_median:,.2f}")
            print(f"  Corrupted range:  ${corrupted_min:,.2f} - ${corrupted_max:,.2f}")

            # Check if it's divide-by-100 error
            ratio = expected_median / corrupted_median if corrupted_median > 0 else 0
            print(f"  Corruption ratio: {ratio:.1f}x")

            if 50 < ratio < 150:
                print(f"  ⚠️  DIVIDE-BY-100 ERROR DETECTED!")
            else:
                print(f"  ⚠️  UNKNOWN CORRUPTION PATTERN")

        print(f"\nHigh Price Analysis:")
        print(f"  Threshold (high): ${threshold_high:,.2f} (150% of expected)")
        print(f"  High price rows:  {high_count:,} ({high_pct:.2f}%)")

        # Overall assessment
        print(f"\n{'='*70}")

        if corrupted_count == 0 and p5 > expected_median * 0.80:
            print("✅ DATA QUALITY: EXCELLENT")
            print("   No corruption detected. All prices in normal range.")
            status = "EXCELLENT"
        elif corrupted_count > 0 and corrupted_pct < 1.0:
            print("⚠️  DATA QUALITY: MINOR ISSUES")
            print(f"   {corrupted_pct:.2f}% of data may be corrupted.")
            print("   Recommend reprocessing from source.")
            status = "MINOR_ISSUES"
        elif corrupted_count > 0 and corrupted_pct >= 1.0:
            print("❌ DATA QUALITY: MAJOR CORRUPTION")
            print(f"   {corrupted_pct:.2f}% of data is corrupted!")
            print("   MUST reprocess from source before training.")
            status = "CORRUPTED"
        elif p5 < expected_median * 0.50:
            print("⚠️  DATA QUALITY: SUSPICIOUS")
            print("   P5 is unusually low. May indicate corruption.")
            print("   Recommend manual inspection.")
            status = "SUSPICIOUS"
        else:
            print("✅ DATA QUALITY: GOOD")
            print("   Prices deviate from expected but within acceptable range.")
            print("   May be due to market movement.")
            status = "GOOD"

        print(f"{'='*70}\n")

        return {
            'status': status,
            'total_rows': total_rows,
            'corrupted_count': corrupted_count,
            'corrupted_pct': corrupted_pct,
            'median': actual_median,
            'p5': p5,
            'p95': p95
        }

    except Exception as e:
        print(f"❌ ERROR analyzing data: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'ERROR', 'error': str(e)}


def main():
    """Main validation function"""
    print("=" * 70)
    print("DATA QUALITY VALIDATION")
    print("=" * 70)

    markets = ['ES', 'NQ', 'YM', 'RTY', 'MES', 'MNQ', 'M2K', 'MYM']
    results = {}

    for market in markets:
        exists, file_path, size_mb = check_file_exists(market, 'D1M')

        if exists:
            stats = analyze_data_quality(market, file_path)
            results[market] = stats
        else:
            print(f"\n⚠️  {market}_D1M.csv not found - skipping")

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        for market, stats in results.items():
            status = stats.get('status', 'UNKNOWN')
            corrupted_pct = stats.get('corrupted_pct', 0)

            if status == "EXCELLENT":
                symbol = "✅"
            elif status in ["GOOD", "MINOR_ISSUES"]:
                symbol = "⚠️ "
            else:
                symbol = "❌"

            print(f"{symbol} {market}: {status} ({corrupted_pct:.2f}% corrupted)")

        # Recommendations
        needs_reprocessing = [m for m, s in results.items() if s.get('status') in ['CORRUPTED', 'MAJOR_ISSUES']]

        if needs_reprocessing:
            print(f"\n❌ ACTION REQUIRED:")
            print(f"   The following markets need reprocessing:")
            for market in needs_reprocessing:
                print(f"     - {market}")
            print(f"\n   Run: python src/reprocess_from_source.py --market {' '.join(needs_reprocessing)}")
        else:
            print(f"\n✅ All checked markets have acceptable data quality!")

    print("=" * 70)


if __name__ == "__main__":
    main()
