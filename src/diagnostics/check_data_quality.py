#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Quality Checker

Validates training and validation data for:
- Timezone correctness (US Eastern Time)
- RTH (Regular Trading Hours) coverage
- Data corruption
- Missing values
- Price anomalies

Usage:
    python src/diagnostics/check_data_quality.py --market NQ
    python src/diagnostics/check_data_quality.py --market ES --detailed
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, time


class DataQualityChecker:
    """Checks data quality for trading environment."""

    RTH_START = time(9, 30)  # 9:30 AM ET
    RTH_END = time(16, 59)   # 4:59 PM ET

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir

    def check_file(self, filepath: str, detailed: bool = False) -> Dict:
        """
        Check quality of a single data file.

        Args:
            filepath: Path to CSV file
            detailed: If True, perform detailed analysis

        Returns:
            Dictionary with quality metrics
        """
        if not os.path.exists(filepath):
            return {"error": f"File not found: {filepath}"}

        try:
            # Load data
            df = pd.read_csv(filepath, index_col='datetime', parse_dates=True)

            results = {
                'file': filepath,
                'total_bars': len(df),
                'date_range': f"{df.index.min()} to {df.index.max()}",
                'columns': list(df.columns),
                'issues': [],
                'warnings': []
            }

            # Check timezone
            tz_info = self._check_timezone(df)
            results.update(tz_info)

            # Check RTH coverage
            rth_info = self._check_rth_coverage(df)
            results.update(rth_info)

            # Check missing values
            missing_info = self._check_missing_values(df)
            results.update(missing_info)

            # Check price anomalies
            price_info = self._check_price_anomalies(df)
            results.update(price_info)

            # Check technical indicators
            indicator_info = self._check_indicators(df)
            results.update(indicator_info)

            if detailed:
                # Detailed statistics
                stats_info = self._detailed_statistics(df)
                results['detailed_stats'] = stats_info

            return results

        except Exception as e:
            return {"error": f"Failed to check {filepath}: {e}"}

    def _check_timezone(self, df: pd.DataFrame) -> Dict:
        """Check if data is in US Eastern Time."""
        info = {}

        if df.index.tz is None:
            info['issues'] = info.get('issues', [])
            info['issues'].append("Data has no timezone information")
            info['timezone'] = "None (Naive)"
        else:
            tz_str = str(df.index.tz)
            info['timezone'] = tz_str

            if 'America/New_York' not in tz_str and 'US/Eastern' not in tz_str:
                info['warnings'] = info.get('warnings', [])
                info['warnings'].append(f"Data timezone is {tz_str}, expected America/New_York")

        return info

    def _check_rth_coverage(self, df: pd.DataFrame) -> Dict:
        """Check Regular Trading Hours coverage."""
        info = {}

        # Convert to ET if needed
        if df.index.tz is None:
            df_et = df.copy()
            df_et.index = df_et.index.tz_localize('UTC').tz_convert('America/New_York')
        elif str(df.index.tz) != 'America/New_York':
            df_et = df.copy()
            df_et.index = df_et.index.tz_convert('America/New_York')
        else:
            df_et = df

        # Filter RTH
        rth_mask = df_et.index.to_series().apply(
            lambda x: self.RTH_START <= x.time() <= self.RTH_END
        )
        rth_bars = rth_mask.sum()

        info['rth_bars'] = int(rth_bars)
        info['rth_percentage'] = float(rth_bars / len(df) * 100)
        info['non_rth_bars'] = int(len(df) - rth_bars)

        if info['rth_percentage'] < 40:
            info['issues'] = info.get('issues', [])
            info['issues'].append(
                f"Low RTH coverage ({info['rth_percentage']:.1f}%) - "
                f"environment expects RTH-only data"
            )
        elif info['non_rth_bars'] > 0:
            info['warnings'] = info.get('warnings', [])
            info['warnings'].append(
                f"Data contains {info['non_rth_bars']} non-RTH bars "
                f"({100 - info['rth_percentage']:.1f}%)"
            )

        return info

    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values."""
        info = {}

        missing = df.isnull().sum()
        total_missing = missing.sum()

        info['total_missing_values'] = int(total_missing)

        if total_missing > 0:
            missing_cols = missing[missing > 0].to_dict()
            info['missing_by_column'] = {k: int(v) for k, v in missing_cols.items()}
            info['issues'] = info.get('issues', [])
            info['issues'].append(
                f"Found {total_missing} missing values across {len(missing_cols)} columns"
            )

        return info

    def _check_price_anomalies(self, df: pd.DataFrame) -> Dict:
        """Check for price anomalies and corruption."""
        info = {}

        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            info['warnings'] = info.get('warnings', [])
            info['warnings'].append("Missing OHLC columns - skipping price checks")
            return info

        # Calculate price statistics
        prices = df['close'].dropna()
        info['price_stats'] = {
            'median': float(prices.median()),
            'mean': float(prices.mean()),
            'std': float(prices.std()),
            'min': float(prices.min()),
            'max': float(prices.max()),
            'p01': float(prices.quantile(0.01)),
            'p05': float(prices.quantile(0.05)),
            'p95': float(prices.quantile(0.95)),
            'p99': float(prices.quantile(0.99))
        }

        # Check for divide-by-100 corruption (common Databento issue)
        median_price = info['price_stats']['median']
        p05 = info['price_stats']['p05']

        # Detect corruption by checking if prices are unusually low
        expected_ranges = {
            'ES': (4000, 8000),
            'NQ': (12000, 28000),
            'YM': (30000, 50000),
            'RTY': (1500, 3000),
            'MES': (4000, 8000),
            'MNQ': (12000, 28000),
            'M2K': (1500, 3000),
            'MYM': (30000, 50000)
        }

        # Try to detect market from filename
        filename = Path(info.get('file', '')).stem
        market = None
        for market_code in expected_ranges.keys():
            if market_code in filename.upper():
                market = market_code
                break

        if market and market in expected_ranges:
            expected_min, expected_max = expected_ranges[market]
            if median_price < expected_min / 100:
                info['issues'] = info.get('issues', [])
                info['issues'].append(
                    f"CRITICAL: Prices appear corrupted (median={median_price:.2f}, "
                    f"expected ~{expected_min}-{expected_max}). "
                    f"Likely divide-by-100 error."
                )
                info['corruption_detected'] = True
                info['suggested_fix'] = "Reprocess data with multiply-by-100 correction"

        # Check OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low'])
        ).sum()

        if invalid_ohlc > 0:
            info['issues'] = info.get('issues', [])
            info['issues'].append(f"Found {invalid_ohlc} bars with invalid OHLC relationships")

        # Check for extreme price jumps
        price_changes = prices.pct_change().abs()
        extreme_moves = (price_changes > 0.10).sum()  # >10% moves
        if extreme_moves > 0:
            info['warnings'] = info.get('warnings', [])
            info['warnings'].append(
                f"Found {extreme_moves} bars with >10% price moves - "
                f"may indicate data quality issues"
            )

        return info

    def _check_indicators(self, df: pd.DataFrame) -> Dict:
        """Check technical indicators."""
        info = {}

        # Check for required indicators
        required_indicators = ['sma_5', 'sma_20', 'rsi', 'atr', 'macd', 'bb_upper', 'bb_lower']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns]

        if missing_indicators:
            info['warnings'] = info.get('warnings', [])
            info['warnings'].append(f"Missing indicators: {', '.join(missing_indicators)}")

        # Check for NaN in indicators (common at start of data)
        indicator_cols = [col for col in df.columns if any(ind in col for ind in ['sma', 'rsi', 'atr', 'macd', 'bb'])]
        if indicator_cols:
            indicator_nans = df[indicator_cols].isnull().sum()
            max_nans = indicator_nans.max()
            if max_nans > 200:  # More than 200 bars with NaN
                info['warnings'] = info.get('warnings', [])
                info['warnings'].append(
                    f"High NaN count in indicators (up to {max_nans} bars) - "
                    f"may reduce training data"
                )

        return info

    def _detailed_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate detailed statistics."""
        stats = {}

        # Volume statistics
        if 'volume' in df.columns:
            vol = df['volume'].dropna()
            stats['volume'] = {
                'mean': float(vol.mean()),
                'median': float(vol.median()),
                'zero_volume_bars': int((vol == 0).sum())
            }

        # Trading session analysis
        if df.index.tz is not None or df.index.tz is None:
            # Convert to ET
            if df.index.tz is None:
                df_et = df.copy()
                df_et.index = df_et.index.tz_localize('UTC').tz_convert('America/New_York')
            else:
                df_et = df.copy()
                df_et.index = df_et.index.tz_convert('America/New_York')

            # Count bars by hour
            hours = df_et.index.hour
            stats['bars_by_hour'] = {int(h): int(c) for h, c in hours.value_counts().sort_index().items()}

        return stats

    def check_train_val_split(self, train_file: str, val_file: str, expected_split: float = 0.7) -> Dict:
        """
        Check train/validation split quality.

        Args:
            train_file: Path to training data
            val_file: Path to validation data
            expected_split: Expected train split ratio

        Returns:
            Dictionary with split analysis
        """
        results = {}

        try:
            train_df = pd.read_csv(train_file, index_col='datetime', parse_dates=True)
            val_df = pd.read_csv(val_file, index_col='datetime', parse_dates=True)

            results['train_bars'] = len(train_df)
            results['val_bars'] = len(val_df)
            results['total_bars'] = len(train_df) + len(val_df)
            results['actual_split'] = len(train_df) / results['total_bars']

            # Check chronological ordering
            train_end = train_df.index.max()
            val_start = val_df.index.min()

            results['train_end'] = str(train_end)
            results['val_start'] = str(val_start)

            if train_end >= val_start:
                results['issues'] = ['Train and validation data overlap - should be chronologically split']
            else:
                gap_days = (val_start - train_end).days
                results['gap_days'] = gap_days
                if gap_days > 7:
                    results['warnings'] = [f"Large gap ({gap_days} days) between train and validation data"]

            # Check split ratio
            split_diff = abs(results['actual_split'] - expected_split)
            if split_diff > 0.05:
                results['warnings'] = results.get('warnings', [])
                results['warnings'].append(
                    f"Split ratio ({results['actual_split']:.1%}) differs from expected ({expected_split:.0%})"
                )

        except Exception as e:
            results['error'] = f"Failed to check split: {e}"

        return results

    def print_report(self, results: Dict, title: str = "Data Quality Report"):
        """Print formatted report."""
        print("\n" + "=" * 80)
        print(f"{title}")
        print("=" * 80)

        if 'error' in results:
            print(f"\n[ERROR] {results['error']}")
            return

        print(f"\n[FILE] {results.get('file', 'N/A')}")
        print(f"Total Bars: {results.get('total_bars', 0):,}")
        print(f"Date Range: {results.get('date_range', 'N/A')}")
        print(f"Timezone: {results.get('timezone', 'N/A')}")

        # RTH coverage
        print(f"\n[REGULAR TRADING HOURS]")
        print(f"RTH Bars: {results.get('rth_bars', 0):,} ({results.get('rth_percentage', 0):.1f}%)")
        print(f"Non-RTH Bars: {results.get('non_rth_bars', 0):,}")

        # Price stats
        if 'price_stats' in results:
            ps = results['price_stats']
            print(f"\n[PRICE STATISTICS]")
            print(f"Median: ${ps['median']:,.2f}  Mean: ${ps['mean']:,.2f}  Std: ${ps['std']:,.2f}")
            print(f"Range: ${ps['min']:,.2f} to ${ps['max']:,.2f}")
            print(f"P05-P95: ${ps['p05']:,.2f} to ${ps['p95']:,.2f}")

        # Missing values
        if results.get('total_missing_values', 0) > 0:
            print(f"\n[MISSING VALUES]")
            print(f"Total: {results['total_missing_values']}")
            if 'missing_by_column' in results:
                for col, count in results['missing_by_column'].items():
                    print(f"  {col}: {count}")

        # Issues
        if results.get('issues'):
            print(f"\n[ISSUES]")
            for i, issue in enumerate(results['issues'], 1):
                print(f"  {i}. {issue}")

        # Warnings
        if results.get('warnings'):
            print(f"\n[WARNINGS]")
            for i, warning in enumerate(results['warnings'], 1):
                print(f"  {i}. {warning}")

        # Detailed stats
        if 'detailed_stats' in results:
            ds = results['detailed_stats']
            if 'volume' in ds:
                print(f"\n[VOLUME]")
                print(f"Mean: {ds['volume']['mean']:,.0f}  Median: {ds['volume']['median']:,.0f}")
                print(f"Zero-volume bars: {ds['volume']['zero_volume_bars']}")

            if 'bars_by_hour' in ds:
                print(f"\n[BARS BY HOUR (ET)]")
                for hour, count in sorted(ds['bars_by_hour'].items()):
                    print(f"  {hour:02d}:00 - {count:,} bars")

        print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Check data quality for training')
    parser.add_argument('--market', type=str, required=True,
                       help='Market symbol (ES, NQ, YM, RTY, etc.)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--detailed', action='store_true',
                       help='Perform detailed analysis')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    args = parser.parse_args()

    checker = DataQualityChecker(data_dir=args.data_dir)

    # Check minute-level data
    minute_file = os.path.join(args.data_dir, f'{args.market}_D1M.csv')
    print(f"\n[INFO] Checking minute-level data: {minute_file}")
    minute_results = checker.check_file(minute_file, detailed=args.detailed)
    checker.print_report(minute_results, f"Minute-Level Data Quality - {args.market}")

    # Check second-level data if exists
    second_file = os.path.join(args.data_dir, f'{args.market}_D1S.csv')
    if os.path.exists(second_file):
        print(f"\n[INFO] Checking second-level data: {second_file}")
        second_results = checker.check_file(second_file, detailed=args.detailed)
        checker.print_report(second_results, f"Second-Level Data Quality - {args.market}")

    # Save results
    if args.output:
        all_results = {
            'market': args.market,
            'minute_data': minute_results
        }
        if os.path.exists(second_file):
            all_results['second_data'] = second_results

        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[INFO] Results saved to: {args.output}")


if __name__ == '__main__':
    main()
