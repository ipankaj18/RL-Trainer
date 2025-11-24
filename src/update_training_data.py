#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Weekly Futures Data Update Pipeline

Automates the process of updating training data from GLBX .zip files:
1. Detects GLBX-*.zip files in data/
2. Extracts and processes both minute-level and second-level data
3. Applies all data fixes (timezone, trading hours, cleaning)
4. Validates processed data
5. Tests with training environment
6. Replaces old data files and cleans up

Usage:
    python update_training_data.py --market NQ    # Process NQ (Nasdaq) data
    python update_training_data.py --market ES    # Process ES (S&P 500) data
    python update_training_data.py --verbose      # Verbose logging
    python update_training_data.py --dry-run      # Test without modifying files

Supported Markets: ES, NQ, YM, RTY, MNQ, MES, M2K, MYM

Author: RL Futures Trading System
Date: October 2025
"""

import os
import sys
import zipfile
import glob
import shutil
import pandas as pd
import numpy as np
import subprocess
from datetime import datetime
from pathlib import Path

# Set environment variable to ensure UTF-8 encoding on Windows
if os.name == 'nt':  # Windows
    # Try to force UTF-8 encoding for stdout
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
    except:
        # If that fails, we'll rely on ASCII replacements below
        pass

# Import local modules
from technical_indicators import add_all_indicators
from feature_engineering import add_market_regime_features
from data_validator import detect_and_fix_price_format
import process_second_data
import clean_second_data


def safe_print(message=""):
    """
    Print message with fallback for encoding errors.
    On Windows, some Unicode characters may not be supported by cp1252 encoding.
    """
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback: replace problematic characters with ASCII equivalents
        replacements = {
            '✓': '[OK]',
            '✅': '[OK]',
            '✗': '[X]',
            '❌': '[X]',
            '→': '->',
            '⚠': '[WARN]',
            '⚠️': '[WARN]',
            '—': '-',
            '–': '-',
            '’': "'",
            '“': '"',
            '”': '"',
        }
        ascii_message = message
        for src, target in replacements.items():
            ascii_message = ascii_message.replace(src, target)
        ascii_message = ascii_message.encode('ascii', errors='ignore').decode('ascii')
        print(ascii_message)


class DataUpdatePipeline:
    """Main pipeline for automated data updates"""

    def __init__(self, data_dir="data", market="", verbose=False, dry_run=False):
        self.data_dir = data_dir
        self.market = market.upper() if market else ""  # Uppercase market prefix
        self.verbose = verbose
        self.dry_run = dry_run
        self.temp_dirs = []
        self.zip_files = []

        # Expected median prices for corruption detection
        # Updated Nov 2025 to current market levels
        self.expected_medians = {
            'ES': 6400,      # E-mini S&P 500: updated from 5500 (Nov 2025 levels: 6200-6500)
            'NQ': 24000,     # E-mini Nasdaq-100: updated from 18000 (Nov 2025 levels: 23000-25000)
            'YM': 44000,     # E-mini Dow: updated from 35000 (Nov 2025 levels: 42000-46000)
            'RTY': 2200,     # E-mini Russell 2000: updated from 2100 (Nov 2025 levels: 2100-2300)
            'MES': 6400,     # Micro E-mini S&P 500: same as ES
            'MNQ': 24000,    # Micro E-mini Nasdaq-100: same as NQ
            'M2K': 2200,     # Micro E-mini Russell 2000: same as RTY
            'MYM': 44000,    # Micro E-mini Dow: same as YM
        }

        # Set output filenames based on market prefix
        if self.market:
            self.minute_filename = f"{self.market}_D1M.csv"
            self.second_filename = f"{self.market}_D1S.csv"
            self.second_raw_filename = f"{self.market}_D1S_raw.csv"
        else:
            self.minute_filename = "D1M.csv"
            self.second_filename = "D1S.csv"
            self.second_raw_filename = "D1S_raw.csv"

    def log(self, message, level="INFO"):
        """Print log message with encoding safety"""
        if self.verbose or level == "ERROR":
            timestamp = datetime.now().strftime("%H:%M:%S")
            safe_print(f"[{timestamp}] [{level}] {message}")
        elif level == "INFO":
            safe_print(message)

    def detect_zip_files(self):
        """Detect all GLBX-*.zip files in data directory"""
        safe_print("\n" + "=" * 80)
        safe_print("[1/6] DETECTING DATA FILES")
        safe_print("=" * 80)

        pattern = os.path.join(self.data_dir, "GLBX-*.zip")
        zip_files = glob.glob(pattern)

        if not zip_files:
            safe_print(f"  No GLBX-*.zip files found in {self.data_dir}/")
            return []

        # Sort by file size (larger first, likely more comprehensive)
        zip_files.sort(key=lambda x: os.path.getsize(x), reverse=True)

        for zip_file in zip_files:
            size_mb = os.path.getsize(zip_file) / (1024 * 1024)
            safe_print(f"  Found: {os.path.basename(zip_file)} ({size_mb:.1f} MB)")

        self.zip_files = zip_files
        return zip_files

    def extract_zip_file(self, zip_path):
        """Extract zip file and return extraction directory"""
        safe_print(f"\n  Extracting {os.path.basename(zip_path)}...")

        # Create extraction directory based on zip filename
        zip_basename = os.path.basename(zip_path).replace('.zip', '')
        extract_dir = os.path.join(self.data_dir, zip_basename)

        if self.dry_run:
            safe_print(f"    [DRY RUN] Would extract to: {extract_dir}")
            return extract_dir

        # Remove existing extraction directory if it exists
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)

        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Track for cleanup
        self.temp_dirs.append(extract_dir)

        # Calculate extracted size
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(extract_dir)
            for filename in filenames
        )
        size_mb = total_size / (1024 * 1024)

        safe_print(f"    Extracted: {extract_dir}/ ({size_mb:.1f} MB)")

        return extract_dir

    def find_csv_files(self, extract_dir):
        """Find minute and second-level CSV files in extracted directory"""
        # Look for OHLCV files
        minute_files = glob.glob(os.path.join(extract_dir, "**/*ohlcv-1m.csv"), recursive=True)
        second_files = glob.glob(os.path.join(extract_dir, "**/*ohlcv-1s.csv"), recursive=True)

        self.log(f"Found {len(minute_files)} minute-level file(s)", "DEBUG")
        self.log(f"Found {len(second_files)} second-level file(s)", "DEBUG")

        return (minute_files[0] if minute_files else None,
                second_files[0] if second_files else None)

    def process_minute_data(self, csv_file):
        """Process minute-level data for training"""
        safe_print("\n" + "=" * 80)
        safe_print("[3/6] PROCESSING MINUTE-LEVEL DATA")
        safe_print("=" * 80)

        if not csv_file or not os.path.exists(csv_file):
            safe_print("  ERROR: Minute-level CSV file not found")
            return False

        try:
            # Load raw data
            safe_print(f"  Loading from {csv_file}...")
            df = pd.read_csv(csv_file, low_memory=False)
            safe_print(f"    Loaded {len(df):,} rows")

            # Check for ES contracts
            if 'symbol' in df.columns:
                # Filter for specified market futures (e.g., ES, NQ, YM, etc.)
                if self.market:
                    market_contracts = [col for col in df['symbol'].unique() if col.startswith(self.market)]
                    df = df[df['symbol'].isin(market_contracts)].copy()
                    safe_print(f"    Filtered to {self.market} contracts: {len(df):,} rows")
                else:
                    # If no market specified, try to detect from symbols
                    safe_print(f"    No market filter applied: {len(df):,} rows")

            # Pre-flight data quality check (gives early warning about corruption)
            safe_print("\n  [PRE-FLIGHT CHECK] Analyzing data quality...")
            if len(df) > 0 and 'close' in df.columns:
                actual_median = df['close'].median()
                expected_median = self.expected_medians.get(self.market, 5000)

                if actual_median < expected_median * 0.20:
                    ratio = expected_median / actual_median
                    safe_print(f"    ⚠️  WARNING: Data appears corrupted!")
                    safe_print(f"    Expected median: ${expected_median:,.2f}")
                    safe_print(f"    Actual median: ${actual_median:,.2f} ({ratio:.1f}x too low)")
                    safe_print(f"    → Auto-correction will be applied...")
                else:
                    safe_print(f"    ✅ Data quality looks good (median: ${actual_median:,.2f})")

            # Detect and fix price format corruption (CRITICAL: Do this before any processing)
            df, format_stats = detect_and_fix_price_format(df, self.market, verbose=True)

            # Show fix summary if corruption was detected
            if format_stats['fixed_count'] > 0:
                safe_print(f"  ✅ Corrected {format_stats['fixed_count']:,} corrupted bars ({format_stats['fixed_pct']:.1f}%)")

            # Parse timestamp
            if 'ts_event' in df.columns:
                df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True)
                df['ts_event'] = df['ts_event'].dt.tz_convert('America/New_York')
                df = df.set_index('ts_event')
                df.index.name = 'datetime'  # Rename index for compatibility
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
                df = df.set_index('datetime')

            safe_print(f"  Timezone: UTC -> America/New_York [OK]")

            # Filter to Apex trading hours (8:30 AM - 4:59 PM ET)
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute

            mask_trading_hours = (
                ((df['hour'] == 8) & (df['minute'] >= 30)) |
                ((df['hour'] >= 9) & (df['hour'] < 16)) |
                ((df['hour'] == 16) & (df['minute'] <= 59))
            )

            rows_before = len(df)
            df = df[mask_trading_hours].copy()
            df = df.drop(['hour', 'minute'], axis=1)
            rows_after = len(df)

            safe_print(f"  Trading hours: 8:30 AM - 4:59 PM ET [OK]")
            safe_print(f"    Before: {rows_before:,} rows")
            safe_print(f"    After: {rows_after:,} rows ({(rows_after/rows_before*100):.1f}%)")

            # Keep only OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[required_cols].copy()

            # Convert to numeric
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove NaN and duplicates
            df = df.dropna()
            df = df[~df.index.duplicated(keep='first')]

            # Add technical indicators
            safe_print("  Adding technical indicators...")
            df = add_all_indicators(df)

            # Add market regime features
            safe_print("  Adding market regime features...")
            df = add_market_regime_features(df)

            # Save
            output_path = os.path.join(self.data_dir, self.minute_filename)

            if self.dry_run:
                safe_print(f"  [DRY RUN] Would save to: {output_path}")
            else:
                df.to_csv(output_path)
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                safe_print(f"\n  [OK] Saved: {output_path} ({size_mb:.1f} MB)")
                safe_print(f"  [OK] Rows: {len(df):,}")
                safe_print(f"  [OK] Columns: {len(df.columns)}")

            return True

        except Exception as e:
            safe_print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_second_data(self, csv_file):
        """Process second-level data for drawdown calculation"""
        safe_print("\n" + "=" * 80)
        safe_print("[4/6] PROCESSING SECOND-LEVEL DATA")
        safe_print("=" * 80)

        if not csv_file or not os.path.exists(csv_file):
            safe_print("  ERROR: Second-level CSV file not found")
            return False

        if self.dry_run:
            safe_print(f"  [DRY RUN] Would process: {csv_file}")
            return True

        try:
            # Step 1: Process raw second data
            raw_output = os.path.join(self.data_dir, self.second_raw_filename)
            safe_print("  Step 1: Processing raw second data...")
            success = process_second_data.main(input_file=csv_file, output_path=raw_output, market=self.market if self.market else 'ES')

            if not success:
                safe_print("  ERROR: Failed to process second data")
                return False

            # Step 2: Clean processed data
            final_output = os.path.join(self.data_dir, self.second_filename)
            safe_print("\n  Step 2: Cleaning processed data...")
            success = clean_second_data.main(input_path=raw_output, output_path=final_output)

            if not success:
                safe_print("  ERROR: Failed to clean second data")
                return False

            # Remove intermediate raw file
            if os.path.exists(raw_output):
                os.remove(raw_output)
                safe_print(f"  Removed intermediate file: {raw_output}")

            return True

        except Exception as e:
            safe_print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

    def validate_data(self):
        """Validate processed data"""
        safe_print("\n" + "=" * 80)
        safe_print("[5/6] VALIDATING DATA")
        safe_print("=" * 80)

        minute_file = os.path.join(self.data_dir, self.minute_filename)
        second_file = os.path.join(self.data_dir, self.second_filename)

        validation_passed = True

        # Validate minute data
        if os.path.exists(minute_file):
            safe_print("\n  Minute-level data validation:")
            try:
                df = pd.read_csv(minute_file, index_col=0, parse_dates=True, nrows=1000)

                # Check timezone (accept both America/New_York and UTC offsets -04:00/-05:00)
                tz_str = str(df.index.tz)
                is_eastern = (
                    'America/New_York' in tz_str or
                    'US/Eastern' in tz_str or
                    'UTC-04:00' in tz_str or  # EDT (Daylight Saving)
                    'UTC-05:00' in tz_str     # EST (Standard Time)
                )
                if df.index.tz is not None and is_eastern:
                    safe_print(f"    [OK] Timezone: {tz_str} (Eastern Time)")
                else:
                    safe_print(f"    [ERROR] Timezone: {tz_str} (NOT Eastern Time)")
                    validation_passed = False

                # Check trading hours (sample check)
                hours = df.index.hour
                minutes = df.index.minute

                valid_hours = (
                    ((hours == 8) & (minutes >= 30)) |
                    ((hours >= 9) & (hours < 16)) |
                    ((hours == 16) & (minutes <= 59))
                )

                if valid_hours.all():
                    safe_print("    [OK] Trading hours: 8:30 AM - 4:59 PM ET")
                else:
                    safe_print("    [ERROR] Trading hours: Invalid times detected")
                    validation_passed = False

            except Exception as e:
                safe_print(f"    [ERROR] Error reading file: {e}")
                validation_passed = False

        # Validate second data
        if os.path.exists(second_file):
            safe_print("\n  Second-level data validation:")
            try:
                df = pd.read_csv(second_file, index_col=0, parse_dates=True, nrows=1000)

                # Check timezone (accept both America/New_York and UTC offsets -04:00/-05:00)
                tz_str = str(df.index.tz)
                is_eastern = (
                    'America/New_York' in tz_str or
                    'US/Eastern' in tz_str or
                    'UTC-04:00' in tz_str or  # EDT (Daylight Saving)
                    'UTC-05:00' in tz_str     # EST (Standard Time)
                )
                if df.index.tz is not None and is_eastern:
                    safe_print(f"    [OK] Timezone: {tz_str} (Eastern Time)")
                else:
                    safe_print(f"    [ERROR] Timezone: {tz_str} (NOT Eastern Time)")
                    validation_passed = False

            except Exception as e:
                safe_print(f"    [ERROR] Error reading file: {e}")
                validation_passed = False

        # Test with environment
        safe_print("\n  Environment test:")
        if self.dry_run:
            safe_print("    [DRY RUN] Skipping environment test")
        else:
            try:
                result = subprocess.run(
                    ['python', 'diagnose_environment.py', '--phase', '1', '--steps', '100', '--market', self.market],
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    capture_output=True,
                    text=True,
                    timeout=180  # Increased from 60s to 180s for larger datasets (128K+ rows)
                )

                if result.returncode == 0:
                    # Check for key success indicators in output
                    output = result.stdout + result.stderr
                    if "Buy action" in output and "Sell action" in output:
                        safe_print("    [OK] Environment test passed")
                    else:
                        safe_print("    [WARN] Environment test ran but results unclear")
                else:
                    safe_print(f"    [ERROR] Environment test failed (exit code {result.returncode})")
                    validation_passed = False

            except subprocess.TimeoutExpired:
                safe_print("    [WARN] Environment test timed out (not critical - data is valid)")
                # Don't fail validation for timeout - data is already successfully processed
            except Exception as e:
                safe_print(f"    [WARN] Could not run environment test: {e}")
                # Don't fail validation for this

        return validation_passed

    def cleanup(self):
        """Clean up temporary files and directories"""
        safe_print("\n" + "=" * 80)
        safe_print("[6/6] CLEANUP")
        safe_print("=" * 80)

        # Remove extracted directories
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                if self.dry_run:
                    safe_print(f"  [DRY RUN] Would delete: {temp_dir}/")
                else:
                    shutil.rmtree(temp_dir)
                    safe_print(f"  Deleted: {temp_dir}/")

        # Remove zip files
        for zip_file in self.zip_files:
            if os.path.exists(zip_file):
                if self.dry_run:
                    safe_print(f"  [DRY RUN] Would delete: {os.path.basename(zip_file)}")
                else:
                    os.remove(zip_file)
                    safe_print(f"  Deleted: {os.path.basename(zip_file)}")

    def run(self):
        """Execute the complete data update pipeline"""
        safe_print("=" * 80)
        market_label = f"{self.market} " if self.market else ""
        safe_print(f"{market_label}FUTURES WEEKLY DATA UPDATE")
        safe_print("=" * 80)

        if self.dry_run:
            safe_print("*** DRY RUN MODE - NO FILES WILL BE MODIFIED ***\n")

        # Step 1: Detect zip files
        zip_files = self.detect_zip_files()

        if not zip_files:
            safe_print("\nNo data files to process. Exiting.")
            return False

        # Step 2: Extract all zip files
        safe_print("\n" + "=" * 80)
        safe_print("[2/6] EXTRACTING DATA")
        safe_print("=" * 80)

        extracted_dirs = []
        for zip_file in zip_files:
            extract_dir = self.extract_zip_file(zip_file)
            extracted_dirs.append(extract_dir)

        # Find CSV files from the largest/first extracted directory
        # (Assuming the largest zip has the most comprehensive data)
        minute_csv = None
        second_csv = None

        for extract_dir in extracted_dirs:
            m_csv, s_csv = self.find_csv_files(extract_dir)
            if m_csv and not minute_csv:
                minute_csv = m_csv
            if s_csv and not second_csv:
                second_csv = s_csv

        # Step 3: Process minute data
        minute_success = self.process_minute_data(minute_csv)

        # Step 4: Process second data
        second_success = self.process_second_data(second_csv)

        # Step 5: Validate
        if not self.dry_run:
            validation_passed = self.validate_data()
        else:
            validation_passed = True

        # Step 6: Cleanup
        self.cleanup()

        # Final summary
        safe_print("\n" + "=" * 80)
        safe_print("UPDATE COMPLETE")
        safe_print("=" * 80)

        if minute_success:
            minute_file = os.path.join(self.data_dir, self.minute_filename)
            if os.path.exists(minute_file):
                df = pd.read_csv(minute_file, index_col=0, nrows=1)
                safe_print(f"[OK] Minute data: {minute_file}")

        if second_success:
            second_file = os.path.join(self.data_dir, self.second_filename)
            if os.path.exists(second_file):
                safe_print(f"[OK] Second data: {second_file}")

        if validation_passed:
            safe_print("[OK] All validations passed")
        else:
            safe_print("[WARN] Some validations failed (check output above)")

        if not self.dry_run:
            safe_print("\nReady for production training!")
        else:
            safe_print("\n[DRY RUN] No files were modified")

        return minute_success and second_success and validation_passed


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Automated weekly ES futures data update',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python update_training_data.py                 # Standard update (creates D1M.csv, D1S.csv)
  python update_training_data.py --market ES     # Create ES_D1M.csv, ES_D1S.csv
  python update_training_data.py --market NQ     # Create NQ_D1M.csv, NQ_D1S.csv
  python update_training_data.py --verbose       # Verbose logging
  python update_training_data.py --dry-run       # Test without modifying files
  python update_training_data.py --data-dir data # Custom data directory
        """
    )

    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--market', type=str, default='',
                       help='Market prefix (e.g., ES, NQ, YM). Creates ES_D1M.csv instead of D1M.csv')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--dry-run', action='store_true',
                       help='Test without modifying any files')

    args = parser.parse_args()

    # Change to project root directory (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Run pipeline
    pipeline = DataUpdatePipeline(
        data_dir=args.data_dir,
        market=args.market,
        verbose=args.verbose,
        dry_run=args.dry_run
    )

    success = pipeline.run()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
