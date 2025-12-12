#!/usr/bin/env python3
"""
Incremental Data Update System for RL Futures Trading

Intelligently updates existing training data by:
1. Detecting new .zip files in data/ directory
2. Identifying date ranges of existing processed data
3. Extracting ONLY new dates from .zip files (avoids reprocessing)
4. Processing new data through full pipeline (indicators, features, validation)
5. Merging with existing data while maintaining chronological order
6. Creating backups before overwriting files

Usage Examples:
    # Automatic incremental update (detects and processes new data)
    python src/incremental_data_updater.py --market NQ

    # Preview changes without applying
    python src/incremental_data_updater.py --market NQ --dry-run

    # Verbose output with detailed logging
    python src/incremental_data_updater.py --market NQ --verbose

    # Force full reprocessing (ignore existing data)
    python src/incremental_data_updater.py --market NQ --force-full

Supported Markets: ES, NQ, YM, RTY, MNQ, MES, M2K, MYM

Author: RL Futures Trading System
Date: November 2025
"""

import os
import sys
import argparse
import zipfile
import shutil
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import namedtuple

# Import local modules
from technical_indicators import add_all_indicators
from feature_engineering import add_market_regime_features
from data_validator import detect_and_fix_price_format, EXPECTED_MEDIAN_PRICES


# Constants
SUPPORTED_MARKETS = ['ES', 'NQ', 'YM', 'RTY', 'MES', 'MNQ', 'M2K', 'MYM']
MINUTE_FILE_SUFFIX = '_D1M.csv'
SECOND_FILE_SUFFIX = '_D1S.csv'
BACKUP_DIR = 'backups'

# Named tuple for date range information
DateRange = namedtuple('DateRange', ['start', 'end', 'row_count'])

# Named tuple for zip file information
ZipFileInfo = namedtuple('ZipFileInfo', ['path', 'timeframe', 'start_date', 'end_date', 'csv_name'])


class IncrementalDataUpdater:
    """
    Manages incremental updates of trading data

    Attributes:
        data_dir (Path): Directory containing data files
        market (str): Market symbol (ES, NQ, etc.)
        verbose (bool): Enable verbose logging
        dry_run (bool): Preview changes without applying
        backup_enabled (bool): Create backups before overwriting
    """

    def __init__(self, data_dir: str = "data", market: str = "",
                 verbose: bool = False, dry_run: bool = False,
                 backup_enabled: bool = True):
        self.data_dir = Path(data_dir)
        self.market = market.upper() if market else ""
        self.verbose = verbose
        self.dry_run = dry_run
        self.backup_enabled = backup_enabled

        # Validate market
        if self.market and self.market not in SUPPORTED_MARKETS:
            raise ValueError(f"Unknown market: {self.market}. Valid markets: {SUPPORTED_MARKETS}")

        # Create backup directory if needed
        self.backup_dir = self.data_dir / BACKUP_DIR
        if not self.dry_run and self.backup_enabled:
            self.backup_dir.mkdir(parents=True, exist_ok=True)

    def print(self, message: str = "", level: str = "INFO"):
        """
        Print message with optional verbosity control

        Args:
            message: Message to print
            level: Log level (INFO, DEBUG, WARN, ERROR)
        """
        if level == "DEBUG" and not self.verbose:
            return

        prefix = {
            "INFO": "",
            "DEBUG": "[DEBUG] ",
            "WARN": "[WARN] ",
            "ERROR": "[ERROR] "
        }.get(level, "")

        print(f"{prefix}{message}")

    # ============================================================
    # DATE RANGE DETECTION FUNCTIONS
    # ============================================================

    def get_existing_data_range(self) -> Optional[DateRange]:
        """
        Get date range of existing processed data

        Returns:
            DateRange(start, end, row_count) or None if no data exists

        Example:
            range_info = updater.get_existing_data_range()
            if range_info:
                print(f"Existing: {range_info.start} to {range_info.end}")
        """
        minute_file = self.data_dir / f"{self.market}{MINUTE_FILE_SUFFIX}"

        if not minute_file.exists():
            self.print(f"No existing data found for {self.market}", "DEBUG")
            return None

        try:
            # Read only first and last rows for efficiency
            df = pd.read_csv(minute_file)

            if len(df) == 0:
                self.print(f"Empty data file: {minute_file}", "WARN")
                return None

            # Parse datetime column
            df['datetime'] = pd.to_datetime(df['datetime'])

            start_date = df['datetime'].min()
            end_date = df['datetime'].max()
            row_count = len(df)

            # Remove timezone info for comparison (convert to date-only comparison)
            start_date = start_date.tz_localize(None) if hasattr(start_date, 'tz') and start_date.tz else start_date
            end_date = end_date.tz_localize(None) if hasattr(end_date, 'tz') and end_date.tz else end_date

            self.print(f"Existing data: {start_date.date()} to {end_date.date()} ({row_count:,} rows)", "DEBUG")

            return DateRange(start=start_date, end=end_date, row_count=row_count)

        except Exception as e:
            self.print(f"Error reading existing data: {e}", "ERROR")
            return None

    def extract_zip_date_range(self, zip_path: Path) -> Optional[Tuple[datetime, datetime, str, str]]:
        """
        Extract date range and CSV filename from zip file

        Parses filename pattern: glbx-mdp3-YYYYMMDD-YYYYMMDD.ohlcv-{1m|1s}.csv

        Args:
            zip_path: Path to .zip file

        Returns:
            Tuple of (start_date, end_date, csv_name, timeframe) or None if parsing fails

        Example:
            start, end, csv_name, tf = extract_zip_date_range(Path("GLBX-20251123-ABC.zip"))
            # Inspects CSV inside: glbx-mdp3-20250822-20251121.ohlcv-1m.csv
            # Returns: (datetime(2025,8,22), datetime(2025,11,21), "glbx...", "1m")
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Find OHLCV CSV file
                csv_files = [name for name in zf.namelist()
                           if name.endswith('.csv') and 'ohlcv' in name.lower()]

                if not csv_files:
                    self.print(f"No OHLCV CSV found in {zip_path.name}", "WARN")
                    return None

                csv_name = csv_files[0]

                # Parse filename: glbx-mdp3-YYYYMMDD-YYYYMMDD.ohlcv-1m.csv
                # Pattern: extract YYYYMMDD-YYYYMMDD part
                pattern = r'(\d{8})-(\d{8})\.ohlcv-(1[ms])'
                match = re.search(pattern, csv_name)

                if not match:
                    self.print(f"Could not parse date from filename: {csv_name}", "WARN")
                    return None

                start_str, end_str, timeframe = match.groups()

                # Parse dates
                start_date = datetime.strptime(start_str, '%Y%m%d')
                end_date = datetime.strptime(end_str, '%Y%m%d')

                self.print(f"Zip file {zip_path.name}: {start_date.date()} to {end_date.date()} ({timeframe})", "DEBUG")

                return (start_date, end_date, csv_name, timeframe)

        except Exception as e:
            self.print(f"Error reading zip file {zip_path.name}: {e}", "ERROR")
            return None

    def detect_zip_files(self) -> Dict[str, List[ZipFileInfo]]:
        """
        Detect all .zip files in data directory and categorize by timeframe

        Returns:
            Dict with keys 'minute' and 'second', each containing list of ZipFileInfo

        Example:
            files = updater.detect_zip_files()
            for info in files['minute']:
                print(f"Minute data: {info.start_date} to {info.end_date}")
        """
        zip_files = {'minute': [], 'second': []}

        # Find all GLBX-*.zip files
        glbx_pattern = self.data_dir / "GLBX-*.zip"
        found_zips = list(self.data_dir.glob("GLBX-*.zip"))

        self.print(f"\n[SCAN] Scanning {self.data_dir} for new .zip files...")
        self.print(f"Found {len(found_zips)} .zip file(s)", "DEBUG")

        for zip_path in found_zips:
            result = self.extract_zip_date_range(zip_path)

            if result:
                start_date, end_date, csv_name, timeframe = result

                info = ZipFileInfo(
                    path=zip_path,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    csv_name=csv_name
                )

                if timeframe == '1m':
                    zip_files['minute'].append(info)
                    self.print(f"  Found: {zip_path.name} (minute data: {start_date.date()} - {end_date.date()})")
                elif timeframe == '1s':
                    zip_files['second'].append(info)
                    self.print(f"  Found: {zip_path.name} (second data: {start_date.date()} - {end_date.date()})")

        return zip_files

    def identify_new_dates(self, existing_range: Optional[DateRange],
                          zip_range: Tuple[datetime, datetime]) -> Optional[Tuple[datetime, datetime]]:
        """
        Identify which dates are new and need to be processed

        Args:
            existing_range: DateRange of current data (or None if no data)
            zip_range: Tuple of (start_date, end_date) from zip file

        Returns:
            Tuple of (new_start, new_end) or None if no new data

        Logic:
            - If no existing data: process all dates
            - If overlap: process only dates after existing end date
            - If gap: warn user and process all new dates
            - If no new data: return None
        """
        zip_start, zip_end = zip_range

        # Case 1: No existing data - process everything
        if existing_range is None:
            self.print("No existing data - will process all dates from .zip", "DEBUG")
            return (zip_start, zip_end)

        # Case 2: Check if zip has any new data
        if zip_end <= existing_range.end:
            self.print(f"Zip data ends {zip_end.date()} but we already have up to {existing_range.end.date()}", "DEBUG")
            return None

        # Case 3: Zip has new data after existing end date
        new_start = existing_range.end + timedelta(days=1)
        new_end = zip_end

        # Check for gap
        gap_days = (new_start - existing_range.end).days - 1
        if gap_days > 0:
            self.print(f"Warning: {gap_days} day gap detected between existing and new data", "WARN")

        return (new_start, new_end)

    def analyze_date_ranges(self, existing_range: Optional[DateRange],
                           zip_files: Dict[str, List[ZipFileInfo]]) -> Dict[str, Any]:
        """
        Analyze date ranges and determine what needs to be processed

        Returns:
            Dict with analysis results including:
            - has_new_data: bool
            - new_date_range: Tuple[datetime, datetime] or None
            - overlap_days: int
            - new_days: int
            - gap_days: int
        """
        analysis = {
            'has_new_data': False,
            'new_date_range': None,
            'overlap_days': 0,
            'new_days': 0,
            'gap_days': 0,
            'estimated_minute_rows': 0,
            'estimated_second_rows': 0
        }

        # Use first minute file for analysis
        if not zip_files['minute']:
            self.print("No minute-level .zip files found", "WARN")
            return analysis

        # Analyze first zip file (assume all have same date range)
        zip_info = zip_files['minute'][0]
        zip_range = (zip_info.start_date, zip_info.end_date)

        new_range = self.identify_new_dates(existing_range, zip_range)

        if new_range:
            analysis['has_new_data'] = True
            analysis['new_date_range'] = new_range

            new_start, new_end = new_range
            analysis['new_days'] = (new_end - new_start).days + 1

            if existing_range:
                # Calculate overlap
                overlap_start = max(existing_range.start, zip_info.start_date)
                overlap_end = min(existing_range.end, zip_info.end_date)
                if overlap_end >= overlap_start:
                    analysis['overlap_days'] = (overlap_end - overlap_start).days + 1

                # Calculate gap
                analysis['gap_days'] = (new_start - existing_range.end).days - 1

            # Estimate new rows (assuming ~310 minute bars per trading day, ~18,600 second bars)
            analysis['estimated_minute_rows'] = analysis['new_days'] * 310
            analysis['estimated_second_rows'] = analysis['new_days'] * 18600

        return analysis

    # ============================================================
    # DATA EXTRACTION & FILTERING FUNCTIONS
    # ============================================================

    def extract_and_filter_zip(self, zip_info: ZipFileInfo,
                               start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Extract CSV from zip file and filter to specific date range

        Args:
            zip_info: ZipFileInfo with path and CSV name
            start_date: Start of date range to extract (inclusive)
            end_date: End of date range to extract (inclusive)

        Returns:
            Filtered DataFrame with only data in the specified date range
        """
        self.print(f"  Extracting {zip_info.csv_name}...", "DEBUG")

        try:
            with zipfile.ZipFile(zip_info.path, 'r') as zf:
                # Read CSV directly from zip
                with zf.open(zip_info.csv_name) as csv_file:
                    df = pd.read_csv(csv_file)

            self.print(f"  Loaded {len(df):,} rows from zip", "DEBUG")

            # Parse timestamp column (handle various column names)
            timestamp_col = None
            for col in ['ts_event', 'timestamp', 'datetime', 'date']:
                if col in df.columns:
                    timestamp_col = col
                    break

            if not timestamp_col:
                raise ValueError(f"No timestamp column found in {zip_info.csv_name}")

            # Convert to datetime
            df['datetime'] = pd.to_datetime(df[timestamp_col])

            # Drop the source timestamp column to avoid duplicate datetime fields later
            if timestamp_col != 'datetime' and timestamp_col in df.columns:
                df = df.drop(columns=[timestamp_col])

            # Filter to date range (dates only, ignore time)
            df['date_only'] = df['datetime'].dt.date
            start_date_only = start_date.date()
            end_date_only = end_date.date()

            mask = (df['date_only'] >= start_date_only) & (df['date_only'] <= end_date_only)
            filtered_df = df[mask].copy()

            # Drop temporary column
            filtered_df = filtered_df.drop(columns=['date_only'])

            self.print(f"  Filtered to {len(filtered_df):,} rows ({start_date.date()} to {end_date.date()})", "DEBUG")

            return filtered_df

        except Exception as e:
            self.print(f"Error extracting data from {zip_info.path.name}: {e}", "ERROR")
            raise

    # ============================================================
    # DATA PROCESSING PIPELINE
    # ============================================================

    def process_raw_data(self, df: pd.DataFrame, timeframe: str = '1m') -> pd.DataFrame:
        """
        Process raw OHLCV data through the full pipeline

        Args:
            df: Raw DataFrame with OHLCV columns
            timeframe: '1m' for minute or '1s' for second data

        Returns:
            Processed DataFrame with indicators and features
        """
        self.print(f"  Processing {timeframe} data through pipeline...", "DEBUG")

        df = df.copy()

        # Step 1: Standardize column names (vendor data may vary)
        column_mapping = {
            'ts_event': 'datetime',
            'timestamp': 'datetime'
        }
        df = df.rename(columns=column_mapping)

        # Ensure required columns exist
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Step 2: Detect and fix price format corruption
        self.print("    Checking for price format corruption...", "DEBUG")
        df, fix_stats = detect_and_fix_price_format(df, self.market, verbose=self.verbose)

        if fix_stats['fixed_count'] > 0:
            self.print(f"    ✓ Fixed {fix_stats['fixed_count']} corrupted bars", "DEBUG")

        # Step 3: Convert timezone to US Eastern
        self.print("    Converting to US Eastern timezone...", "DEBUG")
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Handle timezone conversion with DST handling
        if df['datetime'].dt.tz is None:
            # Assume UTC if no timezone
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')

        # Convert to Eastern time, handling DST ambiguity
        # During DST "fall back", use 'infer' to automatically choose the correct occurrence
        try:
            df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
        except Exception:
            # If standard conversion fails, convert without timezone info first
            # then re-localize to handle ambiguity
            df['datetime'] = df['datetime'].dt.tz_localize(None)
            df['datetime'] = df['datetime'].dt.tz_localize('America/New_York', ambiguous='infer')
        
        # Remove any duplicate timestamps that may still exist after conversion
        # (DST transitions can create duplicates even with 'infer')
        initial_count = len(df)
        df = df[~df['datetime'].duplicated(keep='first')].copy()
        duplicates_removed = initial_count - len(df)
        
        if duplicates_removed > 0:
            self.print(f"    Removed {duplicates_removed} duplicate timestamps (DST transition)", "DEBUG")

        # Step 4: Filter to trading hours (8:30 AM - 4:59 PM ET)
        # Start at 8:30 AM to allow technical indicators to warm up with 1 hour of data
        self.print("    Filtering to trading hours (8:30 AM - 4:59 PM ET)...", "DEBUG")
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute

        mask = (
            ((df['hour'] == 8) & (df['minute'] >= 30)) |
            ((df['hour'] >= 9) & (df['hour'] < 16)) |
            ((df['hour'] == 16) & (df['minute'] <= 59))
        )
        df = df[mask].copy()
        df = df.drop(columns=['hour', 'minute'])

        self.print(f"    Rows after trading hours filter: {len(df):,}", "DEBUG")

        # Step 5: Add technical indicators (only for minute data)
        if timeframe == '1m':
            self.print("    Calculating technical indicators...", "DEBUG")
            df = add_all_indicators(df)

            # Step 6: Add market regime features
            self.print("    Adding market regime features...", "DEBUG")
            df = add_market_regime_features(df)

        # Step 7: Remove NaN rows (from indicator calculation warmup period)
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)

        if dropped_rows > 0:
            self.print(f"    Dropped {dropped_rows} NaN rows (indicator warmup)", "DEBUG")

        # Step 8: Reset index
        df = df.reset_index(drop=True)

        self.print(f"    ✓ Processing complete: {len(df):,} rows", "DEBUG")

        return df

    # ============================================================
    # DATA MERGING & VALIDATION
    # ============================================================

    def _standardize_datetime_column(self, df: pd.DataFrame, context: str) -> pd.DataFrame:
        """
        Ensure the dataframe has a normalized datetime column.

        Args:
            df: DataFrame to normalize
            context: Description for logging (e.g., "old minute data")

        Returns:
            DataFrame with a `datetime` column ready for merging
        """
        df = df.copy()

        if 'datetime' not in df.columns:
            for candidate in ('ts_event', 'timestamp', 'date'):
                if candidate in df.columns:
                    df = df.rename(columns={candidate: 'datetime'})
                    self.print(f"    Normalized datetime column from '{candidate}' for {context}", "DEBUG")
                    break

        if 'datetime' not in df.columns:
            raise KeyError(f"'datetime' column missing in {context} (columns: {list(df.columns)})")

        df['datetime'] = pd.to_datetime(df['datetime'])
        return df

    def merge_with_existing(self, old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge existing data with new data

        Args:
            old_df: Existing processed data
            new_df: New processed data to append

        Returns:
            Merged DataFrame sorted chronologically
        """
        self.print("  Merging with existing data...", "DEBUG")

        # Ensure datetime columns are consistent for both datasets
        old_df = self._standardize_datetime_column(old_df, "existing data")
        new_df = self._standardize_datetime_column(new_df, "new data")

        # Get date ranges for validation
        old_end = old_df['datetime'].max()
        new_start = new_df['datetime'].min()

        self.print(f"    Old data ends: {old_end}", "DEBUG")
        self.print(f"    New data starts: {new_start}", "DEBUG")

        # Check for overlap (shouldn't happen if filtering was correct)
        if new_start <= old_end:
            self.print(f"    Warning: Overlap detected. Removing duplicates...", "WARN")
            # Keep only new data that comes after old data
            new_df = new_df[new_df['datetime'] > old_end].copy()

        # Concatenate
        merged_df = pd.concat([old_df, new_df], ignore_index=True)

        # Sort by datetime
        merged_df = merged_df.sort_values('datetime').reset_index(drop=True)

        self.print(f"    Old rows: {len(old_df):,}", "DEBUG")
        self.print(f"    New rows: {len(new_df):,}", "DEBUG")
        self.print(f"    Merged total: {len(merged_df):,}", "DEBUG")

        return merged_df

    def validate_merged_data(self, df: pd.DataFrame, expected_old_rows: int,
                            expected_new_rows: int) -> bool:
        """
        Validate merged data quality

        Args:
            df: Merged DataFrame
            expected_old_rows: Number of rows in old data
            expected_new_rows: Estimated new rows

        Returns:
            True if validation passed, False otherwise
        """
        self.print("  Validating merged data...", "DEBUG")

        validation_passed = True

        # Check 1: Row count is reasonable
        total_rows = len(df)
        min_expected = expected_old_rows + (expected_new_rows * 0.8)  # Allow 20% variance
        max_expected = expected_old_rows + (expected_new_rows * 1.2)

        if not (min_expected <= total_rows <= max_expected):
            self.print(f"    ⚠ Row count outside expected range: {total_rows:,} (expected {min_expected:.0f}-{max_expected:.0f})", "WARN")

        # Check 2: No duplicate timestamps
        duplicates = df['datetime'].duplicated().sum()
        if duplicates > 0:
            self.print(f"    ✗ Found {duplicates} duplicate timestamps", "ERROR")
            validation_passed = False
        else:
            self.print("    ✓ No duplicate timestamps", "DEBUG")

        # Check 3: Chronological order
        is_sorted = df['datetime'].is_monotonic_increasing
        if not is_sorted:
            self.print("    ✗ Data not in chronological order", "ERROR")
            validation_passed = False
        else:
            self.print("    ✓ Chronological order maintained", "DEBUG")

        # Check 4: Required columns present
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.print(f"    ✗ Missing columns: {missing_cols}", "ERROR")
            validation_passed = False
        else:
            self.print("    ✓ All required columns present", "DEBUG")

        # Check 5: No NaN in critical columns
        critical_cols = ['open', 'high', 'low', 'close']
        for col in critical_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    self.print(f"    ✗ Found {nan_count} NaN values in {col}", "ERROR")
                    validation_passed = False

        if validation_passed:
            self.print("    ✓ All validation checks passed", "DEBUG")

        return validation_passed

    # ============================================================
    # BACKUP & FILE MANAGEMENT
    # ============================================================

    def backup_existing_data(self, file_path: Path) -> Optional[Path]:
        """
        Create timestamped backup of existing data file

        Args:
            file_path: Path to file to backup

        Returns:
            Path to backup file or None if backup failed
        """
        if not file_path.exists():
            self.print(f"  No existing file to backup: {file_path.name}", "DEBUG")
            return None

        try:
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
            backup_path = self.backup_dir / backup_name

            # Copy file
            shutil.copy2(file_path, backup_path)

            file_size = backup_path.stat().st_size / (1024 * 1024)  # MB
            self.print(f"  ✓ Backup created: {backup_name} ({file_size:.1f} MB)")

            return backup_path

        except Exception as e:
            self.print(f"  ✗ Backup failed for {file_path.name}: {e}", "ERROR")
            return None

    def split_train_test(self, df: pd.DataFrame, train_ratio: float = 0.80) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically into train and test sets.
        
        Time-based split is critical for financial data:
        - Training uses EARLIER data (oldest 80%)
        - Testing uses LATER data (newest 20%)
        This prevents look-ahead bias and simulates real deployment.
        
        Args:
            df: Full DataFrame sorted by datetime
            train_ratio: Fraction for training (default 0.80 = 80%)
            
        Returns:
            (train_df, test_df) tuple
        """
        # Ensure data is sorted chronologically
        if 'datetime' in df.columns:
            df = df.sort_values('datetime').reset_index(drop=True)
        
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        # Log split info
        if 'datetime' in df.columns and len(train_df) > 0 and len(test_df) > 0:
            train_start = train_df['datetime'].min()
            train_end = train_df['datetime'].max()
            test_start = test_df['datetime'].min()
            test_end = test_df['datetime'].max()
            self.print(f"    Train: {len(train_df):,} rows ({train_start} to {train_end})", "DEBUG")
            self.print(f"    Test:  {len(test_df):,} rows ({test_start} to {test_end})", "DEBUG")
        
        return train_df, test_df

    def update_data_files(self, minute_df: pd.DataFrame,
                          second_df: Optional[pd.DataFrame] = None) -> Dict[str, bool]:
        """
        Atomically update data files with train/test split.
        
        Creates 6 files total:
        - {MARKET}_D1M.csv (full minute data)
        - {MARKET}_D1M_train.csv (80% oldest for training)
        - {MARKET}_D1M_test.csv (20% newest for evaluation)
        - {MARKET}_D1S.csv (full second data, if provided)
        - {MARKET}_D1S_train.csv (80% oldest for training)
        - {MARKET}_D1S_test.csv (20% newest for evaluation)

        Args:
            minute_df: Processed minute-level DataFrame
            second_df: Processed second-level DataFrame (optional)

        Returns:
            Dict with success status for each file type
        """
        results = {'minute': False, 'minute_train': False, 'minute_test': False,
                   'second': False, 'second_train': False, 'second_test': False}

        # ============================================================
        # MINUTE DATA
        # ============================================================
        minute_file = self.data_dir / f"{self.market}{MINUTE_FILE_SUFFIX}"
        minute_train_file = self.data_dir / f"{self.market}_D1M_train.csv"
        minute_test_file = self.data_dir / f"{self.market}_D1M_test.csv"

        try:
            # Write full minute data
            self.print("  Writing minute data (full)...", "DEBUG")
            minute_df.to_csv(minute_file, index=False)
            file_size = minute_file.stat().st_size / (1024 * 1024)
            self.print(f"  ✓ {minute_file.name} ({len(minute_df):,} rows, {file_size:.1f} MB)")
            results['minute'] = True

            # Split and write train/test
            self.print("  Splitting minute data (80% train / 20% test)...", "DEBUG")
            minute_train, minute_test = self.split_train_test(minute_df, train_ratio=0.80)
            
            minute_train.to_csv(minute_train_file, index=False)
            train_size = minute_train_file.stat().st_size / (1024 * 1024)
            self.print(f"  ✓ {minute_train_file.name} ({len(minute_train):,} rows, {train_size:.1f} MB)")
            results['minute_train'] = True
            
            minute_test.to_csv(minute_test_file, index=False)
            test_size = minute_test_file.stat().st_size / (1024 * 1024)
            self.print(f"  ✓ {minute_test_file.name} ({len(minute_test):,} rows, {test_size:.1f} MB)")
            results['minute_test'] = True

        except Exception as e:
            self.print(f"  ✗ Failed to update minute files: {e}", "ERROR")

        # ============================================================
        # SECOND DATA (if provided)
        # ============================================================
        if second_df is not None:
            second_file = self.data_dir / f"{self.market}{SECOND_FILE_SUFFIX}"
            second_train_file = self.data_dir / f"{self.market}_D1S_train.csv"
            second_test_file = self.data_dir / f"{self.market}_D1S_test.csv"

            try:
                # Write full second data
                self.print("  Writing second data (full)...", "DEBUG")
                second_df.to_csv(second_file, index=False)
                file_size = second_file.stat().st_size / (1024 * 1024)
                self.print(f"  ✓ {second_file.name} ({len(second_df):,} rows, {file_size:.1f} MB)")
                results['second'] = True

                # Split and write train/test
                self.print("  Splitting second data (80% train / 20% test)...", "DEBUG")
                second_train, second_test = self.split_train_test(second_df, train_ratio=0.80)
                
                second_train.to_csv(second_train_file, index=False)
                train_size = second_train_file.stat().st_size / (1024 * 1024)
                self.print(f"  ✓ {second_train_file.name} ({len(second_train):,} rows, {train_size:.1f} MB)")
                results['second_train'] = True
                
                second_test.to_csv(second_test_file, index=False)
                test_size = second_test_file.stat().st_size / (1024 * 1024)
                self.print(f"  ✓ {second_test_file.name} ({len(second_test):,} rows, {test_size:.1f} MB)")
                results['second_test'] = True

            except Exception as e:
                self.print(f"  ✗ Failed to update second files: {e}", "ERROR")

        return results


def main():
    """Main entry point for incremental data updater"""
    parser = argparse.ArgumentParser(
        description='Incrementally update futures trading data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and process new data for NQ
  python src/incremental_data_updater.py --market NQ

  # Preview changes without applying
  python src/incremental_data_updater.py --market NQ --dry-run

  # Verbose output
  python src/incremental_data_updater.py --market NQ --verbose
        """
    )

    parser.add_argument('--market', required=True, choices=SUPPORTED_MARKETS,
                       help='Market symbol to update (ES, NQ, YM, etc.)')
    parser.add_argument('--data-dir', default='data',
                       help='Data directory (default: data/)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without modifying files')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no-backup', action='store_true',
                       help='Disable backups (NOT RECOMMENDED)')
    parser.add_argument('--force-full', action='store_true',
                       help='Force full reprocessing (ignore existing data)')

    args = parser.parse_args()

    # Initialize updater
    updater = IncrementalDataUpdater(
        data_dir=args.data_dir,
        market=args.market,
        verbose=args.verbose,
        dry_run=args.dry_run,
        backup_enabled=not args.no_backup
    )

    # Print banner
    print("\n" + "=" * 60)
    print(f"INCREMENTAL DATA UPDATE - {args.market}")
    print("=" * 60)

    # Step 1: Detect existing data
    print("\n[DETECT] Checking for existing data...")
    existing_range = updater.get_existing_data_range()

    if existing_range:
        print(f"  Found: {args.market}{MINUTE_FILE_SUFFIX}")
        print(f"  Date range: {existing_range.start.date()} to {existing_range.end.date()}")
        print(f"  Rows: {existing_range.row_count:,}")
    else:
        print(f"  No existing data found for {args.market}")
        print("  Will process all data from .zip files")

    # Step 2: Detect zip files
    zip_files = updater.detect_zip_files()

    if not zip_files['minute']:
        print("\n[ERROR] No .zip files found in data directory")
        print("  Place GLBX-*.zip files in data/ directory first")
        return 1

    # Step 3: Analyze date ranges
    print("\n[ANALYZE] Date range analysis...")
    analysis = updater.analyze_date_ranges(existing_range, zip_files)

    if not analysis['has_new_data']:
        print("  Already up-to-date! No new data to process.")
        return 0

    new_start, new_end = analysis['new_date_range']
    print(f"  Existing: {existing_range.start.date() if existing_range else 'None'} - {existing_range.end.date() if existing_range else 'None'}")
    print(f"  New data in .zip: {zip_files['minute'][0].start_date.date()} - {zip_files['minute'][0].end_date.date()}")

    if analysis['overlap_days'] > 0:
        print(f"  Overlap: {analysis['overlap_days']} days (will skip)")
    if analysis['gap_days'] > 0:
        print(f"  ⚠ Gap: {analysis['gap_days']} days (missing data)")

    print(f"  New dates to add: {new_start.date()} - {new_end.date()} ({analysis['new_days']} days)")

    print(f"\n[ESTIMATE] Estimated new rows:")
    print(f"  Minute data: ~{analysis['estimated_minute_rows']:,} rows")
    print(f"  Second data: ~{analysis['estimated_second_rows']:,} rows")

    if args.dry_run:
        print("\n[DRY RUN] Preview complete. Use without --dry-run to apply changes.")
        return 0

    # User confirmation
    if not args.force_full:
        print(f"\nProceed with incremental update? (yes/no): ", end='')
        response = input().strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled by user")
            return 0

    # ============================================================
    # STEP 4: CREATE BACKUPS
    # ============================================================
    if updater.backup_enabled and existing_range:
        print("\n[BACKUP] Creating backups...")
        minute_file = updater.data_dir / f"{args.market}{MINUTE_FILE_SUFFIX}"
        second_file = updater.data_dir / f"{args.market}{SECOND_FILE_SUFFIX}"

        updater.backup_existing_data(minute_file)
        updater.backup_existing_data(second_file)

    # ============================================================
    # STEP 5: EXTRACT AND FILTER NEW DATA
    # ============================================================
    print("\n[PROCESS] Extracting new data from .zip files...")

    try:
        # Extract minute data
        minute_zip = zip_files['minute'][0]
        print(f"  Processing minute data ({new_start.date()} - {new_end.date()})...")
        raw_minute_df = updater.extract_and_filter_zip(minute_zip, new_start, new_end)

        # Extract second data if available
        raw_second_df = None
        if zip_files['second']:
            second_zip = zip_files['second'][0]
            print(f"  Processing second data ({new_start.date()} - {new_end.date()})...")
            raw_second_df = updater.extract_and_filter_zip(second_zip, new_start, new_end)

    except Exception as e:
        print(f"\n[ERROR] Failed to extract data: {e}")
        return 1

    # ============================================================
    # STEP 6: PROCESS THROUGH PIPELINE
    # ============================================================
    print("\n[PIPELINE] Processing new data...")

    try:
        # Process minute data
        print("  Minute data:")
        processed_minute_df = updater.process_raw_data(raw_minute_df, timeframe='1m')

        # Process second data if available
        processed_second_df = None
        if raw_second_df is not None:
            print("\n  Second data:")
            processed_second_df = updater.process_raw_data(raw_second_df, timeframe='1s')

    except Exception as e:
        print(f"\n[ERROR] Failed to process data: {e}")
        return 1

    # ============================================================
    # STEP 7: MERGE WITH EXISTING DATA
    # ============================================================
    print("\n[MERGE] Merging with existing data...")

    try:
        if existing_range:
            # Load existing data
            minute_file = updater.data_dir / f"{args.market}{MINUTE_FILE_SUFFIX}"
            old_minute_df = pd.read_csv(minute_file)

            # Merge minute data
            merged_minute_df = updater.merge_with_existing(old_minute_df, processed_minute_df)

            # Merge second data if exists
            merged_second_df = None
            if processed_second_df is not None:
                second_file = updater.data_dir / f"{args.market}{SECOND_FILE_SUFFIX}"
                if second_file.exists():
                    old_second_df = pd.read_csv(second_file)
                    merged_second_df = updater.merge_with_existing(old_second_df, processed_second_df)
                else:
                    merged_second_df = processed_second_df
        else:
            # No existing data - use processed data as-is
            merged_minute_df = processed_minute_df
            merged_second_df = processed_second_df

    except Exception as e:
        print(f"\n[ERROR] Failed to merge data: {e}")
        return 1

    # ============================================================
    # STEP 8: VALIDATE MERGED DATA
    # ============================================================
    print("\n[VALIDATE] Final validation...")

    old_rows = existing_range.row_count if existing_range else 0
    validation_passed = updater.validate_merged_data(
        merged_minute_df,
        expected_old_rows=old_rows,
        expected_new_rows=analysis['estimated_minute_rows']
    )

    if not validation_passed:
        print("\n[ERROR] Validation failed! Data not saved.")
        print("  Check logs above for details")
        return 1

    # ============================================================
    # STEP 9: SAVE UPDATED FILES
    # ============================================================
    print("\n[SAVE] Updating data files...")

    results = updater.update_data_files(merged_minute_df, merged_second_df)

    if not results['minute']:
        print("\n[ERROR] Failed to save minute data")
        return 1

    # ============================================================
    # SUCCESS SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("INCREMENTAL UPDATE COMPLETE!")
    print("=" * 60)

    print(f"\n[SUCCESS] Added {analysis['new_days']} days of new data")
    print(f"  Date range: {new_start.date()} - {new_end.date()}")
    print(f"  New minute rows: {len(processed_minute_df):,}")
    if processed_second_df is not None:
        print(f"  New second rows: {len(processed_second_df):,}")

    print(f"\n[FILES] Updated:")
    print(f"  ✓ {args.market}{MINUTE_FILE_SUFFIX} ({len(merged_minute_df):,} total rows)")
    print(f"  ✓ {args.market}_D1M_train.csv (80% = {int(len(merged_minute_df) * 0.8):,} rows)")
    print(f"  ✓ {args.market}_D1M_test.csv (20% = {int(len(merged_minute_df) * 0.2):,} rows)")
    if merged_second_df is not None:
        print(f"  ✓ {args.market}{SECOND_FILE_SUFFIX} ({len(merged_second_df):,} total rows)")
        print(f"  ✓ {args.market}_D1S_train.csv (80%)")
        print(f"  ✓ {args.market}_D1S_test.csv (20%)")

    if updater.backup_enabled:
        print(f"\n[BACKUP] Backups saved in {updater.backup_dir}/")

    print(f"\n[NEXT] Ready for training!")
    print(f"  Training will auto-use: {args.market}_D1M_train.csv")
    print(f"  Evaluation will auto-use: {args.market}_D1M_test.csv")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
