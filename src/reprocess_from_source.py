#!/usr/bin/env python3
"""
Reprocess Market Data From Source

This script helps you reprocess data from original vendor zip files with
automatic corruption detection and fixing. Use this when you want to:
1. Fix corrupted data by reprocessing from source
2. Add a new market to your data pipeline
3. Update existing market data with latest vendor data

The script will:
- Delete old corrupted data files
- Extract and process data from zip files
- Automatically detect and fix price format errors
- Validate all data using statistical methods
- Generate clean training-ready data files

Usage Examples:
    # Reprocess NQ data (deletes old NQ files and reprocesses from zip)
    python src/reprocess_from_source.py --market NQ

    # Reprocess multiple markets
    python src/reprocess_from_source.py --market ES NQ YM

    # Reprocess all supported markets
    python src/reprocess_from_source.py --all

    # Dry run (show what would be deleted without actually deleting)
    python src/reprocess_from_source.py --market NQ --dry-run
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple


# Supported markets
SUPPORTED_MARKETS = ['ES', 'NQ', 'YM', 'RTY', 'MES', 'MNQ', 'M2K', 'MYM']


def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent


def get_data_files_for_market(market: str) -> List[Path]:
    """Get all data files associated with a market"""
    data_dir = get_project_root() / "data"
    files = []

    # Training data files
    for suffix in ['_D1M.csv', '_D1S.csv']:
        file_path = data_dir / f"{market}{suffix}"
        if file_path.exists():
            files.append(file_path)

    # Raw/intermediate files
    raw_patterns = [
        f"{market.lower()}_second_level_data_raw.csv",
        f"{market.lower()}_second_level_data.csv",
    ]

    for pattern in raw_patterns:
        file_path = data_dir / pattern
        if file_path.exists():
            files.append(file_path)

    return files


def delete_market_data(market: str, dry_run: bool = False) -> Tuple[List[Path], int]:
    """
    Delete all data files for a market

    Args:
        market: Market symbol (ES, NQ, etc.)
        dry_run: If True, only show what would be deleted

    Returns:
        Tuple of (deleted_files, total_size_mb)
    """
    files = get_data_files_for_market(market)
    total_size = 0

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Deleting {market} data files:")

    if not files:
        print(f"  No existing {market} data files found")
        return [], 0

    for file_path in files:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {'Would delete' if dry_run else 'Deleting'}: {file_path.name} ({size_mb:.2f} MB)")

        if not dry_run:
            file_path.unlink()
            print(f"    ✅ Deleted")

    return files, total_size


def run_update_training_data(market: str) -> bool:
    """
    Run update_training_data.py for a market

    Args:
        market: Market symbol (ES, NQ, etc.)

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING {market} MINUTE DATA")
    print(f"{'='*70}")

    src_dir = get_project_root() / "src"
    script_path = src_dir / "update_training_data.py"

    try:
        # Run update_training_data.py with market argument
        result = subprocess.run(
            [sys.executable, str(script_path), '--market', market],
            cwd=get_project_root(),
            check=True,
            capture_output=False,
            text=True
        )

        print(f"\n✅ {market} minute data processing completed")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ {market} minute data processing failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Error running update_training_data.py: {e}")
        return False


def verify_output_files(market: str) -> Tuple[bool, List[str]]:
    """
    Verify that expected output files were created

    Args:
        market: Market symbol (ES, NQ, etc.)

    Returns:
        Tuple of (success, messages)
    """
    data_dir = get_project_root() / "data"
    messages = []
    success = True

    # Check for main training file
    minute_file = data_dir / f"{market}_D1M.csv"

    if minute_file.exists():
        size_mb = minute_file.stat().st_size / (1024 * 1024)
        messages.append(f"✅ {minute_file.name}: {size_mb:.2f} MB")

        # Verify file has content
        if size_mb < 0.1:
            messages.append(f"  ⚠️  Warning: File seems too small")
            success = False
    else:
        messages.append(f"❌ {minute_file.name}: NOT FOUND")
        success = False

    # Check for second-level data (optional)
    second_file = data_dir / f"{market}_D1S.csv"
    if second_file.exists():
        size_mb = second_file.stat().st_size / (1024 * 1024)
        messages.append(f"✅ {second_file.name}: {size_mb:.2f} MB")

    return success, messages


def reprocess_market(market: str, dry_run: bool = False) -> bool:
    """
    Reprocess a single market from source

    Args:
        market: Market symbol (ES, NQ, etc.)
        dry_run: If True, only show what would be done

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 70)
    print(f"REPROCESSING {market} FROM SOURCE")
    print("=" * 70)

    # Step 1: Delete old data
    deleted_files, total_size = delete_market_data(market, dry_run)

    if deleted_files:
        print(f"\nDeleted {len(deleted_files)} files ({total_size:.2f} MB total)")
    else:
        print(f"\nNo existing {market} data files to delete")

    if dry_run:
        print("\n[DRY RUN] Would now run:")
        print(f"  python src/update_training_data.py --market {market}")
        return True

    # Step 2: Process data from source
    success = run_update_training_data(market)

    if not success:
        print(f"\n❌ Failed to reprocess {market}")
        return False

    # Step 3: Verify output files
    print(f"\n{'='*70}")
    print(f"VERIFICATION")
    print(f"{'='*70}")

    verified, messages = verify_output_files(market)

    for msg in messages:
        print(msg)

    if verified:
        print(f"\n✅ {market} reprocessing completed successfully!")
    else:
        print(f"\n⚠️  {market} reprocessing completed with warnings")

    return verified


def main():
    parser = argparse.ArgumentParser(
        description='Reprocess market data from source with automatic corruption fixing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --market NQ                    # Reprocess NQ
  %(prog)s --market ES NQ YM              # Reprocess multiple markets
  %(prog)s --all                          # Reprocess all markets
  %(prog)s --market NQ --dry-run          # Show what would be done
        """
    )

    parser.add_argument(
        '--market',
        type=str,
        nargs='+',
        choices=SUPPORTED_MARKETS,
        help='Market(s) to reprocess (ES, NQ, YM, RTY, MES, MNQ, M2K, MYM)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Reprocess all supported markets'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually doing it'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.market and not args.all:
        parser.error("Either --market or --all must be specified")

    # Determine which markets to process
    if args.all:
        markets = SUPPORTED_MARKETS
    else:
        markets = args.market

    print("=" * 70)
    print("DATA REPROCESSING UTILITY")
    print("=" * 70)
    print(f"Markets to reprocess: {', '.join(markets)}")
    print(f"Dry run: {'YES' if args.dry_run else 'NO'}")

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No actual changes will be made")

    # Confirm with user
    if not args.dry_run:
        print("\n⚠️  WARNING: This will DELETE existing data files for these markets!")
        print("Data will be reprocessed from source zip files with corruption fixes.")
        response = input("\nContinue? (yes/no): ").strip().lower()

        if response not in ['yes', 'y']:
            print("Aborted by user")
            return 1

    # Process each market
    results = {}
    for market in markets:
        results[market] = reprocess_market(market, args.dry_run)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)

    for market, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{market}: {status}")

    print(f"\nCompleted: {success_count}/{total_count} markets")

    if args.dry_run:
        print("\n[DRY RUN] No actual changes were made")
        return 0

    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
