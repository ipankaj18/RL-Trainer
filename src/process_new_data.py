#!/usr/bin/env python3
"""
Process New Market Data (with Automatic Corruption Detection & Fixing)

USE THIS SCRIPT when you download new data from Databento!

This script automatically:
1. ✅ Detects if data is corrupted (divide-by-100 errors)
2. ✅ Fixes corruption automatically
3. ✅ Validates data quality
4. ✅ Generates clean training-ready data

You don't need to check if data is good or bad - the script does it for you!

Usage Examples:
    # Process new NQ data (automatic corruption detection & fixing)
    python src/process_new_data.py --market NQ

    # Process new ES data
    python src/process_new_data.py --market ES

    # Process multiple markets
    python src/process_new_data.py --market ES NQ YM

    # See what would happen without actually processing
    python src/process_new_data.py --market NQ --dry-run
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


# Supported markets
SUPPORTED_MARKETS = ['ES', 'NQ', 'YM', 'RTY', 'MES', 'MNQ', 'M2K', 'MYM']


def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent


def check_data_exists(market: str) -> bool:
    """Check if training data already exists for this market"""
    data_dir = get_project_root() / "data"
    data_file = data_dir / f"{market}_D1M.csv"
    return data_file.exists()


def process_market_data(market: str, dry_run: bool = False) -> bool:
    """
    Process new market data with automatic corruption detection and fixing

    Args:
        market: Market symbol (ES, NQ, etc.)
        dry_run: If True, only show what would be done

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 70)
    print(f"PROCESSING NEW {market} DATA")
    print("=" * 70)

    # Check if data already exists
    existing = check_data_exists(market)
    if existing:
        print(f"\n⚠️  Warning: {market}_D1M.csv already exists")
        print(f"This will OVERWRITE existing data for {market}")
        response = input("\nContinue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled by user")
            return False

    if dry_run:
        print("\n[DRY RUN] Would run:")
        print(f"  python src/update_training_data.py --market {market}")
        print("\nWith automatic:")
        print("  - Corruption detection (divide-by-100 errors)")
        print("  - Automatic fixing (multiply by 100)")
        print("  - Statistical validation (IQR method)")
        print("  - Data cleaning (NaN, duplicates, outliers)")
        return True

    # Run update_training_data.py which has auto-detection and auto-fixing
    src_dir = get_project_root() / "src"
    script_path = src_dir / "update_training_data.py"

    print(f"\nRunning data processing with automatic corruption detection...")
    print(f"(Any corruption will be detected and fixed automatically)\n")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), '--market', market],
            cwd=get_project_root(),
            check=True
        )

        print(f"\n{'='*70}")
        print(f"✅ {market} DATA PROCESSING COMPLETE")
        print(f"{'='*70}")

        # Verify output
        data_dir = get_project_root() / "data"
        data_file = data_dir / f"{market}_D1M.csv"

        if data_file.exists():
            size_mb = data_file.stat().st_size / (1024 * 1024)
            print(f"\nOutput file: {data_file.name}")
            print(f"Size: {size_mb:.2f} MB")
            print(f"\n✅ Clean data ready for training!")
        else:
            print(f"\n⚠️  Warning: Expected output file not found")
            return False

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Processing failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Process new market data with automatic corruption detection and fixing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --market NQ                    # Process new NQ data
  %(prog)s --market ES NQ                 # Process ES and NQ
  %(prog)s --market NQ --dry-run          # Preview without processing

What this script does automatically:
  ✅ Detects corrupted data (divide-by-100 errors)
  ✅ Fixes corruption by multiplying by 100
  ✅ Validates prices using statistical methods
  ✅ Removes NaN values and duplicates
  ✅ Detects anomalous price jumps
  ✅ Generates clean training-ready data

You don't need to check if data is good or bad - it's all automatic!
        """
    )

    parser.add_argument(
        '--market',
        type=str,
        nargs='+',
        required=True,
        choices=SUPPORTED_MARKETS,
        help='Market(s) to process (ES, NQ, YM, RTY, MES, MNQ, M2K, MYM)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually doing it'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("NEW DATA PROCESSING (AUTO-CORRECTION ENABLED)")
    print("=" * 70)
    print(f"\nMarkets: {', '.join(args.market)}")
    print(f"Dry run: {'YES' if args.dry_run else 'NO'}")

    if not args.dry_run:
        print("\n📥 Processing data from Databento zip files...")
        print("🔍 Automatic corruption detection enabled")
        print("🔧 Automatic corruption fixing enabled")

    # Process each market
    results = {}
    for market in args.market:
        results[market] = process_market_data(market, args.dry_run)

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
        print("\n[DRY RUN] No actual processing was done")
        return 0

    if success_count == total_count:
        print("\n🎉 All markets processed successfully!")
        print("✅ Data is clean and ready for training")
        return 0
    else:
        print("\n⚠️  Some markets failed to process")
        return 1


if __name__ == "__main__":
    sys.exit(main())
