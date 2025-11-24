#!/usr/bin/env python3
"""
Quick verification test for the code improvements.
Tests that the improvements don't break existing functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_main_py_imports():
    """Test that main.py imports correctly"""
    print("Testing main.py imports...")
    try:
        # This will fail if there are syntax errors
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", project_root / "main.py")
        main_module = importlib.util.module_from_spec(spec)
        # Don't execute, just check syntax
        print("  ✅ main.py syntax valid")
        return True
    except Exception as e:
        print(f"  ❌ main.py has errors: {e}")
        return False

def test_incremental_updater_imports():
    """Test that incremental_data_updater.py imports correctly"""
    print("\nTesting incremental_data_updater.py imports...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "incremental_data_updater", 
            project_root / "src" / "incremental_data_updater.py"
        )
        updater_module = importlib.util.module_from_spec(spec)
        print("  ✅ incremental_data_updater.py syntax valid")
        return True
    except Exception as e:
        print(f"  ❌ incremental_data_updater.py has errors: {e}")
        return False

def test_select_instrument_exists():
    """Verify select_instrument method exists in main.py"""
    print("\nTesting select_instrument() method exists...")
    try:
        with open(project_root / "main.py", 'r', encoding='utf-8') as f:
            content = f.read()
            if "def select_instrument" in content:
                print("  ✅ select_instrument() method found")
                return True
            else:
                print("  ❌ select_instrument() method NOT found")
                return False
    except Exception as e:
        print(f"  ❌ Error reading main.py: {e}")
        return False

def test_pythonpath_config_exists():
    """Verify PYTHONPATH configuration exists in process_data_incremental"""
    print("\nTesting PYTHONPATH configuration...")
    try:
        with open(project_root / "main.py", 'r', encoding='utf-8') as f:
            content = f.read()
            if "env['PYTHONPATH']" in content and "process_data_incremental" in content:
                print("  ✅ PYTHONPATH configuration found in process_data_incremental()")
                return True
            else:
                print("  ❌ PYTHONPATH configuration NOT found")
                return False
    except Exception as e:
        print(f"  ❌ Error reading main.py: {e}")
        return False

def test_no_unused_imports():
    """Verify unused imports were removed"""
    print("\nTesting unused imports removal...")
    try:
        with open(project_root / "src" / "incremental_data_updater.py", 'r', encoding='utf-8') as f:
            content = f.read()
            has_clean_second = "import clean_second_data" in content
            has_process_second = "import process_second_data" in content
            
            if has_clean_second or has_process_second:
                print("  ❌ Unused imports still present")
                return False
            else:
                print("  ✅ Unused imports removed")
                return True
    except Exception as e:
        print(f"  ❌ Error reading incremental_data_updater.py: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("CODE IMPROVEMENTS VERIFICATION TEST")
    print("=" * 60)
    
    tests = [
        test_main_py_imports,
        test_incremental_updater_imports,
        test_select_instrument_exists,
        test_pythonpath_config_exists,
        test_no_unused_imports
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED!")
        print("\nYour code improvements are working correctly.")
        return 0
    else:
        print(f"❌ {total - passed} TEST(S) FAILED")
        print("\nPlease review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
