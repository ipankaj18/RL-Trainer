#!/usr/bin/env python3
"""
Diagnostic script to check package installation status.

This script helps debug why packages appear as "missing" even when installed.
"""

import sys
import subprocess


def check_package_methods(package_name):
    """Check package using multiple methods."""
    print(f"\n{'='*60}")
    print(f"Checking: {package_name}")
    print(f"{'='*60}")
    
    results = {}
    
    # Method 1: Direct import
    try:
        __import__(package_name.replace('-', '_'))
        results['direct_import'] = '✅ SUCCESS'
    except ImportError as e:
        results['direct_import'] = f'❌ FAILED: {e}'
    
    # Method 2: importlib.metadata
    try:
        from importlib.metadata import distribution
        dist = distribution(package_name)
        results['importlib.metadata'] = f'✅ SUCCESS (version: {dist.version})'
    except Exception as e:
        results['importlib.metadata'] = f'❌ FAILED: {e}'
    
    # Method 3: pkg_resources
    try:
        import pkg_resources
        dist = pkg_resources.get_distribution(package_name)
        results['pkg_resources'] = f'✅ SUCCESS (version: {dist.version})'
    except Exception as e:
        results['pkg_resources'] = f'❌ FAILED: {e}'
    
    # Method 4: pip list
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--format=freeze'],
            capture_output=True,
            text=True,
            check=True
        )
        pip_packages = result.stdout.lower()
        if package_name.lower() in pip_packages:
            results['pip_list'] = '✅ FOUND in pip list'
        else:
            results['pip_list'] = '❌ NOT FOUND in pip list'
    except Exception as e:
        results['pip_list'] = f'❌ FAILED: {e}'
    
    # Print results
    for method, result in results.items():
        print(f"  {method:20s}: {result}")
    
    return results


def main():
    """Main diagnostic function."""
    print("="*60)
    print("PACKAGE INSTALLATION DIAGNOSTIC")
    print("="*60)
    print(f"\nPython executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Packages to check
    packages_to_check = [
        'peft',
        'verl',
        'transformers',
        'torch',
        'accelerate',
        'bitsandbytes',
    ]
    
    all_results = {}
    for pkg in packages_to_check:
        all_results[pkg] = check_package_methods(pkg)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    for pkg, results in all_results.items():
        # Check if at least one method succeeded
        any_success = any('✅' in str(v) for v in results.values())
        status = '✅ INSTALLED' if any_success else '❌ NOT INSTALLED'
        print(f"{pkg:20s}: {status}")
        
        # Show which methods worked
        if any_success:
            working_methods = [k for k, v in results.items() if '✅' in str(v)]
            print(f"{'':20s}  Working: {', '.join(working_methods)}")
    
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}\n")
    
    # Check for packages that show in pip but not importable
    for pkg, results in all_results.items():
        pip_ok = '✅' in str(results.get('pip_list', ''))
        import_ok = '✅' in str(results.get('direct_import', ''))
        
        if pip_ok and not import_ok:
            print(f"⚠️  {pkg}: Installed via pip but not importable")
            print(f"   Try: pip install --force-reinstall {pkg}")
            print()


if __name__ == "__main__":
    main()
