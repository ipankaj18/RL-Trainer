#!/usr/bin/env python3
"""
LoRA Dependencies Verification Script

Checks if all required packages for LoRA fine-tuning are installed.
Run this after installing requirements.txt to verify Phase 3 is ready.

Usage:
    python verify_lora_dependencies.py
"""

import sys
from typing import List, Tuple

def check_import(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Check if a package can be imported.

    Args:
        package_name: Name of the package (for display)
        import_name: Actual import name (defaults to package_name)

    Returns:
        Tuple of (success: bool, version: str)
    """
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)


def main():
    """Run dependency checks."""
    print("=" * 70)
    print("LoRA FINE-TUNING DEPENDENCIES VERIFICATION")
    print("=" * 70)
    print()

    # Define required packages
    dependencies = [
        ("PyTorch", "torch"),
        ("Transformers", "transformers"),
        ("PEFT (LoRA)", "peft"),
        ("Accelerate", "accelerate"),
        ("BitsAndBytes", "bitsandbytes"),
        ("Safetensors", "safetensors"),
        ("Hugging Face Hub", "huggingface_hub"),
        ("SentencePiece", "sentencepiece"),
    ]

    results: List[Tuple[str, bool, str]] = []

    for package_name, import_name in dependencies:
        success, info = check_import(package_name, import_name)
        results.append((package_name, success, info))

    # Display results
    print("Package Versions:")
    print("-" * 70)

    all_installed = True
    for package_name, success, info in results:
        if success:
            status = "✅"
            message = f"v{info}"
        else:
            status = "❌"
            message = "NOT INSTALLED"
            all_installed = False

        print(f"{status} {package_name:<20} {message}")

    print()
    print("=" * 70)

    # Additional checks
    print("\nAdditional Checks:")
    print("-" * 70)

    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
            print(f"✅ CUDA Available: Yes (v{cuda_version})")
            print(f"   GPU Devices: {device_count}")
            print(f"   GPU 0: {device_name}")
        else:
            print("⚠️  CUDA Available: No (CPU-only mode)")
            print("   Phase 3 training will be VERY slow without GPU")
    except Exception as e:
        print(f"❌ CUDA Check Failed: {e}")

    # Check PEFT components
    if any(pkg[0] == "PEFT (LoRA)" and pkg[1] for pkg in results):
        try:
            from peft import LoraConfig, get_peft_model, PeftModel
            print("✅ PEFT Components: All imports successful")
            print("   - LoraConfig")
            print("   - get_peft_model")
            print("   - PeftModel")
        except ImportError as e:
            print(f"❌ PEFT Components: Import error - {e}")

    print()
    print("=" * 70)

    # Final verdict
    if all_installed:
        print("✅ ALL DEPENDENCIES INSTALLED - Ready for Phase 3 LoRA training!")
        print()
        print("Next steps:")
        print("  1. Verify Phi-3 model exists: ls -lh Phi-3-mini-4k-instruct/")
        print("  2. Test Phase 3 training:    python src/train_phase3_llm.py --test")
        print("  3. Check LoRA setup in logs: Look for 'Target: all-linear'")
        return 0
    else:
        print("❌ MISSING DEPENDENCIES - Install missing packages")
        print()
        print("To install all requirements:")
        print("  pip install -r requirements.txt")
        print()
        print("To install just PEFT:")
        print("  pip install peft>=0.7.1")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
