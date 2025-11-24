#!/usr/bin/env python3
"""
Install verl and SofT-GRPO Stage 2 dependencies in the pod.

This script automates the installation of verl and all required dependencies
for Phase 3 Stage 2 (SofT-GRPO reasoning training).

Usage:
    python scripts/install_verl_pod.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with return code {e.returncode}")
        return False


def verify_installation():
    """Verify that all required packages are installed."""
    print(f"\n{'='*60}")
    print("Verifying installation...")
    print(f"{'='*60}\n")
    
    packages = [
        ("verl", "verl"),
        ("fastapi", "FastAPI"),
        ("ray", "Ray"),
        ("sglang", "SGLang"),
        ("hydra", "Hydra"),
        ("wandb", "Weights & Biases"),
    ]
    
    all_installed = True
    for import_name, display_name in packages:
        try:
            __import__(import_name)
            print(f"✅ {display_name} installed")
        except ImportError:
            print(f"❌ {display_name} NOT installed")
            all_installed = False
    
    return all_installed


def verify_versions():
    """Verify that critical packages meet version requirements."""
    print(f"\n{'='*60}")
    print("Verifying package versions...")
    print(f"{'='*60}\n")
    
    try:
        import transformers
        from packaging import version
        
        transformers_version = transformers.__version__
        min_version = "4.46.0"
        
        if version.parse(transformers_version) >= version.parse(min_version):
            print(f"✅ transformers {transformers_version} (>= {min_version})")
        else:
            print(f"❌ transformers {transformers_version} (< {min_version} - INCOMPATIBLE)")
            return False
    except ImportError:
        print("❌ transformers not installed")
        return False
    except Exception as e:
        print(f"⚠️  Could not verify transformers version: {e}")
    
    try:
        import peft
        print(f"✅ peft {peft.__version__}")
    except ImportError:
        print("❌ peft not installed")
        return False
    except Exception as e:
        print(f"⚠️  Could not verify peft version: {e}")
    
    # Test the critical import
    try:
        from transformers.modeling_layers import GradientCheckpointingLayer
        print("✅ transformers.modeling_layers import successful")
        return True
    except ImportError as e:
        print(f"❌ transformers.modeling_layers import failed: {e}")
        return False



def main():
    """Main installation process."""
    print("="*60)
    print("Installing verl for SofT-GRPO Stage 2")
    print("="*60)
    
    # Get project root (assuming script is in scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    verl_path = project_root / "SofT-GRPO-master-main" / "verl-0.4.x"
    
    # Check if verl directory exists
    if not verl_path.exists():
        print(f"❌ Error: verl directory not found at {verl_path}")
        print("Please ensure SofT-GRPO-master-main/verl-0.4.x exists")
        sys.exit(1)
    
    print(f"\n✅ Found verl at: {verl_path}")
    
    # Step 1: Upgrade transformers to fix compatibility
    if not run_command(
        "pip install --upgrade 'transformers>=4.46.0'",
        "[1/4] Upgrading transformers for compatibility"
    ):
        print("\n❌ transformers upgrade failed. Aborting.")
        sys.exit(1)
    
    # Step 2: Install verl
    if not run_command(
        f"pip install -e {verl_path}",
        "[2/4] Installing verl from bundled repository"
    ):
        print("\n❌ verl installation failed. Aborting.")
        sys.exit(1)
    
    # Step 3: Install dependencies
    dependencies = [
        "transformers>=4.46.0",  # Required for peft compatibility
        "peft",  # Ensure latest compatible version
        "fastapi",
        "uvicorn",
        "openai",
        "ray[default]>=2.10",
        "hydra-core",
        "datasets",
        "pyarrow>=19.0.0",
        "wandb",
        "codetiming",
        "dill",
        "liger-kernel",
        "tensordict<=0.6.2",
        "torchdata",
        "torchvision",
        "packaging>=20.0",
        "pybind11",
        "pylatexenc",
        "sglang[all]==0.4.6.post5",
        "torch-memory-saver>=0.0.5"
    ]
    
    # Install dependencies using subprocess directly to avoid shell escaping issues
    print(f"\n{'='*60}")
    print("[3/4] Installing verl dependencies")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install"] + dependencies,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ [3/4] Installing verl dependencies completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ [3/4] Installing verl dependencies failed with return code {e.returncode}")
        print("\n⚠️  Warning: Some dependencies may have failed to install.")
        print("Continuing with verification...")
    
    # Step 4: Verify installation
    print()
    
    # First verify versions
    versions_ok = verify_versions()
    
    # Then verify general installation
    packages_ok = verify_installation()
    
    if versions_ok and packages_ok:
        print("\n" + "="*60)
        print("✅ verl installation complete!")
        print("="*60)
        print("\nYou can now run Phase 3 Stage 2 (SofT-GRPO) training.")
        return 0
    else:
        print("\n" + "="*60)
        print("⚠️  Installation completed with warnings")
        print("="*60)
        print("\nSome packages may not have installed correctly.")
        print("Please check the output above for errors.")
        return 1



if __name__ == "__main__":
    sys.exit(main())
