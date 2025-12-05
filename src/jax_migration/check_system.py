#!/usr/bin/env python3
import sys
import subprocess
import platform
import os

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()

def check_system():
    print("="*60)
    print("JAX System Prerequisite Check")
    print("="*60)

    # 1. Python Version
    print(f"\n[1] Python Version: {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        print("  WARNING: JAX requires Python 3.9+")
    else:
        print("  OK")

    # 2. OS Check
    print(f"\n[2] Operating System: {platform.system()} {platform.release()}")
    if platform.system() != "Linux":
        print("  NOTE: JAX is best supported on Linux/macOS. Windows support is experimental via WSL.")

    # 3. NVIDIA Driver / GPU Check
    print("\n[3] Checking NVIDIA GPU...")
    has_nvidia, output = run_command("nvidia-smi")
    if has_nvidia:
        print("  OK: NVIDIA Driver found.")
        # Extract GPU name
        for line in output.split('\n'):
            if "NVIDIA" in line and "MiB" in line:
                print(f"  GPU: {line.strip()}")
                break
        
        # Check CUDA version reported by driver
        try:
            cuda_version = output.split("CUDA Version:")[1].split()[0]
            print(f"  Driver CUDA Version: {cuda_version}")
        except:
            pass
    else:
        print("  WARNING: 'nvidia-smi' failed. No NVIDIA GPU detected or drivers missing.")
        print("  JAX will only run in CPU mode without a GPU.")

    # 4. Check for nvcc (CUDA Toolkit) - Optional but good to know
    print("\n[4] Checking CUDA Toolkit (nvcc)...")
    has_nvcc, output = run_command("nvcc --version")
    if has_nvcc:
        print("  OK: CUDA Toolkit found.")
        for line in output.split('\n'):
            if "release" in line:
                print(f"  {line.strip()}")
    else:
        print("  NOTE: 'nvcc' not found. This is okay if you install jax[cuda] wheels which include CUDA runtime.")

    # 5. Check pip
    print("\n[5] Checking pip...")
    has_pip, _ = run_command("pip --version")
    if has_pip:
        print("  OK: pip is installed.")
    else:
        print("  ERROR: pip is missing.")

    print("\n" + "="*60)
    print("RECOMMENDATION:")
    if has_nvidia:
        print("System appears ready for JAX with GPU support.")
        print("Run: pip install -r src/jax_migration/requirements_jax.txt")
    else:
        print("System lacks GPU support. You can still run JAX on CPU, but training will be slow.")
        print("Run: pip install jax jaxlib flax optax")
    print("="*60)

if __name__ == "__main__":
    check_system()
