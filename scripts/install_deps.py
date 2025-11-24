import subprocess
import sys
import platform
import os

def install(args):
    cmd = [sys.executable, "-m", "pip", "install"] + args
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    print(f"Platform: {platform.system()}")
    
    # Path to requirements.txt (one level up)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(script_dir, "..", "requirements.txt")
    
    if not os.path.exists(requirements_path):
        print(f"Error: requirements.txt not found at {requirements_path}")
        sys.exit(1)

    # 1. Install flash-attn with --no-build-isolation on Linux
    # This must be done before the general requirements to avoid build isolation issues
    # when torch is present in the environment but not visible to the build process.
    if platform.system() != "Windows":
        print("Detected non-Windows system. Installing flash-attn with --no-build-isolation...")
        try:
            # Install flash-attn explicitly
            install(["flash-attn>=2.5.8", "--no-build-isolation"])
        except subprocess.CalledProcessError:
            print("Failed to install flash-attn. It might be incompatible or require CUDA. Continuing...")
    else:
        print("Windows detected. Skipping flash-attn (not supported/required).")

    # 2. Install the rest
    print(f"Installing requirements from {requirements_path}...")
    install(["-r", requirements_path])
    
    print("Installation complete.")

if __name__ == "__main__":
    main()
