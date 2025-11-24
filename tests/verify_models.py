
import zipfile
import numpy as np
import torch
import os

def check_zip(path):
    print(f"Checking {path}...")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    try:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            bad_file = zip_ref.testzip()
            if bad_file:
                print(f"CORRUPTED: {bad_file}")
            else:
                print("OK: Zip file is valid.")
    except Exception as e:
        print(f"ERROR: {e}")

def check_npz(path):
    print(f"Checking {path}...")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    try:
        data = np.load(path)
        print(f"OK: Loaded npz with keys: {list(data.keys())}")
    except Exception as e:
        print(f"ERROR: {e}")

def check_torch_load(path):
    print(f"Checking torch load {path}...")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    try:
        # Just try to load the header or map_location='cpu'
        torch.load(path, map_location='cpu')
        print("OK: Torch load successful.")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    check_zip("models/phase1_foundational_fixed.zip")
    check_npz("logs/phase1/evaluations.npz")
    # Also check the pkl file
    check_torch_load("models/phase1_foundational_fixed_vecnorm.pkl")
