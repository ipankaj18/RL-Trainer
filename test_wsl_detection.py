import platform
import os
import sys

def supports_subproc_vec_env() -> bool:
    if os.name == 'nt':
        print("os.name is nt")
        return False
    
    try:
        release = platform.release().lower()
        print(f"Release string: {release}")
        if 'microsoft' in release or 'wsl' in release:
            print("Detected WSL via release string")
            return False
    except Exception as e:
        print(f"Exception checking release: {e}")
        pass
        
    return True

print(f"supports_subproc_vec_env returns: {supports_subproc_vec_env()}")
