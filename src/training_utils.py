import os
import platform

def supports_subproc_vec_env() -> bool:
    """
    Detect whether SubprocVecEnv is safe on this host.

    Windows / WSL builds often struggle with forked environments, so we disable
    it automatically there while keeping it enabled for native Linux pods.
    """
    if os.name == 'nt':
        return False
    
    # Check for WSL (Windows Subsystem for Linux)
    # WSL usually has 'microsoft' or 'wsl' in the release string
    try:
        release = platform.release().lower()
        if 'microsoft' in release or 'wsl' in release:
            return False
    except Exception:
        pass
        
    return True
