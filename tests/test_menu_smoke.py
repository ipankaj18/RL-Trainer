import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

try:
    from main import RLTrainerMenu
    print("Successfully imported RLTrainerMenu")
    
    menu = RLTrainerMenu()
    print("Successfully instantiated RLTrainerMenu")
    
    # Check if methods exist
    methods = [
        "continue_training_from_model",
        "run_evaluation",
        "run_jax_training_menu",
        "launch_jax_training",
        "install_requirements",
        "process_data_incremental",
        "run_pipeline"
    ]
    
    for method in methods:
        if hasattr(menu, method):
            print(f"Method {method} exists")
        else:
            print(f"ERROR: Method {method} missing")
            sys.exit(1)
            
    print("Smoke test passed!")
    
except Exception as e:
    print(f"Smoke test failed: {e}")
    sys.exit(1)
