
import sys
from pathlib import Path
sys.path.append("src")
from pipeline.soft_grpo_validation import validate_soft_grpo_requirements, print_validation_report

def test():
    # Mock config with a repo_root that definitely doesn't have verl
    config = {"soft_grpo": {"repo_root": "C:/NonExistentPath"}}
    project_root = Path.cwd()
    stage1 = {
        "base_model": {"local_path": "mock_model"},
        "lora_adapter_path": "mock_adapter"
    } 
    
    print("Testing validation logic with mock data...")
    try:
        checks = validate_soft_grpo_requirements(config, project_root, stage1, training_enabled=True)
        print_validation_report(checks)
    except Exception as e:
        print(f"Validation raised exception (expected if checks fail): {e}")

if __name__ == "__main__":
    test()
