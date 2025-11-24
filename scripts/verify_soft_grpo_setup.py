
import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from pipeline.soft_grpo_config import load_soft_grpo_config, get_soft_grpo_paths
from pipeline.soft_grpo_validation import validate_soft_grpo_requirements, print_validation_report
from pipeline.soft_grpo_manifest import list_markets_with_stage1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_soft_grpo")

def main():
    print("="*60)
    print("SofT-GRPO Setup Verification")
    print("="*60)

    # 1. Load Config
    try:
        config = load_soft_grpo_config()
        print("[OK] Config loaded successfully")
    except Exception as e:
        print(f"[FAIL] Failed to load config: {e}")
        return

    # 2. Check Paths
    paths = get_soft_grpo_paths(config)
    print(f"\nChecking paths:")
    for name, path in paths.items():
        exists = path.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"{status} {name}: {path}")
        
        if name == "repo_root" and not exists:
            print(f"  [WARN] Stage 2 repo not found at {path}. Clone it or update config.")

    # 3. Check Stage 1 Metadata
    print(f"\nChecking Stage 1 Metadata (Phase 3 completion):")
    markets = list_markets_with_stage1(paths["model_root"])
    if not markets:
        print("[WARN] No markets found with Stage 1 completion.")
        print("       Run Phase 3 training first: python src/train_phase3_llm.py --market NQ --test")
    else:
        for m in markets:
            print(f"[OK] Found Stage 1 data for market: {m['market']}")
            
            # Run Validation for this market
            print(f"\nRunning Validation for {m['market']}...")
            checks = validate_soft_grpo_requirements(
                config=config,
                project_root=Path.cwd(),
                stage1=m['stage1'],
                training_enabled=True
            )
            print_validation_report(checks)

    # 4. Check Dependencies (verl)
    print(f"\nChecking Dependencies:")
    repo_root = paths["repo_root"]
    if repo_root.exists():
        # Check if verl is importable or present
        verl_path = repo_root / "verl"
        if verl_path.exists():
             print(f"[OK] 'verl' package found in {repo_root}")
        else:
             print(f"[WARN] 'verl' package NOT found in {repo_root}")
    
    print("\nVerification Complete.")

if __name__ == "__main__":
    main()
