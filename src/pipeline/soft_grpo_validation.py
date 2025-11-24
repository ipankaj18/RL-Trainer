"""Validation utilities for SofT-GRPO Stage 2."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SoftGrpoValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_soft_grpo_requirements(
    config: Dict[str, Any],
    project_root: Path,
    stage1: Dict[str, Any],
    training_enabled: bool,
) -> List[Dict[str, Any]]:
    """
    Validate requirements for SofT-GRPO Stage 2.

    Args:
        config: Loaded configuration dictionary.
        project_root: Path to the project root.
        stage1: Stage 1 metadata from the manifest.
        training_enabled: Whether training is enabled (vs dataset export only).

    Returns:
        List of validation check results (dict with 'name', 'passed', 'message').
    """
    checks = []

    # 1. Check Config Structure
    checks.append({
        "name": "Config Structure",
        "passed": bool(config),
        "message": "Config is empty" if not config else "Config loaded"
    })

    if not config:
        return checks

    # 2. Check Model Root
    # FIXED: Check soft_grpo.model_root instead of paths.model_root
    model_root = config.get("soft_grpo", {}).get("model_root")
    if model_root:
        # Resolve model root relative to project root if not absolute
        model_root_path = Path(model_root)
        if not model_root_path.is_absolute():
            model_root_path = project_root / model_root_path
        
        checks.append({
            "name": "Model Root Directory",
            "passed": model_root_path.exists(),
            "message": f"Found at {model_root_path}" if model_root_path.exists() else f"Missing at {model_root_path}"
        })
    else:
        checks.append({
            "name": "Model Root Directory",
            "passed": False,
            "message": "soft_grpo.model_root not defined in config"
        })

    # 3. Check Stage 1 Data
    checks.append({
        "name": "Stage 1 Metadata",
        "passed": bool(stage1),
        "message": "Stage 1 metadata present" if stage1 else "Stage 1 metadata missing"
    })

    # 4. Training Specific Checks
    if training_enabled and stage1:
        # Check Base Model
        base_model = stage1.get("base_model", {})
        local_path = base_model.get("local_path")
        
        if local_path:
            local_path_obj = Path(local_path)
            if not local_path_obj.is_absolute():
                local_path_obj = project_root / local_path_obj
                
            checks.append({
                "name": "Base Model Path",
                "passed": local_path_obj.exists(),
                "message": f"Found at {local_path_obj}" if local_path_obj.exists() else f"Missing at {local_path_obj}"
            })
        else:
             checks.append({
                "name": "Base Model Path",
                "passed": False,
                "message": "base_model.local_path not found in Stage 1 metadata"
            })

        # Check LoRA Adapter
        lora_path = stage1.get("lora_adapter_path")
        if lora_path:
            lora_path_obj = Path(lora_path)
            if not lora_path_obj.is_absolute():
                lora_path_obj = project_root / lora_path_obj
                
            checks.append({
                "name": "LoRA Adapter Path",
                "passed": lora_path_obj.exists(),
                "message": f"Found at {lora_path_obj}" if lora_path_obj.exists() else f"Missing at {lora_path_obj}"
            })
        else:
             checks.append({
                "name": "LoRA Adapter Path",
                "passed": False,
                "message": "lora_adapter_path not found in Stage 1 metadata"
            })

    # Check Dependencies
    # FIXED: Check if verl is installed as a Python package
    try:
        import verl
        checks.append({
            "name": "Dependency: verl",
            "passed": True,
            "message": f"verl package installed (version: {getattr(verl, '__version__', 'unknown')})"
        })
    except ImportError:
        repo_root = Path(config.get("soft_grpo", {}).get("repo_root", project_root))
        verl_install_path = repo_root / "verl-0.4.x"
        checks.append({
            "name": "Dependency: verl",
            "passed": False,
            "message": f"verl not installed. Install with: cd {verl_install_path} && pip install -e ."
        })

    # Check for flash_attn (optional but recommended)
    try:
        import flash_attn
        if getattr(flash_attn, "__version__", None) == "unavailable":
            raise RuntimeError("flash_attn stub active")
        checks.append({
            "name": "Dependency: flash_attn",
            "passed": True,
            "message": f"flash_attn installed (v{flash_attn.__version__})"
        })
    except Exception:
        checks.append({
            "name": "Dependency: flash_attn",
            "passed": True,  # Warning only, so we don't fail validation
            "message": "flash_attn unavailable; falling back to eager attention."
        })

    return checks


def print_validation_report(checks: List[Dict[str, Any]]) -> None:
    """
    Print a formatted report of validation checks.

    Args:
        checks: List of check results from validate_soft_grpo_requirements.
    
    Raises:
        SoftGrpoValidationError: If any check failed.
    """
    print("\n[SofT-GRPO] Validation Report:")
    print("=" * 60)
    
    all_passed = True
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['name']}: {check['message']}")
        if not check["passed"]:
            all_passed = False
            
    print("=" * 60)
    print()

    if not all_passed:
        failed_checks = [c['name'] for c in checks if not c['passed']]
        raise SoftGrpoValidationError(f"Validation failed for: {', '.join(failed_checks)}")
