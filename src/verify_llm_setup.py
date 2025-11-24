#!/usr/bin/env python3
"""
LLM Setup Verification Helper

Ensures base model weights and the LoRA adapter are staged
correctly before running Phase 3 training.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import yaml
try:
    from src.llm_utils import ensure_model_downloaded
except ImportError:
    # Handle running as script from src/
    from llm_utils import ensure_model_downloaded

logger = logging.getLogger("verify_llm_setup")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_paths() -> Tuple[Path, Path, str]:
    """Read llm_config.yaml to determine base + adapter paths."""
    config_path = PROJECT_ROOT / "config" / "llm_config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    llm_cfg = cfg.get("llm_model", {})

    base_path = Path(llm_cfg.get("local_path") or "Base_Model")
    
    adapter_val = llm_cfg.get("adapter_path")
    adapter_path = Path(adapter_val) if adapter_val else None

    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_path
    if adapter_path and not adapter_path.is_absolute():
        adapter_path = PROJECT_ROOT / adapter_path

    return base_path, adapter_path, llm_cfg.get("name", "microsoft/Phi-3-mini-128k-instruct")


def check_dependencies() -> bool:
    """Verify core Python packages are installed."""
    logger.info("=" * 60)
    logger.info("STEP 1: Checking dependencies")
    logger.info("=" * 60)
    required = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers",
        "peft": "PEFT (LoRA)",
        "bitsandbytes": "bitsandbytes (quantized loading)",
        "huggingface_hub": "huggingface_hub (auto-download)"
    }
    missing = []
    for module, description in required.items():
        try:
            __import__(module)
            logger.info("✅ %s", description)
        except ImportError:
            logger.error("❌ %s missing (pip install %s)", description, module)
            missing.append(module)
    logger.info("")
    return not missing


def check_base_model(base_path: Path, model_name: str) -> bool:
    """Ensure base model assets exist locally."""
    logger.info("=" * 60)
    logger.info("STEP 2: Checking base model files")
    logger.info("=" * 60)
    # Use shared utility to check/download
    if ensure_model_downloaded(model_name, base_path):
        # Additional version check (optional, but good to keep)
        config_path = base_path / "config.json"
        if config_path.exists():
            try:
                import json
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                max_pos = config_data.get("max_position_embeddings", 0)
                expected_4k = "4k" in model_name.lower()
                expected_128k = "128k" in model_name.lower()
                
                if expected_4k and max_pos > 8192:
                    logger.warning("⚠️  Found 128k model but config requests 4k model.")
                elif expected_128k and max_pos < 10000:
                    logger.warning("⚠️  Found 4k model but config requests 128k model.")
                    
            except Exception as e:
                logger.warning("⚠️  Could not verify model version from config.json: %s", e)
        
        return True
    
    return False


def check_adapter(adapter_path: Path) -> bool:
    """Ensure the LoRA adapter exists."""
    logger.info("=" * 60)
    logger.info("STEP 3: Checking LoRA adapter files")
    logger.info("=" * 60)
    
    if adapter_path is None:
        logger.info("ℹ️  No adapter configured (training from scratch). Skipping check.")
        logger.info("")
        return True

    if not adapter_path.exists():
        logger.error("❌ Adapter directory not found: %s", adapter_path)
        return False
    if not (adapter_path / "adapter_config.json").exists():
        logger.error("❌ adapter_config.json missing in %s", adapter_path)
        return False
    if not list((adapter_path).glob("adapter_model*.safetensors")) and not list(
        (adapter_path).glob("adapter_model*.bin")
    ):
        logger.error("❌ Adapter weights missing (adapter_model.safetensors/bin) in %s", adapter_path)
        return False
    logger.info("✅ Adapter present at %s", adapter_path)
    logger.info("")
    return True


def check_token(auth_env: str = "HF_TOKEN") -> bool:
    """Verify HF token environment variable is set."""
    logger.info("=" * 60)
    logger.info("STEP 4: Checking Hugging Face authentication")
    logger.info("=" * 60)
    token = os.getenv(auth_env) or os.getenv("HF_TOKEN")
    if not token:
        logger.warning("⚠️  %s environment variable not set", auth_env)
        logger.warning("    Some models require license acceptance + HF token.")
        logger.info("    (Skipping failure since model might be local)")
        logger.info("")
        return True
    if not token.startswith("hf_"):
        logger.warning("⚠️  Token does not start with 'hf_'; double-check value.")
        logger.info("")
        return False
    logger.info("✅ Hugging Face token detected (env: %s)", auth_env or "HF_TOKEN")
    logger.info("")
    return True


def test_model_loading(base_path: Path, adapter_path: Path) -> bool:
    """Attempt a minimal load + generation cycle."""
    logger.info("=" * 60)
    logger.info("STEP 5: Testing model + adapter loading (INT4)")
    logger.info("=" * 60)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        logger.error("❌ Required packages missing for load test: %s", exc)
        return False

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(base_path),
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        if adapter_path:
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
            logger.info("   Attached LoRA adapter.")
        else:
            logger.info("   Using base model (no adapter).")
            
        model.eval()
        prompt = "Market momentum is bullish. Action:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.1,
                do_sample=False
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("✅ Inference OK. Sample response: %s", response[len(prompt):].strip())
        logger.info("")
        return True
    except Exception as exc:
        logger.error("❌ Model loading test failed: %s", exc)
        logger.info("")
        return False


def main() -> int:
    base_path, adapter_path, model_name = _load_paths()
    auth_env = "HF_TOKEN"

    checks = [
        ("Dependencies", check_dependencies),
        ("Base Model", lambda: check_base_model(base_path, model_name)),
        ("LoRA Adapter", lambda: check_adapter(adapter_path)),
        ("HF Token", lambda: check_token(auth_env)),
        ("Model Loading", lambda: test_model_loading(base_path, adapter_path))
    ]

    results = []
    for name, func in checks:
        passed = func()
        results.append((name, passed))

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info("%s: %s", status, name)
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\nAll checks passed. Ready for Phase 3 training.")
        return 0
    logger.error("\nSome checks failed. Resolve issues before launching Phase 3.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
