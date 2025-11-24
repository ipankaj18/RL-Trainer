"""
LLM Utilities
"""
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def ensure_model_downloaded(model_name: str, local_path: Path) -> bool:
    """
    Ensure the LLM model is present at local_path.
    If not, download it from Hugging Face.
    
    Args:
        model_name: Hugging Face model ID (e.g., "microsoft/Phi-3-mini-4k-instruct")
        local_path: Local path to store the model
        
    Returns:
        True if model is present (or successfully downloaded), False otherwise.
    """
    # Check for critical files
    required_files = ["config.json", "tokenizer.model", "tokenizer_config.json"]
    
    # If path exists, check contents
    if local_path.exists():
        missing = [f for f in required_files if not (local_path / f).exists()]
        
        # Also check for weights (safetensors OR bin)
        has_weights = list(local_path.glob("*.safetensors")) or list(local_path.glob("*.bin")) or list(local_path.glob("*.index.json"))
        
        if not missing and has_weights:
            logger.info(f"[LLM] ✅ Model found at {local_path}")
            return True
        
        if missing:
            logger.warning(f"[LLM] ⚠️  Missing model files at {local_path}: {missing}")
        if not has_weights:
            logger.warning(f"[LLM] ⚠️  No model weights found at {local_path}")
            
    logger.info(f"[LLM] ⬇️  Model not found or incomplete at {local_path}. Downloading {model_name}...")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Ensure directory exists
        local_path.mkdir(parents=True, exist_ok=True)
        
        snapshot_download(
            repo_id=model_name,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        logger.info(f"[LLM] ✅ Download completed to {local_path}")
        return True
        
    except ImportError:
        logger.error("[LLM] ❌ huggingface_hub not installed. Cannot download model.")
        logger.error("Run: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"[LLM] ❌ Download failed: {e}")
        return False
