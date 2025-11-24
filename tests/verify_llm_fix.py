import sys
import os
import logging
import yaml
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from llm_reasoning import LLMReasoningModule

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,  # Changed to DEBUG for detailed output
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Suppress third-party logs
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

def load_config():
    with open('config/llm_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def check_model_files(base_path='Base_Model'):
    """Check if essential model files exist."""
    logger = logging.getLogger(__name__)
    required_files = [
        'config.json',
        'modeling_phi3.py',
        'tokenizer_config.json'
    ]
    
    path = Path(base_path)
    if not path.exists():
        logger.error(f"❌ Model directory not found: {base_path}")
        return False
        
    missing = []
    for f in required_files:
        if not (path / f).exists():
            missing.append(f)
            
    # Check for weights
    has_weights = list(path.glob("*.safetensors")) or list(path.glob("*.bin"))
    if not has_weights:
        missing.append("model weights (.safetensors or .bin)")
        
    if missing:
        logger.error(f"❌ Missing model files in {base_path}: {', '.join(missing)}")
        return False
        
    logger.info(f"✅ All essential model files found in {base_path}")
    return True

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting LLM verification...")
    
    # Check files first
    if not check_model_files():
        logger.error("Aborting verification due to missing files.")
        return

    try:
        config = load_config()
        
        # Initialize LLM module
        llm = LLMReasoningModule(config)
        
        # Test query with proper numpy array and realistic market context
        observation = np.zeros(261, dtype=np.float32)
        # Set normalized values (z-scores) so denormalizer produces realistic physical values
        # Mean values (0.0) correspond to: ADX=25, RSI=50, Price=5000, etc.
        observation[228] = 0.0  # ADX (Mean=25)
        observation[229] = 0.0  # VWAP (Mean=5000)
        observation[240] = 0.0  # RSI (Mean=50)
        observation[232] = 1.0  # Momentum (Mean=0, Std=0.5 -> 0.5) -> Bullish
        observation[0] = 0.0    # Price (Mean=5000)
        
        position_state = {
            'trade_history': [],
            'position': 0,
            'balance': 50000.0
        }
        market_context = {
            'market_name': 'NQ',
            'current_price': 5000.0,
            'position_status': 'FLAT',
            'unrealized_pnl': 0.0,
            'win_rate': 0.5,
            'consecutive_losses': 0,
            'balance': 50000.0
        }
        
        logger.info("Sending test query to LLM...")
        # FIXED: Unpack 4 values (action, confidence, reasoning, query_id)
        action, confidence, reasoning, query_id = llm.query(
            observation=observation,
            position_state=position_state,
            market_context=market_context,
            available_actions=[0, 1, 2]
        )
        
        logger.info(f"Response received:")
        logger.info(f"Action: {action}")
        logger.info(f"Confidence: {confidence}")
        logger.info(f"Reasoning: {reasoning}")
        logger.info(f"Query ID: {query_id}")
        
        if reasoning == "Empty response fallback":
            logger.warning("⚠️  Fallback triggered! The model returned an empty response.")
        elif not reasoning:
             logger.error("❌ Reasoning is empty!")
        else:
            logger.info("✅ Valid response received.")
            
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
