import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_path = "Base_Model"

try:
    logger.info(f"Loading tokenizer from {base_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    logger.info("Tokenizer loaded.")

    logger.info(f"Loading model from {base_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    logger.info("Model loaded successfully.")
    
    logger.info("Testing generation...")
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
    logger.info(f"Generation successful: {response}")

except Exception as e:
    logger.error(f"Failed to load model: {e}")
    import traceback
    traceback.print_exc()
