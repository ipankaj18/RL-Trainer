import sys
import os
import yaml
from transformers import AutoTokenizer

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

def load_config():
    with open('config/llm_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    print("Checking Tokenizer Special Tokens...")
    
    try:
        config = load_config()
        llm_cfg = config.get('llm_model', {})
        local_path = llm_cfg.get('local_path', 'Base_Model')
        if not os.path.isabs(local_path):
            local_path = os.path.join(os.getcwd(), local_path)
            
        print(f"Loading tokenizer from: {local_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            trust_remote_code=True
        )
        
        print(f"\nTokenizer class: {tokenizer.__class__.__name__}")
        print(f"Vocab size: {tokenizer.vocab_size}")
        
        special_tokens = tokenizer.all_special_tokens
        print(f"\nAll special tokens ({len(special_tokens)}):")
        print(special_tokens)
        
        required_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>"
        ]
        
        print("\nVerifying Llama-3 tokens:")
        missing = []
        for token in required_tokens:
            if token in special_tokens:
                print(f"✅ Found: {token} (ID: {tokenizer.convert_tokens_to_ids(token)})")
            else:
                print(f"❌ MISSING: {token}")
                missing.append(token)
                
        # Test tokenization of a special token string
        test_str = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        print(f"\nTest tokenization of: '{test_str}'")
        ids = tokenizer.encode(test_str, add_special_tokens=False)
        print(f"IDs: {ids}")
        
        decoded = tokenizer.decode(ids)
        print(f"Decoded: '{decoded}'")
        
        if len(ids) > 5: # Should be roughly 3-4 tokens if special tokens work
            print("\n⚠️  WARNING: Token count is high. Special tokens might be tokenized as text!")
        else:
            print("\n✅ Token count looks correct.")

    except Exception as e:
        print(f"\n❌ Check failed: {e}")

if __name__ == "__main__":
    main()