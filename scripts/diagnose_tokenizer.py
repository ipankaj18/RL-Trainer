import sys
import os
import yaml
from pathlib import Path
from transformers import AutoTokenizer

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

def load_config():
    with open('config/llm_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    print("Diagnosing Tokenizer Configuration...")
    
    try:
        config = load_config()
        llm_cfg = config.get('llm_model', {})
        
        # Resolve path similar to LLMReasoningModule
        local_path = llm_cfg.get('local_path', 'Base_Model')
        if not os.path.isabs(local_path):
            local_path = os.path.join(os.getcwd(), local_path)
            
        print(f"Loading tokenizer from: {local_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            trust_remote_code=True
        )
        
        print(f"\nTokenizer loaded: {tokenizer.__class__.__name__}")
        print(f"Chat template attribute present: {hasattr(tokenizer, 'chat_template')}")
        print(f"Chat template value: {tokenizer.chat_template!r}")
        
        # Test formatting
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        print("\nTesting apply_chat_template...")
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                print("\n--- Formatted Output Start ---")
                print(formatted)
                print("--- Formatted Output End ---")
                
                # Check for special tokens
                if "<|begin_of_text|>" in formatted:
                    print("\n✅ Llama-3 special tokens detected.")
                else:
                    print("\n❌ Llama-3 special tokens MISSING.")
            except Exception as e:
                print(f"\n❌ Error applying chat template: {e}")
        else:
            print("\n❌ Tokenizer does not support apply_chat_template")
            
    except Exception as e:
        print(f"\n❌ Diagnosis failed: {e}")

if __name__ == "__main__":
    main()