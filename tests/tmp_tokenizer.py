from transformers import AutoTokenizer
from pathlib import Path
path = str(Path('fingpt-mt_llama3-8b_lora/Base_Model').resolve())
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
print('chat template (first 500 chars):')
print(tokenizer.chat_template[:500])
print('special tokens map:', tokenizer.special_tokens_map)
print('bos token:', tokenizer.bos_token)
print('eos token:', tokenizer.eos_token)
print('start_header_id id:', tokenizer.convert_tokens_to_ids('<|start_header_id|>'))
