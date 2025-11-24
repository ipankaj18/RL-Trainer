from pathlib import Path
import yaml
from transformers import AutoTokenizer

config = yaml.safe_load(Path('config/llm_config.yaml').read_text())
context_template = config['prompts']['context_template']
system_prompt = config['prompts']['system']
strict_instruction = (
    "\nOUTPUT (one line only): ACTION | confidence | reason\n"
    "Allowed ACTION values: HOLD, BUY, SELL, MOVE_TO_BE, ENABLE_TRAIL, DISABLE_TRAIL.\n"
    "Confidence must be a decimal between 0 and 1 (e.g., 0.72).\n"
    "Example: HOLD | 0.20 | Risk protection after 3 losses\n"
    "BEGIN YOUR ANSWER IMMEDIATELY AFTER THIS INSTRUCTION WITH THE ACTION WORD.\n"
    "Do NOT include markdown, headings, or extra text."
)
context = context_template.format(
    market_name='NQ',
    current_time='15:45 ET',
    current_price=11800.54,
    adx=32.5,
    trend_interpretation='Strong uptrend',
    vwap_distance=0.013,
    vwap_interpretation='Above VWAP',
    rsi=58.3,
    rsi_interpretation='Neutral',
    momentum=75.2,
    momentum_interpretation='Bullish',
    position_status='FLAT',
    unrealized_pnl=120.0,
    win_rate=0.52,
    win_rate_warning='',
    consecutive_losses=1,
    loss_streak_warning='',
    balance=105000.0,
    last_3_trades='BUY +$90; SELL +$45; HOLD'
)
user_prompt = f"{context}{strict_instruction}"
chat_template = Path('llm_templates/meta_llama3_chat_template.txt').read_text()
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B', trust_remote_code=True)
tokenizer.chat_template = chat_template
messages = [
    {'role': 'system', 'content': system_prompt.strip()},
    {'role': 'user', 'content': user_prompt.strip()},
]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print('prompt length chars', len(formatted_prompt))
inputs = tokenizer(formatted_prompt, return_tensors='pt', padding=True, truncation=True, max_length=2048)
print('max_length=2048 token count', inputs['input_ids'].shape[-1])
inputs4096 = tokenizer(formatted_prompt, return_tensors='pt', padding=True, truncation=True, max_length=4096)
print('max_length=4096 token count', inputs4096['input_ids'].shape[-1])
