"""
LLM Reasoning Module

Purpose: LLM model loading, inference, and prompt management for trading decisions.

Uses: Phi-3-mini-4k-instruct (3.8B parameters, ~4GB VRAM with INT8 quantization)
Inference: ~15-20ms per query on RTX 3060

Features:
- INT8 quantization for reduced VRAM usage
- Selective querying to reduce latency
- Structured prompt templates
- Response parsing and validation
"""

import os
import logging
import time
import json
import re
import yaml
from pathlib import Path
from typing import Tuple, Dict, Optional
from collections import deque
import numpy as np
# Support running both as a package (src.llm_reasoning) and as a script import from project root.
try:
    from .llm_utils import ensure_model_downloaded
except ImportError:
    from llm_utils import ensure_model_downloaded

# Conditional imports - only load if LLM dependencies available
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # PHASE 2: LoRA imports for fine-tuning
    try:
        from peft import LoraConfig, get_peft_model, PeftModel
        LORA_AVAILABLE = True
    except ImportError:
        LORA_AVAILABLE = False
        logging.warning("PEFT not available. LoRA fine-tuning disabled. Install with: pip install peft")
    
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LORA_AVAILABLE = False
    logging.warning("LLM dependencies not available. Install with: pip install transformers torch accelerate bitsandbytes peft")


class LLMReasoningModule:
    """
    LLM inference engine for trading advice.
    
    Loads and manages Phi-3-mini model for generating trading recommendations
    based on market context and observations.
    """
    
    def __init__(self, config_path: str = "config/llm_config.yaml", mock_mode: bool = False, enable_fine_tuning: bool = True):
        """
        Initialize LLM reasoning module.

        Args:
            config_path: Path to LLM configuration file
            mock_mode: Use mock LLM for testing (no GPU required)
            enable_fine_tuning: Enable LoRA fine-tuning capability (PHASE 2)
        """
        self.logger = logging.getLogger(__name__)
        self.mock_mode = mock_mode or not LLM_AVAILABLE
        self.enable_fine_tuning = enable_fine_tuning and LORA_AVAILABLE and not mock_mode
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.logger.info(f"[LLM] Configuration loaded from {config_path}")
        except Exception as e:
            self.logger.error(f"[LLM] Failed to load config: {e}")
            self.config = self._get_default_config()
        
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # Statistics tracking
        self.total_queries = 0
        self.error_count = 0
        self.avg_latency_ms = 0.0
        self.last_error_time = 0
        self.error_cooldown_active = False
        
        # Cache for responses
        self.response_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # PHASE 2: Fine-tuning components
        if self.enable_fine_tuning:
            self.experience_buffer = LLMExperienceBuffer(max_size=10000)
            self.fine_tuning_steps = 0
            self.fine_tune_optimizer = None  # Optimizer created once and reused
            self.fine_tune_scheduler = None  # Learning rate scheduler
            self.logger.info("[LLM] LoRA fine-tuning enabled")
        else:
            self.experience_buffer = None
            self.fine_tuning_steps = 0
            self.fine_tune_optimizer = None
            self.fine_tune_scheduler = None
            if not self.mock_mode and not LORA_AVAILABLE:
                self.logger.info("[LLM] LoRA not available, fine-tuning disabled")

        if self.mock_mode:
            self.logger.info("[LLM] Running in MOCK mode (no GPU required)")
        else:
            self._load_model()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if config file not found."""
        return {
            'llm_model': {
                'name': 'microsoft/Phi-3-mini-4k-instruct',
                'quantization': 'int8',
                'device': 'auto',
                'max_new_tokens': 50,
                'temperature': 0.1,
                'top_p': 0.9
            },
            'fusion': {
                'llm_weight': 0.3,
                'confidence_threshold': 0.7,
                'use_selective_querying': True,
                'query_interval': 5
            },
            'prompts': {
                'system': 'You are a professional futures trader. Provide concise trading advice.',
                'context_template': 'Market: {market_name}\nPrice: {current_price:.2f}\nPosition: {position_status}'
            },
            'fallback': {
                'enable_rl_fallback': True,
                'max_llm_errors': 10
            }
        }
    
    def _load_model(self):
        """Load LLM model with quantization from Phi-3-mini-4k-instruct folder."""
        if not LLM_AVAILABLE:
            raise RuntimeError(
                "[LLM] PyTorch/Transformers dependencies not available. "
                "Install required packages: pip install torch transformers"
            )

        try:
            # Always use the manually downloaded LLM from Phi-3-mini-4k-instruct folder
            configured_path = self.config['llm_model'].get('local_path', 'Phi-3-mini-4k-instruct')
            local_path = Path(configured_path)

            # Universal project-root resolution (works in local, RunPod, any cwd)
            if not local_path.is_absolute():
                # Always resolve relative to project root (src/llm_reasoning.py -> src/ -> project/)
                project_root = Path(__file__).parent.parent
                local_path = project_root / configured_path

            # Auto-download if missing
            model_name = self.config['llm_model']['name']
            if not ensure_model_downloaded(model_name, local_path):
                raise FileNotFoundError(
                    f"[LLM] Model not found and download failed at: {local_path}\n"
                    f"Please ensure internet access or manually download {model_name} to:\n"
                    f"  {local_path.absolute()}"
                )

            model_name = str(local_path)
            self.logger.info(f"[LLM] Loading model from: {model_name}")

            quantization = self.config['llm_model']['quantization']
            
            self.logger.info(f"[LLM] Loading {model_name} with {quantization} quantization...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=False  # Use native Phi-3 implementation (transformers 4.56+)
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            load_kwargs = {
                'pretrained_model_name_or_path': model_name,
                'trust_remote_code': False,  # Use native Phi-3 implementation (transformers 4.56+)
                'torch_dtype': torch.float16,
                'device_map': self.config['llm_model']['device']
            }
            
            if quantization == 'int8':
                load_kwargs['load_in_8bit'] = True
                self.logger.info("[LLM] Using INT8 quantization (~4GB VRAM)")
            elif quantization == 'int4':
                load_kwargs['load_in_4bit'] = True
                self.logger.info("[LLM] Using INT4 quantization (~2GB VRAM)")
            
            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
            
            # Get device
            if hasattr(self.model, 'device'):
                self.device = self.model.device
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.logger.info(f"[LLM] Model loaded successfully on {self.device}")
            
            # PHASE 2: Setup LoRA adapters if fine-tuning enabled
            if self.enable_fine_tuning:
                self._setup_lora_adapters()
            
        except Exception as e:
            self.logger.error(f"[LLM] Failed to load model: {e}")
            raise RuntimeError(
                f"[LLM] Model initialization failed. Error: {e}\n"
                f"Ensure Phi-3-mini-4k-instruct is properly downloaded and all dependencies are installed."
            ) from e
    
    def _setup_lora_adapters(self, adapter_path: Optional[str] = None):
        """
        PHASE 2: Setup LoRA adapters for efficient fine-tuning.

        LoRA Configuration:
        - Target modules: ALL linear layers (attention + MLP)
        - Rank: 16 (tradeoff between capacity and efficiency)
        - Alpha: 32 (scaling factor)
        - Dropout: 0.1
        - Auto-loads existing adapters if available

        Args:
            adapter_path: Path to existing adapters (optional, auto-detected if None)
        """
        if not self.enable_fine_tuning or not LORA_AVAILABLE:
            return

        self.logger.info("[LLM] Setting up LoRA adapters for fine-tuning...")

        # Check for existing adapters
        if adapter_path is None:
            adapter_path = self._find_latest_lora_adapter()

        if adapter_path and os.path.exists(adapter_path):
            try:
                self.logger.info(f"[LLM] Loading existing LoRA adapters from {adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    adapter_path,
                    is_trainable=True  # Keep adapters trainable
                )
                self.logger.info("[LLM] ✅ LoRA adapters loaded successfully")

                # Count trainable parameters
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                all_params = sum(p.numel() for p in self.model.parameters())

                self.logger.info(f"[LLM] Loaded adapter stats:")
                self.logger.info(f"      Trainable params: {trainable_params:,} ({trainable_params/all_params*100:.2f}%)")
                self.logger.info(f"      Total params: {all_params:,}")
                return

            except Exception as e:
                self.logger.warning(f"[LLM] Failed to load adapters: {e}. Creating new ones...")

        # Create new adapters
        self.logger.info("[LLM] Creating new LoRA adapters...")

        lora_config = LoraConfig(
            r=16,  # Rank - controls adapter capacity
            lora_alpha=32,  # Scaling factor
            target_modules="all-linear",  # ALL linear layers (matches official Phi-3 sample)
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Wrap model with LoRA
        self.model = get_peft_model(self.model, lora_config)

        # Enable gradient tracking
        self.model.train()

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())

        self.logger.info(f"[LLM] LoRA adapters created:")
        self.logger.info(f"      Trainable params: {trainable_params:,} ({trainable_params/all_params*100:.2f}%)")
        self.logger.info(f"      Total params: {all_params:,}")
        self.logger.info(f"      Target: all-linear (attention + MLP layers)")
        self.logger.info("[LLM] ✅ LLM ready for fine-tuning")

    def _find_latest_lora_adapter(self) -> Optional[str]:
        """Find the most recent LoRA adapter checkpoint."""
        import glob
        from pathlib import Path

        # Search in models directory
        models_dir = Path("models")
        if not models_dir.exists():
            return None

        # Look for adapter directories
        adapter_patterns = [
            "models/*_lora",
            "models/phase3*lora*",
            "models/lora_adapters*"
        ]

        adapter_dirs = []
        for pattern in adapter_patterns:
            adapter_dirs.extend(glob.glob(pattern))

        if not adapter_dirs:
            return None

        # Find newest by modification time
        latest = max(adapter_dirs, key=lambda p: os.path.getmtime(p))

        # Verify it contains adapter files
        adapter_path = Path(latest)
        if (adapter_path / "adapter_config.json").exists():
            self.logger.info(f"[LLM] Found existing adapter: {adapter_path}")
            return str(adapter_path)

        return None
    
    def query(self, observation: np.ndarray, position_state: Dict, market_context: Dict) -> Tuple[int, float, str, Optional[int]]:
        """
        PHASE 2: Query LLM for trading advice with experience tracking.
        
        Args:
            observation: (261,) numpy array with enhanced features
            position_state: Dictionary with position information
            market_context: Dictionary with market state
            
        Returns:
            Tuple of (action, confidence, reasoning, query_id)
        """
        self.total_queries += 1
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(observation, position_state, market_context)
            if cache_key in self.response_cache:
                self.cache_hits += 1
                cached_response = self.response_cache[cache_key]
                # Decay confidence for cached responses
                cached_response['confidence'] *= self.config.get('fusion', {}).get('cache_decay_rate', 0.8)
                return cached_response['action'], cached_response['confidence'], cached_response['reasoning'], None
            
            self.cache_misses += 1
            
            # Build prompt
            prompt = self._build_prompt(observation, position_state, market_context)
            
            # Generate response
            response = self._generate_response(
                prompt,
                position_state=position_state,
                market_context=market_context
            )
            latency = (time.time() - start_time) * 1000
            
            # Update statistics
            self._update_latency_stats(latency)
            
            # Parse response
            action, confidence, reasoning = self._parse_response(response)
            
            # Cache the response
            self.response_cache[cache_key] = {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'timestamp': time.time()
            }
            
            # PHASE 2: Track for fine-tuning
            query_id = None
            if self.enable_fine_tuning:
                query_id = self._add_to_experience_buffer(
                    prompt=prompt,
                    response=response,
                    action=action,
                    observation=observation,
                    position_state=position_state,
                    market_context=market_context
                )
            
            # Clean old cache entries
            self._clean_cache()
            
            return action, confidence, reasoning, query_id
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"[LLM] Query failed: {e}")
            
            # Check error cooldown
            if self._should_enter_error_cooldown():
                self.error_cooldown_active = True
                self.last_error_time = time.time()
            
            # Fallback behavior
            fallback_action = self._get_fallback_action(position_state)
            return fallback_action, 0.0, f"LLM_ERROR: {str(e)}", None
    
    def _get_cache_key(self, observation: np.ndarray, position_state: Dict, market_context: Dict) -> str:
        """Generate cache key for LLM query."""
        # Use key features that identify the market state
        key_features = [
            int(observation[228] * 100),  # ADX slope
            int(observation[229] * 100),  # VWAP distance
            int(observation[240]),  # RSI
            position_state.get('position', 0),
            int(position_state.get('balance', 0) / 1000)  # Balance in thousands
        ]
        return str(key_features)
    
    def _clean_cache(self, max_age: float = 300.0):
        """Clean cache entries older than max_age seconds."""
        current_time = time.time()
        self.response_cache = {
            k: v for k, v in self.response_cache.items()
            if current_time - v['timestamp'] < max_age
        }
    
    def _should_enter_error_cooldown(self) -> bool:
        """Check if we should enter error cooldown period."""
        max_errors = self.config.get('fallback', {}).get('max_llm_errors', 10)
        return self.error_count >= max_errors
    
    def _get_fallback_action(self, position_state: Dict) -> int:
        """Get fallback action when LLM fails."""
        # Default to HOLD (0) for safety
        # Could be enhanced to use RL prediction or simple heuristics
        return 0
    
    def query_with_cot(self, observation: np.ndarray, position_state: Dict, market_context: Dict, available_actions: list) -> Tuple[int, float, str, Optional[int]]:
        """
        PHASE 3: Query LLM with chain-of-thought reasoning.
        
        Returns:
            action: int
            confidence: float
            reasoning: str (full reasoning chain as text)
            query_id: int (for outcome tracking)
        """
        if not hasattr(self, 'cot_reasoner'):
            try:
                from .chain_of_thought import ChainOfThoughtReasoner
            except ImportError:
                from chain_of_thought import ChainOfThoughtReasoner
            self.cot_reasoner = ChainOfThoughtReasoner(self)
        
        # Execute CoT
        action, confidence, reasoning_chain = self.cot_reasoner.reason(
            observation, position_state, market_context, available_actions
        )

        reasoning_text = self._format_reasoning_chain(reasoning_chain)

        # PHASE 2: Track for fine-tuning (use full reasoning chain text)
        query_id = None
        if self.enable_fine_tuning:
            query_id = self._add_to_experience_buffer(
                prompt=reasoning_text,
                response=reasoning_chain.get('step4_decision', reasoning_text),
                action=action,
                observation=observation,
                position_state=position_state,
                market_context=market_context
            )

        return action, confidence, reasoning_text, query_id
    
    def _update_latency_stats(self, latency: float):
        """Update average latency statistics."""
        if self.total_queries > 0:
            self.avg_latency_ms = (self.avg_latency_ms * (self.total_queries - 1) + latency) / self.total_queries
        else:
            self.avg_latency_ms = latency
    
    def _build_prompt(self, obs: np.ndarray, position_state: Dict, market_context: Dict) -> str:
        """Build LLM prompt from observation and context with interpreted indicators."""
        try:
            template = self.config['prompts']['context_template']

            # Extract key features from observation
            # obs[228:238] = extended market context
            # obs[238:246] = multi-timeframe indicators
            # obs[246:256] = pattern recognition
            # obs[256:261] = risk context

            # Safe extraction with defaults
            adx = float(obs[228]) if len(obs) > 228 else 25.0
            vwap_distance = float(obs[229]) if len(obs) > 229 else 0.0
            rsi = float(obs[240]) if len(obs) > 240 else 50.0
            momentum = float(obs[232]) if len(obs) > 232 else 0.0

            # === INTERPRET INDICATORS ===
            # Trend strength interpretation
            if adx > 30:
                trend_interpretation = "STRONG (trade with trend)"
            elif adx > 20:
                trend_interpretation = "MODERATE (trade cautiously)"
            else:
                trend_interpretation = "WEAK (prefer HOLD)"

            # Momentum interpretation
            if momentum > 0.5:
                momentum_interpretation = "BULLISH (upward pressure)"
            elif momentum < -0.5:
                momentum_interpretation = "BEARISH (downward pressure)"
            else:
                momentum_interpretation = "NEUTRAL (no clear direction)"

            # VWAP interpretation
            if vwap_distance > 0.01:  # > 1% above VWAP
                vwap_interpretation = "Above VWAP (bullish bias)"
            elif vwap_distance < -0.01:  # > 1% below VWAP
                vwap_interpretation = "Below VWAP (bearish bias)"
            else:
                vwap_interpretation = "Near VWAP (neutral)"

            # RSI interpretation
            if rsi > 70:
                rsi_interpretation = "OVERBOUGHT (avoid BUY)"
            elif rsi < 30:
                rsi_interpretation = "OVERSOLD (avoid SELL)"
            elif 40 <= rsi <= 60:
                rsi_interpretation = "Neutral (rely on trend)"
            else:
                rsi_interpretation = f"{'Bullish bias' if rsi > 60 else 'Bearish bias'}"

            # Risk warnings
            win_rate = market_context.get('win_rate', 0.5)
            consecutive_losses = market_context.get('consecutive_losses', 0)

            if win_rate < 0.4:
                win_rate_warning = "⚠️ LOW - Reduce confidence"
            else:
                win_rate_warning = ""

            if consecutive_losses >= 3:
                loss_streak_warning = "🛑 CRITICAL - HOLD ONLY"
            elif consecutive_losses >= 2:
                loss_streak_warning = "⚠️ WARNING - Be cautious"
            else:
                loss_streak_warning = ""

            # Format last trades
            last_trades = self._format_last_trades(position_state)

            # Build prompt with all interpretations
            prompt = template.format(
                market_name=market_context.get('market_name', 'NQ'),
                current_time=market_context.get('current_time', '10:30'),
                current_price=market_context.get('current_price', 5000.0),
                adx=adx,
                trend_interpretation=trend_interpretation,
                vwap_distance=vwap_distance,
                vwap_interpretation=vwap_interpretation,
                rsi=rsi,
                rsi_interpretation=rsi_interpretation,
                momentum=momentum,
                momentum_interpretation=momentum_interpretation,
                position_status=market_context.get('position_status', 'FLAT'),
                unrealized_pnl=market_context.get('unrealized_pnl', 0.0),
                win_rate=win_rate,
                win_rate_warning=win_rate_warning,
                consecutive_losses=consecutive_losses,
                loss_streak_warning=loss_streak_warning,
                balance=market_context.get('balance', 50000.0),
                last_3_trades=last_trades
            )

            # Add system prompt
            system_prompt = self.config['prompts']['system']
            strict_instruction = (
                "\nOUTPUT (one line only): ACTION | confidence | reason\n"
                "Allowed ACTION values: HOLD, BUY, SELL, MOVE_TO_BE, ENABLE_TRAIL, DISABLE_TRAIL.\n"
                "Confidence must be a decimal between 0 and 1 (e.g., 0.72).\n"
                "Example: HOLD | 0.20 | Risk protection after 3 losses\n"
                "BEGIN YOUR ANSWER IMMEDIATELY AFTER THIS INSTRUCTION WITH THE ACTION WORD.\n"
                "Do NOT include markdown, headings, or extra text."
            )
            full_prompt = f"{system_prompt}\n\n{prompt}{strict_instruction}"

            return full_prompt

        except Exception as e:
            self.logger.error(f"[LLM] Error building prompt: {e}")
            # Return a minimal prompt as fallback
            return f"Market context unavailable. What action do you recommend?"
    

    def _generate_response(self,
                            prompt: str,
                            *,
                            position_state: Optional[Dict] = None,
                            market_context: Optional[Dict] = None,
                            max_new_tokens: Optional[int] = None) -> str:
        """Generate response using the LLM model."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "[LLM] Model not loaded. Cannot generate response. "
                "Ensure model initialization succeeded."
            )
        raw_response = self._generate_raw(prompt, max_new_tokens=max_new_tokens)
        if '|' not in (raw_response or ''):
            preview = (raw_response or "").strip().splitlines()
            preview = preview[0] if preview else ""
            self.logger.warning(
                f"[LLM] Non-compliant response detected, applying fallback: '{preview}'"
            )
            return "HOLD | 0.0 | FORMAT_FALLBACK"
        return raw_response

    def _generate_raw(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate LLM response from the model using Phi-3 chat template."""
        try:
            # CRITICAL FIX: Use Phi-3 chat template for proper instruction following
            # Phi-3 requires messages in <|system|>...<|end|><|user|>...<|end|><|assistant|> format

            # Split prompt into system and user parts
            # The prompt is built as: system_prompt + context
            if '\n\n' in prompt:
                parts = prompt.split('\n\n', 1)
                system_content = parts[0]
                user_content = parts[1] if len(parts) > 1 else parts[0]
            else:
                # Fallback: use default system message
                system_content = "You are an expert futures trader. Provide trading advice in format: ACTION | confidence | reason"
                user_content = prompt

            # Format as chat messages
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]

            # Apply Phi-3 chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # Adds <|assistant|> tag
            )

            # Tokenize the formatted prompt
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            generation_kwargs = {
                'max_new_tokens': max_new_tokens or self.config['llm_model']['max_new_tokens'],
                'temperature': self.config['llm_model']['temperature'],
                'top_p': self.config['llm_model']['top_p'],
                'do_sample': self.config['llm_model'].get('do_sample', True),
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'use_cache': True  # Enable KV cache for faster inference
            }

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )

            # Decode only the generated tokens (skip input prompt)
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],  # Only decode new tokens
                skip_special_tokens=True
            ).strip()

            return response

        except Exception as e:
            self.logger.error(f"[LLM] Generation error: {e}")
            raise
    
    
    def _parse_response(self, response: str) -> Tuple[int, float, str]:
        """
        Parse LLM response: "BUY | 0.85 | Strong uptrend"
        
        Returns:
            Tuple of (action, confidence, reasoning)
        """
        try:
            response = (response or "").strip()
            if not response:
                raise ValueError("Empty response")

            parsed = self._parse_pipe_format(response)
            if parsed:
                return parsed

            parsed = self._parse_json_like_response(response)
            if parsed:
                return parsed

            parsed = self._parse_regex_response(response)
            if parsed:
                return parsed

            raise ValueError(f"Invalid format: {response}")

        except Exception as e:
            self.logger.warning(f"[LLM] Parse error: {e}, response: '{response}'")
            return 0, 0.0, f"PARSE_ERROR: {response}"

    def _parse_pipe_format(self, response: str) -> Optional[Tuple[int, float, str]]:
        parts = [p.strip() for p in response.split('|')]
        if len(parts) < 3:
            return None
        return self._coerce_parsed_values(parts[0], parts[1], parts[2])

    def _parse_json_like_response(self, response: str) -> Optional[Tuple[int, float, str]]:
        try:
            json_start = response.find('{')
            json_end = response.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                snippet = response[json_start:json_end + 1]
                data = json.loads(snippet)
                action = data.get('action')
                confidence = data.get('confidence', data.get('confidence_score'))
                reasoning = data.get('reasoning') or data.get('explanation') or ''
                if action is None or confidence is None:
                    return None
                return self._coerce_parsed_values(str(action), str(confidence), reasoning)
        except Exception:
            return None
        return None

    def _parse_regex_response(self, response: str) -> Optional[Tuple[int, float, str]]:
        action_match = re.search(
            r"\b(HOLD|BUY|SELL|MOVE_TO_BE|MOVE_SL_TO_BE|ENABLE_TRAIL|DISABLE_TRAIL)\b",
            response,
            re.IGNORECASE,
        )
        conf_match = re.search(r"(\d+(?:\.\d+)?|\.\d+)", response)
        if not action_match or not conf_match:
            return None
        action_str = action_match.group(1).upper()
        confidence = conf_match.group(1)
        reasoning_start = action_match.end()
        reasoning = response[reasoning_start:].strip()
        return self._coerce_parsed_values(action_str, confidence, reasoning)

    def _coerce_parsed_values(
        self, action_str: str, confidence_value: str, reasoning: str
    ) -> Tuple[int, float, str]:
        action_map = {
            'HOLD': 0,
            'BUY': 1,
            'SELL': 2,
            'MOVE_TO_BE': 3,
            'MOVE_SL_TO_BE': 3,
            'ENABLE_TRAIL': 4,
            'DISABLE_TRAIL': 5,
        }
        action = action_map.get(action_str.strip().upper(), 0)
        try:
            confidence = float(confidence_value)
        except ValueError:
            self.logger.warning(f"[LLM] Invalid confidence format: {confidence_value}")
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        clean_reasoning = (reasoning or "").strip() or "LLM provided no reasoning"
        return action, confidence, clean_reasoning
    
    def _format_last_trades(self, position_state: Dict) -> str:
        """Format last 3 trades for prompt."""
        try:
            history = position_state.get('trade_history', [])
            if not history:
                return "No trades yet"
            
            # Get last 3 trades
            last_3 = history[-3:]
            formatted = []
            
            for trade in last_3:
                if isinstance(trade, dict):
                    pnl = trade.get('pnl', 0)
                    result = "WIN" if pnl > 0 else "LOSS"
                    formatted.append(f"{result} (${pnl:.0f})")
                else:
                    # Handle simple list of P&L values
                    result = "WIN" if trade > 0 else "LOSS"
                    formatted.append(f"{result} (${trade:.0f})")
            
            return ", ".join(formatted)
        
        except Exception as e:
            self.logger.error(f"[LLM] Error formatting trades: {e}")
            return "Error formatting trades"
    
    def get_stats(self) -> Dict:
        """Get LLM usage statistics."""
        total = self.total_queries
        if total == 0:
            return {
                'total_queries': 0,
                'error_count': self.error_count,
                'error_rate': 0.0,
                'avg_latency_ms': 0.0,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': 0.0
            }
        
        return {
            'total_queries': total,
            'error_count': self.error_count,
            'error_rate': self.error_count / total * 100,
            'avg_latency_ms': self.avg_latency_ms,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / total * 100
        }
    
    def clear_cache(self):
        """Clear response cache."""
        self.response_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    # PHASE 2: Fine-tuning methods
    
    def _format_reasoning_chain(self, reasoning_chain: Dict[str, str]) -> str:
        """Format reasoning chain into a multi-line string for logging/storage."""
        return (
            f"ANALYSIS: {reasoning_chain.get('step1_analysis', '').strip()}\n\n"
            f"OPTIONS: {reasoning_chain.get('step2_options', '').strip()}\n\n"
            f"RISKS: {reasoning_chain.get('step3_risks', '').strip()}\n\n"
            f"DECISION: {reasoning_chain.get('step4_decision', '').strip()}"
        )

    def _add_to_experience_buffer(self,
                                  prompt,
                                  response,
                                  action,
                                  observation,
                                  position_state,
                                  market_context=None):
        """
        PHASE 2: Add query to experience buffer for later fine-tuning.
        
        Returns:
            query_id: int - unique ID for this query
        """
        if not self.enable_fine_tuning or self.experience_buffer is None:
            return None
        
        query_id = len(self.experience_buffer)

        prompt_text = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
        response_text = response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)

        experience = {
            'id': query_id,
            'prompt': prompt_text,
            'response': response_text,
            'action': int(action),
            'observation': observation.copy(),
            'position_state': position_state.copy(),
            'market_context': (market_context.copy() if isinstance(market_context, dict) else {}),
            'timestamp': time.time(),
            'outcome': None  # Will be filled later
        }

        self.experience_buffer.add(experience)
        
        return query_id
    
    def update_outcome(self, query_id, reward, final_pnl):
        """
        PHASE 2: Update query outcome after trade completes.
        
        Args:
            query_id: int - ID from query()
            reward: float - immediate reward
            final_pnl: float - final P&L of trade
        """
        if (query_id is not None and 
            self.enable_fine_tuning and 
            self.experience_buffer is not None and
            query_id < len(self.experience_buffer)):
            
            self.experience_buffer.buffer[query_id]['outcome'] = {
                'reward': reward,
                'pnl': final_pnl,
                'success': (final_pnl > 0)
            }
    
    def fine_tune_step(self, batch_size=8, learning_rate=5e-5):
        """
        PHASE 2: Single fine-tuning step on successful trading outcomes.

        Strategy:
        1. Sample experiences where outcome is known
        2. Weight by success (winning trades emphasized)
        3. Fine-tune LLM to reproduce successful reasoning
        4. Use gradient accumulation for stability
        5. Persistent optimizer with learning rate scheduling

        Returns:
            loss: float - training loss
            accuracy: float - how often LLM reproduces successful action
        """
        if not self.enable_fine_tuning or self.experience_buffer is None:
            return None, None

        if len(self.experience_buffer) < batch_size:
            return None, None

        # Sample batch (weighted by outcome)
        batch = self.experience_buffer.sample_weighted(batch_size)

        if batch is None:
            return None, None

        # Create optimizer once and reuse (CRITICAL FIX)
        if self.fine_tune_optimizer is None:
            self.fine_tune_optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
            # Add cosine annealing scheduler
            self.fine_tune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.fine_tune_optimizer,
                T_max=1000,
                eta_min=learning_rate * 0.1
            )
            self.logger.info(f"[LLM] Optimizer created: AdamW(lr={learning_rate}, weight_decay=0.01)")

        # Prepare for training
        self.model.train()

        total_loss = 0.0
        correct = 0

        # Zero gradients before loop
        self.fine_tune_optimizer.zero_grad()

        for exp in batch:
            outcome = exp.get('outcome')
            if not outcome:
                continue

            inputs = self.tokenizer(
                exp['prompt'],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            targets = self.tokenizer(
                exp['response'],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)

            # Forward pass
            outputs = self.model(**inputs, labels=targets.input_ids)
            loss = outputs.loss

            # Weight loss by outcome (improved weighting)
            pnl = outcome.get('pnl', 0.0)
            weight = max(1.0 + pnl / 100.0, 0.1)  # Positive weight for winners
            weighted_loss = loss * weight / batch_size  # Normalize by batch size

            # Backward (accumulate gradients)
            weighted_loss.backward()

            total_loss += loss.item()

            # FIXED: Validation logic (generate correctly)
            with torch.no_grad():
                # Generate prediction
                gen_inputs = self.tokenizer(
                    exp['prompt'],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                gen_outputs = self.model.generate(
                    **gen_inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=False,  # Greedy for validation
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

                predicted_response = self.tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
                # Remove prompt from response
                if exp['prompt'] in predicted_response:
                    predicted_response = predicted_response[len(exp['prompt']):].strip()

                predicted_action, _, _ = self._parse_response(predicted_response)
                if predicted_action == exp['action']:
                    correct += 1

        # Update weights once
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.fine_tune_optimizer.step()
        self.fine_tune_scheduler.step()

        self.fine_tuning_steps += 1

        avg_loss = total_loss / batch_size
        accuracy = correct / batch_size
        current_lr = self.fine_tune_scheduler.get_last_lr()[0]

        if self.fine_tuning_steps % 10 == 0:
            self.logger.info(
                f"[LLM] Fine-tune step {self.fine_tuning_steps}: "
                f"loss={avg_loss:.4f}, acc={accuracy:.1%}, lr={current_lr:.2e}"
            )

        return avg_loss, accuracy
    
    def save_lora_adapters(self, path: Optional[str] = None):
        """
        Save LoRA adapters with automatic versioning and metadata.

        Args:
            path: Optional custom path (auto-generated with timestamp if None)
        """
        if not self.enable_fine_tuning or self.model is None:
            return

        if path is None:
            # Auto-generate path with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"models/lora_adapters_step{self.fine_tuning_steps}_{timestamp}"

        # Ensure models directory exists
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save adapter weights
        self.model.save_pretrained(path)

        # Save metadata
        metadata = {
            'fine_tuning_steps': self.fine_tuning_steps,
            'total_queries': self.total_queries,
            'experience_buffer_size': len(self.experience_buffer) if self.experience_buffer else 0,
            'timestamp': datetime.datetime.now().isoformat(),
            'lora_config': {
                'r': 16,
                'lora_alpha': 32,
                'target_modules': 'all-linear',
                'lora_dropout': 0.1
            },
            'statistics': self.get_stats()
        }

        import json
        with open(Path(path) / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"[LLM] LoRA adapters saved to {path}")
        self.logger.info(f"      Steps: {self.fine_tuning_steps}, Queries: {self.total_queries}")
        self.logger.info(f"      Buffer size: {metadata['experience_buffer_size']}")
    
    def load_lora_adapters(self, path):
        """Load fine-tuned LoRA adapters."""
        if self.enable_fine_tuning and self.model is not None and os.path.exists(path):
            self.model = PeftModel.from_pretrained(self.model, path)
            self.logger.info(f"[LLM] LoRA adapters loaded from {path}")


class LLMExperienceBuffer:
    """
    PHASE 2: Buffer for LLM query-outcome pairs.
    
    Used to fine-tune LLM on successful trading outcomes.
    """
    
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        """Add experience dict."""
        self.buffer.append(experience)
    
    def sample_weighted(self, batch_size):
        """
        PHASE 2: Sample batch weighted by outcome quality.

        Improved Strategy:
        - Only sample experiences with known outcomes
        - Weight by P&L quality (normalized and clipped)
        - Factor in reward/risk ratio (Sharpe-like metric)
        - Learn from both successes AND mistakes
        - Allow duplicates for very high-quality experiences

        Returns:
            List of experiences or None if insufficient data
        """
        # Filter to experiences with outcomes
        completed = [exp for exp in self.buffer if exp.get('outcome') is not None]

        if len(completed) < batch_size:
            return None

        # Compute improved sampling weights
        weights = []
        for exp in completed:
            pnl = exp['outcome']['pnl']
            reward = exp['outcome'].get('reward', 0.0)

            # Base weight from P&L (normalized and clipped)
            pnl_normalized = np.clip(pnl / 100.0, -3.0, 5.0)

            # Reward quality factor (Sharpe-like: reward per unit risk)
            if abs(pnl) > 1e-6:
                quality = reward / abs(pnl)
            else:
                quality = 0.0

            # Final weight calculation
            if pnl > 0:
                # Winning trades: higher weight for better quality
                weight = 1.0 + pnl_normalized + 0.5 * quality
            else:
                # Losing trades: learn from big mistakes
                weight = 0.2 + abs(pnl_normalized) * 0.3

            weights.append(max(weight, 0.1))  # Floor at 0.1

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Sample with replacement (allows duplicates for very good experiences)
        indices = np.random.choice(len(completed), size=batch_size, replace=True, p=weights)
        batch = [completed[i] for i in indices]

        return batch
    
    def __len__(self):
        return len(self.buffer)


