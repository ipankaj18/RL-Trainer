#!/usr/bin/env python3
"""
Comprehensive LLM Fix Test Suite

This test validates all the fixes implemented to address the empty response issue
where the LLM only generates token ID 128001 (beginning-of-sentence token).
"""

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
    """Setup detailed logging for test output."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/test_llm_fix.log', 'w')
        ]
    )
    # Suppress third-party logs
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

def load_config():
    """Load LLM configuration."""
    with open('config/llm_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_test_observation():
    """Create realistic test observation."""
    obs = np.zeros(261, dtype=np.float32)
    # Set realistic indicator values (de-normalized scale)
    obs[228] = 25.0  # ADX
    obs[229] = 5000.0  # VWAP
    obs[240] = 50.0  # RSI
    obs[232] = 0.0  # Momentum
    obs[0] = 5000.0  # Price
    return obs

def create_test_position_state():
    """Create test position state."""
    return {
        'trade_history': [],
        'position': 0,
        'balance': 50000.0
    }

def create_test_market_context():
    """Create test market context."""
    return {
        'market_name': 'ES',
        'current_price': 5000.0,
        'position_status': 'FLAT',
        'unrealized_pnl': 0.0,
        'win_rate': 0.5,
        'consecutive_losses': 0,
        'balance': 50000.0
    }

def test_basic_generation(llm):
    """Test basic LLM generation."""
    print("\n=== Testing Basic LLM Generation ===")
    
    obs = create_test_observation()
    position_state = create_test_position_state()
    market_context = create_test_market_context()
    
    try:
        action, confidence, reasoning, query_id = llm.query(
            observation=obs,
            position_state=position_state,
            market_context=market_context,
            available_actions=[0, 1, 2]
        )
        
        print(f"✅ Basic generation successful!")
        print(f"   Action: {action}")
        print(f"   Confidence: {confidence}")
        print(f"   Reasoning: {reasoning}")
        print(f"   Query ID: {query_id}")
        
        # Validate response format
        if reasoning and "Empty response fallback" not in reasoning:
            print("✅ Response format is valid")
            return True
        else:
            print("❌ Response format is invalid or empty")
            return False
            
    except Exception as e:
        print(f"❌ Basic generation failed: {e}")
        return False

def test_prompt_formatting(llm):
    """Test different prompt formatting scenarios."""
    print("\n=== Testing Prompt Formatting ===")
    
    # Test 1: Native chat template
    print("\n1. Testing native Llama-3 chat template...")
    try:
        prompt = llm._build_prompt(
            create_test_observation(),
            create_test_position_state(),
            create_test_market_context()
        )
        print(f"✅ Native template built successfully")
        print(f"   System prompt length: {len(prompt.get('system', ''))}")
        print(f"   User prompt length: {len(prompt.get('user', ''))}")
        
        # Check for assistant marker
        if 'assistant' in str(prompt):
            print("✅ Assistant marker found in prompt")
        else:
            print("⚠️  Assistant marker missing from prompt")
            
    except Exception as e:
        print(f"❌ Native template failed: {e}")
    
    # Test 2: Manual format fallback
    print("\n2. Testing manual Llama-3 format fallback...")
    try:
        manual_prompt = llm._manual_llama3_format(
            "You are a trading assistant.",
            "What is your recommendation?"
        )
        print(f"✅ Manual format built successfully")
        print(f"   Contains BOS token: {'<|begin_of_text|>' in manual_prompt}")
        print(f"   Contains system header: {'<|start_header_id|>system<|end_header_id|>' in manual_prompt}")
        print(f"   Contains user header: {'<|start_header_id|>user<|end_header_id|>' in manual_prompt}")
        print(f"   Contains assistant header: {'<|start_header_id|>assistant<|end_header_id|>' in manual_prompt}")
        
    except Exception as e:
        print(f"❌ Manual format failed: {e}")

def test_generation_parameters(llm):
    """Test generation parameter validation."""
    print("\n=== Testing Generation Parameters ===")
    
    # Test with different temperature values
    temperatures = [0.1, 0.3, 0.55, 0.8, 1.0]
    
    for temp in temperatures:
        print(f"\nTesting with temperature={temp}...")
        
        # Temporarily modify config
        original_temp = llm.config['llm_model']['temperature']
        llm.config['llm_model']['temperature'] = temp
        
        try:
            obs = create_test_observation()
            position_state = create_test_position_state()
            market_context = create_test_market_context()
            
            action, confidence, reasoning, query_id = llm.query(
                observation=obs,
                position_state=position_state,
                market_context=market_context,
                available_actions=[0, 1, 2]
            )
            
            if reasoning and "Empty response" not in reasoning and len(reasoning.strip()) > 0:
                print(f"✅ Temperature {temp} generated valid response")
            else:
                print(f"❌ Temperature {temp} failed or generated empty response")
                
        except Exception as e:
            print(f"❌ Temperature {temp} test failed: {e}")
        finally:
            # Restore original temperature
            llm.config['llm_model']['temperature'] = original_temp

def test_tokenizer_handling(llm):
    """Test tokenizer special token handling."""
    print("\n=== Testing Tokenizer Special Token Handling ===")
    
    try:
        # Check special tokens
        print(f"Pad token ID: {llm.tokenizer.pad_token_id}")
        print(f"EOS token ID: {llm.tokenizer.eos_token_id}")
        print(f"BOS token ID: {llm.tokenizer.bos_token_id}")
        
        # Test tokenization
        test_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>Test<|eot_id|><|start_header_id|>user<|end_header_id|>Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        tokens = llm.tokenizer(test_text, return_tensors="pt")
        
        print(f"✅ Tokenization successful")
        print(f"   Input tokens shape: {tokens['input_ids'].shape}")
        print(f"   Attention mask shape: {tokens['attention_mask'].shape}")
        
        # Test decoding
        decoded = llm.tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
        print(f"   Decoded (skip_special=True): '{decoded}'")
        
        decoded_with_special = llm.tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=False)
        print(f"   Decoded (skip_special=False): '{decoded_with_special}'")
        
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")

def test_error_handling(llm):
    """Test error handling and fallback mechanisms."""
    print("\n=== Testing Error Handling ===")
    
    # Test 1: Invalid observation
    print("\n1. Testing with invalid observation...")
    try:
        invalid_obs = np.array([-999.9] * 261)  # Invalid values
        action, confidence, reasoning, query_id = llm.query(
            observation=invalid_obs,
            position_state=create_test_position_state(),
            market_context=create_test_market_context(),
            available_actions=[0, 1, 2]
        )
        
        if action == 0 and confidence < 0.5:  # Should fallback safely
            print("✅ Invalid observation handled gracefully")
        else:
            print("⚠️  Invalid observation may not have been handled correctly")
            
    except Exception as e:
        print(f"❌ Invalid observation test failed: {e}")
    
    # Test 2: Empty market context
    print("\n2. Testing with empty market context...")
    try:
        empty_context = {}
        action, confidence, reasoning, query_id = llm.query(
            observation=create_test_observation(),
            position_state=create_test_position_state(),
            market_context=empty_context,
            available_actions=[0, 1, 2]
        )
        
        if reasoning and len(reasoning.strip()) > 0:
            print("✅ Empty context handled gracefully")
        else:
            print("⚠️  Empty context may not have been handled correctly")
            
    except Exception as e:
        print(f"❌ Empty context test failed: {e}")

def main():
    """Run comprehensive LLM fix validation."""
    print("🔧 Comprehensive LLM Fix Validation Suite")
    print("=" * 50)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize LLM with fixes enabled
        print("\n🚀 Initializing LLM with comprehensive fixes...")
        llm = LLMReasoningModule(config)
        
        if llm.mock_mode:
            print("⚠️  Running in MOCK mode - tests will be limited")
            return
        
        # Run all test suites
        test_results = []
        
        # Test 1: Basic generation
        test_results.append(test_basic_generation(llm))
        
        # Test 2: Prompt formatting
        test_results.append(test_prompt_formatting(llm))
        
        # Test 3: Generation parameters
        test_results.append(test_generation_parameters(llm))
        
        # Test 4: Tokenizer handling
        test_results.append(test_tokenizer_handling(llm))
        
        # Test 5: Error handling
        test_results.append(test_error_handling(llm))
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        
        print(f"Tests passed: {passed_tests}/{total_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! The comprehensive LLM fix appears to be working.")
        elif passed_tests >= total_tests * 0.8:
            print("✅ MOST TESTS PASSED! The comprehensive LLM fix is mostly working.")
        else:
            print("⚠️  SOME TESTS FAILED! The comprehensive LLM fix needs more work.")
        
        # Get LLM statistics
        stats = llm.get_stats()
        print(f"\n📈 LLM Statistics:")
        print(f"   Total queries: {stats['total_queries']}")
        print(f"   Error count: {stats['error_count']}")
        print(f"   Error rate: {stats['error_rate']:.1f}%")
        print(f"   Average latency: {stats['avg_latency_ms']:.1f}ms")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        print(f"❌ Test suite failed: {e}")

if __name__ == "__main__":
    main()