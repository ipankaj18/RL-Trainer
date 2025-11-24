#!/usr/bin/env python3
"""
Quick test script to verify LLM loading fix.

Tests that Phi-3 loads without 'seen_tokens' errors.
"""

import sys
import logging
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

def test_llm_loading():
    """Test LLM loading with fixed trust_remote_code."""
    print("=" * 70)
    print("Testing LLM Loading Fix")
    print("=" * 70)

    try:
        # Import LLM module
        from src.llm_reasoning import LLMReasoningModule

        print("\n[1/4] Importing LLM module... ✅")

        # Load LLM
        print("\n[2/4] Loading Phi-3 model (this may take 30-60 seconds)...")
        llm = LLMReasoningModule(
            config_path='config/llm_config.yaml',
            mock_mode=False,
            enable_fine_tuning=False  # Disable for quick test
        )

        print("[2/4] Model loaded successfully! ✅")

        # Test query
        print("\n[3/4] Testing LLM query (generation)...")

        # Create dummy observation (261D)
        observation = np.random.randn(261).astype(np.float32)
        position_state = {
            'position': 0,
            'balance': 50000.0,
            'unrealized_pnl': 0.0,
            'trade_history': []
        }
        market_context = {
            'market_name': 'NQ',
            'current_time': '10:30',
            'current_price': 20150.0,
            'position_status': 'FLAT',
            'unrealized_pnl': 0.0,
            'win_rate': 0.5,
            'consecutive_losses': 0,
            'balance': 50000.0
        }

        # Query LLM
        action, confidence, reasoning, query_id = llm.query(
            observation=observation,
            position_state=position_state,
            market_context=market_context
        )

        print(f"[3/4] Query successful! ✅")
        print(f"      Action: {action}")
        print(f"      Confidence: {confidence:.2f}")
        print(f"      Reasoning: {reasoning[:80]}...")

        # Get stats
        print("\n[4/4] Checking LLM statistics...")
        stats = llm.get_stats()
        print(f"      Total queries: {stats['total_queries']}")
        print(f"      Error count: {stats['error_count']}")
        print(f"      Avg latency: {stats['avg_latency_ms']:.1f}ms")

        if stats['error_count'] == 0:
            print("[4/4] No errors! ✅")
        else:
            print(f"[4/4] ⚠️ {stats['error_count']} errors detected")
            return False

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED - LLM Fix Verified!")
        print("=" * 70)
        print("\nYou can now run Phase 3 training:")
        print("  python src/train_phase3_llm.py --test")
        print()

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_llm_loading()
    sys.exit(0 if success else 1)
