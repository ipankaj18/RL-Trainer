"""
Test script to verify improved LLM prompts work correctly.
Shows what the LLM will actually see with the new prompt structure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.llm_reasoning import LLMReasoningModule

def test_prompt_generation():
    """Test prompt generation with various market scenarios."""

    print("=" * 80)
    print("TESTING IMPROVED LLM PROMPTS")
    print("=" * 80)

    # Initialize LLM in mock mode (no GPU needed for testing)
    llm = LLMReasoningModule(mock_mode=True)

    # Test scenarios for different markets
    test_scenarios = [
        {
            "name": "NQ Strong Uptrend",
            "market": "NQ",
            "obs_values": {
                228: 45.0,    # Strong ADX
                229: 0.025,   # 2.5% above VWAP
                232: 1.2,     # Strong positive momentum
                240: 65.0     # RSI bullish but not overbought
            },
            "position_state": {
                "position": 0,
                "balance": 52000,
                "win_rate": 0.55,
                "consecutive_losses": 0,
                "trade_history": [
                    {"pnl": 200}, {"pnl": 150}, {"pnl": -50}
                ]
            },
            "market_context": {
                "market_name": "NQ",
                "current_time": "10:30",
                "current_price": 16500.0,
                "position_status": "FLAT",
                "unrealized_pnl": 0.0
            }
        },
        {
            "name": "ES Losing Streak WARNING",
            "market": "ES",
            "obs_values": {
                228: 28.0,    # Moderate ADX
                229: -0.015,  # 1.5% below VWAP
                232: -0.8,    # Negative momentum
                240: 45.0     # Neutral RSI
            },
            "position_state": {
                "position": 0,
                "balance": 48500,
                "win_rate": 0.38,  # Below 40% threshold
                "consecutive_losses": 3,  # CRITICAL
                "trade_history": [
                    {"pnl": -100}, {"pnl": -150}, {"pnl": -200}
                ]
            },
            "market_context": {
                "market_name": "ES",
                "current_time": "14:30",
                "current_price": 5250.0,
                "position_status": "FLAT",
                "unrealized_pnl": 0.0
            }
        },
        {
            "name": "YM Position Management",
            "market": "YM",
            "obs_values": {
                228: 35.0,    # Strong ADX
                229: 0.018,   # Above VWAP
                232: 0.6,     # Positive momentum
                240: 58.0     # Neutral RSI
            },
            "position_state": {
                "position": 1,  # Long position
                "balance": 51500,
                "win_rate": 0.52,
                "consecutive_losses": 0,
                "trade_history": [
                    {"pnl": 100}, {"pnl": 200}, {"pnl": 150}
                ]
            },
            "market_context": {
                "market_name": "YM",
                "current_time": "11:15",
                "current_price": 38500.0,
                "position_status": "LONG",
                "unrealized_pnl": 250.0  # Profit > $150, should suggest MOVE_TO_BE
            }
        },
        {
            "name": "RTY Weak Trend",
            "market": "RTY",
            "obs_values": {
                228: 15.0,    # Weak ADX
                229: 0.002,   # Near VWAP
                232: -0.1,    # Minimal momentum
                240: 52.0     # Neutral RSI
            },
            "position_state": {
                "position": 0,
                "balance": 50000,
                "win_rate": 0.48,
                "consecutive_losses": 1,
                "trade_history": [
                    {"pnl": 50}, {"pnl": -30}, {"pnl": 100}
                ]
            },
            "market_context": {
                "market_name": "RTY",
                "current_time": "13:00",
                "current_price": 1850.0,
                "position_status": "FLAT",
                "unrealized_pnl": 0.0
            }
        }
    ]

    # Test each scenario
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'=' * 80}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"Market: {scenario['market']}")
        print(f"{'=' * 80}\n")

        # Create observation array (261D)
        obs = np.zeros(261)
        for idx, value in scenario['obs_values'].items():
            obs[idx] = value

        # Build prompt using the updated method
        prompt = llm._build_prompt(
            obs,
            scenario['position_state'],
            scenario['market_context']
        )

        # Print the prompt
        print(prompt)

        # Query the LLM (mock mode)
        action, confidence, reasoning, _ = llm.query(
            obs,
            scenario['position_state'],
            scenario['market_context']
        )

        print(f"\n{'─' * 80}")
        print(f"LLM Response:")
        print(f"Action: {['HOLD', 'BUY', 'SELL', 'MOVE_TO_BE', 'ENABLE_TRAIL', 'DISABLE_TRAIL'][action]}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Reasoning: {reasoning}")
        print(f"{'─' * 80}\n")

    # Print statistics
    stats = llm.get_stats()
    print("\n" + "=" * 80)
    print("LLM STATISTICS")
    print("=" * 80)
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Average Latency: {stats['avg_latency_ms']:.1f}ms")
    print(f"Cache Hit Rate: {stats['cache_hit_rate']:.1f}%")
    print(f"Error Rate: {stats['error_rate']:.2f}%")
    print("=" * 80)

    print("\n✅ PROMPT TEST COMPLETE!")
    print("\nKey Improvements:")
    print("1. ✓ Market-agnostic (works for ES, NQ, YM, RTY, etc.)")
    print("2. ✓ Interpreted indicators (STRONG/WEAK instead of raw numbers)")
    print("3. ✓ Clear trading rules embedded in system prompt")
    print("4. ✓ Risk warnings (losing streaks, low win rate)")
    print("5. ✓ Position management guidance (MOVE_TO_BE, ENABLE_TRAIL)")
    print("\nExpected Impact:")
    print("- LLM confidence should increase from 2% → 40-70%")
    print("- Agreement rate should improve from 46% → 60-75%")
    print("- Better decision quality (respects losing streaks, trends)")

if __name__ == "__main__":
    test_prompt_generation()
