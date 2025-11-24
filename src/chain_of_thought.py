"""
Chain-of-Thought Reasoning Module

Implements multi-step deliberation for trading decisions.
"""

from typing import Dict, Tuple, List, Optional
import re
import numpy as np
import time


class ChainOfThoughtPrompts:
    """
    Prompt templates for 4-step reasoning.

    Step 1: ANALYZE - Understand current market state
    Step 2: OPTIONS - Generate possible actions
    Step 3: RISKS - Evaluate risks of each option
    Step 4: DECIDE - Make final decision with reasoning
    """

    STEP1_ANALYZE = """You are a professional trader analyzing market conditions.

Current Market State:
- Price: ${price:.2f}
- Trend: {trend}
- RSI: {rsi:.1f}
- At Support: {at_support}
- At Resistance: {at_resistance}
- Volume: {volume_trend}
- Recent Pattern: {pattern}

Position State:
- Current Position: {position}
- Entry Price: ${entry_price:.2f} (if in position)
- Unrealized P&L: ${unrealized_pnl:.2f}
- Time in Position: {time_in_position} bars
- Consecutive Losses: {consecutive_losses}

Task: Analyze the current market state in 2-3 sentences. Focus on:
1. Trend direction and strength
2. Key support/resistance levels
3. Momentum and volume
4. Overall market regime (trending/ranging/choppy)

Analysis:"""

    STEP2_OPTIONS = """Given your analysis:
"{analysis}"

Generate 3-4 possible trading actions. For each option, briefly describe the rationale.

Format:
Option A: [Action] - [1 sentence rationale]
Option B: [Action] - [1 sentence rationale]
Option C: [Action] - [1 sentence rationale]

Available actions: {available_actions}

Options:"""

    STEP3_RISKS = """Given your analysis and options:

Analysis: "{analysis}"

Options:
{options}

For each option, evaluate the key risks in 1 sentence.

Format:
Option A Risk: [Key risk]
Option B Risk: [Key risk]
Option C Risk: [Key risk]

Risks:"""

    STEP4_DECIDE = """Now make your final decision.

Analysis: "{analysis}"

Options:
{options}

Risks:
{risks}

Based on the complete analysis, choose the best action and explain why in 2-3 sentences.

Format:
Decision: [ACTION]
Confidence: [0.0-1.0]
Reasoning: [2-3 sentence explanation of why this is the best choice given the analysis, options, and risks]

Final Decision:"""


class ChainOfThoughtReasoner:
    """
    Executes chain-of-thought reasoning process.

    Manages 4-step deliberation and caching.
    """

    def __init__(self, llm_model, cache_size=100):
        self.llm = llm_model
        self.cache = {}  # Cache reasoning chains
        self.cache_size = cache_size
        self.prompts = ChainOfThoughtPrompts()

    def reason(self, observation, position_state, market_context, available_actions):
        """
        Execute full chain-of-thought reasoning.

        Args:
            observation: np.array [261] - market features
            position_state: dict - current position info
            market_context: dict - market state description
            available_actions: list[str] - valid actions (e.g., ["HOLD", "BUY"])

        Returns:
            action: int (0-5)
            confidence: float (0-1)
            reasoning_chain: dict - full 4-step reasoning
        """
        # Check cache (if market state very similar, reuse)
        cache_key = self._get_cache_key(observation, position_state)
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            return cached['action'], cached['confidence'], cached['reasoning']

        # STEP 1: Analyze market state
        step1_prompt = self._build_step1_prompt(observation, position_state, market_context)
        analysis = self.llm._generate_response(step1_prompt)

        # STEP 2: Generate options
        step2_prompt = self._build_step2_prompt(analysis, available_actions)
        options = self.llm._generate_response(step2_prompt)

        # STEP 3: Evaluate risks
        step3_prompt = self._build_step3_prompt(analysis, options)
        risks = self.llm._generate_response(step3_prompt)

        # STEP 4: Make decision
        step4_prompt = self._build_step4_prompt(analysis, options, risks)
        decision = self.llm._generate_response(step4_prompt)

        # Parse final decision
        action, confidence = self._parse_decision(decision, available_actions)

        # Build reasoning chain
        reasoning_chain = {
            'step1_analysis': analysis,
            'step2_options': options,
            'step3_risks': risks,
            'step4_decision': decision,
            'action': action,
            'confidence': confidence
        }

        # Cache
        self._add_to_cache(cache_key, action, confidence, reasoning_chain)

        return action, confidence, reasoning_chain

    def _build_step1_prompt(self, observation, position_state, market_context):
        """Build Step 1 prompt - ANALYZE."""
        return self.prompts.STEP1_ANALYZE.format(
            price=market_context.get('price', 0),
            trend=market_context.get('trend', 'unknown'),
            rsi=market_context.get('rsi', 50),
            at_support=market_context.get('at_support', False),
            at_resistance=market_context.get('at_resistance', False),
            volume_trend=market_context.get('volume_trend', 'normal'),
            pattern=market_context.get('pattern', 'none'),
            position=position_state.get('position', 0),
            entry_price=position_state.get('entry_price', 0),
            unrealized_pnl=position_state.get('unrealized_pnl', 0),
            time_in_position=position_state.get('time_in_position', 0),
            consecutive_losses=position_state.get('consecutive_losses', 0)
        )

    def _build_step2_prompt(self, analysis, available_actions):
        """Build Step 2 prompt - OPTIONS."""
        actions_str = ", ".join(available_actions)
        return self.prompts.STEP2_OPTIONS.format(
            analysis=analysis,
            available_actions=actions_str
        )

    def _build_step3_prompt(self, analysis, options):
        """Build Step 3 prompt - RISKS."""
        return self.prompts.STEP3_RISKS.format(
            analysis=analysis,
            options=options
        )

    def _build_step4_prompt(self, analysis, options, risks):
        """Build Step 4 prompt - DECIDE."""
        return self.prompts.STEP4_DECIDE.format(
            analysis=analysis,
            options=options,
            risks=risks
        )

    def _parse_decision(self, decision_text, available_actions):
        """
        Parse final decision text to extract action and confidence.

        Expected format:
        Decision: BUY
        Confidence: 0.75
        Reasoning: ...
        """
        action = 0  # Default HOLD
        confidence = 0.5  # Default medium confidence

        # Extract decision
        decision_match = re.search(r'Decision:\s*(\w+)', decision_text, re.IGNORECASE)
        if decision_match:
            action_str = decision_match.group(1).upper()

            # Map to action index
            action_map = {
                'HOLD': 0, 'BUY': 1, 'SELL': 2,
                'MOVE_SL_TO_BE': 3, 'ENABLE_TRAIL': 4, 'DISABLE_TRAIL': 5
            }
            action = action_map.get(action_str, 0)

        # Extract confidence
        conf_match = re.search(r'Confidence:\s*([\d.]+)', decision_text)
        if conf_match:
            confidence = float(conf_match.group(1))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1

        return action, confidence

    def _get_cache_key(self, observation, position_state):
        """Generate cache key from observation."""
        # Use rounded features for cache key (allows similar states to match)
        key_features = observation[228:240]  # Use LLM features for caching
        rounded = tuple(np.round(key_features, decimals=1))
        position = position_state.get('position', 0)
        return (rounded, position)

    def _add_to_cache(self, key, action, confidence, reasoning):
        """Add reasoning to cache."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest (FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning
        }
