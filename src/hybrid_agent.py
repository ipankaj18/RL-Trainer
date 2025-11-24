"""
Hybrid Trading Agent

Combines RL (PPO) agent with LLM advisor using intelligent decision fusion.

Decision Fusion Strategies:
1. Agreement: Both agree     take action
2. High Confidence: One very confident     follow it
3. Disagreement: Use weighted voting based on confidence
4. Risk Veto: Override if risk too high (consecutive losses, near DD limit)

Features:
- Selective LLM querying to reduce latency
- Confidence-based decision weighting
- Risk-aware action veto
- Comprehensive statistics tracking
"""

import numpy as np
import logging
from collections import defaultdict
from typing import Tuple, Dict, Optional
import torch

# Import fusion components
try:
    from .fusion_network import FusionNetwork, FusionExperienceBuffer
except ImportError:
    from fusion_network import FusionNetwork, FusionExperienceBuffer


class HybridTradingAgent:
    """
    Hybrid agent combining RL (PPO) + LLM reasoning.
    
    Wraps RL model and LLM advisor to provide fused trading decisions
    with risk management and performance tracking.
    """
    
    def __init__(self, rl_model, llm_model, config: Dict):
        """
        Initialize hybrid trading agent.

        Args:
            rl_model: MaskablePPO model for RL predictions (can be None initially, set later)
            llm_model: LLMReasoningModule for LLM advice
            config: Configuration dictionary with fusion parameters
        """
        self.logger = logging.getLogger(__name__)

        # Core components
        # Note: rl_model can be None initially and set later via set_rl_model()
        self.rl_agent = rl_model  # MaskablePPO or None
        self.llm_advisor = llm_model  # LLMReasoningModule
        self.config = config
        
        self.device = torch.device('cpu')

        # Fusion parameters
        fusion_config = config.get('fusion', {})
        self.llm_weight = fusion_config.get('llm_weight', 0.3)
        self.confidence_threshold = fusion_config.get('confidence_threshold', 0.7)
        self.use_selective_querying = fusion_config.get('use_selective_querying', True)
        self.query_interval = fusion_config.get('query_interval', 5)
        self.query_cooldown = fusion_config.get('query_cooldown', 3)
        llm_disabled = getattr(llm_model, 'mock_mode', False) or self.llm_weight <= 0.0
        if llm_disabled:
            # Pure RL / mock mode: force-disable querying and weights
            self.llm_weight = 0.0
            self.use_selective_querying = False
        
        # PHASE 1: Initialize fusion network (if enabled)
        self.use_neural_fusion = fusion_config.get('use_neural_fusion', True)
        if self.use_neural_fusion:
            self.logger.info("[FUSION] Initializing neural fusion network...")
            self.fusion_network = FusionNetwork(
                input_dim=20,
                hidden_dims=[128, 64, 32],
                num_actions=6
            )
            self.fusion_buffer = FusionExperienceBuffer(max_size=100000)
            self.logger.info("[FUSION] ✅ Neural fusion network initialized")
        else:
            self.fusion_network = None
            self.fusion_buffer = None
            self.logger.info("[FUSION] Using rule-based fusion (neural fusion disabled)")
        
        # PHASE 4: Setup async LLM for always-on thinking
        self.always_on_thinking = fusion_config.get('always_on_thinking', not llm_disabled)
        if self.always_on_thinking and not llm_disabled:
            self.logger.info("[ASYNC] Setting up async LLM inference...")
            try:
                from .async_llm import AsyncLLMInference, BatchedAsyncLLM
            except ImportError:
                from async_llm import AsyncLLMInference, BatchedAsyncLLM
            
            if config.get('use_batched_llm', True):
                self.async_llm = BatchedAsyncLLM(
                    llm_model,
                    max_batch_size=config.get('llm_batch_size', 8),
                    batch_timeout_ms=config.get('llm_batch_timeout_ms', 10)
                )
            else:
                self.async_llm = AsyncLLMInference(
                    llm_model,
                    max_workers=config.get('llm_workers', 4)
                )
            self.logger.info("[ASYNC] ✅ Async LLM inference initialized")
        else:
            self.async_llm = None
            if llm_disabled:
                self.logger.info("[ASYNC] LLM disabled (mock mode or zero weight) - async queries skipped")
            else:
                self.logger.info("[ASYNC] Using synchronous LLM (always-on thinking disabled)")
        
        # Risk parameters
        risk_config = config.get('risk', {})
        self.max_consecutive_losses = risk_config.get('max_consecutive_losses', 3)
        self.min_win_rate_threshold = risk_config.get('min_win_rate_threshold', 0.4)
        self.dd_buffer_threshold = risk_config.get('dd_buffer_threshold', 0.2)
        self.enable_risk_veto = risk_config.get('enable_risk_veto', True)
        
        # Statistics tracking
        self.last_llm_query_id: Optional[int] = None
        self.stats = {
            'total_decisions': 0,
            'rl_only': 0,
            'llm_only': 0,
            'agreement': 0,
            'disagreement': 0,
            'risk_veto': 0,
            'llm_queries': 0,
            'cache_hits': 0
        }
        
        # Caching (for selective querying)
        self.last_llm_action = 0
        self.last_llm_confidence = 0.0
        self.last_llm_reasoning = ""
        self.steps_since_llm_query = defaultdict(int)
        self.last_position_state_by_env: Dict[int, Dict] = {}
        
        # Performance tracking
        self.rl_confidences = []
        self.llm_confidences = []
        
        self.logger.info(f"[HYBRID] Initialized with LLM weight: {self.llm_weight}")

    def set_rl_model(self, rl_model):
        """
        Set or update the RL model.

        This allows creating the hybrid agent before the model exists,
        then setting the model later once environments are ready.

        Args:
            rl_model: MaskablePPO model for RL predictions
        """
        self.rl_agent = rl_model
        self._sync_components_to_device()
        self.logger.info(f"[HYBRID] RL model updated (device={self.device})")

    @property
    def rl_model(self):
        """Alias for rl_agent (backward compatibility)."""
        return self.rl_agent

    @rl_model.setter
    def rl_model(self, model):
        """Alias for set_rl_model (backward compatibility)."""
        self.set_rl_model(model)

    def _get_model_device(self) -> torch.device:
        """Infer the RL model's device for hybrid components."""
        if self.rl_agent is None:
            return torch.device('cpu')
        device = getattr(self.rl_agent, 'device', None)
        if device is None:
            return torch.device('cpu')
        if isinstance(device, str):
            return torch.device(device)
        return device

    def _sync_components_to_device(self):
        """Keep hybrid-side modules aligned with the RL model's device."""
        target_device = self._get_model_device()
        self.device = target_device

        if self.use_neural_fusion and self.fusion_network is not None:
            self.fusion_network.to(target_device)

    def predict(self, observation: np.ndarray, action_mask: np.ndarray,
                position_state: Dict, market_context: Dict, env_id: int = 0) -> Tuple[int, Dict]:
        """
        PHASE 4: Make decision with always-on async LLM thinking.

        Args:
            observation: (261,) array with enhanced features
            action_mask: (6,) boolean array of valid actions
            position_state: Current position information
            market_context: Market state information
            env_id: Environment ID (for parallel environments)

        Returns:
            Tuple of (action, metadata_dict)
        """
        # Validate RL model is set
        if self.rl_agent is None:
            raise ValueError("RL model not set! Call set_rl_model() before using predict().")

        self.stats['total_decisions'] += 1

        # 1. Get RL recommendation (instant)
        rl_obs = observation
        policy = getattr(self.rl_agent, 'policy', None)
        has_adapter = getattr(policy, 'adapter', None) is not None if policy is not None else False
        if policy is not None and not has_adapter:
            obs_space = getattr(policy, 'observation_space', None)
            if obs_space is not None and hasattr(obs_space, 'shape') and len(obs_space.shape) > 0:
                target_dim = int(obs_space.shape[0])
                if len(observation) > target_dim:
                    rl_obs = observation[:target_dim]

        # FIX: Use _rl_only_predict to avoid infinite recursion
        # The rl_agent's policy might be a HybridAgentPolicy which would call back to us
        # _rl_only_predict bypasses the hybrid logic and goes straight to the RL network
        if hasattr(self.rl_agent, 'policy') and hasattr(self.rl_agent.policy, '_rl_only_predict'):
            # Hybrid policy detected - use the RL-only fallback method
            rl_action, _ = self.rl_agent.policy._rl_only_predict(rl_obs, action_mask)
            rl_value = 0.0  # Fallback method doesn't return value, use neutral
        else:
            # Standard policy - use normal predict
            rl_action, rl_value = self.rl_agent.predict(rl_obs, action_masks=action_mask)
        rl_confidence = self._calculate_rl_confidence(rl_value, rl_action, action_mask)
        
        # 2. Get LLM recommendation (from previous step - zero latency!)
        llm_result = self.async_llm.get_latest_result(env_id, timeout_ms=5) if self.async_llm else None
        result_is_new = bool(llm_result and llm_result.get('is_new', True))

        if llm_result and llm_result['success']:
            llm_action = llm_result['action']
            llm_confidence = llm_result['confidence']
            llm_reasoning = llm_result['reasoning']
            llm_query_id = llm_result['query_id']
            self.logger.debug(f"[HYBRID] LLM result received for env {env_id}: action={llm_action}, confidence={llm_confidence:.2f}")
            if result_is_new:
                self.stats['llm_queries'] += 1  # Track successful LLM queries
        else:
            # No LLM result yet (first step or LLM failed)
            llm_action = rl_action
            llm_confidence = 0.0
            llm_reasoning = "No LLM result available"
            llm_query_id = None
            if llm_result:
                self.logger.debug(f"[HYBRID] LLM result failed for env {env_id}: {llm_result.get('reasoning', 'Unknown error')}")
            else:
                self.logger.debug(f"[HYBRID] No LLM result available for env {env_id}")
        
        # 3. Submit new LLM query for NEXT step (non-blocking, selective)
        llm_query_submitted = False
        if self.always_on_thinking and self.async_llm:
            available_actions = self._get_available_actions(action_mask)
            if self._should_query_llm(env_id, rl_confidence, position_state, action_mask):
                self.logger.debug(f"[HYBRID] Submitting LLM query for env {env_id}, available_actions={available_actions}")
                self.async_llm.submit_query(
                    env_id, observation, position_state, market_context, available_actions
                )
                self.steps_since_llm_query[env_id] = 0
                llm_query_submitted = True
            else:
                self.steps_since_llm_query[env_id] += 1
        else:
            self.logger.debug(f"[HYBRID] LLM querying disabled for env {env_id}")
            self.steps_since_llm_query[env_id] += 1
        
        # 4. Fusion (uses results from current step)
        final_action, fusion_meta = self._fuse_decisions(
            rl_action, rl_confidence,
            llm_action, llm_confidence,
            action_mask, position_state
        )
        
        # PHASE 1: Track fusion decision for training
        if self.use_neural_fusion and self.fusion_buffer is not None:
            fusion_input = self._build_fusion_input(
                rl_action, rl_confidence,
                llm_action, llm_confidence,
                action_mask, position_state
            )
            self.last_fusion_context = {
                'input': fusion_input,
                'rl_action': rl_action,
                'llm_action': llm_action,
                'final_action': final_action,
                'timestamp': position_state.get('timestamp', 0)
            }
        
        # 5. Apply risk veto
        final_action, risk_veto = self._apply_risk_veto(
            final_action, position_state, market_context
        )

        if (self.use_neural_fusion and
            self.fusion_buffer is not None and
            hasattr(self, 'last_fusion_context') and
            self.last_fusion_context is not None):
            self.last_fusion_context['final_action'] = final_action

        # Track latest LLM query reference for outcome attribution
        self.last_llm_query_id = llm_query_id

        if risk_veto:
            self.stats['risk_veto'] += 1
            fusion_meta['risk_veto'] = True
        
        # Update statistics
        self._update_stats(fusion_meta['fusion_method'])
        
        # Track confidences
        self.rl_confidences.append(rl_confidence)
        self.llm_confidences.append(llm_confidence)
        
        # Keep last position state for selective querying
        self.last_position_state_by_env[env_id] = position_state.copy()
        
        return final_action, {
            'rl_action': rl_action,
            'rl_confidence': rl_confidence,
            'llm_action': llm_action,
            'llm_confidence': llm_confidence,
            'llm_reasoning': llm_reasoning,
            'fusion_method': fusion_meta['fusion_method'],
            'final_action': final_action,
            'risk_veto': risk_veto,
            'llm_queried': result_is_new or llm_query_submitted,
            'llm_query_id': llm_query_id
        }
    
    def _should_query_llm(self, env_id: int, rl_confidence: float,
                          position_state: Dict, action_mask: np.ndarray) -> bool:
        """
        Determine if we should query LLM this step.
        
        Query LLM if:
        1. Selective querying is disabled (always query)
        2. RL is uncertain (confidence < threshold)
        3. Entry decision (position == 0 and action is entry)
        4. Interval reached (every N steps)
        5. Position state changed significantly
        """
        if not self.use_selective_querying:
            return True  # Always query
        
        steps_since = self.steps_since_llm_query.get(env_id, self.query_interval)

        # Enforce cooldown before issuing another query unless state changed dramatically
        if steps_since < self.query_cooldown:
            if self._position_state_changed(env_id, position_state):
                return True
            return False

        # Check if RL is uncertain
        if rl_confidence < self.confidence_threshold:
            return True
        
        # Check if it's an entry decision
        position = position_state.get('position', 0)
        if position == 0:
            # Check if RL is suggesting entry
            # This would require predicting RL action first, which we already do
            pass
        
        # Check interval
        if steps_since >= self.query_interval:
            return True
        
        # Check if position state changed significantly
        if self._position_state_changed(env_id, position_state):
            return True
        
        return False  # Use cached LLM response
    
    def _position_state_changed(self, env_id: int, current_state: Dict) -> bool:
        """Check if position state changed significantly."""
        last_state = self.last_position_state_by_env.get(env_id)
        if not last_state:
            return True
        
        # Check key metrics that might warrant fresh LLM input
        key_metrics = ['position', 'balance', 'win_rate', 'consecutive_losses']
        
        for metric in key_metrics:
            if current_state.get(metric) != last_state.get(metric):
                return True
        
        return False
    
    def _calculate_rl_confidence(self, value: float, action: int, action_mask: np.ndarray) -> float:
        """
        Estimate RL confidence from value estimate and action probabilities.

        HIGH PRIORITY FIX #7: Improved confidence calculation (less arbitrary).

        Args:
            value: Value estimate from RL model
            action: Chosen action
            action_mask: Valid action mask

        Returns:
            Confidence score (0-1)
        """
        try:
            # HIGH PRIORITY FIX #7: Better confidence calculation
            # Value estimates typically range from -100 to +100 during training
            # We use a sigmoid-like function instead of arbitrary division by 10

            # Approach 1: Use normalized value (works across training stages)
            # Typical value range observed: [-50, +50] for trained models
            normalized_value = np.tanh(value / 20.0)  # Smooth sigmoid
            base_confidence = (abs(normalized_value) + 0.1) / 1.1  # Range: [0.09, 1.0]

            # Approach 2: Penalize if only few actions valid (limited choice)
            num_valid_actions = np.sum(action_mask)
            if num_valid_actions <= 1:
                # Only one valid action = no real choice = low confidence
                base_confidence *= 0.7

            # Approach 3: Could be enhanced with action probability distribution
            # For future: Extract policy logits to get true action probabilities
            # confidence = max(action_probs) would be ideal

            # Clamp to valid range
            confidence = np.clip(base_confidence, 0.0, 1.0)

            return float(confidence)

        except Exception as e:
            self.logger.error(f"[HYBRID] Error calculating RL confidence: {e}")
            return 0.5  # Neutral confidence
    
    def _fuse_decisions(self, rl_action: int, rl_conf: float, 
                       llm_action: int, llm_conf: float,
                       action_mask: np.ndarray, position_state: Dict) -> Tuple[int, Dict]:
        """
        PHASE 1: Neural fusion - learned adaptive weighting.
        
        Uses trained fusion network to determine optimal action.
        Falls back to rule-based fusion if neural fusion disabled or not trained.
        """
        # Use neural fusion if enabled and trained
        if self.use_neural_fusion and self.fusion_network is not None and len(self.fusion_buffer) >= 100:
            return self._fuse_neural(rl_action, rl_conf, llm_action, llm_conf, action_mask, position_state)
        else:
            # Fallback to rule-based fusion
            return self._fuse_rule_based(rl_action, rl_conf, llm_action, llm_conf, action_mask, position_state)
    
    def _fuse_neural(self, rl_action: int, rl_conf: float, 
                    llm_action: int, llm_conf: float,
                    action_mask: np.ndarray, position_state: Dict) -> Tuple[int, Dict]:
        """
        Neural fusion - learned adaptive weighting.
        
        Uses trained fusion network to determine optimal action.
        """
        # Build fusion input (20D)
        fusion_input = self._build_fusion_input(
            rl_action, rl_conf, llm_action, llm_conf,
            action_mask, position_state
        )

        # Get fusion network prediction
        final_action, rl_trust, llm_trust = self.fusion_network.predict_action(
            fusion_input,
            deterministic=True
        )

        # Validate action is legal
        if not action_mask[final_action]:
            self.logger.warning(f"[FUSION] Network predicted invalid action {final_action}")
            # Fallback: use RL if valid, else HOLD
            final_action = rl_action if action_mask[rl_action] else 0
            rl_trust, llm_trust = 1.0, 0.0

        # Update stats
        self.stats['fusion_decisions'] += 1
        self.stats['avg_rl_trust'] = 0.9 * self.stats.get('avg_rl_trust', 0.5) + 0.1 * rl_trust
        self.stats['avg_llm_trust'] = 0.9 * self.stats.get('avg_llm_trust', 0.5) + 0.1 * llm_trust

        # Metadata
        metadata = {
            'source': 'fusion_network',
            'rl_action': rl_action,
            'rl_confidence': rl_conf,
            'llm_action': llm_action,
            'llm_confidence': llm_conf,
            'rl_trust': rl_trust,
            'llm_trust': llm_trust,
            'agreement': (rl_action == llm_action)
        }

        return final_action, metadata
    
    def _fuse_rule_based(self, rl_action: int, rl_conf: float, 
                        llm_action: int, llm_conf: float,
                        action_mask: np.ndarray, position_state: Dict) -> Tuple[int, Dict]:
        """
        Original rule-based fusion for fallback.
        
        Priority:
        1. Agreement     take action
        2. High confidence RL (>0.9)     follow RL
        3. High confidence LLM (>0.9)     follow LLM
        4. Both uncertain (<0.5)     HOLD
        5. Weighted decision by confidence
        """
        # Agreement
        if rl_action == llm_action:
            self.stats['agreement'] += 1
            return rl_action, {'fusion_method': 'agreement'}
        
        # High confidence RL
        if rl_conf > 0.9 and llm_conf <= 0.6:
            self.stats['rl_only'] += 1
            return rl_action, {'fusion_method': 'rl_confident'}
        
        # High confidence LLM
        if llm_conf > 0.9 and rl_conf <= 0.6:
            self.stats['llm_only'] += 1
            return llm_action, {'fusion_method': 'llm_confident'}
        
        # Both uncertain     HOLD for safety
        if rl_conf < 0.5 and llm_conf < 0.5:
            return 0, {'fusion_method': 'both_uncertain'}
        
        # Weighted decision
        self.stats['disagreement'] += 1
        
        # Apply LLM weight from config
        rl_weight = rl_conf * (1 - self.llm_weight)
        llm_weight = llm_conf * self.llm_weight
        
        if llm_weight > rl_weight:
            return llm_action, {'fusion_method': 'llm_weighted'}
        else:
            return rl_action, {'fusion_method': 'rl_weighted'}
    
    def _build_fusion_input(self, rl_action, rl_conf, llm_action, llm_conf,
                           action_mask, position_state):
        """
        Build 20D input for fusion network.

        Returns:
            np.array [20] - fusion context
        """
        # One-hot encode actions (6D each)
        rl_action_onehot = np.zeros(6)
        rl_action_onehot[rl_action] = 1.0

        llm_action_onehot = np.zeros(6)
        llm_action_onehot[llm_action] = 1.0

        # Context features (8D)
        agreement = float(rl_action == llm_action)
        market_volatility = position_state.get('volatility', 0.5)
        recent_win_rate = position_state.get('win_rate', 0.5)
        consecutive_losses = min(position_state.get('consecutive_losses', 0) / 5.0, 1.0)
        distance_from_dd = position_state.get('distance_from_drawdown', 1.0)
        time_in_position = min(position_state.get('time_in_position', 0) / 100.0, 1.0)

        # Concatenate (6 + 1 + 6 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 20D)
        fusion_input = np.concatenate([
            rl_action_onehot,      # 6D
            [rl_conf],             # 1D
            llm_action_onehot,     # 6D
            [llm_conf],            # 1D
            [agreement],           # 1D
            [market_volatility],   # 1D
            [recent_win_rate],     # 1D
            [consecutive_losses],  # 1D
            [distance_from_dd],    # 1D
            [time_in_position]     # 1D
        ])

        return fusion_input.astype(np.float32)
    
    def _get_available_actions(self, action_mask: np.ndarray) -> list:
        """
        Get list of available action names based on action mask.
        
        Args:
            action_mask: np.array [6] - 1 if action allowed, 0 if not
            
        Returns:
            list[str]: Names of available actions
        """
        action_names = ['HOLD', 'BUY', 'SELL', 'MOVE_SL_TO_BE', 'ENABLE_TRAIL', 'DISABLE_TRAIL']
        return [name for i, name in enumerate(action_names) if action_mask[i] == 1]
    
    def _apply_risk_veto(self, action: int, position_state: Dict, market_context: Dict) -> Tuple[int, bool]:
        """
        Apply risk-based veto to proposed action.
        
        Veto conditions:
        1. Consecutive losses     threshold     Only allow HOLD or exits
        2. Near trailing DD limit     Only allow HOLD or risk-reducing actions
        3. Win rate < threshold     Increase selectivity for entries
        
        Returns:
            Tuple of (final_action, veto_applied)
        """
        if not self.enable_risk_veto:
            return action, False
        
        consecutive_losses = position_state.get('consecutive_losses', 0)
        dd_buffer = position_state.get('dd_buffer_ratio', 1.0)
        win_rate = position_state.get('win_rate', 0.5)
        current_position = position_state.get('position', 0)
        
        # Define risk-reducing actions
        risk_reducing_actions = [0, 3, 5]  # HOLD, MOVE_TO_BE, DISABLE_TRAIL
        exit_actions = [3, 5]  # Actions that reduce risk
        
        # Veto 1: Consecutive losses threshold
        if consecutive_losses >= self.max_consecutive_losses:
            if action in [1, 2]:  # BUY or SELL (new entries)
                self.logger.info(f"[RISK_VETO] Blocking entry due to {consecutive_losses} consecutive losses")
                return 0, True  # Force HOLD
        
        # Veto 2: Drawdown buffer threshold
        if dd_buffer < self.dd_buffer_threshold:
            if action not in risk_reducing_actions:
                self.logger.info(f"[RISK_VETO] Blocking action {action} due to DD proximity ({dd_buffer:.1%})")
                return 0, True  # Force HOLD
        
        # Veto 3: Low win rate threshold
        if win_rate < self.min_win_rate_threshold and current_position == 0:
            if action in [1, 2]:  # New entries only
                self.logger.info(f"[RISK_VETO] Blocking entry due to low win rate ({win_rate:.1%})")
                return 0, True  # Force HOLD
        
        return action, False  # No veto
    
    def _update_stats(self, fusion_method: str):
        """Update fusion statistics."""
        # Stats are updated in _fuse_decisions, this method is for future extensions
        pass
    
    def get_stats(self) -> Dict:
        """Get decision fusion statistics."""
        total = self.stats['total_decisions']
        if total == 0:
            return self.stats
        
        # Calculate percentages
        stats_with_pct = {**self.stats}
        
        stats_with_pct['agreement_pct'] = self.stats['agreement'] / total * 100
        stats_with_pct['disagreement_pct'] = self.stats['disagreement'] / total * 100
        stats_with_pct['rl_only_pct'] = self.stats['rl_only'] / total * 100
        stats_with_pct['llm_only_pct'] = self.stats['llm_only'] / total * 100
        stats_with_pct['risk_veto_pct'] = self.stats['risk_veto'] / total * 100
        stats_with_pct['llm_query_rate'] = self.stats['llm_queries'] / total * 100
        stats_with_pct['cache_hit_rate'] = self.stats['cache_hits'] / max(self.stats['llm_queries'], 1) * 100
        
        # Add confidence averages
        if self.rl_confidences:
            stats_with_pct['avg_rl_confidence'] = np.mean(self.rl_confidences)
        if self.llm_confidences:
            stats_with_pct['avg_llm_confidence'] = np.mean(self.llm_confidences)
        
        return stats_with_pct
    
    def reset_stats(self):
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0
        
        self.rl_confidences.clear()
        self.llm_confidences.clear()
    
    def get_llm_stats(self) -> Dict:
        """Get LLM-specific statistics."""
        return self.llm_advisor.get_stats() if hasattr(self.llm_advisor, 'get_stats') else {}
    
    def update_fusion_outcome(self, reward: float):
        """
        PHASE 1: Update fusion buffer with outcome after trade completes.
        
        Called by environment when trade finishes.
        
        Args:
            reward: float - reward received from the trade
        """
        if (self.use_neural_fusion and
            self.fusion_buffer is not None and
            hasattr(self, 'last_fusion_context') and
            self.last_fusion_context is not None):
            
            self.fusion_buffer.add(
                self.last_fusion_context['input'],
                self.last_fusion_context['rl_action'],
                self.last_fusion_context['llm_action'],
                self.last_fusion_context['final_action'],
                reward
            )

            # Clear the cached decision context once recorded
            self.last_fusion_context = None

    def pop_last_llm_query_id(self) -> Optional[int]:
        """
        Retrieve and clear the most recent LLM query identifier.
        """
        query_id = self.last_llm_query_id
        self.last_llm_query_id = None
        return query_id
    
    def update_llm_outcome(self, query_id: Optional[int], reward: float, final_pnl: float):
        """
        PHASE 2: Update LLM query outcome after trade completes.
        
        Called by environment when trade finishes.
        
        Args:
            query_id: int - ID from query()
            reward: float - immediate reward
            final_pnl: float - final P&L of trade
        """
        resolved_query_id = query_id if query_id is not None else self.pop_last_llm_query_id()
        if hasattr(self, 'llm_advisor') and resolved_query_id is not None:
            self.llm_advisor.update_outcome(resolved_query_id, reward, final_pnl)

