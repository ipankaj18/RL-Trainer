"""
Hybrid Policy Wrapper for Stable Baselines 3

Enables LLM integration during training by routing predictions through HybridTradingAgent.
This is the core architectural fix for Phase 3 LLM integration.
"""

import logging
import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

# Setup logger for this module
logger = logging.getLogger(__name__)


class HybridAgentPolicy(MaskableActorCriticPolicy):
    """
    SB3-compatible policy that integrates HybridTradingAgent into the training loop.
    
    This policy wrapper routes all predictions through the hybrid agent, enabling:
    - LLM queries during training (not just inference)
    - Decision fusion in the training loop
    - Risk-aware action selection
    - Comprehensive statistics tracking
    
    Architecture:
    - Maintains original RL model architecture for gradient flow
    - Overrides forward() to route through hybrid agent
    - Preserves action masking compatibility
    - Enables async LLM inference during rollouts
    """
    
    def __init__(self, *args, hybrid_agent=None, **kwargs):
        """
        Initialize hybrid policy.

        Args:
            *args: Arguments passed to parent policy
            hybrid_agent: HybridTradingAgent instance for LLM integration
            **kwargs: Keyword arguments passed to parent policy
        """
        super().__init__(*args, **kwargs)

        self.hybrid_agent = hybrid_agent
        self._validate_hybrid_agent()

        # Statistics for monitoring state access
        self._state_access_stats = {
            'position_state_actual': 0,
            'position_state_fallback': 0,
            'market_context_actual': 0,
            'market_context_fallback': 0
        }

        logger.info(f"[HYBRID_POLICY] Initialized with hybrid agent: {hybrid_agent is not None}")

    def _get_policy_device(self) -> torch.device:
        """
        Return the device the policy is currently residing on.

        Falls back to CPU if the policy hasn't registered parameters yet.
        """
        if hasattr(self, 'device'):
            return self.device
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def _validate_hybrid_agent(self):
        """Validate that hybrid agent is properly configured."""
        if self.hybrid_agent is None:
            logger.warning("[HYBRID_POLICY] No hybrid agent provided - using RL-only mode")
            return

        # Verify hybrid agent has required methods
        required_methods = ['predict', 'get_stats', 'reset_stats']
        for method in required_methods:
            if not hasattr(self.hybrid_agent, method):
                raise ValueError(f"Hybrid agent missing required method: {method}")

        logger.info("[HYBRID_POLICY] ✅ Hybrid agent validation passed")
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False, 
                action_masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that routes through hybrid agent for LLM integration.
        
        This is the key method that enables LLM participation during training.
        Called by SB3 during rollout collection.
        
        Args:
            obs: Observations tensor [batch_size, obs_dim]
            deterministic: Whether to use deterministic actions
            action_masks: Action masks tensor [batch_size, n_actions]
            
        Returns:
            Tuple of (actions, values, log_probs)
        """
        if self.hybrid_agent is None:
            # Fallback to RL-only mode
            return super().forward(obs, deterministic, action_masks)
        
        # Get device from policy parameters (ensure all tensors are on policy device)
        device = self._get_policy_device()
        
        # Ensure input tensor is on the same device as policy
        obs = obs.to(device)
        
        # CRITICAL: Keep original action_masks for return - don't modify
        original_action_masks = action_masks
        
        batch_size = obs.shape[0]
        
        # For processing with hybrid agent, create a working copy
        if action_masks is not None:
            if isinstance(action_masks, torch.Tensor):
                action_masks_processed = action_masks.to(device)
                action_masks_np = action_masks_processed.detach().cpu().numpy()
            else:
                action_masks_processed = torch.tensor(action_masks, dtype=torch.bool, device=device)
                action_masks_np = action_masks_processed.detach().cpu().numpy()
        else:
            action_masks_processed = None
            action_masks_np = np.ones((batch_size, self.action_space.n), dtype=bool)
        
        # Convert observation to numpy for hybrid agent
        obs_np = obs.detach().cpu().numpy()

        # Process each observation in batch
        actions_list = []
        
        for i in range(batch_size):
            # Route through hybrid agent (activates LLM!)
            action, _ = self._predict_with_hybrid_agent(
                obs_np[i], action_masks_np[i], env_id=i
            )
            actions_list.append(action)
        
        # Convert actions back to tensor on policy device
        actions = torch.tensor(actions_list, dtype=torch.long, device=device)
        
        # Get values and log_probs from base policy for gradient flow
        # This ensures RL components still receive gradients
        with torch.no_grad():
            # Extract features first (required for proper observation processing)
            features = self.extract_features(obs)
            # Use base policy to get values (critic)
            latent_pi, latent_vf = self.mlp_extractor(features)
            values = self.value_net(latent_vf)

            # Get log_probs for policy gradient
            distribution = self._get_action_dist_from_latent(latent_pi)
            if action_masks is not None:
                # Use the processed tensor version for masking
                distribution.apply_masking(action_masks_processed)

            log_probs = distribution.log_prob(actions)
        
        return actions, values, log_probs
    
    def _predict_with_hybrid_agent(self, observation: np.ndarray, 
                                   action_mask: np.ndarray, env_id: int = 0) -> Tuple[int, Dict]:
        """
        Route prediction through hybrid agent to enable LLM integration.
        
        Args:
            observation: Single observation array
            action_mask: Action mask array
            env_id: Environment ID for async LLM
            
        Returns:
            Tuple of (action, info_dict)
        """
        try:
            # Get position state from environment (need to access it)
            # This requires the environment to be accessible - we'll use a fallback
            position_state = self._build_position_state(env_id)
            market_context = self._build_market_context(env_id)
            
            # Route through hybrid agent (THIS ACTIVATES LLM!)
            action, meta = self.hybrid_agent.predict(
                observation, action_mask, position_state, market_context, env_id
            )
            
            return action, meta
            
        except Exception as e:
            logger.error(f"[HYBRID_POLICY] Error in hybrid prediction: {e}")
            # Fallback to RL-only
            return self._rl_only_predict(observation, action_mask)
    
    def _rl_only_predict(self, observation: np.ndarray,
                        action_mask: np.ndarray) -> Tuple[int, Dict]:
        """
        Fallback RL-only prediction (no LLM).

        This fallback handles whichever observation dimension the current policy expects.
        Adapter-enabled policies can consume the full 261D Phase 3 observation, while
        legacy policies automatically operate on their configured observation space.

        Args:
            observation: Observation array (261D for Phase 3)
            action_mask: Action mask array

        Returns:
            Tuple of (action, empty_info)
        """
        # Get device from policy parameters
        device = self._get_policy_device()

        # Convert to tensor on correct device, preserving the full observation
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)

        # Get latent representation
        with torch.no_grad():
            # Extract features first (required for proper observation processing)
            features = self.extract_features(obs_tensor)
            latent_pi, _ = self.mlp_extractor(features)
            distribution = self._get_action_dist_from_latent(latent_pi)
            distribution.apply_masking(mask_tensor)

            # Sample action
            action = distribution.sample()[0].item()

        return action, {'fusion_method': 'rl_only_fallback'}
    
    def _build_position_state(self, env_id: int) -> Dict:
        """
        Build position state dictionary for hybrid agent.

        Attempts to retrieve actual position state from registered environment.
        Falls back to safe defaults if environment not available.

        Args:
            env_id: Environment ID

        Returns:
            Position state dictionary
        """
        # Try to get actual environment state from registry
        env = get_environment(env_id)

        if env is not None and hasattr(env, '_get_position_state'):
            try:
                # Get actual position state from environment
                position_state = env._get_position_state()
                self._state_access_stats['position_state_actual'] += 1
                return position_state
            except Exception as e:
                # Log error but continue with fallback
                logger.warning(f"[HYBRID_POLICY] Could not get position state from env {env_id}: {e}")

        # Fallback to safe defaults if environment not available or error occurred
        self._state_access_stats['position_state_fallback'] += 1
        return {
            'position': 0,
            'balance': 50000.0,
            'win_rate': 0.5,
            'consecutive_losses': 0,
            'dd_buffer_ratio': 1.0,
            'time_in_position': 0,
            'unrealized_pnl': 0.0,
            'timestamp': 0
        }
    
    def _build_market_context(self, env_id: int) -> Dict:
        """
        Build market context dictionary for hybrid agent.

        Attempts to retrieve actual market context from registered environment.
        Falls back to safe defaults if environment not available.

        Args:
            env_id: Environment ID

        Returns:
            Market context dictionary
        """
        # Try to get actual environment for market context
        env = get_environment(env_id)

        if env is not None:
            try:
                # Build market context from environment attributes
                context = {}

                # Get market name if available
                if hasattr(env, 'market_spec'):
                    context['market_name'] = env.market_spec.symbol
                elif hasattr(env, 'market_name'):
                    context['market_name'] = env.market_name
                else:
                    context['market_name'] = 'Unknown'

                # Get current time if available
                if hasattr(env, '_current_step') and hasattr(env, 'data'):
                    try:
                        current_idx = env._current_step
                        if current_idx < len(env.data):
                            timestamp = env.data.index[current_idx]
                            context['current_time'] = timestamp.strftime('%H:%M')
                        else:
                            context['current_time'] = '00:00'
                    except:
                        context['current_time'] = '00:00'
                else:
                    context['current_time'] = '00:00'

                # Get current price if available
                if hasattr(env, 'data') and hasattr(env, '_current_step'):
                    try:
                        current_idx = env._current_step
                        if current_idx < len(env.data):
                            context['current_price'] = float(env.data.iloc[current_idx]['close'])
                        else:
                            context['current_price'] = 0.0
                    except:
                        context['current_price'] = 0.0
                else:
                    context['current_price'] = 0.0

                self._state_access_stats['market_context_actual'] += 1
                return context

            except Exception as e:
                # Log error but continue with fallback
                logger.warning(f"[HYBRID_POLICY] Could not get market context from env {env_id}: {e}")

        # Fallback to safe defaults if environment not available or error occurred
        self._state_access_stats['market_context_fallback'] += 1
        return {
            'market_name': 'Unknown',
            'current_time': '00:00',
            'current_price': 0.0
        }
    
    def get_state_access_stats(self) -> Dict:
        """
        Get statistics on environment state access.

        Returns:
            Dictionary with state access statistics
        """
        total_position_state = (self._state_access_stats['position_state_actual'] +
                               self._state_access_stats['position_state_fallback'])
        total_market_context = (self._state_access_stats['market_context_actual'] +
                               self._state_access_stats['market_context_fallback'])

        position_actual_pct = (self._state_access_stats['position_state_actual'] / total_position_state * 100
                              if total_position_state > 0 else 0)
        market_actual_pct = (self._state_access_stats['market_context_actual'] / total_market_context * 100
                            if total_market_context > 0 else 0)

        return {
            **self._state_access_stats,
            'position_state_actual_pct': position_actual_pct,
            'market_context_actual_pct': market_actual_pct,
            'total_accesses': total_position_state
        }

    def validate_registry(self) -> bool:
        """
        Validate that environment registry is properly configured.

        Returns:
            True if registry has environments, False otherwise
        """
        if not ENVIRONMENT_REGISTRY:
            logger.warning("[HYBRID_POLICY] Environment registry is empty!")
            logger.warning("[HYBRID_POLICY] Position/market state will use fallback defaults.")
            logger.warning("[HYBRID_POLICY] Call register_environment() to fix this.")
            return False

        logger.info(f"[HYBRID_POLICY] Registry validated: {len(ENVIRONMENT_REGISTRY)} environments registered")
        return True

    def predict(self, observation: np.ndarray, state: Optional[Any] = None,
                episode_start: Optional[np.ndarray] = None, deterministic: bool = False,
                action_masks: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Public predict method (used during evaluation).
        
        Args:
            observation: Observation array
            state: Hidden state (for recurrent policies)
            episode_start: Episode start mask
            deterministic: Whether to use deterministic actions
            action_masks: Action masks
            
        Returns:
            Tuple of (actions, states)
        """
        if self.hybrid_agent is None:
            # Fallback to parent
            return super().predict(observation, state, episode_start, deterministic, action_masks)
        
        # Handle batch predictions
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)
            if action_masks is not None:
                action_masks = action_masks.reshape(1, -1)
        
        batch_size = observation.shape[0]
        actions = []
        
        for i in range(batch_size):
            if action_masks is not None:
                mask = action_masks[i]
            else:
                # Create a default mask allowing all actions if none is provided
                mask = np.ones(self.action_space.n, dtype=bool)
            
            action, _ = self._predict_with_hybrid_agent(observation[i], mask, env_id=i)
            actions.append(action)
        
        return np.array(actions), state


class HybridPolicyWrapper:
    """
    Utility wrapper to inject hybrid agent into existing SB3 models.
    
    Usage:
        model = MaskablePPO.load("model.zip")
        hybrid_agent = HybridTradingAgent(model, llm, config)
        wrapped_model = HybridPolicyWrapper(model, hybrid_agent)
        wrapped_model.learn(...)  # Now uses LLM!
    """
    
    def __init__(self, model, hybrid_agent):
        self.model = model
        self.hybrid_agent = hybrid_agent
        
        # Replace policy with hybrid version
        original_policy = model.policy
        
        # Create hybrid policy with same architecture
        self.model.policy = HybridAgentPolicy(
            observation_space=original_policy.observation_space,
            action_space=original_policy.action_space,
            lr_schedule=original_policy.lr_schedule,
            hybrid_agent=hybrid_agent,
            **original_policy.policy_kwargs
        )
        
        # Copy weights from original policy
        self.model.policy.load_state_dict(original_policy.state_dict())
        
        logger.info("[HYBRID_POLICY_WRAPPER] Model wrapped with hybrid policy")
    
    def learn(self, *args, **kwargs):
        """Delegate to model learn with hybrid policy."""
        return self.model.learn(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        """Delegate to model predict."""
        return self.model.predict(*args, **kwargs)
    
    def save(self, *args, **kwargs):
        """Delegate to model save."""
        return self.model.save(*args, **kwargs)


# Global registry for environment references (needed for position state)
# This is a workaround for multiprocessing environments
ENVIRONMENT_REGISTRY = {}


def register_environment(env_id: int, env):
    """Register environment for hybrid agent access."""
    ENVIRONMENT_REGISTRY[env_id] = env
    logger.debug(f"[HYBRID_POLICY] Registered environment {env_id}")


def get_environment(env_id: int):
    """Get registered environment."""
    return ENVIRONMENT_REGISTRY.get(env_id)
