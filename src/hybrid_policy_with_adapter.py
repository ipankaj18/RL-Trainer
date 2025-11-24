"""
Hybrid Policy with Adapter Layer for Phase 2 → Phase 3 Transfer Learning

This policy extends HybridAgentPolicy with a learned adapter layer that projects
261D Phase 3 observations to 228D Phase 2 representation space, enabling
proper transfer learning while preserving all Phase 2 knowledge.

Architecture:
    Phase 3 Observation (261D)
        ↓
    [Adapter Layer: 261D → 228D] ← Learnable projection
        ↓
    [Phase 2 Network: 228D → actions] ← Transferred unchanged
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

# Import existing hybrid policy for inheritance
try:
    from .hybrid_policy import (
        HybridAgentPolicy,
        get_environment,
        register_environment
    )
except ImportError:
    from hybrid_policy import (
        HybridAgentPolicy,
        get_environment,
        register_environment
    )

# Setup logger
logger = logging.getLogger(__name__)


class HybridAgentPolicyWithAdapter(HybridAgentPolicy):
    """
    Hybrid policy with adapter layer for Phase 2 → Phase 3 transfer learning.

    This policy adds a learnable 261D→228D projection layer before the Phase 2
    network, allowing:
    - Full Phase 2 weight transfer (no dimension mismatch)
    - Learned projection of 33 LLM features into Phase 2 representation space
    - Gradual training: adapter first, then full network
    - Preservation of all Phase 2 knowledge

    Key Differences from HybridAgentPolicy:
    - Adds adapter layer (Linear 261D → 228D)
    - Modifies observation space handling
    - Overrides extract_features() to apply adapter
    - Keeps all LLM decision fusion functionality
    """

    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete,
                 lr_schedule,
                 hybrid_agent=None,
                 base_obs_dim: int = 228,
                 **kwargs):
        """
        Initialize hybrid policy with adapter layer.

        Args:
            observation_space: Phase 3 observation space (261D)
            action_space: Action space (6 discrete actions)
            lr_schedule: Learning rate schedule from model
            hybrid_agent: HybridTradingAgent for LLM integration
            base_obs_dim: Base observation dimension (228D for Phase 2)
            **kwargs: Additional policy kwargs
        """
        # Store dimensions
        self.base_obs_dim = base_obs_dim  # 228D (Phase 2)
        self.full_obs_dim = observation_space.shape[0]  # 261D (Phase 3)

        if self.full_obs_dim < self.base_obs_dim:
            raise ValueError(
                f"Full observation dimension ({self.full_obs_dim}) must be >= "
                f"base dimension ({self.base_obs_dim})"
            )

        logger.info(
            f"[ADAPTER] Initializing adapter policy: "
            f"{self.full_obs_dim}D → {self.base_obs_dim}D"
        )

        # Create adapted observation space (228D for base network)
        # This is what the parent policy will use
        adapted_obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.base_obs_dim,),
            dtype=np.float32
        )

        # Initialize parent with 228D observation space
        # This creates all the networks expecting 228D input
        super().__init__(
            adapted_obs_space,
            action_space,
            lr_schedule,
            hybrid_agent=hybrid_agent,
            **kwargs
        )

        # Now add the adapter layer AFTER parent initialization
        # This adapter will project 261D → 228D before passing to base network
        self.adapter = nn.Linear(self.full_obs_dim, self.base_obs_dim)

        # Initialize adapter with identity-like projection
        self._initialize_adapter()

        # FIX: Move adapter to same device as policy parameters
        # This ensures the adapter moves with the model when .to(device) is called
        policy_device = self._get_policy_device()
        self.adapter.to(policy_device)
        logger.info(f"[ADAPTER] Moved adapter to device: {policy_device}")

        # Store original observation space for reference
        self.full_observation_space = observation_space

        logger.info("[ADAPTER] ✅ Adapter layer initialized successfully")

    def _initialize_adapter(self):
        """
        Initialize adapter weights with identity projection + zero padding.

        Strategy:
        - First 228 weights: Identity matrix (preserve base features)
        - Last 33 weights: Zero (LLM features start with no influence)
        - Bias: Zero

        This ensures:
        - Initial behavior matches Phase 2 (adapter is "transparent")
        - LLM features start with no contribution
        - Adapter can learn optimal projection during training
        """
        with torch.no_grad():
            # Initialize weight matrix to zeros
            self.adapter.weight.zero_()

            # Set first 228x228 block to identity (preserve base features)
            identity_size = min(self.base_obs_dim, self.full_obs_dim)
            self.adapter.weight[:identity_size, :identity_size] = torch.eye(identity_size)

            # Last 33 columns (LLM features) remain zero - will be learned
            # This means LLM features initially contribute nothing

            # Zero bias
            self.adapter.bias.zero_()

        logger.info(
            f"[ADAPTER] Initialized: Identity projection for first {identity_size}D, "
            f"zero weights for {self.full_obs_dim - identity_size}D LLM features"
        )

    def _ensure_adapter_device(self) -> torch.device:
        """Keep adapter weights on the same device as the policy."""
        device = self._get_policy_device()
        if self.adapter.weight.device != device:
            self.adapter.to(device)
        return device

    def _apply_adapter(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Apply the adapter layer when observations are 261D.

        Returns:
            Tensor shaped [batch, base_obs_dim]
        """
        device = self._ensure_adapter_device()
        obs = obs.to(device)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        if obs.shape[-1] == self.full_obs_dim:
            return self.adapter(obs)
        if obs.shape[-1] == self.base_obs_dim:
            # Already adapted (e.g., RL-only fallback)
            logger.debug(
                "[ADAPTER] Received %sD observation, expected %sD. Passing through without projection.",
                self.base_obs_dim,
                self.full_obs_dim,
            )
            return obs

        raise ValueError(
            f"Unexpected observation dimension: {obs.shape[-1]}. "
            f"Expected {self.full_obs_dim}D or {self.base_obs_dim}D"
        )

    def extract_features(
        self,
        obs: torch.Tensor,
        features_extractor: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """
        Override extract_features to apply adapter layer.

        Flow:
        1. Apply adapter: 261D → 228D
        2. Call parent extract_features with 228D observation
        3. Return features for mlp_extractor

        Args:
            obs: Observation tensor [batch_size, 261] or [batch_size, 228]
            features_extractor: Optional override (SB3 uses separate heads)

        Returns:
            Features tensor [batch_size, 228]
        """
        if features_extractor is None:
            features_extractor = self.features_extractor

        adapted_obs = self._apply_adapter(obs)
        return features_extractor(adapted_obs)

    def forward(self, obs: torch.Tensor, deterministic: bool = False,
                action_masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with adapter applied to observations.

        This override ensures adapter is applied before hybrid agent prediction.
        The observation will be automatically adapted in extract_features().

        Args:
            obs: Observations tensor [batch_size, 261]
            deterministic: Whether to use deterministic actions
            action_masks: Action masks tensor [batch_size, n_actions]

        Returns:
            Tuple of (actions, values, log_probs)
        """
        # Validate observation dimension
        if obs.shape[-1] != self.full_obs_dim:
            logger.warning(
                f"[ADAPTER] Expected {self.full_obs_dim}D observation, "
                f"got {obs.shape[-1]}D"
            )

        # Call parent forward - it will use our overridden extract_features
        # which applies the adapter
        return super().forward(obs, deterministic, action_masks)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Ensure value predictions also pass through the adapter.
        """
        features = self.extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def _rl_only_predict(self, observation: np.ndarray,
                        action_mask: np.ndarray) -> Tuple[int, Dict]:
        """
        Fallback RL-only prediction with adapter applied.

        This override ensures adapter is used even in fallback mode.

        Args:
            observation: Observation array (261D for Phase 3)
            action_mask: Action mask array

        Returns:
            Tuple of (action, info_dict)
        """
        # Get device
        device = self._get_policy_device()

        # Convert to tensor (full 261D observation)
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)

        # Get latent representation
        # extract_features will apply adapter automatically
        with torch.no_grad():
            features = self.extract_features(obs_tensor)  # Adapter applied here
            latent_pi, _ = self.mlp_extractor(features)
            distribution = self._get_action_dist_from_latent(latent_pi)
            distribution.apply_masking(mask_tensor)

            # Sample action
            action = distribution.sample()[0].item()

        return action, {'fusion_method': 'rl_only_fallback_with_adapter'}

    def get_adapter_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the adapter layer.

        Useful for monitoring adapter training progress.

        Returns:
            Dictionary with adapter statistics
        """
        with torch.no_grad():
            adapter_weight = self.adapter.weight
            adapter_bias = self.adapter.bias

            # Compute statistics
            stats = {
                'adapter_weight_mean': adapter_weight.mean().item(),
                'adapter_weight_std': adapter_weight.std().item(),
                'adapter_weight_max': adapter_weight.max().item(),
                'adapter_weight_min': adapter_weight.min().item(),
                'adapter_bias_mean': adapter_bias.mean().item(),
                'adapter_bias_std': adapter_bias.std().item(),

                # Check if adapter has changed from initialization
                'base_features_identity': self._check_identity_preservation(),
                'llm_features_learned': self._check_llm_feature_learning(),
            }

        return stats

    def _check_identity_preservation(self) -> float:
        """
        Check how much the first 228x228 block deviates from identity.

        Returns:
            Deviation from identity (0.0 = perfect identity)
        """
        with torch.no_grad():
            identity_block = self.adapter.weight[:self.base_obs_dim, :self.base_obs_dim]
            identity_matrix = torch.eye(self.base_obs_dim, device=identity_block.device)
            deviation = (identity_block - identity_matrix).abs().mean().item()
        return deviation

    def _check_llm_feature_learning(self) -> float:
        """
        Check how much the LLM feature weights (last 33 columns) have changed from zero.

        Returns:
            Magnitude of LLM feature weights (0.0 = still zero, >0.0 = learning)
        """
        with torch.no_grad():
            llm_weights = self.adapter.weight[:, self.base_obs_dim:]
            magnitude = llm_weights.abs().mean().item()
        return magnitude


# Export for use in other modules
__all__ = ['HybridAgentPolicyWithAdapter']
