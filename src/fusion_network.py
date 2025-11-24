"""
Neural Fusion Network - Learns when to trust RL vs LLM

Purpose: Replace rule-based fusion in hybrid_agent.py with adaptive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Dict, Any
from collections import deque


class FusionNetwork(nn.Module):
    """
    Neural network that learns optimal fusion of RL and LLM decisions.

    Architecture:
        Input (20D):
            - RL action (6D one-hot)
            - RL confidence (1D)
            - LLM action (6D one-hot)
            - LLM confidence (1D)
            - Action agreement (1D boolean)
            - Market volatility (1D)
            - Recent win rate (1D)
            - Consecutive losses (1D)
            - Distance from drawdown (1D)
            - Time in position (1D)

        Hidden: [128, 64, 32]

        Output (8D):
            - Final action logits (6D)
            - RL trust score (1D, 0-1)
            - LLM trust score (1D, 0-1)
    """

    def __init__(self, input_dim=20, hidden_dims=[128, 64, 32], num_actions=6):
        super(FusionNetwork, self).__init__()

        self.num_actions = num_actions

        # Feature encoder
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output heads
        self.action_head = nn.Linear(prev_dim, num_actions)
        self.rl_trust_head = nn.Linear(prev_dim, 1)
        self.llm_trust_head = nn.Linear(prev_dim, 1)

    def forward(self, fusion_input):
        """
        Forward pass.

        Args:
            fusion_input: Tensor [batch, 20] - fusion context

        Returns:
            action_logits: Tensor [batch, 6] - final action distribution
            rl_trust: Tensor [batch, 1] - how much to trust RL (0-1)
            llm_trust: Tensor [batch, 1] - how much to trust LLM (0-1)
        """
        features = self.encoder(fusion_input)

        action_logits = self.action_head(features)
        rl_trust = torch.sigmoid(self.rl_trust_head(features))
        llm_trust = torch.sigmoid(self.llm_trust_head(features))

        return action_logits, rl_trust, llm_trust

    def predict_action(self, fusion_input, deterministic=True):
        """
        Predict final action.

        Args:
            fusion_input: Numpy array [20] or Tensor [batch, 20]
            deterministic: If True, return argmax. If False, sample.

        Returns:
            action: int (0-5)
            rl_trust: float (0-1)
            llm_trust: float (0-1)
        """
        if isinstance(fusion_input, np.ndarray):
            fusion_input = torch.from_numpy(fusion_input).float()
        else:
            fusion_input = fusion_input.float()

        if fusion_input.dim() == 1:
            fusion_input = fusion_input.unsqueeze(0)

        device = next(self.encoder.parameters()).device
        fusion_input = fusion_input.to(device)

        with torch.no_grad():
            action_logits, rl_trust, llm_trust = self.forward(fusion_input)

            if deterministic:
                action = torch.argmax(action_logits, dim=1).item()
            else:
                probs = torch.softmax(action_logits, dim=1)
                action = torch.multinomial(probs, 1).item()

        return action, rl_trust.item(), llm_trust.item()


class FusionExperienceBuffer:
    """
    Buffer for storing fusion decisions and outcomes.

    Used to train fusion network via supervised learning from outcomes.
    """

    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self,
            fusion_input: np.ndarray,
            rl_action: int,
            llm_action: int,
            final_action: int,
            outcome_reward: float):
        """
        Add experience.

        Args:
            fusion_input: np.array [20] - context when decision made
            rl_action: int - RL agent's proposed action
            llm_action: int - LLM advisor's proposed action
            final_action: int - action executed after fusion/risk veto
            outcome_reward: float - reward received (for weighting)
        """
        self.buffer.append({
            'input': fusion_input.astype(np.float32),
            'rl_action': int(rl_action),
            'llm_action': int(llm_action),
            'final_action': int(final_action),
            'reward': float(outcome_reward)
        })

    def sample(self, batch_size=256) -> Dict[str, torch.Tensor]:
        """
        Sample batch for training.

        Returns:
            Dictionary of tensors containing inputs, actions and metadata.
        """
        current_size = len(self.buffer)
        if current_size == 0:
            raise ValueError("FusionExperienceBuffer is empty.")

        sample_size = min(batch_size, current_size)
        indices = np.random.choice(current_size, sample_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        inputs = torch.FloatTensor([exp['input'] for exp in batch])
        final_actions = torch.LongTensor([exp['final_action'] for exp in batch])
        rl_actions = torch.LongTensor([exp['rl_action'] for exp in batch])
        llm_actions = torch.LongTensor([exp['llm_action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])

        # Weight samples by absolute outcome magnitude (ensure a minimum weight)
        weights = torch.clamp(rewards.abs(), 0.0, 10.0) / 10.0 + 0.1

        return {
            'inputs': inputs,
            'final_actions': final_actions,
            'rl_actions': rl_actions,
            'llm_actions': llm_actions,
            'rewards': rewards,
            'weights': weights
        }

    def __len__(self):
        return len(self.buffer)


class FusionTrainer:
    """
    Trainer for fusion network.

    Uses supervised learning from successful trading outcomes.
    """

    def __init__(self, fusion_network, learning_rate=3e-4, device='auto'):
        # Handle 'auto' device selection
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.network = fusion_network.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.action_criterion = nn.CrossEntropyLoss(reduction='none')  # Per-sample loss
        self.trust_criterion = nn.BCELoss(reduction='none')

    def train_step(self, batch: Dict[str, torch.Tensor]):
        """
        Single training step using a batch sampled from the experience buffer.

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        inputs = batch['inputs'].to(self.device)
        final_actions = batch['final_actions'].to(self.device)
        rl_actions = batch['rl_actions'].to(self.device)
        llm_actions = batch['llm_actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        weights = batch['weights'].to(self.device)

        self.optimizer.zero_grad()

        # Forward pass
        action_logits, rl_trust, llm_trust = self.network(inputs)

        # Action loss
        action_loss_per_sample = self.action_criterion(action_logits, final_actions)
        action_loss = (action_loss_per_sample * weights).mean()

        # Trust targets derived from outcome and which advisor supplied the final action
        reward_scale = torch.tanh(rewards / 50.0)  # normalize reward magnitude to [-1, 1]
        same_rl = (final_actions == rl_actions).float()
        same_llm = (final_actions == llm_actions).float()

        target_rl = 0.5 + 0.5 * reward_scale * (same_rl - same_llm)
        target_llm = 0.5 + 0.5 * reward_scale * (same_llm - same_rl)
        target_rl = torch.clamp(target_rl, 0.0, 1.0).unsqueeze(1)
        target_llm = torch.clamp(target_llm, 0.0, 1.0).unsqueeze(1)

        trust_weight = weights.unsqueeze(1)
        rl_trust_loss = (self.trust_criterion(rl_trust, target_rl) * trust_weight).mean()
        llm_trust_loss = (self.trust_criterion(llm_trust, target_llm) * trust_weight).mean()
        trust_loss = 0.5 * (rl_trust_loss + llm_trust_loss)

        total_loss = action_loss + trust_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            predictions = torch.argmax(action_logits, dim=1)
            action_accuracy = (predictions == final_actions).float().mean().item()
            rl_trust_error = torch.abs(rl_trust - target_rl).mean().item()
            llm_trust_error = torch.abs(llm_trust - target_llm).mean().item()

        metrics = {
            'action_accuracy': action_accuracy,
            'action_loss': action_loss.item(),
            'trust_loss': trust_loss.item(),
            'rl_trust_error': rl_trust_error,
            'llm_trust_error': llm_trust_error
        }

        return total_loss.item(), metrics
