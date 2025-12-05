
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.train_phase2 import apply_smallworld_rewiring

import gymnasium as gym
from gymnasium import spaces

def create_dummy_model(action_space_size, device='cpu'):
    """Create a dummy MaskablePPO model."""
    class DummyEnv(gym.Env):
        def __init__(self, n_actions):
            self.action_space = spaces.Discrete(n_actions)
            self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
            self.num_envs = 1
        
        def reset(self, seed=None, options=None):
            return np.zeros(10, dtype=np.float32), {}
            
        def step(self, action):
            return np.zeros(10, dtype=np.float32), 0.0, False, False, {}
            
        def action_masks(self):
            return np.ones(self.action_space.n, dtype=bool)
    
    env = DummyEnv(action_space_size)
    
    model = MaskablePPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(net_arch=[32, 32]),
        device=device,
        verbose=0
    )
    return model

def test_transfer_logic():
    print("="*60)
    print("TESTING TRANSFER LOGIC")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. Create Phase 1 model (3 actions)
    print("\n[1] Creating Phase 1 model (3 actions)...")
    p1_model = create_dummy_model(3, device)
    
    # Set specific weights for action head to verify transfer
    # Action net is the last layer. For MlpPolicy, it's usually model.policy.action_net
    # But in SB3 it might be wrapped. Let's check.
    print(f"Phase 1 action net type: {type(p1_model.policy.action_net)}")
    
    with torch.no_grad():
        # Set weights to a known pattern: 1.0 for all
        p1_model.policy.action_net.weight.fill_(1.0)
        p1_model.policy.action_net.bias.fill_(0.5)
    
    print("Phase 1 action weights set to 1.0, bias to 0.5")

    # 2. Create Phase 2 model (6 actions)
    print("\n[2] Creating Phase 2 model (6 actions)...")
    p2_model = create_dummy_model(6, device)
    
    # Initialize with zeros to verify transfer
    with torch.no_grad():
        p2_model.policy.action_net.weight.fill_(0.0)
        p2_model.policy.action_net.bias.fill_(0.0)
        
    print("Phase 2 action weights initialized to 0.0")

    # 3. Simulate Transfer Logic (Copying what we plan to implement)
    print("\n[3] Executing Transfer Logic...")
    
    # --- IMPLEMENTATION OF PROPOSED FIX ---
    with torch.no_grad():
        # Get action nets
        p1_action_net = p1_model.policy.action_net
        p2_action_net = p2_model.policy.action_net
        
        # Verify shapes
        print(f"P1 Action Net: {p1_action_net.weight.shape} (out, in)")
        print(f"P2 Action Net: {p2_action_net.weight.shape} (out, in)")
        
        # Transfer common actions (0, 1, 2)
        # Indices: 0=Hold, 1=Buy, 2=Sell
        common_indices = [0, 1, 2]
        
        # Copy weights for common indices
        # Shape is (n_actions, hidden_dim)
        p2_action_net.weight[common_indices, :] = p1_action_net.weight[common_indices, :].clone()
        p2_action_net.bias[common_indices] = p1_action_net.bias[common_indices].clone()
        
        print(f"Transferred weights for indices {common_indices}")
        
    # --------------------------------------

    # 4. Verify Results
    print("\n[4] Verifying Transfer...")
    
    p2_weights = p2_model.policy.action_net.weight
    p2_bias = p2_model.policy.action_net.bias
    
    # Check common actions (should be 1.0 and 0.5)
    common_weights_mean = p2_weights[0:3].mean().item()
    common_bias_mean = p2_bias[0:3].mean().item()
    
    print(f"Common actions (0-2) mean weight: {common_weights_mean:.4f} (Expected: 1.0000)")
    print(f"Common actions (0-2) mean bias:   {common_bias_mean:.4f} (Expected: 0.5000)")
    
    # Check new actions (should be 0.0)
    new_weights_mean = p2_weights[3:].mean().item()
    new_bias_mean = p2_bias[3:].mean().item()
    
    print(f"New actions (3-5) mean weight:    {new_weights_mean:.4f} (Expected: 0.0000)")
    print(f"New actions (3-5) mean bias:      {new_bias_mean:.4f} (Expected: 0.0000)")
    
    if np.isclose(common_weights_mean, 1.0) and np.isclose(new_weights_mean, 0.0):
        print("\nSUCCESS: Partial weight transfer verified!")
    else:
        print("\nFAILURE: Weight transfer incorrect.")

if __name__ == "__main__":
    test_transfer_logic()
