
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from environment_phase1 import TradingEnvironmentPhase1

def verify_action_masking():
    print("Verifying Action Masking Fix...")
    
    # Create dummy data (outside RTH)
    dates = pd.date_range(start="2023-01-01 04:00:00", periods=100, freq="1min", tz="America/New_York")
    data = pd.DataFrame({
        'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000,
        'sma_5': 100.0, 'sma_20': 100.0, 'rsi': 50.0, 'macd': 0.0, 'momentum': 0.0, 'atr': 1.0
    }, index=dates)
    
    # Initialize environment with disable_rth_check=True
    env = TradingEnvironmentPhase1(
        data=data,
        disable_rth_check=True
    )
    
    # Reset
    obs, info = env.reset()
    
    # Check action mask
    mask = env.action_masks() # This method exists in environment_phase2.py but phase1 might not have it exposed directly?
    # Wait, Phase 1 environment doesn't have action_masks() method in the snippet I saw?
    # Let me check environment_phase1.py again. 
    # It has _compute_rth_mask but maybe not action_masks method exposed for PPO?
    # Actually, MaskablePPO requires action_masks(). 
    # Phase 1 env usually implements it or it's added by a wrapper.
    # In environment_phase2.py it was explicit.
    # Let's check if Phase 1 has it.
    
    # If Phase 1 doesn't have it, I should check _rth_mask directly.
    
    rth_mask_val = env._rth_mask[env.current_step]
    print(f"RTH Mask Value at step {env.current_step} (04:00 AM): {rth_mask_val}")
    
    if rth_mask_val:
        print("✅ PASS: RTH check disabled, trading allowed at 04:00 AM.")
    else:
        print("❌ FAIL: RTH check still blocking trading at 04:00 AM.")

if __name__ == "__main__":
    verify_action_masking()
