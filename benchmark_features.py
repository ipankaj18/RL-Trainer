
import time
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.llm_features import LLMFeatureBuilder
from src.feature_engineering import add_market_regime_features

def create_mock_data(n_rows=10000):
    dates = pd.date_range('2024-01-01 09:30', periods=n_rows, freq='1min', tz='America/New_York')
    df = pd.DataFrame({
        'open': np.random.uniform(4000, 4100, n_rows),
        'high': np.random.uniform(4100, 4150, n_rows),
        'low': np.random.uniform(3950, 4000, n_rows),
        'close': np.random.uniform(4000, 4100, n_rows),
        'volume': np.random.randint(100, 1000, n_rows),
        'atr': np.random.uniform(10, 30, n_rows)
    }, index=dates)
    return df

class MockEnv:
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.position = 0
        self.balance = 50000
        self.peak_balance = 50000
        self.consecutive_losses = 0
        self.trade_pnl_history = []
        self.max_adverse_excursion = 0
        self.max_favorable_excursion = 0

    def _calculate_unrealized_pnl(self):
        return 0.0

def benchmark():
    print("Generating mock data...")
    df = create_mock_data(5000)
    
    print("Pre-calculating features (Vectorized)...")
    start_time = time.time()
    df = add_market_regime_features(df)
    vectorized_time = time.time() - start_time
    print(f"Feature Engineering (Vectorized) took: {vectorized_time:.4f}s")

    env = MockEnv(df)
    builder = LLMFeatureBuilder()
    base_obs = np.zeros(228, dtype=np.float32)
    
    print("\nBenchmarking LLMFeatureBuilder (Iterative)...")
    n_steps = 1000
    start_time = time.time()
    
    # Simulate 1000 steps
    for i in range(200, 200 + n_steps):
        env.current_step = i
        builder.build_enhanced_observation(env, base_obs)
        
    iterative_time = time.time() - start_time
    print(f"LLMFeatureBuilder (1000 calls) took: {iterative_time:.4f}s")
    print(f"Average time per step: {iterative_time / n_steps:.6f}s")
    
    # Projection
    steps_per_second = n_steps / iterative_time
    print(f"Steps per second (single core): {steps_per_second:.2f}")
    
    # With 64 envs, effective FPS would be steps_per_second / 64
    print(f"Estimated max FPS with 64 envs (CPU bound): {steps_per_second / 64:.2f}")

if __name__ == "__main__":
    benchmark()
