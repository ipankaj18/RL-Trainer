#!/usr/bin/env python3
"""Export JAX checkpoint to CPU-compatible pickle format."""
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax
print('JAX devices:', jax.devices())

from flax.training import checkpoints
import pickle
import numpy as np
from pathlib import Path

base_dir = Path('/home/javlo/Code Projects/RL Trainner & Executor System/AI Trainer')
checkpoint_path = str(base_dir / 'models/phase2_jax_nq/phase2_jax_final_3051')
output_path = str(base_dir / 'Agent_temp/model/phase2_jax_nq/params_cpu.pkl')

print(f'Loading from: {checkpoint_path}')

try:
    restored = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)
    if restored:
        print('Loaded! Type:', type(restored))
        params = restored.get('params', restored)
        
        def to_numpy(x):
            if hasattr(x, 'numpy'): 
                return np.array(x)
            elif isinstance(x, dict): 
                return {k: to_numpy(v) for k, v in x.items()}
            else: 
                return x
        
        numpy_params = to_numpy(params)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(numpy_params, f)
        print(f'✅ Saved to: {output_path}')
        print(f'   Size: {os.path.getsize(output_path)} bytes')
    else:
        print('❌ Failed to load')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
