# JAX Setup Guide

This project pins the JAX migration stack to the 0.4.35 series **for GPU use only**. Install inside a clean virtual environment on Linux/WSL with CUDA 12 available; no CPU-only path is supported.

## GPU-only install (Linux/WSL with CUDA 12)
1. Create and activate a venv (bash in WSL):  
   `python3 -m venv .venv-jax-gpu && source .venv-jax-gpu/bin/activate`
2. Upgrade pip: `python -m pip install --upgrade pip`
3. Install the GPU stack (requirements include the CUDA find-links):  
   `python -m pip install -r src/jax_migration/requirements_jax.txt`
4. Set `CUDA_ROOT` so JAX can find the bundled CUDA toolkit from the pip wheels:  
   `export CUDA_ROOT="$(pwd)/.venv-jax-gpu/lib/python3.12/site-packages/nvidia/cuda_runtime"`  
   (Add this to your shell/activate script for persistence.)
5. Validate GPU pickup:  
   `python - <<'PY'\nimport jax\nprint(jax.__version__, jax.devices())\nPY`  
   Expect to see a `CudaDevice` in the output.

## Shell notes
- Run these commands from bash inside WSL so the Linux CUDA wheels are resolved.
- If PowerShell is needed, use `wsl.exe` to enter bash, then run the commands above (PowerShell uses `;` as a separator, Bash uses `&&`).
- Always stay inside the dedicated GPU venv to avoid conflicts with global packages that may require newer JAX versions.

## Checkpoint path compatibility (Windows/WSL)
- Orbax/TensorStore cannot read or write Windows UNC paths like `\\wsl.localhost\Ubuntu\home\...`.
- Run training/evaluation from inside WSL (bash) so checkpoint paths look like `/home/...`.
- If you must run from PowerShell, pass a Windows-native checkpoint directory (e.g., `C:\Users\<user>\AppData\Local\Temp`) via flags such as `--checkpoint-dir` in `src/jax_migration/evaluate_phase2_jax.py`.
