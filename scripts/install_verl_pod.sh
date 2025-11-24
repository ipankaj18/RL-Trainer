#!/bin/bash
# Install verl and SofT-GRPO Stage 2 dependencies in the pod
# This script should be run inside the RunPod container

set -e  # Exit on error

echo "=========================================="
echo "Installing verl for SofT-GRPO Stage 2"
echo "=========================================="

# Navigate to project root (assuming script is in scripts/)
cd "$(dirname "$0")/.."

echo ""
echo "[1/3] Installing verl from bundled repository..."
cd SofT-GRPO-master-main/verl-0.4.x
pip install -e .
cd ../..

echo ""
echo "[2/3] Installing verl dependencies..."
pip install fastapi uvicorn openai "ray[default]>=2.10" hydra-core datasets \
    "pyarrow>=19.0.0" wandb codetiming dill liger-kernel "tensordict<=0.6.2" \
    torchdata torchvision "packaging>=20.0" pybind11 pylatexenc \
    "sglang[all]==0.4.6.post5" "torch-memory-saver>=0.0.5"

echo ""
echo "[3/3] Verifying installation..."
python -c "import verl; import fastapi; import ray; import sglang; print('✅ All dependencies installed successfully')"

echo ""
echo "=========================================="
echo "✅ verl installation complete!"
echo "=========================================="
echo ""
echo "You can now run Phase 3 Stage 2 (SofT-GRPO) training."
