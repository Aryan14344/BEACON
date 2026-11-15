#!/usr/bin/env bash
# ============================================================
# PhishBP — One-shot environment setup for Arch Linux
# RTX 2050 / CUDA 13.4 driver / CUDA 12.1 toolkit target
#
# Run this ONCE from your terminal:
#   bash /home/aryan/ML_DS_Pro/phishbp/scripts/setup_env.sh
# ============================================================

set -e
VENV=/home/aryan/ML_DS_Pro/phishbp/.venv
PIP=$VENV/bin/pip
PYTHON=$VENV/bin/python

echo "==> Creating virtual environment at $VENV"
python3 -m venv $VENV --prompt phishbp

echo "==> Upgrading pip"
$PIP install --upgrade pip setuptools wheel

echo "==> Installing PyTorch 2.3.0 + CUDA 12.1"
$PIP install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing PyTorch Geometric"
$PIP install torch-geometric

echo "==> Installing PyG optional deps (scatter, sparse) for CUDA 12.1"
$PIP install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

echo "==> Installing project requirements"
$PIP install -r /home/aryan/ML_DS_Pro/phishbp/requirements.txt

echo "==> Verifying installation"
$PYTHON -c "
import torch, torch_geometric
print(f'torch:        {torch.__version__}')
print(f'CUDA avail:   {torch.cuda.is_available()}')
print(f'GPU:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
print(f'VRAM:         {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB' if torch.cuda.is_available() else '')
print(f'torch_geometric: {torch_geometric.__version__}')
print('Setup complete ✓')
"

echo ""
echo "==> To activate the venv in your shell:"
echo "    source $VENV/bin/activate"
echo ""
echo "==> To run Stage 1 tests:"
echo "    cd /home/aryan/ML_DS_Pro/phishbp"
echo "    source .venv/bin/activate"
echo "    pytest tests/ -v"
