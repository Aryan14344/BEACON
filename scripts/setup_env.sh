#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  PhishBP — Environment Setup using UV (Python 3.12 + PyTorch cu121)
#  Run this in your terminal:
#    bash /home/aryan/ML_DS_Pro/phishbp/scripts/setup_env.sh
# ═══════════════════════════════════════════════════════════════
set -e

cd /home/aryan/ML_DS_Pro/phishbp

echo "▶ Step 1: Cleaning previous broken venv"
rm -rf .venv

echo "▶ Step 2: Creating clean Python 3.12 venv with uv"
uv venv --python 3.12 .venv

PYTHON="/home/aryan/ML_DS_Pro/phishbp/.venv/bin/python"
PIP="/home/aryan/ML_DS_Pro/phishbp/.venv/bin/pip"

echo "▶ Step 3: Installing PyTorch 2.3.0 + cu121 into Python 3.12 venv"
uv pip install \
    --python "$PYTHON" \
    torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo "▶ Step 4: Verifying PyTorch GPU access"
$PYTHON -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:', torch.cuda.get_device_name(0))
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('WARNING: CUDA is not available to PyTorch.')
"

echo "▶ Step 5: Installing PyTorch Geometric & dependencies"
uv pip install \
    --python "$PYTHON" \
    torch-geometric

uv pip install \
    --python "$PYTHON" \
    torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html || echo "Note: optional binary wheels install skipped, pure PyG works."

echo "▶ Step 6: Installing remaining ML/DS libraries"
uv pip install \
    --python "$PYTHON" \
    numpy scipy scikit-learn pandas networkx \
    lightgbm optuna shap \
    dnspython aiohttp tldextract publicsuffixlist \
    python-whois tqdm regex \
    fastapi uvicorn pydantic \
    matplotlib seaborn plotly \
    wandb hydra-core omegaconf \
    pytest pytest-asyncio

echo "▶ Step 7: Running Stage 1 validation"
$PYTHON scripts/validate_bp_numpy.py

echo "▶ Step 8: Running full test suite"
$PYTHON -m pytest tests/ -v

echo ""
echo "════════════════════════════════════════════════════"
echo "  All set! To activate your environment:"
echo "    source /home/aryan/ML_DS_Pro/phishbp/.venv/bin/activate"
echo "════════════════════════════════════════════════════"
