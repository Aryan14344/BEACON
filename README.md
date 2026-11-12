# PhishBP: Belief Propagation × GNN Hybrid for Phishing Detection

> **Status:** Stage 1 complete — BP math validated ✓

## Research Questions
- **RQ1**: How do BP, GNNs, and hybrids compare as edge homophily, label rate, and feature quality change?
- **RQ2**: Does a hybrid match or beat the better of the two in every regime?
- **RQ3**: How do the methods hold up under realistic graph-structure attacks?
- **RQ4** (extension): Do the answers change on page-level graphs?

## Core Novel Contribution (H3)
A `BPMessagePassingLayer` with **learnable per-relation 2×2 compatibility potentials** Ψ_r, trained end-to-end on top of a Heterogeneous Graph Transformer (HGT). Under graph-structure attacks, the model learns to down-weight compromised relation types.

## Quick Start

### Stage 1 Validation (NumPy only — no GPU needed)
```bash
cd phishbp
python3 scripts/validate_bp_numpy.py
```

### Full Setup (requires CUDA GPU)
```bash
# 1. Install PyTorch for your CUDA version (example: CUDA 12.1)
pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Run full test suite
pytest tests/ -v
```

## Project Structure
```
phishbp/
├── src/
│   ├── data/           # Graph construction, toy graph
│   ├── models/         # BP baseline, GNN baselines, H1-H3 hybrids
│   ├── attacks/        # MimicIP, MintA, URL-padding
│   ├── eval/           # Metrics, robustness curves
│   └── server/         # FastAPI inference server
├── extension/          # Chrome Manifest V3 extension (TypeScript)
├── experiments/        # Experiment A (homophily) and B (robustness)
├── paper/              # LaTeX source
└── tests/              # pytest suite
```

## Stage Progress
- [x] Stage 1: BP math validated, toy graph, unit tests
- [ ] Stage 2: Real data pipeline (PDNS-Net, PhishTank)
- [ ] Stage 3: All baselines + eval harness
- [ ] Stage 4: H3 novel method
- [ ] Stage 5: Experiment A (homophily study)
- [ ] Stage 6: Experiment B (robustness)
- [ ] Stage 7: Chrome extension + FastAPI server
- [ ] Stage 8: Paper writing
