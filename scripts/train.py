"""
PhishBP — Large-Scale Mini-Batch Training Script for H3 Hybrid
==============================================================
Trains the PhishBPHybridH3 model (Feature Encoder + Learnable BP Layer)
using GPU mini-batches across large-scale datasets (100,000+ URLs).

Features:
  - Mini-batch DataLoader with configurable batch size (e.g. 512, 1024)
  - Multi-batch GPU gradient descent on RTX 2050
  - Live epoch and batch progress tracking with running loss
  - Validation metrics (ROC-AUC, PR-AUC, Accuracy) every epoch
  - Saves best checkpoint to checkpoints/h3_trained.pt

Usage:
  python scripts/train.py --epochs 25 --batch_size 512 --lr 0.003
  python scripts/train.py --data_path data/processed/large_hetero_graph.pt
"""

import os
import sys
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.bp_layer import PhishBPHybridH3
from src.eval.metrics import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("TrainH3")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(data_path: str):
    """
    Loads either the large-scale graph or unified dataset.
    """
    path = Path(data_path)
    if not path.exists():
        fallback = Path("data/processed/hetero_graph.pt")
        if fallback.exists():
            logger.warning(f"Specified path {path} not found. Falling back to {fallback}.")
            path = fallback
        else:
            logger.error(f"No dataset found at {path}. Run scripts/download_large_dataset.py first.")
            sys.exit(1)

    logger.info(f"Loading dataset from: {path} (size: {path.stat().st_size / 1e6:.2f} MB)...")
    data = torch.load(path, weights_only=False)

    if hasattr(data, 'edge_types') and ('url', 'co_located', 'url') in data.edge_types:
        # Large graph format
        features = data['url'].x
        labels = data['url'].y
        edge_index = data[('url', 'co_located', 'url')].edge_index
        edge_type = data[('url', 'co_located', 'url')].edge_type
        num_rel = 2
    elif hasattr(data, 'node_types') and 'url' in data.node_types:
        # Standard HeteroData format
        node_types = ['url', 'domain', 'ip', 'nameserver']
        offsets = {}
        total_nodes = 0
        for nt in node_types:
            if nt in data.node_types:
                offsets[nt] = total_nodes
                total_nodes += data[nt].num_nodes
            else:
                offsets[nt] = total_nodes

        all_src, all_dst, all_etype = [], [], []
        rel_map = {
            ('url', 'resolves_to', 'domain'): 0,
            ('domain', 'resolves_to', 'ip'): 1,
            ('domain', 'uses_ns', 'nameserver'): 2,
        }
        for etype, rel_id in rel_map.items():
            if etype in data.edge_types:
                ei = data[etype].edge_index
                src_t, _, dst_t = etype
                all_src.append(ei[0] + offsets[src_t])
                all_dst.append(ei[1] + offsets[dst_t])
                all_etype.append(torch.full((ei.shape[1],), rel_id, dtype=torch.long))

        if all_src:
            edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)
            edge_type = torch.cat(all_etype)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)

        url_x = data['url'].x
        features = url_x
        labels = data['url'].y
        num_rel = 3
    else:
        raise ValueError(f"Unrecognized data format in {path}")

    return features, labels, edge_index, edge_type, num_rel


def main():
    parser = argparse.ArgumentParser(description="Large-Scale Mini-Batch Training for PhishBP")
    parser.add_argument("--data_path", default="data/processed/large_hetero_graph.pt",
                        help="Path to graph dataset (will auto-fallback to hetero_graph.pt if not built)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Mini-batch size for GPU training")
    parser.add_argument("--lr", type=float, default=0.003, help="Adam learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Encoder hidden dimension")
    parser.add_argument("--bp_iters", type=int, default=4, help="Unrolled BP iterations")
    parser.add_argument("--output_path", default="checkpoints/h3_trained.pt", help="Checkpoint save path")
    args = parser.parse_args()

    # Load data
    features, labels, edge_index, edge_type, num_rel = load_dataset(args.data_path)
    N = features.shape[0]
    in_dim = features.shape[1]

    # Split 80% train, 20% validation
    rng = np.random.default_rng(42)
    indices = rng.permutation(N)
    n_train = int(N * 0.8)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]

    logger.info("=" * 70)
    logger.info("PHISHBP LARGE-SCALE TRAINING INITIALIZED")
    logger.info("=" * 70)
    logger.info(f"Compute Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    logger.info(f"Dataset Size:   {N:,} total samples ({int((labels == 1).sum()):,} phishing, {int((labels == 0).sum()):,} benign)")
    logger.info(f"Training Set:   {len(train_idx):,} samples | Validation Set: {len(val_idx):,} samples")
    logger.info(f"Batch Size:     {args.batch_size} ({len(train_idx) // args.batch_size + 1} mini-batches per epoch)")
    logger.info(f"Total Epochs:   {args.epochs} ({args.epochs * (len(train_idx) // args.batch_size + 1):,} gradient updates)")
    logger.info("=" * 70)

    # Mini-batch DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False)

    # Initialize H3 Model
    model = PhishBPHybridH3(
        in_features=in_dim,
        hidden_dim=args.hidden_dim,
        num_relations=num_rel,
        bp_iters=args.bp_iters,
        damping=0.5
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.NLLLoss()

    best_val_auc = 0.0
    start_time = time.time()

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = len(train_loader)
        epoch_start = time.time()

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader, 1):
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            # Encoder produces prior logits
            log_priors = F.log_softmax(model.encoder(batch_x), dim=-1)

            # In mini-batch training, construct local neighborhood connections
            B = batch_x.size(0)
            # Efficient cyclic mini-batch adjacency for message passing
            src_b = torch.arange(0, B - 1, device=DEVICE)
            dst_b = torch.arange(1, B, device=DEVICE)
            mini_ei = torch.stack([src_b, dst_b], dim=0)
            mini_et = torch.zeros(B - 1, dtype=torch.long, device=DEVICE)

            # Pass through differentiable BP layer
            log_beliefs = model.bp_layer(log_priors, mini_ei, mini_et)
            loss = criterion(log_beliefs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % max(1, n_batches // 4) == 0 or batch_idx == n_batches:
                progress = (batch_idx / n_batches) * 100
                logger.info(
                    f"Epoch [{epoch:02d}/{args.epochs:02d}] - "
                    f"Batch [{batch_idx:03d}/{n_batches:03d}] ({progress:4.1f}%) | "
                    f"Batch Loss: {loss.item():.4f} | Running Avg: {total_loss / batch_idx:.4f}"
                )

        # Validation Phase
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(DEVICE)
                logits = model.encoder(bx)
                probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(by.numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        eval_res = evaluate(y_true=val_targets, y_score=val_preds, method="PhishBP_Large")
        val_auc = getattr(eval_res, "roc_auc", 0.0)
        val_pr = getattr(eval_res, "pr_auc", 0.0)
        epoch_sec = time.time() - epoch_start

        logger.info(
            f"─── Epoch {epoch:02d} Summary [{epoch_sec:.1f}s] ─── "
            f"Train Loss: {total_loss/n_batches:.4f} | "
            f"Val ROC-AUC: {val_auc:.4f} | Val PR-AUC: {val_pr:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            out_p = Path(args.output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_p)
            logger.info(f"  ★ New best model checkpoint saved! (Val AUC: {val_auc:.4f})")

    total_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"✓ TRAINING FINISHED in {total_time:.1f} seconds ({total_time / 60:.2f} minutes)")
    logger.info(f"✓ Best Validation ROC-AUC: {best_val_auc:.4f}")
    logger.info(f"✓ Trained checkpoint ready at: {args.output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
