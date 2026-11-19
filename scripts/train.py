"""
PhishBP — Standalone Training Script for H3 Hybrid Model
=========================================================
Trains the PhishBPHybridH3 model (Encoder + Differentiable BP Layer)
on the processed heterogeneous graph and saves the checkpoint.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 80 --lr 0.005 --batch_size 128
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def extract_unified_graph(data):
    """
    Extracts unified homogeneous edge tensors and maps node features.
    """
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

    labels = torch.full((total_nodes,), -1, dtype=torch.long)
    for nt in node_types:
        if nt in data.node_types and hasattr(data[nt], 'y'):
            labels[offsets[nt]: offsets[nt] + data[nt].num_nodes] = data[nt].y

    url_x = data['url'].x
    in_dim = url_x.shape[1]
    features = torch.zeros((total_nodes, in_dim))
    features[offsets['url']: offsets['url'] + data['url'].num_nodes] = url_x

    return edge_index, edge_type, labels, features, offsets, total_nodes, len(rel_map)


def main():
    parser = argparse.ArgumentParser(description="Train PhishBP H3 Model")
    parser.add_argument("--data_path", default="data/processed/hetero_graph.pt", help="Path to processed graph")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Encoder hidden dimension")
    parser.add_argument("--bp_iters", type=int, default=5, help="Number of unrolled BP message passing iterations")
    parser.add_argument("--damping", type=float, default=0.5, help="BP message damping factor")
    parser.add_argument("--output_path", default="checkpoints/h3_trained.pt", help="Checkpoint save path")
    args = parser.parse_args()

    graph_file = Path(args.data_path)
    if not graph_file.exists():
        logger.error(f"Graph file not found at {graph_file}. Run scripts/run_data_collection.py first.")
        sys.exit(1)

    logger.info(f"Loading graph from {graph_file}...")
    data = torch.load(graph_file, weights_only=False)
    edge_index, edge_type, labels, features, offsets, total_nodes, num_rel = extract_unified_graph(data)

    url_start = offsets['url']
    url_count = data['url'].num_nodes
    url_indices = np.arange(url_start, url_start + url_count)

    # 75% train, 25% test split
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(url_indices))
    n_train = int(len(url_indices) * 0.75)
    train_idx = url_indices[perm[:n_train]]
    test_idx = url_indices[perm[n_train:]]

    y_train = labels[train_idx]
    y_test = labels[test_idx].numpy()

    logger.info(f"Dataset summary:")
    logger.info(f"  Total graph nodes: {total_nodes} (URLs: {url_count}, Domains: {offsets.get('ip', 0) - offsets.get('domain', 0)})")
    logger.info(f"  Total edges: {edge_index.shape[1]}")
    logger.info(f"  Training URLs: {len(train_idx)} | Testing URLs: {len(test_idx)}")
    logger.info(f"  Class balance: {int((y_train == 1).sum())} phish / {int((y_train == 0).sum())} benign in train")

    # Instantiate model
    in_dim = features.shape[1]
    model = PhishBPHybridH3(
        in_features=in_dim,
        hidden_dim=args.hidden_dim,
        num_relations=num_rel,
        bp_iters=args.bp_iters,
        damping=args.damping,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Move tensors to device
    x_dev = features.to(DEVICE)
    ei_dev = edge_index.to(DEVICE)
    et_dev = edge_type.to(DEVICE)
    train_idx_t = torch.tensor(train_idx, device=DEVICE)
    y_train_t = y_train.to(DEVICE)

    logger.info("\nStarting training loop...")
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass: Encoder -> Log-Priors -> Differentiable BP Message Passing
        log_beliefs = model(x_dev, ei_dev, et_dev)
        loss = F.nll_loss(log_beliefs[train_idx_t], y_train_t)

        loss.backward()
        optimizer.step()

        # Evaluate periodically
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                test_beliefs = model(x_dev, ei_dev, et_dev)
                probs = test_beliefs.exp()[:, 1].cpu().numpy()
                test_probs = probs[test_idx]

                metrics = evaluate(y_true=y_test, y_score=test_probs, method="H3_Train")
                auc = metrics.get("roc_auc", 0.0)
                pr_auc = metrics.get("pr_auc", 0.0)

                logger.info(
                    f"Epoch [{epoch:02d}/{args.epochs:02d}]  "
                    f"Loss: {loss.item():.4f}  |  "
                    f"Val ROC-AUC: {auc:.4f}  |  "
                    f"Val PR-AUC: {pr_auc:.4f}"
                )

                if auc > best_auc:
                    best_auc = auc

    # Save model checkpoint
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    logger.info(f"\n✓ Training complete! Best Val ROC-AUC: {best_auc:.4f}")
    logger.info(f"✓ Saved trained model checkpoint to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
