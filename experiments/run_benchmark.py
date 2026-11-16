"""
Comprehensive Benchmark Suite for PhishBP:
Trains and compares all methods across evaluation splits:
  1. Feature-only: LightGBM
  2. Pure Graph Inference: Loopy BP (with seed priors)
  3. GNN Baselines: GraphSAGE, HGT
  4. Hybrids:
     - H1: GNN Priors + Loopy BP Refinement
     - H2: Correct-and-Smooth (C&S)
     - H3: PhishBPHybridH3 (Learnable BP layer trained end-to-end)

Runs multiple seeds, computes all metrics (ROC-AUC, PR-AUC, TPR@0.1%FPR, ECE),
and prints the formal comparison table.
"""

import sys
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.bp_baseline import LoopyBP, BPGraph, homophily_potential
from src.models.gnn_baselines import GraphSAGEModel, HeteroHGT
from src.models.bp_layer import PhishBPHybridH3
from src.models.hybrids import H1Hybrid, CorrectAndSmooth
from src.eval.metrics import evaluate, aggregate_results, compare_methods

import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Benchmark")


def extract_unified_graph(data: HeteroData):
    """
    Convert HeteroData into a unified homogeneous indexed graph with relation types.
    Maps all nodes (url, domain, ip, nameserver) to global integer IDs [0, Total_N).
    """
    node_types = ['url', 'domain', 'ip', 'nameserver']
    offsets = {}
    total_nodes = 0
    for nt in node_types:
        offsets[nt] = total_nodes
        total_nodes += data[nt].num_nodes

    all_src = []
    all_dst = []
    all_etype = []

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

    edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)
    edge_type = torch.cat(all_etype)

    # Collect labels and URL features
    labels = torch.full((total_nodes,), -1, dtype=torch.long)
    for nt in node_types:
        labels[offsets[nt]: offsets[nt] + data[nt].num_nodes] = data[nt].y

    # Node features: URL nodes get 21-dim features, others get zero-filled or projected embeddings
    url_x = data['url'].x
    in_dim = url_x.shape[1]
    features = torch.zeros((total_nodes, in_dim))
    features[offsets['url']: offsets['url'] + data['url'].num_nodes] = url_x

    return edge_index, edge_type, labels, features, offsets, total_nodes, len(rel_map)


def run_benchmark(data_path="data/processed/hetero_graph.pt", n_seeds=3, epochs=60):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using compute device: {device}")

    data = torch.load(data_path, weights_only=False)
    edge_index, edge_type, labels, features, offsets, total_nodes, num_rel = extract_unified_graph(data)

    url_start = offsets['url']
    url_count = data['url'].num_nodes
    url_indices = np.arange(url_start, url_start + url_count)
    url_labels = labels[url_indices].numpy()

    # Create stratified temporal-like train/test splits over seeds
    all_results = {
        "LightGBM (Feature-only)": [],
        "Loopy BP (Pure Graph)": [],
        "GraphSAGE": [],
        "H1 Hybrid (GNN+BP)": [],
        "H2 Hybrid (C&S)": [],
        "H3 Hybrid (Learnable BP end-to-end)": [],
    }

    for seed in range(n_seeds):
        logger.info(f"\n--- Running Seed {seed + 1}/{n_seeds} ---")
        rng = np.random.default_rng(seed + 100)
        
        # 70% train, 30% test on URL nodes
        perm = rng.permutation(len(url_indices))
        n_train = int(len(url_indices) * 0.7)
        train_idx = url_indices[perm[:n_train]]
        test_idx = url_indices[perm[n_train:]]

        y_train = labels[train_idx].numpy()
        y_test = labels[test_idx].numpy()
        X_train = features[train_idx].numpy()
        X_test = features[test_idx].numpy()

        # -------------------------------------------------------------
        # 1. Feature-only: LightGBM
        # -------------------------------------------------------------
        lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=seed, verbose=-1)
        lgb_model.fit(X_train, y_train)
        lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
        all_results["LightGBM (Feature-only)"].append(
            evaluate(y_true=y_test, y_score=lgb_probs, method="LightGBM", window=0, seed=seed)
        )

        # -------------------------------------------------------------
        # 2. Pure Graph Inference: Loopy BP with Seed Priors
        # -------------------------------------------------------------
        bp_graph_potentials = [homophily_potential(strength=2.0) for _ in range(num_rel)]
        log_priors = torch.zeros(total_nodes, 2)
        # Set seeds
        conf = 10.0
        for idx in train_idx:
            lbl = labels[idx].item()
            if lbl == 1:
                log_priors[idx] = torch.tensor([-conf, conf])
            else:
                log_priors[idx] = torch.tensor([conf, -conf])

        bp_graph = BPGraph(
            num_nodes=total_nodes,
            edge_index=edge_index,
            edge_type=edge_type,
            log_priors=log_priors,
            num_rel=num_rel,
            potentials=bp_graph_potentials
        )
        bp = LoopyBP(max_iter=15, damping=0.5)
        bp_all_probs = bp.infer(bp_graph).cpu().numpy()
        bp_test_probs = bp_all_probs[test_idx]
        all_results["Loopy BP (Pure Graph)"].append(
            evaluate(y_true=y_test, y_score=bp_test_probs, method="LoopyBP", window=0, seed=seed)
        )

        # -------------------------------------------------------------
        # 3. GNN Baseline: GraphSAGE
        # -------------------------------------------------------------
        sage = GraphSAGEModel(in_channels=features.shape[1], hidden_channels=32, num_layers=2).to(device)
        optimizer = torch.optim.Adam(sage.parameters(), lr=0.01, weight_decay=1e-4)
        x_dev = features.to(device)
        ei_dev = edge_index.to(device)
        train_idx_t = torch.tensor(train_idx, device=device)
        test_idx_t = torch.tensor(test_idx, device=device)
        y_train_t = labels[train_idx].float().to(device)

        sage.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = sage(x_dev, ei_dev)
            loss = F.binary_cross_entropy_with_logits(logits[train_idx_t], y_train_t)
            loss.backward()
            optimizer.step()

        sage.eval()
        with torch.no_grad():
            sage_all_logits = sage(x_dev, ei_dev)
            sage_all_probs = torch.sigmoid(sage_all_logits).cpu().numpy()
            sage_test_probs = sage_all_probs[test_idx]

        all_results["GraphSAGE"].append(
            evaluate(y_true=y_test, y_score=sage_test_probs, method="GraphSAGE", window=0, seed=seed)
        )

        # -------------------------------------------------------------
        # 4. H1 Hybrid: GNN Logits -> Loopy BP Refinement
        # -------------------------------------------------------------
        h1 = H1Hybrid(num_relations=num_rel, max_iter=12, damping=0.5, strength=1.5)
        # Convert GNN logits into 2-dim log-priors
        gnn_probs_2d = torch.stack([1.0 - torch.sigmoid(sage_all_logits), torch.sigmoid(sage_all_logits)], dim=-1)
        gnn_log_priors = torch.log(gnn_probs_2d + 1e-8).cpu()
        # Clamp seed training nodes to hard priors
        for idx in train_idx:
            lbl = labels[idx].item()
            gnn_log_priors[idx] = torch.tensor([-conf, conf]) if lbl == 1 else torch.tensor([conf, -conf])

        h1_probs = h1.infer(gnn_log_priors, edge_index, edge_type).cpu().numpy()
        all_results["H1 Hybrid (GNN+BP)"].append(
            evaluate(y_true=y_test, y_score=h1_probs[test_idx], method="H1_Hybrid", window=0, seed=seed)
        )

        # -------------------------------------------------------------
        # 5. H2 Hybrid: Correct and Smooth (C&S)
        # -------------------------------------------------------------
        cs = CorrectAndSmooth(num_correct_layers=2, num_smooth_layers=2)
        base_probs_t = gnn_probs_2d.cpu()
        train_mask_t = torch.zeros(total_nodes, dtype=torch.bool)
        train_mask_t[train_idx] = True
        cs_smoothed = cs.post_process(base_probs_t, labels, train_mask_t, edge_index)
        cs_test_probs = cs_smoothed[test_idx, 1].numpy()
        all_results["H2 Hybrid (C&S)"].append(
            evaluate(y_true=y_test, y_score=cs_test_probs, method="H2_CS", window=0, seed=seed)
        )

        # -------------------------------------------------------------
        # 6. Novel Core Method H3: End-to-End Learnable BP Layer
        # -------------------------------------------------------------
        h3_model = PhishBPHybridH3(
            in_features=features.shape[1],
            hidden_dim=32,
            num_relations=num_rel,
            bp_iters=4,
            damping=0.5
        ).to(device)
        h3_opt = torch.optim.Adam(h3_model.parameters(), lr=0.008, weight_decay=1e-4)
        et_dev = edge_type.to(device)
        y_train_long = labels[train_idx].to(device)

        h3_model.train()
        for epoch in range(epochs):
            h3_opt.zero_grad()
            log_beliefs = h3_model(x_dev, ei_dev, et_dev)
            loss = F.nll_loss(log_beliefs[train_idx_t], y_train_long)
            loss.backward()
            h3_opt.step()

        h3_model.eval()
        with torch.no_grad():
            h3_beliefs = h3_model(x_dev, ei_dev, et_dev)
            h3_all_probs = h3_beliefs.exp()[:, 1].cpu().numpy()
            h3_test_probs = h3_all_probs[test_idx]

        all_results["H3 Hybrid (Learnable BP end-to-end)"].append(
            evaluate(y_true=y_test, y_score=h3_test_probs, method="H3_Hybrid", window=0, seed=seed)
        )

    # Formal comparison table
    logger.info("\n" + "="*80)
    logger.info("FINAL BENCHMARK RESULTS (Across Multiple Seeds)")
    logger.info("="*80)
    compare_methods(all_results, primary_metric="roc_auc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/processed/hetero_graph.pt")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    run_benchmark(data_path=args.data_path, n_seeds=args.seeds, epochs=args.epochs)
