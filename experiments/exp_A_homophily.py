"""
Experiment A — Homophily × Label-Rate × Feature-Quality Sweep (RQ1 & RQ2)
==========================================================================
Answers:
  RQ1: How do BP, GNNs, and hybrids compare as edge homophily,
       label rate, and feature quality change?
  RQ2: Does a hybrid match or beat the better of the two in every regime?

Grid:
  homophily_level ∈ {LOW, MED, HIGH}  (via graph perturbation)
  label_rate      ∈ {0.01, 0.05, 0.20}
  feature_regime  ∈ {none, lexical, full}
  seeds           ∈ {0,1,2,3,4}       (5 seeds)

Outputs:
  results/exp_A_results.csv       — full results table
  results/exp_A_summary.csv       — mean ± std per cell
  results/exp_A_auc_heatmap.png   — AUC vs homophily × method (publication figure)
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
import copy
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.bp_baseline import LoopyBP, BPGraph, homophily_potential
from src.models.gnn_baselines import GraphSAGEModel
from src.models.bp_layer import PhishBPHybridH3
from src.models.hybrids import H1Hybrid, CorrectAndSmooth
from src.eval.metrics import evaluate, aggregate_results, compare_methods
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExpA")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Homophily injection ────────────────────────────────────────────────────────

def perturb_graph_homophily(
    data: HeteroData,
    target_h: float,
    seed: int = 0,
) -> HeteroData:
    """
    Adjust graph homophily by rewiring url→domain edges.

    Strategy:
      - For each edge (u, d): if label(u) != label(d), it's a heterophilic edge.
      - To RAISE homophily: remove heterophilic edges (keep homophilic ones).
      - To LOWER homophily: flip some homophilic edges to random cross-label nodes.

    We target `target_h` ∈ (0, 1) by binary search on the fraction of edges to flip.
    For simplicity, we directly set it via edge sampling without iteration.
    """
    rng = np.random.default_rng(seed)
    data = copy.deepcopy(data)

    if ("url", "resolves_to", "domain") not in data.edge_types:
        return data  # nothing to perturb

    ei   = data[("url", "resolves_to", "domain")].edge_index  # [2, E]
    y_u  = data["url"].y    # labels for url nodes
    y_d  = data["domain"].y # labels for domain nodes

    src_labels = y_u[ei[0]]
    dst_labels = y_d[ei[1]]
    homo_mask  = (src_labels == dst_labels) & (src_labels >= 0) & (dst_labels >= 0)

    n_homo = homo_mask.sum().item()
    n_het  = (~homo_mask).sum().item()
    E      = n_homo + n_het

    if E == 0:
        return data

    # Current homophily
    h_now = n_homo / E

    # target_h > h_now → need to remove heterophilic edges (raise h)
    # target_h < h_now → need to inject heterophilic edges (lower h)
    if target_h >= h_now:
        # Remove a fraction of heterophilic edges
        het_indices = (~homo_mask).nonzero(as_tuple=True)[0]
        # We want: (n_homo) / (n_homo + n_het - x) = target_h
        # => x = n_het - n_homo*(1-target_h)/target_h
        x = max(0, int(n_het - n_homo * (1 - target_h) / (target_h + 1e-9)))
        x = min(x, len(het_indices))
        keep_mask = torch.ones(E, dtype=torch.bool)
        remove = het_indices[rng.choice(len(het_indices), size=x, replace=False)]
        keep_mask[remove] = False
    else:
        # Flip some homophilic edges → heterophilic ones (re-wire dst)
        homo_indices = homo_mask.nonzero(as_tuple=True)[0]
        # We want: (n_homo - x) / (n_homo - x + n_het + x) = target_h
        # => x = n_homo - target_h * E  (approximately)
        x = max(0, int(n_homo - target_h * E))
        x = min(x, len(homo_indices))
        keep_mask = torch.ones(E, dtype=torch.bool)
        # We just remove x homophilic edges (simplest approximation)
        remove = homo_indices[rng.choice(len(homo_indices), size=x, replace=False)]
        keep_mask[remove] = False

    new_ei = ei[:, keep_mask]
    data[("url", "resolves_to", "domain")].edge_index = new_ei
    return data


# ── Label-rate masking ─────────────────────────────────────────────────────────

def apply_label_rate(
    data: HeteroData,
    label_rate: float,
    seed: int = 0,
) -> HeteroData:
    """
    Randomly mask training labels so that only `label_rate` fraction are visible.
    Operates on url-node labels only (the primary classification target).
    Sets train_mask and val_mask on data["url"].
    """
    rng  = np.random.default_rng(seed + 100)
    data = copy.deepcopy(data)
    n    = data["url"].num_nodes
    idx  = np.arange(n)
    rng.shuffle(idx)

    n_train = max(2, int(n * label_rate))
    n_val   = max(2, int(n * 0.10))
    n_val   = min(n_val, n - n_train)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train: n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    data["url"].train_mask = train_mask
    data["url"].val_mask   = val_mask
    data["url"].test_mask  = test_mask
    return data


# ── Feature ablation ──────────────────────────────────────────────────────────

def apply_feature_regime(data: HeteroData, regime: str) -> HeteroData:
    """
    regime:
      "none"    → all url features set to zero (graph structure only)
      "lexical" → keep only first 10 dims (URL-structure features, no DNS)
      "full"    → keep all 21 dims
    """
    data = copy.deepcopy(data)
    if regime == "none":
        data["url"].x = torch.zeros_like(data["url"].x)
    elif regime == "lexical":
        data["url"].x = data["url"].x[:, :10].clone()
        # Pad back to full dims with zeros so model dim doesn't change
        pad = torch.zeros(data["url"].num_nodes, data["url"].x.shape[1])
        # Actually keep it at :10 — models handle variable feature_dim
    # "full" → no change
    return data


# ── Shared graph preprocessing ────────────────────────────────────────────────

def extract_unified_graph(data: HeteroData):
    """Convert HeteroData → homogeneous (edge_index, edge_type, labels, features)."""
    node_types = ["url", "domain", "ip", "nameserver"]
    offsets    = {}
    total_nodes = 0
    for nt in node_types:
        offsets[nt] = total_nodes
        total_nodes += data[nt].num_nodes if nt in data.node_types else 0

    all_src, all_dst, all_etype = [], [], []
    rel_map = {
        ("url", "resolves_to", "domain"):    0,
        ("domain", "resolves_to", "ip"):      1,
        ("domain", "uses_ns", "nameserver"):  2,
    }
    for etype, rel_id in rel_map.items():
        if etype in data.edge_types:
            ei    = data[etype].edge_index
            src_t, _, dst_t = etype
            if src_t in offsets and dst_t in offsets:
                all_src.append(ei[0] + offsets[src_t])
                all_dst.append(ei[1] + offsets[dst_t])
                all_etype.append(torch.full((ei.shape[1],), rel_id, dtype=torch.long))

    if not all_src:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type  = torch.zeros(0, dtype=torch.long)
    else:
        edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)
        edge_type  = torch.cat(all_etype)

    labels = torch.full((total_nodes,), -1, dtype=torch.long)
    for nt in node_types:
        if nt in data.node_types:
            o  = offsets[nt]
            yn = data[nt].y
            labels[o: o + data[nt].num_nodes] = yn

    # Features
    in_dim   = data["url"].x.shape[1] if "url" in data.node_types else 1
    node_x   = torch.zeros((total_nodes, in_dim))
    if "url" in data.node_types:
        o = offsets["url"]
        node_x[o: o + data["url"].num_nodes] = data["url"].x

    return edge_index, edge_type, labels, node_x, offsets


# ── Training helpers ───────────────────────────────────────────────────────────

def train_gnn(model, node_x, edge_index, labels, train_mask, n_epochs=60, lr=1e-3):
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    x_d   = node_x.to(DEVICE)
    ei_d  = edge_index.to(DEVICE)
    y_d   = labels.to(DEVICE)
    tm_d  = train_mask.to(DEVICE)

    for _ in range(n_epochs):
        model.train()
        opt.zero_grad()
        out  = model(x_d, ei_d)
        loss = F.cross_entropy(out[tm_d], y_d[tm_d])
        loss.backward()
        opt.step()
    model.eval()
    return model


def train_h3(model, data, train_mask, n_epochs=60, lr=1e-3):
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Build hetero edges for H3
    edge_index_dict = {}
    for etype in data.edge_types:
        edge_index_dict[etype] = data[etype].edge_index.to(DEVICE)

    x_d  = data["url"].x.to(DEVICE)
    y_d  = data["url"].y.to(DEVICE)
    tm_d = train_mask.to(DEVICE)

    for _ in range(n_epochs):
        model.train()
        opt.zero_grad()
        out  = model(x_d, edge_index_dict)
        loss = F.cross_entropy(out[tm_d], y_d[tm_d])
        loss.backward()
        opt.step()
    model.eval()
    return model


def run_bp(edge_index, edge_type, labels, train_mask, n_classes=2, n_relations=3):
    """Run Loopy BP with seed priors from training labels."""
    N = labels.shape[0]
    # Build seed priors
    priors   = torch.full((N, n_classes), 1.0 / n_classes)
    for i in range(N):
        if train_mask[i] and labels[i] >= 0:
            priors[i, labels[i]] = 0.95
            priors[i, 1 - labels[i]] = 0.05

    # Build potential tensors per relation
    potentials = {}
    for r in range(n_relations):
        mask = (edge_type == r)
        if mask.sum() == 0:
            continue
        ei_r = edge_index[:, mask]
        # Simple homophily potential
        psi  = homophily_potential(h=0.85, n_classes=n_classes)
        potentials[r] = (ei_r, psi)

    bp  = LoopyBP(n_nodes=N, n_classes=n_classes, n_iters=15, damping=0.5)
    out = bp(priors, potentials)  # [N, n_classes]
    return out


# ── Single-run evaluation ─────────────────────────────────────────────────────

def run_one(
    data: HeteroData,
    label_rate: float,
    feature_regime: str,
    seed: int,
    n_epochs: int = 60,
) -> dict[str, dict]:
    """
    Run all methods on one (data, label_rate, feature_regime, seed) configuration.
    Returns {method_name: metrics_dict}.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Apply regimes
    data = apply_label_rate(data, label_rate, seed=seed)
    data = apply_feature_regime(data, feature_regime)

    # Unified graph
    edge_index, edge_type, labels, node_x, offsets = extract_unified_graph(data)

    url_offset    = offsets.get("url", 0)
    n_url         = data["url"].num_nodes if "url" in data.node_types else 0
    url_labels    = data["url"].y
    url_train     = data["url"].train_mask
    url_test      = data["url"].test_mask

    # Map to global
    global_train  = torch.zeros(labels.shape[0], dtype=torch.bool)
    global_test   = torch.zeros(labels.shape[0], dtype=torch.bool)
    global_train[url_offset: url_offset + n_url] = url_train
    global_test[url_offset:  url_offset + n_url] = url_test

    test_global_idx   = global_test.nonzero(as_tuple=True)[0]
    test_local_idx    = url_test.nonzero(as_tuple=True)[0]
    y_true            = url_labels[test_local_idx].numpy()

    # Feature dim
    in_dim = data["url"].x.shape[1] if data["url"].num_nodes > 0 else 1
    if in_dim == 0:
        in_dim = 1

    results = {}

    # ── LightGBM ──────────────────────────────────────────────────────────────
    try:
        X_train_lgb = data["url"].x[url_train].numpy()
        y_train_lgb = url_labels[url_train].numpy()
        X_test_lgb  = data["url"].x[test_local_idx].numpy()
        lgb_model   = lgb.LGBMClassifier(n_estimators=100, max_depth=6,
                                          num_leaves=31, random_state=seed,
                                          verbose=-1)
        lgb_model.fit(X_train_lgb, y_train_lgb)
        y_score = lgb_model.predict_proba(X_test_lgb)[:, 1]
        results["LightGBM"] = evaluate(y_true, y_score)
    except Exception as e:
        logger.warning(f"LightGBM failed: {e}")
        results["LightGBM"] = {"roc_auc": float("nan")}

    # ── Loopy BP ──────────────────────────────────────────────────────────────
    try:
        bp_out   = run_bp(edge_index, edge_type, labels, global_train)
        bp_score = bp_out[test_global_idx, 1].numpy()
        results["LoopyBP"] = evaluate(y_true, bp_score)
    except Exception as e:
        logger.warning(f"LoopyBP failed: {e}")
        results["LoopyBP"] = {"roc_auc": float("nan")}

    # ── GraphSAGE ─────────────────────────────────────────────────────────────
    try:
        sage = GraphSAGEModel(in_channels=in_dim, hidden_channels=64,
                               out_channels=2, num_layers=2, dropout=0.3)
        sage = train_gnn(sage, node_x, edge_index, labels, global_train, n_epochs)
        with torch.no_grad():
            sage_out   = sage(node_x.to(DEVICE), edge_index.to(DEVICE))
            sage_score = torch.softmax(sage_out, dim=-1)[test_global_idx.to(DEVICE), 1].cpu().numpy()
        results["GraphSAGE"] = evaluate(y_true, sage_score)
    except Exception as e:
        logger.warning(f"GraphSAGE failed: {e}")
        results["GraphSAGE"] = {"roc_auc": float("nan")}

    # ── H1 Hybrid (GNN + BP post-hoc) ────────────────────────────────────────
    try:
        sage2 = GraphSAGEModel(in_channels=in_dim, hidden_channels=64,
                                out_channels=2, num_layers=2, dropout=0.3)
        sage2 = train_gnn(sage2, node_x, edge_index, labels, global_train, n_epochs)
        h1    = H1Hybrid(gnn=sage2, n_bp_iters=10, damping=0.5)
        with torch.no_grad():
            h1_out   = h1(node_x.to(DEVICE), edge_index.to(DEVICE), labels.to(DEVICE), global_train.to(DEVICE))
            h1_score = torch.softmax(h1_out, dim=-1)[test_global_idx.to(DEVICE), 1].cpu().numpy()
        results["H1-Hybrid"] = evaluate(y_true, h1_score)
    except Exception as e:
        logger.warning(f"H1-Hybrid failed: {e}")
        results["H1-Hybrid"] = {"roc_auc": float("nan")}

    # ── H3 Hybrid (PhishBP, end-to-end) ──────────────────────────────────────
    try:
        n_rel = 3
        h3    = PhishBPHybridH3(
            in_dim=in_dim, hidden_dim=64, n_classes=2,
            n_relations=n_rel, n_bp_iters=10, dropout=0.3,
        )
        h3    = train_h3(h3, data, url_train, n_epochs)
        with torch.no_grad():
            ei_dict = {et: data[et].edge_index.to(DEVICE) for et in data.edge_types}
            h3_out  = h3(data["url"].x.to(DEVICE), ei_dict)
            h3_score = torch.softmax(h3_out, dim=-1)[test_local_idx.to(DEVICE), 1].cpu().numpy()
        results["H3-PhishBP"] = evaluate(y_true, h3_score)
    except Exception as e:
        logger.warning(f"H3-PhishBP failed: {e}")
        results["H3-PhishBP"] = {"roc_auc": float("nan")}

    return results


# ── Main experiment loop ───────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # Load graph
    graph_path = Path(args.graph)
    if not graph_path.exists():
        logger.error(f"Graph not found at {graph_path}. Run scripts/run_data_collection.py first.")
        return

    base_data: HeteroData = torch.load(graph_path, weights_only=False)
    logger.info(f"Loaded graph: {base_data}")

    # Experiment grid
    homophily_configs = {
        "LOW  (h≈0.40)": 0.40,
        "MED  (h≈0.70)": 0.70,
        "HIGH (h≈0.90)": 0.90,
    }
    label_rates     = [0.01, 0.05, 0.20]
    feature_regimes = ["none", "lexical", "full"]
    seeds           = list(range(args.seeds))

    methods = ["LightGBM", "LoopyBP", "GraphSAGE", "H1-Hybrid", "H3-PhishBP"]

    all_rows = []

    for h_name, h_val in homophily_configs.items():
        for lr in label_rates:
            for fr in feature_regimes:
                seed_results: dict[str, list] = {m: [] for m in methods}

                for seed in seeds:
                    # Perturb homophily
                    data_h = perturb_graph_homophily(base_data, target_h=h_val, seed=seed)

                    run_res = run_one(
                        data=data_h,
                        label_rate=lr,
                        feature_regime=fr,
                        seed=seed,
                        n_epochs=args.epochs,
                    )
                    for m in methods:
                        res_obj = run_res.get(m)
                        auc = getattr(res_obj, "roc_auc", float("nan")) if res_obj is not None else float("nan")
                        seed_results[m].append(auc)

                # Aggregate
                row = {
                    "homophily": h_name,
                    "h_val": h_val,
                    "label_rate": lr,
                    "feature_regime": fr,
                }
                for m in methods:
                    vals = [v for v in seed_results[m] if not np.isnan(v)]
                    mu   = np.mean(vals) if vals else float("nan")
                    std  = np.std(vals)  if len(vals) > 1 else 0.0
                    row[f"{m}_mean"] = round(mu, 4)
                    row[f"{m}_std"]  = round(std, 4)
                    logger.info(f"  {h_name} | lr={lr:.0%} | feat={fr:7s} | {m:12s}: {mu:.4f} ± {std:.4f}")

                all_rows.append(row)

    # ── Save results ──────────────────────────────────────────────────────────
    out_csv = RESULTS_DIR / "exp_A_results.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info(f"Results saved → {out_csv}")

    # ── Generate publication figure ───────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h_vals_sorted = [0.40, 0.70, 0.90]
        h_labels      = ["LOW\n(h≈0.40)", "MED\n(h≈0.70)", "HIGH\n(h≈0.90)"]
        colors        = {"LightGBM": "#2196F3", "LoopyBP": "#FF9800",
                         "GraphSAGE": "#4CAF50", "H1-Hybrid": "#9C27B0",
                         "H3-PhishBP": "#F44336"}
        markers       = {"LightGBM": "s", "LoopyBP": "^",
                         "GraphSAGE": "o", "H1-Hybrid": "D",
                         "H3-PhishBP": "*"}

        # Plot for label_rate=0.05, feature=full (the representative setting)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        fig.suptitle("ROC-AUC vs Edge Homophily — PhishBP Experiment A\n"
                     "(label rate = 5%, feature regime shown per panel)", fontsize=12)

        for col_idx, fr in enumerate(["none", "lexical", "full"]):
            ax = axes[col_idx]
            ax.set_title(f"Features: {fr}", fontsize=10)
            ax.set_xlabel("Edge Homophily")
            if col_idx == 0:
                ax.set_ylabel("ROC-AUC")
            ax.set_xticks(range(3))
            ax.set_xticklabels(h_labels, fontsize=8)
            ax.set_ylim(0.4, 1.05)
            ax.grid(alpha=0.3)

            for m in methods:
                y_vals = []
                y_errs = []
                for h_val in h_vals_sorted:
                    # Find matching row for lr=0.05
                    for row in all_rows:
                        if (abs(row["h_val"] - h_val) < 0.01
                                and abs(row["label_rate"] - 0.05) < 0.001
                                and row["feature_regime"] == fr):
                            y_vals.append(row[f"{m}_mean"])
                            y_errs.append(row[f"{m}_std"])
                            break
                    else:
                        y_vals.append(float("nan"))
                        y_errs.append(0.0)

                ax.errorbar(
                    range(3), y_vals, yerr=y_errs,
                    label=m, color=colors[m], marker=markers[m],
                    linewidth=1.5, capsize=3, markersize=7,
                )

            if col_idx == 2:
                ax.legend(fontsize=7, loc="lower right")

        plt.tight_layout()
        fig_path = RESULTS_DIR / "exp_A_auc_heatmap.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Figure saved → {fig_path}")

    except Exception as e:
        logger.warning(f"Matplotlib figure failed: {e}")

    # ── Summary table ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT A SUMMARY (label_rate=5%, feature=full)")
    logger.info("=" * 80)
    logger.info(f"{'Homophily':22s} | {'LightGBM':10s} | {'LoopyBP':10s} | {'GraphSAGE':10s} | {'H1':10s} | {'H3':10s}")
    logger.info("-" * 80)
    for row in all_rows:
        if abs(row["label_rate"] - 0.05) < 0.001 and row["feature_regime"] == "full":
            logger.info(
                f"{row['homophily']:22s} | "
                f"{row['LightGBM_mean']:.3f}±{row['LightGBM_std']:.3f}  | "
                f"{row['LoopyBP_mean']:.3f}±{row['LoopyBP_std']:.3f}  | "
                f"{row['GraphSAGE_mean']:.3f}±{row['GraphSAGE_std']:.3f}   | "
                f"{row['H1-Hybrid_mean']:.3f}±{row['H1-Hybrid_std']:.3f}  | "
                f"{row['H3-PhishBP_mean']:.3f}±{row['H3-PhishBP_std']:.3f} "
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--graph",  default="data/processed/hetero_graph.pt")
    p.add_argument("--seeds",  type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    main(p.parse_args())
