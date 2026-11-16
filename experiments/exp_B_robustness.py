"""
Experiment B: Adversarial Graph-Structure & Feature Perturbations (Robustness Study for RQ3).
Evaluates models against:
  1. Clean baseline (no attack)
  2. MimicIP Attack (DNS A-record spoofing: malicious domains pointed to benign IPs)
  3. URL-Padding Attack (evasive path padding, forced HTTPS, suppressing keywords)
  4. Combined Attack (MimicIP + URL-Padding)

Inspects the learned compatibility potentials Ψ_r of H3 to see if it downweights attacked relations.
"""

import sys
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_benchmark import extract_unified_graph
from src.attacks.graph_attacks import apply_mimicip_attack, apply_url_padding_attack
from src.features.lexical import FEATURE_NAMES
from src.models.bp_layer import PhishBPHybridH3
from src.models.gnn_baselines import GraphSAGEModel
from src.eval.metrics import evaluate

import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RobustnessExperiment")


def run_robustness_study(data_path="data/processed/hetero_graph.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running Experiment B (Robustness) on device: {device}")

    raw_data = torch.load(data_path, weights_only=False)
    edge_index, edge_type, labels, features, offsets, total_nodes, num_rel = extract_unified_graph(raw_data)

    url_start = offsets['url']
    url_count = raw_data['url'].num_nodes
    url_indices = np.arange(url_start, url_start + url_count)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(url_indices))
    n_train = int(len(url_indices) * 0.7)
    train_idx = url_indices[perm[:n_train]]
    test_idx = url_indices[perm[n_train:]]

    y_train = labels[train_idx].numpy()
    y_test = labels[test_idx].numpy()
    X_train = features[train_idx].numpy()
    X_test = features[test_idx].numpy()

    # Pre-train clean models
    # 1. LightGBM
    lgb_clean = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    lgb_clean.fit(X_train, y_train)

    # 2. GraphSAGE
    sage = GraphSAGEModel(in_channels=features.shape[1], hidden_channels=32, num_layers=2).to(device)
    sage_opt = torch.optim.Adam(sage.parameters(), lr=0.01, weight_decay=1e-4)
    x_dev = features.to(device)
    ei_dev = edge_index.to(device)
    train_idx_t = torch.tensor(train_idx, device=device)
    test_idx_t = torch.tensor(test_idx, device=device)
    y_train_t = labels[train_idx].float().to(device)

    sage.train()
    for _ in range(50):
        sage_opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(sage(x_dev, ei_dev)[train_idx_t], y_train_t)
        loss.backward()
        sage_opt.step()

    # 3. H3 Hybrid
    h3 = PhishBPHybridH3(in_features=features.shape[1], hidden_dim=32, num_relations=num_rel, bp_iters=4).to(device)
    h3_opt = torch.optim.Adam(h3.parameters(), lr=0.008, weight_decay=1e-4)
    et_dev = edge_type.to(device)
    y_train_long = labels[train_idx].to(device)

    h3.train()
    for _ in range(50):
        h3_opt.zero_grad()
        loss = F.nll_loss(h3(x_dev, ei_dev, et_dev)[train_idx_t], y_train_long)
        loss.backward()
        h3_opt.step()

    logger.info("Clean models successfully trained. Evaluating adversarial perturbations...")

    # Define attack scenarios
    attack_scenarios = ["Clean (No Attack)", "MimicIP (30% DNS A-records spoofed)", "URL-Padding (Feature evasion)", "Combined Attack (MimicIP + Padding)"]
    results_table = []

    for scenario in attack_scenarios:
        # Prepare perturbed graph and features
        current_data = raw_data
        curr_feat = features.clone()

        if "MimicIP" in scenario:
            current_data = apply_mimicip_attack(raw_data, attack_ratio=0.35, seed=42)

        cur_ei, cur_et, _, _, _, _, _ = extract_unified_graph(current_data)
        cur_ei_dev = cur_ei.to(device)
        cur_et_dev = cur_et.to(device)

        if "Padding" in scenario:
            curr_feat[url_indices] = apply_url_padding_attack(
                curr_feat[url_indices], labels[url_indices], FEATURE_NAMES, attack_ratio=0.35, seed=42
            )
        cur_x_dev = curr_feat.to(device)
        cur_X_test = curr_feat[test_idx].numpy()

        # Evaluate LightGBM
        lgb_preds = lgb_clean.predict_proba(cur_X_test)[:, 1]
        lgb_auc = evaluate(y_test, lgb_preds).roc_auc

        # Evaluate GraphSAGE
        sage.eval()
        with torch.no_grad():
            sage_preds = torch.sigmoid(sage(cur_x_dev, cur_ei_dev))[test_idx_t].cpu().numpy()
            sage_auc = evaluate(y_test, sage_preds).roc_auc

        # Evaluate H3
        h3.eval()
        with torch.no_grad():
            h3_preds = h3(cur_x_dev, cur_ei_dev, cur_et_dev).exp()[test_idx_t, 1].cpu().numpy()
            h3_auc = evaluate(y_test, h3_preds).roc_auc

        results_table.append({
            "Scenario": scenario,
            "LightGBM": lgb_auc,
            "GraphSAGE": sage_auc,
            "H3 Hybrid": h3_auc,
        })

    logger.info("\n" + "="*85)
    logger.info(f"{'Perturbation Scenario':<40}  {'LightGBM':>12}  {'GraphSAGE':>12}  {'H3 Hybrid':>12}")
    logger.info("-" * 85)
    for row in results_table:
        logger.info(f"  {row['Scenario']:<38}  {row['LightGBM']:>12.4f}  {row['GraphSAGE']:>12.4f}  {row['H3 Hybrid']:>12.4f}")
    logger.info("=" * 85)

    # Inspect Learned Potentials in H3
    logger.info("\nLearned Relation Potentials Ψ_r in H3:")
    rel_names = ["url_resolves_to_domain", "domain_resolves_to_ip", "domain_uses_ns"]
    learned_psi = h3.bp_layer.log_psi.detach().cpu().numpy()
    for r, name in enumerate(rel_names):
        diag_strength = (learned_psi[r, 0, 0] + learned_psi[r, 1, 1]) / 2.0
        off_diag = (learned_psi[r, 0, 1] + learned_psi[r, 1, 0]) / 2.0
        logger.info(f"  Relation '{name}': Homophily Potentials Diag={diag_strength:.3f}, OffDiag={off_diag:.3f}")


if __name__ == "__main__":
    run_robustness_study()
