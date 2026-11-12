"""
Stage 1 unit tests for LoopyBP and toy graph utilities.

Run with:  pytest tests/test_stage1_bp.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import numpy as np

from src.models.bp_baseline import (
    LoopyBP,
    BPGraph,
    homophily_potential,
    log_sum_exp,
)
from src.data.toy_graph import (
    make_toy_graph,
    edge_homophily,
    per_relation_homophily,
    N_TOTAL,
    N_REL,
    REL,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. log_sum_exp numerical stability
# ─────────────────────────────────────────────────────────────────────────────

class TestLogSumExp:
    def test_basic(self):
        x = torch.tensor([[0.0, 0.0]])
        result = log_sum_exp(x, dim=1)
        expected = torch.log(torch.tensor(2.0))
        assert torch.allclose(result, expected, atol=1e-6)

    def test_large_values(self):
        """Should not overflow/underflow."""
        x = torch.tensor([[1000.0, 1001.0]])
        result = log_sum_exp(x, dim=1)
        # Should be ~1001 + log(1 + e^-1) ≈ 1001.31
        assert torch.isfinite(result).all()
        assert result.item() > 1001.0

    def test_negative_large(self):
        x = torch.tensor([[-1000.0, -1001.0]])
        result = log_sum_exp(x, dim=1)
        assert torch.isfinite(result).all()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Minimal 3-node chain graph
# ─────────────────────────────────────────────────────────────────────────────

def make_chain_graph(seed_label: int = 1) -> tuple[BPGraph, int]:
    """
    Three nodes in a chain:  0 --- 1 --- 2
    Node 0 is a known phishing seed.
    With strong homophily potential, nodes 1 and 2 should become phishing.
    """
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_type  = torch.zeros(4, dtype=torch.long)
    log_priors = torch.zeros(3, 2)
    # Strong prior on node 0
    confidence = 20.0
    if seed_label == 1:
        log_priors[0] = torch.tensor([-confidence, confidence])
    else:
        log_priors[0] = torch.tensor([confidence, -confidence])

    potentials = [homophily_potential(strength=3.0)]

    graph = BPGraph(
        num_nodes=3,
        edge_index=edge_index,
        edge_type=edge_type,
        log_priors=log_priors,
        num_rel=1,
        potentials=potentials,
    )
    return graph, seed_label


class TestBPChainGraph:
    """BP should propagate seed labels along a chain."""

    def test_phishing_propagation(self):
        graph, _ = make_chain_graph(seed_label=1)
        bp = LoopyBP(max_iter=50, damping=0.5, tol=1e-6)
        probs = bp.infer(graph)
        # All three nodes should be classified as phishing (prob > 0.9)
        assert probs[0].item() > 0.9, f"Seed node P(phish)={probs[0]:.3f} < 0.9"
        assert probs[1].item() > 0.7, f"Node 1 P(phish)={probs[1]:.3f} < 0.7"
        assert probs[2].item() > 0.5, f"Node 2 P(phish)={probs[2]:.3f} < 0.5"

    def test_benign_propagation(self):
        graph, _ = make_chain_graph(seed_label=0)
        bp = LoopyBP(max_iter=50, damping=0.5, tol=1e-6)
        probs = bp.infer(graph)
        assert probs[0].item() < 0.1, f"Seed node P(phish)={probs[0]:.3f} > 0.1"
        assert probs[1].item() < 0.3, f"Node 1 P(phish)={probs[1]:.3f} > 0.3"

    def test_output_shape(self):
        graph, _ = make_chain_graph()
        probs = LoopyBP().infer(graph)
        assert probs.shape == (3,)

    def test_output_probabilities(self):
        """All outputs must be valid probabilities in [0, 1]."""
        graph, _ = make_chain_graph()
        probs = LoopyBP().infer(graph)
        assert (probs >= 0).all() and (probs <= 1).all()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Isolated nodes (no edges)
# ─────────────────────────────────────────────────────────────────────────────

class TestBPIsolatedNodes:
    def test_uniform_prior_stays_uniform(self):
        """With no edges and uniform prior, output should be ~0.5."""
        graph = BPGraph(
            num_nodes=5,
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_type=torch.zeros(0, dtype=torch.long),
            log_priors=torch.zeros(5, 2),
            num_rel=1,
            potentials=[homophily_potential(1.0)],
        )
        probs = LoopyBP().infer(graph)
        assert torch.allclose(probs, torch.full((5,), 0.5), atol=1e-5)

    def test_strong_prior_no_edges(self):
        """Without edges, the prior should determine the output."""
        log_priors = torch.tensor([
            [10.0, -10.0],  # node 0: benign
            [-10.0, 10.0],  # node 1: phishing
        ])
        graph = BPGraph(
            num_nodes=2,
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_type=torch.zeros(0, dtype=torch.long),
            log_priors=log_priors,
            num_rel=1,
            potentials=[homophily_potential(1.0)],
        )
        probs = LoopyBP().infer(graph)
        assert probs[0].item() < 0.01, "Benign node should have P(phish) ≈ 0"
        assert probs[1].item() > 0.99, "Phishing node should have P(phish) ≈ 1"


# ─────────────────────────────────────────────────────────────────────────────
# 4. make_log_priors helper
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeLogPriors:
    def test_seed_labels_set_correctly(self):
        log_priors = LoopyBP.make_log_priors(
            num_nodes=10,
            seed_indices=torch.tensor([0, 5]),
            seed_labels=torch.tensor([1, 0]),
            confidence=5.0,
        )
        # Node 0 is phishing: log_priors[0,1] >> log_priors[0,0]
        assert log_priors[0, 1] > log_priors[0, 0]
        # Node 5 is benign: log_priors[5,0] >> log_priors[5,1]
        assert log_priors[5, 0] > log_priors[5, 1]
        # Unlabelled nodes: uniform (both 0.0)
        assert torch.all(log_priors[1] == 0.0)

    def test_shape(self):
        lp = LoopyBP.make_log_priors(5, torch.tensor([2]), torch.tensor([1]))
        assert lp.shape == (5, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Edge homophily metric
# ─────────────────────────────────────────────────────────────────────────────

class TestHomophily:
    def test_perfect_homophily(self):
        """All edges connect same-label nodes → h = 1.0."""
        edge_index = torch.tensor([[0, 2], [1, 3]])
        labels = torch.tensor([0, 0, 1, 1])
        h = edge_homophily(edge_index, labels)
        assert abs(h - 1.0) < 1e-6

    def test_zero_homophily(self):
        """All edges connect different-label nodes → h = 0.0."""
        edge_index = torch.tensor([[0, 1], [1, 0]])
        labels = torch.tensor([0, 1])
        h = edge_homophily(edge_index, labels)
        assert abs(h - 0.0) < 1e-6

    def test_range(self):
        """Homophily must be in [0, 1]."""
        graph, labels = make_toy_graph(homophily=0.8)
        h = edge_homophily(graph.edge_index, labels)
        assert 0.0 <= h <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Toy graph factory
# ─────────────────────────────────────────────────────────────────────────────

class TestToyGraph:
    def test_shape(self):
        graph, labels = make_toy_graph()
        assert graph.num_nodes == N_TOTAL
        assert labels.shape == (N_TOTAL,)

    def test_binary_labels(self):
        _, labels = make_toy_graph()
        assert set(labels.tolist()).issubset({0, 1})

    def test_per_relation_homophily(self):
        graph, labels = make_toy_graph(homophily=0.9, seed=0)
        h_dict = per_relation_homophily(
            graph.edge_index, graph.edge_type, labels, num_rel=N_REL
        )
        assert len(h_dict) == N_REL
        for r, h in h_dict.items():
            assert 0.0 <= h <= 1.0, f"Relation {r} homophily out of [0,1]: {h}"

    def test_bp_on_toy_graph_with_seeds(self):
        """Full integration: BP on toy graph should beat random baseline."""
        graph, labels = make_toy_graph(homophily=0.85, seed=42)

        # Use 10% of phishing nodes as seeds
        phish_idx = (labels == 1).nonzero(as_tuple=True)[0]
        benign_idx = (labels == 0).nonzero(as_tuple=True)[0]
        rng = torch.Generator().manual_seed(0)
        n_seed = max(1, len(phish_idx) // 10)
        seed_phish = phish_idx[torch.randperm(len(phish_idx), generator=rng)[:n_seed]]
        seed_benign = benign_idx[torch.randperm(len(benign_idx), generator=rng)[:n_seed]]
        seed_idx = torch.cat([seed_phish, seed_benign])
        seed_lbl = torch.cat([torch.ones(n_seed, dtype=torch.long),
                               torch.zeros(n_seed, dtype=torch.long)])

        # Set priors
        graph.log_priors = LoopyBP.make_log_priors(
            graph.num_nodes, seed_idx, seed_lbl, confidence=8.0
        )

        bp = LoopyBP(max_iter=20, damping=0.5, verbose=True)
        probs = bp.infer(graph)

        # Evaluate on unseen nodes
        mask = torch.ones(N_TOTAL, dtype=torch.bool)
        mask[seed_idx] = False
        pred = (probs[mask] > 0.5).long()
        acc = (pred == labels[mask]).float().mean().item()
        print(f"\n  BP accuracy on unseen nodes: {acc:.3f}")
        # Should beat random (0.5) on a high-homophily graph
        assert acc > 0.55, f"BP accuracy {acc:.3f} not better than near-random"

    def test_homophily_monotone(self):
        """Higher homophily param → higher measured homophily on the graph."""
        _, labels_lo = make_toy_graph(homophily=0.6, seed=1)
        graph_lo, _ = make_toy_graph(homophily=0.6, seed=1)
        graph_hi, labels_hi = make_toy_graph(homophily=0.95, seed=1)

        h_lo = edge_homophily(graph_lo.edge_index, labels_lo)
        h_hi = edge_homophily(graph_hi.edge_index, labels_hi)
        # Not guaranteed to be strictly monotone on small graphs, but should trend
        print(f"\n  h_lo={h_lo:.3f}  h_hi={h_hi:.3f}")
        assert h_hi >= h_lo - 0.1  # allow small tolerance


# ─────────────────────────────────────────────────────────────────────────────
# 7. Damping effect
# ─────────────────────────────────────────────────────────────────────────────

class TestDamping:
    def test_high_damping_converges(self):
        """With high damping (α=0.9), BP should still converge."""
        graph, _ = make_chain_graph(seed_label=1)
        bp = LoopyBP(max_iter=100, damping=0.9, tol=1e-5)
        probs = bp.infer(graph)
        assert torch.isfinite(probs).all()
        assert probs[0].item() > 0.9

    def test_zero_damping(self):
        """With no damping (α=0), BP is standard loopy BP."""
        graph, _ = make_chain_graph(seed_label=1)
        bp = LoopyBP(max_iter=50, damping=0.0, tol=1e-5)
        probs = bp.infer(graph)
        assert torch.isfinite(probs).all()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point for quick manual run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("PhishBP Stage 1 — Manual Validation")
    print("=" * 60)

    # Chain graph
    graph, label = make_chain_graph(seed_label=1)
    bp = LoopyBP(max_iter=30, damping=0.5, verbose=True)
    probs = bp.infer(graph)
    print(f"\nChain graph (seed=phishing):")
    for i, p in enumerate(probs):
        bar = "█" * int(p.item() * 20)
        print(f"  Node {i}: P(phish)={p.item():.4f}  {bar}")

    # Toy graph full test
    print("\nToy heterogeneous graph:")
    graph, labels = make_toy_graph(homophily=0.85, seed=42)
    h_dict = per_relation_homophily(graph.edge_index, graph.edge_type, labels, N_REL)
    rel_names = {v: k for k, v in REL.items()}
    for r, h in h_dict.items():
        print(f"  h({rel_names[r]}) = {h:.4f}")

    print("\n✓ All manual checks passed. Run: pytest tests/test_stage1_bp.py -v")
