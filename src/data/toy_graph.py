"""
Synthetic heterogeneous DNS-like graph for Stage 1 validation.

Topology:
  Node types: URL (200), Domain (100), IP (50), NS (20)  — total 370
  Edge types:
    url_domain : URL → Domain   (each URL belongs to one domain)
    domain_ip  : Domain → IP    (DNS A records)
    domain_ns  : Domain → NS    (DNS NS records)

Labels: 0 = benign, 1 = phishing
Label generation respects controllable edge homophily per relation.

Usage
-----
>>> from src.data.toy_graph import make_toy_graph
>>> graph, labels = make_toy_graph(homophily=0.8, seed=42)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from src.models.bp_baseline import BPGraph, homophily_potential


# ─────────────────────────────────────────────────────────────────────────────
# Node type registry
# ─────────────────────────────────────────────────────────────────────────────

NODE_TYPES = ["url", "domain", "ip", "ns"]
N_URL    = 200
N_DOMAIN = 100
N_IP     = 50
N_NS     = 20
N_TOTAL  = N_URL + N_DOMAIN + N_IP + N_NS

OFFSETS = {
    "url":    0,
    "domain": N_URL,
    "ip":     N_URL + N_DOMAIN,
    "ns":     N_URL + N_DOMAIN + N_IP,
}

# Relation ids
REL = {"url_domain": 0, "domain_ip": 1, "domain_ns": 2}
N_REL = len(REL)


# ─────────────────────────────────────────────────────────────────────────────
# Homophily computation
# ─────────────────────────────────────────────────────────────────────────────

def edge_homophily(edge_index: Tensor, labels: Tensor) -> float:
    """
    Fraction of edges (u,v) where label_u == label_v.

    Parameters
    ----------
    edge_index : (2, E) — global node indices
    labels     : (N,)
    """
    src, dst = edge_index
    same = (labels[src] == labels[dst]).float()
    return same.mean().item()


def per_relation_homophily(
    edge_index: Tensor,
    edge_type: Tensor,
    labels: Tensor,
    num_rel: int,
) -> dict[int, float]:
    """Return edge homophily for each relation type separately."""
    result = {}
    for r in range(num_rel):
        mask = edge_type == r
        if mask.sum() == 0:
            result[r] = float("nan")
            continue
        ei = edge_index[:, mask]
        result[r] = edge_homophily(ei, labels)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Toy graph factory
# ─────────────────────────────────────────────────────────────────────────────

def make_toy_graph(
    homophily: float = 0.8,
    phishing_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[BPGraph, Tensor]:
    """
    Build a small synthetic heterogeneous graph with known labels.

    The `homophily` parameter controls how strongly same-label nodes
    are co-located in the graph.  It affects all three relation types.

    Parameters
    ----------
    homophily     : target edge homophily ∈ [0.5, 1.0]
    phishing_ratio: fraction of phishing nodes
    seed          : random seed for reproducibility

    Returns
    -------
    graph  : BPGraph ready for LoopyBP.infer()
    labels : (N_TOTAL,) int64 ground-truth labels
    """
    rng = np.random.default_rng(seed)

    # ── 1. Assign labels ─────────────────────────────────────────────────────
    labels = np.zeros(N_TOTAL, dtype=np.int64)
    # Phishing assignment: domains first (the core unit), then propagate
    phish_domains = rng.choice(N_DOMAIN, size=int(N_DOMAIN * phishing_ratio), replace=False)
    labels[OFFSETS["domain"] + phish_domains] = 1

    # ── 2. Build URL → Domain edges ──────────────────────────────────────────
    # Each URL maps to exactly one domain.
    # With probability `homophily`, a phishing URL maps to a phishing domain.
    url_to_domain = _assign_with_homophily(
        rng, N_URL, phish_domains, N_DOMAIN, homophily
    )
    # Propagate labels: URL inherits domain label
    for u in range(N_URL):
        labels[OFFSETS["url"] + u] = labels[OFFSETS["domain"] + url_to_domain[u]]

    src_ud = torch.tensor(
        [OFFSETS["url"] + u for u in range(N_URL)], dtype=torch.long
    )
    dst_ud = torch.tensor(
        [OFFSETS["domain"] + url_to_domain[u] for u in range(N_URL)], dtype=torch.long
    )

    # ── 3. Build Domain → IP edges ───────────────────────────────────────────
    # Phishing domains share IPs; controlled by homophily.
    phish_ips = rng.choice(N_IP, size=max(1, int(N_IP * phishing_ratio)), replace=False)
    domain_to_ip = _assign_with_homophily(
        rng, N_DOMAIN, phish_ips, N_IP, homophily
    )
    # Propagate IP labels (IP is phishing if any phishing domain uses it)
    for d in range(N_DOMAIN):
        if labels[OFFSETS["domain"] + d] == 1:
            labels[OFFSETS["ip"] + domain_to_ip[d]] = 1

    src_di = torch.tensor(
        [OFFSETS["domain"] + d for d in range(N_DOMAIN)], dtype=torch.long
    )
    dst_di = torch.tensor(
        [OFFSETS["ip"] + domain_to_ip[d] for d in range(N_DOMAIN)], dtype=torch.long
    )

    # ── 4. Build Domain → NS edges ───────────────────────────────────────────
    phish_ns = rng.choice(N_NS, size=max(1, int(N_NS * phishing_ratio)), replace=False)
    domain_to_ns = _assign_with_homophily(
        rng, N_DOMAIN, phish_ns, N_NS, homophily
    )
    for d in range(N_DOMAIN):
        if labels[OFFSETS["domain"] + d] == 1:
            labels[OFFSETS["ns"] + domain_to_ns[d]] = 1

    src_dns_e = torch.tensor(
        [OFFSETS["domain"] + d for d in range(N_DOMAIN)], dtype=torch.long
    )
    dst_dns_e = torch.tensor(
        [OFFSETS["ns"] + domain_to_ns[d] for d in range(N_DOMAIN)], dtype=torch.long
    )

    # ── 5. Assemble edge_index and edge_type ──────────────────────────────────
    edge_index = torch.stack([
        torch.cat([src_ud, dst_ud, src_di, dst_di, src_dns_e, dst_dns_e]),
        torch.cat([dst_ud, src_ud, dst_di, src_di, dst_dns_e, src_dns_e]),
    ], dim=0)  # undirected: each edge appears in both directions

    n_ud = N_URL
    n_di = N_DOMAIN
    n_dn = N_DOMAIN
    edge_type = torch.cat([
        torch.full((2 * n_ud,), REL["url_domain"], dtype=torch.long),
        torch.full((2 * n_di,), REL["domain_ip"],  dtype=torch.long),
        torch.full((2 * n_dn,), REL["domain_ns"],  dtype=torch.long),
    ])

    # ── 6. Build BPGraph ──────────────────────────────────────────────────────
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_priors = torch.zeros(N_TOTAL, 2)  # uniform

    potentials = [homophily_potential(strength=1.5) for _ in range(N_REL)]

    graph = BPGraph(
        num_nodes=N_TOTAL,
        edge_index=edge_index,
        edge_type=edge_type,
        log_priors=log_priors,
        num_rel=N_REL,
        potentials=potentials,
    )

    return graph, labels_t


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _assign_with_homophily(
    rng: np.random.Generator,
    n_src: int,
    phish_targets: np.ndarray,
    n_target: int,
    homophily: float,
) -> np.ndarray:
    """
    Assign each of `n_src` source nodes to a target node ∈ [0, n_target).
    phish_targets: indices of 'phishing' target nodes.
    With probability `homophily`, phishing sources go to phishing targets.
    """
    benign_targets = np.setdiff1d(np.arange(n_target), phish_targets)
    # We don't know source labels yet, so we build phishing-source mapping first.
    # We'll assign: sources 0..n_phish_src-1 are phishing (approximation for toy).
    n_phish_src = max(1, int(n_src * 0.3))
    assignments = np.empty(n_src, dtype=np.int64)

    for i in range(n_src):
        is_phish_src = i < n_phish_src
        if is_phish_src and len(phish_targets) > 0:
            if rng.random() < homophily:
                assignments[i] = rng.choice(phish_targets)
            else:
                assignments[i] = rng.choice(n_target)
        else:
            if rng.random() < homophily and len(benign_targets) > 0:
                assignments[i] = rng.choice(benign_targets)
            else:
                assignments[i] = rng.choice(n_target)
    return assignments
