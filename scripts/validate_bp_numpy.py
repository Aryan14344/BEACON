"""
Stage 1 quick-start validation script — runs with NumPy ONLY (no PyTorch).
This lets you verify the math before installing PyTorch + PyG.

Run:  python3 scripts/validate_bp_numpy.py
"""

from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Pure-NumPy LoopyBP (mirrors bp_baseline.py exactly, same math)
# ─────────────────────────────────────────────────────────────────────────────

def log_sum_exp(x: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    m = x.max(axis=axis, keepdims=True)
    return np.log(np.exp(x - m).sum(axis=axis, keepdims=True)) + m


def homophily_potential(strength: float = 1.5) -> np.ndarray:
    return np.array([[ strength, -strength],
                     [-strength,  strength]], dtype=np.float64)


def loopy_bp(
    num_nodes: int,
    edge_index: np.ndarray,       # (2, E)  undirected edges (each listed once)
    edge_type:  np.ndarray,       # (E,)
    log_priors: np.ndarray,       # (N, 2)
    potentials: list[np.ndarray], # list of (2,2)
    max_iter: int = 20,
    damping:  float = 0.5,
    tol: float = 1e-4,
    verbose: bool = True,
) -> np.ndarray:
    """Return P(x_i = 1) for all nodes."""
    src, dst = edge_index                   # (E,)
    E = edge_index.shape[1]

    # Extend to both directions
    full_src   = np.concatenate([src,  dst])
    full_dst   = np.concatenate([dst,  src])
    full_etype = np.concatenate([edge_type, edge_type])
    rev_idx    = np.concatenate([np.arange(E) + E, np.arange(E)])

    log_msgs = np.zeros((2 * E, 2))        # initialise uniform

    for t in range(max_iter):
        new_msgs = np.zeros_like(log_msgs)

        for e in range(2 * E):
            s = full_src[e]
            r = full_etype[e]
            psi = potentials[r]             # (2, 2)

            in_edges = np.where(full_dst == s)[0]
            in_edges = in_edges[in_edges != rev_idx[e]]

            inc = log_msgs[in_edges].sum(axis=0) if len(in_edges) > 0 else np.zeros(2)
            factor = log_priors[s] + inc    # (2,)

            joint = psi + factor[:, np.newaxis]   # (2, 2)
            msg   = log_sum_exp(joint, axis=0).squeeze(axis=0)  # (2,)
            msg  -= log_sum_exp(msg[np.newaxis, :], axis=1).squeeze()
            new_msgs[e] = msg

        new_msgs = damping * new_msgs + (1 - damping) * log_msgs
        diff = np.abs(new_msgs - log_msgs)
        delta = diff.max() if diff.size > 0 else 0.0
        log_msgs = new_msgs

        if verbose:
            print(f"  iter {t+1:3d}  |Δ| = {delta:.8f}")
        if delta < tol:
            if verbose:
                print(f"  → Converged at iteration {t+1}")
            break
    else:
        if verbose:
            print(f"  ⚠ Did not converge (|Δ| = {delta:.8f})")

    # Compute beliefs
    log_beliefs = log_priors.copy()
    for e in range(2 * E):
        d = full_dst[e]
        log_beliefs[d] += log_msgs[e]

    log_beliefs -= log_sum_exp(log_beliefs, axis=1)
    return np.exp(log_beliefs[:, 1])


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: 3-node chain
# ─────────────────────────────────────────────────────────────────────────────

def test_chain():
    print("\n" + "="*55)
    print("TEST 1: 3-node chain  (0 — 1 — 2), node 0 = phishing")
    print("="*55)

    # 0 ─ 1 ─ 2  (undirected, one edge type)
    edge_index = np.array([[0, 1], [1, 2]])   # (2, 2)  two undirected edges
    edge_type  = np.array([0, 0])
    log_priors = np.zeros((3, 2))
    confidence = 15.0
    log_priors[0] = [-confidence, confidence]  # node 0 = phishing

    potentials = [homophily_potential(3.0)]

    probs = loopy_bp(3, edge_index, edge_type, log_priors, potentials,
                     max_iter=30, damping=0.5, tol=1e-6, verbose=True)

    for i, p in enumerate(probs):
        bar = "█" * int(p * 30)
        label = "phishing" if i == 0 else "unknown"
        print(f"  Node {i} ({label}): P(phish)={p:.4f}  {bar}")

    assert probs[0] > 0.99, f"Seed node failed: {probs[0]:.4f}"
    assert probs[1] > 0.7,  f"Neighbour failed: {probs[1]:.4f}"
    assert probs[2] > 0.5,  f"2-hop node: {probs[2]:.4f}"
    print("  ✓ PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Isolated nodes respect prior
# ─────────────────────────────────────────────────────────────────────────────

def test_isolated():
    print("\n" + "="*55)
    print("TEST 2: Isolated nodes follow prior exactly")
    print("="*55)

    log_priors = np.array([[10., -10.],  # node 0: benign
                           [-10., 10.]]) # node 1: phishing
    edge_index = np.zeros((2, 0), dtype=np.int64)
    edge_type  = np.zeros(0, dtype=np.int64)

    probs = loopy_bp(2, edge_index, edge_type, log_priors,
                     [homophily_potential(1.0)], verbose=False)
    print(f"  Node 0 (benign): P(phish) = {probs[0]:.6f}  (expect ≈ 0)")
    print(f"  Node 1 (phish) : P(phish) = {probs[1]:.6f}  (expect ≈ 1)")
    assert probs[0] < 0.01
    assert probs[1] > 0.99
    print("  ✓ PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Heterophily potential flips beliefs
# ─────────────────────────────────────────────────────────────────────────────

def test_heterophily():
    print("\n" + "="*55)
    print("TEST 3: Heterophily potential — seed phish, neighbour should be benign")
    print("="*55)

    edge_index = np.array([[0], [1]])
    edge_type  = np.array([0])
    log_priors = np.zeros((2, 2))
    log_priors[0] = [-15., 15.]   # node 0 = phishing

    # Heterophily: different labels preferred (strength < 0)
    potentials = [homophily_potential(-2.5)]
    probs = loopy_bp(2, edge_index, edge_type, log_priors, potentials,
                     max_iter=30, damping=0.5, verbose=False)
    print(f"  Node 0 (phish seed): P(phish) = {probs[0]:.4f}")
    print(f"  Node 1 (neighbour) : P(phish) = {probs[1]:.4f} (expect < 0.5)")
    assert probs[0] > 0.9
    assert probs[1] < 0.5
    print("  ✓ PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Edge homophily metric
# ─────────────────────────────────────────────────────────────────────────────

def test_homophily_metric():
    print("\n" + "="*55)
    print("TEST 4: Edge homophily metric")
    print("="*55)

    # Perfect homophily
    edge_index = np.array([[0, 2], [1, 3]])
    labels = np.array([0, 0, 1, 1])
    h = np.mean(labels[edge_index[0]] == labels[edge_index[1]])
    print(f"  Perfect homophily: h = {h:.4f} (expect 1.0)")
    assert abs(h - 1.0) < 1e-6

    # Zero homophily
    edge_index2 = np.array([[0, 1], [1, 0]])
    labels2 = np.array([0, 1])
    h2 = np.mean(labels2[edge_index2[0]] == labels2[edge_index2[1]])
    print(f"  Zero homophily:    h = {h2:.4f} (expect 0.0)")
    assert abs(h2 - 0.0) < 1e-6

    print("  ✓ PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Full toy graph with seeds
# ─────────────────────────────────────────────────────────────────────────────

def make_toy_graph_numpy(homophily: float = 0.8, seed: int = 42):
    """Minimal version of make_toy_graph in pure numpy."""
    rng = np.random.default_rng(seed)
    N_URL, N_DOM, N_IP, N_NS = 60, 30, 15, 6
    N = N_URL + N_DOM + N_IP + N_NS
    off = {"url": 0, "dom": N_URL, "ip": N_URL+N_DOM, "ns": N_URL+N_DOM+N_IP}

    labels = np.zeros(N, dtype=np.int64)
    phish_dom = rng.choice(N_DOM, size=N_DOM//3, replace=False)
    labels[off["dom"] + phish_dom] = 1

    # Build URL→Domain edges
    domain_assign = np.array([
        rng.choice(phish_dom if rng.random() < homophily and i < N_URL//3
                   else np.arange(N_DOM))
        for i in range(N_URL)
    ])
    for u in range(N_URL):
        labels[off["url"] + u] = labels[off["dom"] + domain_assign[u]]

    src_ud = np.arange(N_URL) + off["url"]
    dst_ud = domain_assign   + off["dom"]

    edge_src   = np.concatenate([src_ud, dst_ud])
    edge_dst   = np.concatenate([dst_ud, src_ud])
    edge_index = np.stack([edge_src, edge_dst])
    edge_type  = np.zeros(len(edge_src), dtype=np.int64)

    log_priors = np.zeros((N, 2))
    return edge_index, edge_type, log_priors, labels, N


def test_toy_graph_bp():
    print("\n" + "="*55)
    print("TEST 5: BP on toy heterogeneous graph (seeds → inference)")
    print("="*55)
    rng = np.random.default_rng(99)

    edge_index, edge_type, log_priors, labels, N = make_toy_graph_numpy(
        homophily=0.85, seed=42
    )

    phish_idx = np.where(labels == 1)[0]
    benign_idx = np.where(labels == 0)[0]
    n_seed = max(2, len(phish_idx) // 8)

    seed_p = rng.choice(phish_idx, size=n_seed, replace=False)
    seed_b = rng.choice(benign_idx, size=n_seed, replace=False)

    conf = 10.0
    for idx in seed_p:
        log_priors[idx] = [-conf, conf]
    for idx in seed_b:
        log_priors[idx] = [conf, -conf]

    probs = loopy_bp(N, edge_index, edge_type, log_priors,
                     [homophily_potential(2.0)],
                     max_iter=20, damping=0.5, verbose=False)

    seed_set = set(seed_p.tolist()) | set(seed_b.tolist())
    unseen   = [i for i in range(N) if i not in seed_set]
    preds    = (probs[unseen] > 0.5).astype(int)
    acc      = np.mean(preds == labels[unseen])

    print(f"  Seeds:   {len(seed_set)} nodes  ({n_seed} phish + {n_seed} benign)")
    print(f"  Unseen:  {len(unseen)} nodes")
    print(f"  Accuracy on unseen: {acc:.3f} (expect > 0.55 for high-homophily graph)")
    assert acc > 0.55, f"BP accuracy {acc:.3f} not better than near-random"
    print("  ✓ PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "█"*55)
    print("  PhishBP — Stage 1 Validation (NumPy only)")
    print("█"*55)

    test_chain()
    test_isolated()
    test_heterophily()
    test_homophily_metric()
    test_toy_graph_bp()

    print("\n" + "█"*55)
    print("  ALL TESTS PASSED ✓")
    print("  Next: install PyTorch + PyG and run pytest tests/")
    print("█"*55)
