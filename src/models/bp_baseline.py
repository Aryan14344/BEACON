"""
Loopy Belief Propagation (sum-product) baseline for binary node classification.

Mathematical foundation
-----------------------
Variables:  x_i ∈ {0 (benign), 1 (phishing)}
Node potential (prior): φ_i(x_i)   — from seed labels or feature scores
Edge potential:         Ψ(x_i, x_j) — 2×2 compatibility matrix, one per relation

Sum-product messages (log-domain for numerical stability):
  log μ_{i→j}(x_j) = log Σ_{x_i} exp[log Ψ(x_i, x_j) + log φ_i(x_i)
                         + Σ_{k∈N(i)\j} log μ_{k→i}(x_i)]

Using the log-sum-exp trick:
  log Σ_x exp(a_x) = max_x(a_x) + log Σ_x exp(a_x - max_x(a_x))

Damping (Heskes, 2004):
  log μ_new = α·log μ_computed + (1-α)·log μ_old

Belief (posterior):
  log b_i(x_i) = log φ_i(x_i) + Σ_{j∈N(i)} log μ_{j→i}(x_i)

After normalisation:  P(x_i=1) = softmax(log b_i)[1]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BPGraph:
    """
    A simple container for a binary-classification BP graph.

    Attributes
    ----------
    num_nodes   : total number of nodes N
    edge_index  : (2, E) int64 — COO format, **undirected** (each edge listed once)
    edge_type   : (E,) int64  — relation id for each edge
    log_priors  : (N, 2) float32 — log φ_i(x_i); rows sum to 0 for uniform prior
    num_rel     : number of distinct relation types
    potentials  : list of (2,2) float32 tensors, one per relation (log-domain)
                  potentials[r][x_i, x_j] = log Ψ_r(x_i, x_j)
    """
    num_nodes: int
    edge_index: Tensor          # (2, E)
    edge_type: Tensor           # (E,)
    log_priors: Tensor          # (N, 2)
    num_rel: int
    potentials: list[Tensor]    # len == num_rel, each (2, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Log-sum-exp helper
# ─────────────────────────────────────────────────────────────────────────────

def log_sum_exp(x: Tensor, dim: int) -> Tensor:
    """Numerically stable log-sum-exp along `dim`."""
    m = x.max(dim=dim, keepdim=True).values
    return (x - m).exp().sum(dim=dim, keepdim=True).log() + m


# ─────────────────────────────────────────────────────────────────────────────
# Core BP engine
# ─────────────────────────────────────────────────────────────────────────────

class LoopyBP:
    """
    Damped loopy belief propagation on a heterogeneous graph.

    Parameters
    ----------
    max_iter : int
        Maximum number of message-passing rounds.
    damping  : float ∈ [0, 1)
        α in μ_new = α·μ_computed + (1-α)·μ_old.
        Higher values slow convergence but improve stability.
    tol      : float
        Convergence threshold on max absolute message change (log-domain).
    verbose  : bool
    """

    def __init__(
        self,
        max_iter: int = 20,
        damping: float = 0.5,
        tol: float = 1e-4,
        verbose: bool = False,
    ) -> None:
        self.max_iter = max_iter
        self.damping = damping
        self.tol = tol
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def infer(self, graph: BPGraph) -> Tensor:
        """
        Run loopy BP and return posterior probabilities P(x_i = 1).

        Returns
        -------
        probs : (N,) float32 Tensor  — P(x_i = phishing)
        """
        N = graph.num_nodes
        E = graph.edge_index.shape[1]

        # Edge case: If there are no edges, beliefs are simply the normalized node priors
        if E == 0:
            log_beliefs = graph.log_priors - log_sum_exp(graph.log_priors, dim=1)
            return log_beliefs[:, 1].exp()

        src, dst = graph.edge_index  # (E,)

        # Messages are stored **per directed edge** in COO order.
        # We store both directions: for undirected edge (u,v), we keep
        #   msg[e]       : u→v  message  (shape 2: values for x_v ∈ {0,1})
        #   msg[e + E]   : v→u  message
        # But for simplicity we store a flat (2*E, 2) tensor.
        # Index convention: forward edge e is at position e,
        #                   reverse edge e is at position E + e.

        # Initialise all messages to log(0.5) = log-uniform
        log_msgs = torch.zeros(2 * E, 2)  # (2E, 2)

        # Augment edge_index with reverse edges for easy neighbourhood lookup
        full_src = torch.cat([src, dst])          # (2E,)
        full_dst = torch.cat([dst, src])          # (2E,)
        full_etype = torch.cat([graph.edge_type, graph.edge_type])  # (2E,)
        # Reverse edge index: for edge e (forward), reverse is E+e and v.v.
        rev_idx = torch.cat([
            torch.arange(E) + E,
            torch.arange(E),
        ])  # (2E,)  rev_idx[e] = index of the reverse of edge e

        for iteration in range(self.max_iter):
            new_log_msgs = torch.zeros_like(log_msgs)

            # Compute new message for each directed edge e: src[e] → dst[e]
            # log μ_{s→d}(x_d) = log Σ_{x_s} [
            #     log Ψ_r(x_s, x_d)
            #     + log φ_s(x_s)
            #     + Σ_{k→s, k≠d} log μ_{k→s}(x_s)
            # ]
            for e in range(2 * E):
                s = full_src[e].item()
                r = full_etype[e].item()
                log_psi = graph.potentials[r]  # (2, 2): psi[x_s, x_d]

                # Aggregate incoming messages to s, excluding the reverse
                # of edge e (i.e., exclude the message from d→s)
                in_edges_to_s = (full_dst == s).nonzero(as_tuple=True)[0]
                # Exclude the reverse edge
                rev_e = rev_idx[e]
                mask = in_edges_to_s != rev_e
                in_edges_to_s = in_edges_to_s[mask]

                # Sum of incoming log-messages at s:  (2,)
                if len(in_edges_to_s) > 0:
                    inc = log_msgs[in_edges_to_s].sum(dim=0)  # (2,)
                else:
                    inc = torch.zeros(2)

                # log factor at s: log φ_s(x_s) + incoming sum
                log_factor_s = graph.log_priors[s] + inc  # (2,)

                # For each x_d: sum over x_s
                # shape: (x_s=2, x_d=2)
                joint = log_psi + log_factor_s.unsqueeze(1)  # (2, 2)
                new_msg = log_sum_exp(joint, dim=0).squeeze(0)  # (2,)

                # Normalise (subtract log-partition of this message)
                new_msg = new_msg - log_sum_exp(new_msg.unsqueeze(0), dim=1).squeeze()

                new_log_msgs[e] = new_msg

            # Damping
            new_log_msgs = (
                self.damping * new_log_msgs
                + (1 - self.damping) * log_msgs
            )

            # Convergence check
            delta = (new_log_msgs - log_msgs).abs().max().item()
            log_msgs = new_log_msgs

            if self.verbose:
                logger.debug(f"  BP iter {iteration+1:3d}  |Δmsg|_∞ = {delta:.6f}")

            if delta < self.tol:
                if self.verbose:
                    logger.debug(f"  BP converged at iteration {iteration+1}")
                break
        else:
            if self.verbose:
                logger.warning(
                    f"BP did NOT converge in {self.max_iter} iterations "
                    f"(final |Δ| = {delta:.6f})"
                )

        # Compute beliefs
        # For each node n: sum over all incoming directed edges (k→n)
        log_beliefs = graph.log_priors.clone()  # (N, 2)
        for e in range(2 * E):
            d = full_dst[e].item()
            log_beliefs[d] += log_msgs[e]

        # Normalise beliefs → probabilities
        # P(x_i=1) = softmax(log_beliefs)[1]
        log_beliefs = log_beliefs - log_sum_exp(log_beliefs, dim=1)
        probs = log_beliefs[:, 1].exp()  # (N,)
        return probs

    # ------------------------------------------------------------------ #
    #  Convenience: set seed labels as strong priors                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def make_log_priors(
        num_nodes: int,
        seed_indices: Tensor,
        seed_labels: Tensor,
        confidence: float = 10.0,
    ) -> Tensor:
        """
        Build log-prior tensor.

        Seed nodes get a strong one-sided prior (± confidence).
        Unlabelled nodes get a uniform prior log(0.5, 0.5).

        Parameters
        ----------
        confidence : float
            Log-odds magnitude for seed nodes.  Higher = harder constraint.
        """
        log_priors = torch.zeros(num_nodes, 2)
        for idx, label in zip(seed_indices.tolist(), seed_labels.tolist()):
            if label == 1:  # phishing
                log_priors[idx] = torch.tensor([-confidence, confidence])
            else:           # benign
                log_priors[idx] = torch.tensor([confidence, -confidence])
        return log_priors


# ─────────────────────────────────────────────────────────────────────────────
# Default compatibility potentials
# ─────────────────────────────────────────────────────────────────────────────

def homophily_potential(strength: float = 1.5) -> Tensor:
    """
    Log-domain 2×2 compatibility matrix favouring same-label edges.

      Ψ = [[strength, -strength],
           [-strength,  strength]]   (log domain)

    strength > 0 ⟹ homophily (same-label edges preferred).
    strength < 0 ⟹ heterophily.
    strength = 0 ⟹ uninformative (all messages stay at prior).
    """
    return torch.tensor([
        [ strength, -strength],
        [-strength,  strength],
    ])
