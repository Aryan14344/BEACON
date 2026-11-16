"""
Hybrids implementation:
- H1: GNN priors, then standard Loopy BP refinement (post-hoc, decoupled).
- H2: Correct-and-Smooth (C&S) post-processing on top of base predictions.
- H3: PhishBPHybridH3 (core novel method: differentiable BP layer with learnable potentials Ψ_r).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.bp_baseline import LoopyBP, BPGraph, homophily_potential
from src.models.bp_layer import BPMessagePassingLayer, PhishBPHybridH3


class H1Hybrid:
    """
    H1: GNN Priors + Post-hoc Loopy BP Refinement.
    1. Base classifier (or GNN) predicts posterior logits for each node.
    2. Convert logits to log-priors: log φ_i(x_i) = log_softmax(logits_i).
    3. Run standard damped loopy BP over the multi-relational graph with fixed homophily potentials.
    """
    def __init__(self, num_relations: int, max_iter: int = 15, damping: float = 0.5, strength: float = 2.0):
        self.num_relations = num_relations
        self.max_iter = max_iter
        self.damping = damping
        self.strength = strength
        self.bp = LoopyBP(max_iter=max_iter, damping=damping)

    def infer(self, log_priors: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        """
        Args:
            log_priors: (N, 2) tensor of class log-probabilities
            edge_index: (2, E) undirected edge indices
            edge_type: (E,) relation type id
        Returns:
            probs: (N,) P(phishing)
        """
        num_nodes = log_priors.size(0)
        potentials = [homophily_potential(strength=self.strength) for _ in range(self.num_relations)]
        
        graph = BPGraph(
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_type=edge_type,
            log_priors=log_priors,
            num_rel=self.num_relations,
            potentials=potentials,
        )
        return self.bp.infer(graph)


class CorrectAndSmooth:
    """
    H2: Correct-and-Smooth (Huang et al., ICLR 2021).
    Post-processing framework:
      1. Correct step: Propagate training residual errors (y_train - y_pred) across graph edges.
      2. Smooth step: Propagate combined predictions across graph edges.
    """
    def __init__(self, num_correct_layers: int = 3, correct_alpha: float = 0.5,
                 num_smooth_layers: int = 3, smooth_alpha: float = 0.5):
        self.num_correct_layers = num_correct_layers
        self.correct_alpha = correct_alpha
        self.num_smooth_layers = num_smooth_layers
        self.smooth_alpha = smooth_alpha

    def _propagate(self, x: Tensor, adj_norm: torch.sparse.Tensor, alpha: float, num_layers: int) -> Tensor:
        out = x.clone()
        for _ in range(num_layers):
            out = (1.0 - alpha) * torch.sparse.mm(adj_norm, out) + alpha * x
        return out

    def build_normalized_adj(self, edge_index: Tensor, num_nodes: int) -> torch.sparse.Tensor:
        """Symmetric normalized adjacency D^{-1/2} A D^{-1/2}."""
        src, dst = edge_index
        # Make sure it's undirected
        full_src = torch.cat([src, dst])
        full_dst = torch.cat([dst, src])
        
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.index_add_(0, full_src, torch.ones_like(full_src, dtype=torch.float))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        edge_weight = deg_inv_sqrt[full_src] * deg_inv_sqrt[full_dst]
        indices = torch.stack([full_src, full_dst])
        adj = torch.sparse_coo_tensor(indices, edge_weight, (num_nodes, num_nodes)).coalesce()
        return adj

    def post_process(self, base_probs: Tensor, y_train: Tensor, train_mask: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            base_probs: (N, 2) probability predictions from base classifier
            y_train: (N,) target labels (0 or 1)
            train_mask: (N,) boolean mask of training nodes
            edge_index: (2, E) undirected edge index
        Returns:
            smoothed_probs: (N, 2)
        """
        N = base_probs.size(0)
        adj = self.build_normalized_adj(edge_index, N)

        # 1. Error correction
        y_one_hot = F.one_hot(y_train[train_mask], num_classes=2).float()
        error = torch.zeros_like(base_probs)
        error[train_mask] = y_one_hot - base_probs[train_mask]

        smoothed_error = self._propagate(error, adj, self.correct_alpha, self.num_correct_layers)
        
        # Scale error by training error magnitude
        corrected = base_probs + smoothed_error
        corrected = torch.clamp(corrected, min=0.0)
        corrected = corrected / (corrected.sum(dim=-1, keepdim=True) + 1e-8)

        # Clamp training nodes back to true labels for smooth step
        corrected[train_mask] = y_one_hot

        # 2. Smooth step
        smoothed = self._propagate(corrected, adj, self.smooth_alpha, self.num_smooth_layers)
        smoothed = torch.clamp(smoothed, min=0.0)
        smoothed = smoothed / (smoothed.sum(dim=-1, keepdim=True) + 1e-8)
        return smoothed
