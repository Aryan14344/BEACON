"""
Core Novel Method H3:
Belief Propagation layer with learnable per-relation compatibility potentials (Ψ_r)
trained end-to-end with GNN representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


def log_sum_exp(x: torch.Tensor, dim: int) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    return (x - m).exp().sum(dim=dim, keepdim=True).log() + m


class BPMessagePassingLayer(nn.Module):
    """
    Differentiable Loopy Belief Propagation in log domain.
    Learns 2x2 compatibility potential matrix Ψ_r for each relation r.
    """
    def __init__(self, num_relations: int, num_iterations: int = 5, damping: float = 0.5):
        super().__init__()
        self.num_relations = num_relations
        self.num_iterations = num_iterations
        self.damping = damping

        # Learnable log-potentials: shape (num_relations, 2, 2)
        # Initialized with homophily prior: higher diagonal weights
        initial_psi = torch.zeros(num_relations, 2, 2)
        for r in range(num_relations):
            initial_psi[r] = torch.tensor([[1.5, -1.5], [-1.5, 1.5]])
        self.log_psi = nn.Parameter(initial_psi)

    def forward(self, log_priors: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor):
        """
        Args:
            log_priors: (N, 2) tensor from GNN or seed logits
            edge_index: (2, E) undirected edge indices
            edge_type: (E,) relation type id
        Returns:
            log_beliefs: (N, 2) posterior log-beliefs
        """
        N = log_priors.size(0)
        E = edge_index.size(1)

        src, dst = edge_index[0], edge_index[1]
        full_src = torch.cat([src, dst], dim=0)
        full_dst = torch.cat([dst, src], dim=0)
        full_etype = torch.cat([edge_type, edge_type], dim=0)
        num_dir_edges = 2 * E

        # Directed messages: (2E, 2)
        log_msgs = torch.zeros((num_dir_edges, 2), device=log_priors.device)

        # Vectorized edge-neighborhood message passing
        rev_idx = torch.cat([torch.arange(E, 2*E, device=edge_index.device),
                             torch.arange(0, E, device=edge_index.device)])

        for _ in range(self.num_iterations):
            # Sum incoming messages per node: (N, 2)
            node_incoming = torch.zeros((N, 2), device=log_priors.device)
            node_incoming.index_add_(0, full_dst, log_msgs)

            # Incoming to source node 's' minus the reverse message (from d -> s)
            s = full_src
            inc_to_s = node_incoming[s] - log_msgs[rev_idx]  # (2E, 2)
            factor_s = log_priors[s] + inc_to_s             # (2E, 2)

            # Apply relation potential: log_psi is (R, 2, 2) -> full_etype maps to (2E, 2, 2)
            psi_e = self.log_psi[full_etype]                 # (2E, 2, 2)
            # joint(x_s, x_d) = psi(x_s, x_d) + factor_s(x_s)
            joint = psi_e + factor_s.unsqueeze(2)           # (2E, 2, 2)

            # Marginalize out source x_s via log-sum-exp -> new message for x_d
            new_msg = log_sum_exp(joint, dim=1).squeeze(1)  # (2E, 2)
            # Normalize message
            new_msg = new_msg - log_sum_exp(new_msg, dim=1)

            # Apply damping
            log_msgs = self.damping * new_msg + (1.0 - self.damping) * log_msgs

        # Final belief aggregation
        node_incoming = torch.zeros((N, 2), device=log_priors.device)
        node_incoming.index_add_(0, full_dst, log_msgs)
        log_beliefs = log_priors + node_incoming
        log_beliefs = log_beliefs - log_sum_exp(log_beliefs, dim=1)
        return log_beliefs


class PhishBPHybridH3(nn.Module):
    """
    End-to-end H3 Hybrid Model:
    Feature/GNN Encoder -> Initial Log-Prior Beliefs -> Differentiable BP Layer -> Posteriors
    """
    def __init__(self, in_features: int, hidden_dim: int, num_relations: int, bp_iters: int = 5, damping: float = 0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Generates 2-class logits (log-priors)
        )
        self.bp_layer = BPMessagePassingLayer(num_relations=num_relations, num_iterations=bp_iters, damping=damping)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor):
        # 1. Feature encoder generates node-level priors
        log_priors = F.log_softmax(self.encoder(x), dim=-1)

        # 2. End-to-end BP message passing refines priors with graph topology and learned potentials
        log_beliefs = self.bp_layer(log_priors, edge_index, edge_type)
        return log_beliefs
