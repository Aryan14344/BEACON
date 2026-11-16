"""
Unit tests for Stage 3 & 4:
- GraphSAGE model execution & forward pass
- HeteroHGT model execution & forward pass
- BPMessagePassingLayer (H3) gradient flow and learning
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from torch_geometric.data import HeteroData

from src.models.gnn_baselines import GraphSAGEModel, HeteroHGT
from src.models.bp_layer import BPMessagePassingLayer, PhishBPHybridH3


class TestGNNBaselines:
    def test_graphsage_forward(self):
        N = 100
        in_dim = 16
        x = torch.randn(N, in_dim)
        edge_index = torch.randint(0, N, (2, 300))
        
        model = GraphSAGEModel(in_channels=in_dim, hidden_channels=32, out_channels=1)
        out = model(x, edge_index)
        assert out.shape == (N,)
        assert torch.isfinite(out).all()

    def test_hgt_forward(self):
        # Create minimal synthetic HeteroData
        data = HeteroData()
        data['domain'].x = torch.randn(50, 8)
        data['ip'].x = torch.randn(20, 8)
        
        data['domain', 'resolves_to', 'ip'].edge_index = torch.stack([
            torch.randint(0, 50, (60,)),
            torch.randint(0, 20, (60,))
        ])
        data['ip', 'rev_resolves_to', 'domain'].edge_index = torch.stack([
            data['domain', 'resolves_to', 'ip'].edge_index[1],
            data['domain', 'resolves_to', 'ip'].edge_index[0]
        ])
        
        in_channels_dict = {'domain': 8, 'ip': 8}
        model = HeteroHGT(data.metadata(), in_channels_dict=in_channels_dict, hidden_channels=16, num_heads=2)
        logits, h_dict = model(data.x_dict, data.edge_index_dict)
        assert logits.shape == (50,)
        assert 'domain' in h_dict and 'ip' in h_dict


class TestH3HybridBP:
    def test_bp_message_passing_layer(self):
        N = 20
        E = 40
        num_relations = 3
        
        log_priors = torch.randn(N, 2)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, num_relations, (E,))
        
        layer = BPMessagePassingLayer(num_relations=num_relations, num_iterations=4, damping=0.5)
        beliefs = layer(log_priors, edge_index, edge_type)
        
        assert beliefs.shape == (N, 2)
        assert torch.isfinite(beliefs).all()
        # Verify normalization (log beliefs should sum to 1 in probability space)
        probs = beliefs.exp().sum(dim=1)
        assert torch.allclose(probs, torch.ones(N), atol=1e-5)

    def test_h3_gradient_flow(self):
        """Verify that gradients flow back into the learnable potentials log_psi."""
        N = 30
        E = 50
        in_dim = 10
        num_relations = 2
        
        x = torch.randn(N, in_dim, requires_grad=True)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, num_relations, (E,))
        
        model = PhishBPHybridH3(in_features=in_dim, hidden_dim=16, num_relations=num_relations, bp_iters=3)
        log_beliefs = model(x, edge_index, edge_type)
        
        # Loss on dummy binary targets
        targets = torch.randint(0, 2, (N,))
        loss = torch.nn.functional.nll_loss(log_beliefs, targets)
        loss.backward()
        
        # Potentials must receive non-zero gradients
        assert model.bp_layer.log_psi.grad is not None
        assert torch.norm(model.bp_layer.log_psi.grad).item() > 0.0
        assert x.grad is not None
