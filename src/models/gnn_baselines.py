"""
GraphSAGE and Heterogeneous Graph Transformer (HGT) baselines for PhishBP.
Compatible with PyTorch Geometric (PyG).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HGTConv, Linear


class GraphSAGEModel(nn.Module):
    """
    Inductive 2/3-layer GraphSAGE baseline.
    Operates on a homogeneous projection or single node/edge type representation.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 1, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.classifier(x)
        return logits.squeeze(-1)


class HeteroHGT(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) for multi-relational graphs:
    Node types: URL, Domain, IP, Nameserver, ASN
    """
    def __init__(self, metadata, in_channels_dict: dict, hidden_channels: int = 64, out_channels: int = 1, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.lin_dict = nn.ModuleDict()
        for node_type, in_dim in in_channels_dict.items():
            self.lin_dict[node_type] = Linear(in_dim, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        self.dropout = dropout
        self.out_lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Initial projection to common hidden dimension
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = F.relu(self.lin_dict[node_type](x))

        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {key: F.dropout(F.relu(val), p=self.dropout, training=self.training) for key, val in h_dict.items()}

        # Predict target node type logits (e.g. 'domain' or 'url')
        target_repr = h_dict.get('domain', list(h_dict.values())[0])
        logits = self.out_lin(target_repr).squeeze(-1)
        return logits, h_dict
