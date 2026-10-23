"""
Graph Neural Network (GNN) branch for chemical structure representation learning.
Processes molecular graphs with 78-dimensional atom features into a fixed-length
embedding vector using GCN, GAT, or GIN with batch normalization and global pooling.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)


class MolecularGNN(nn.Module):
    """
    3-layer Molecular Graph Neural Network branch.

    Args:
        input_dim (int): Dimensionality of input node/atom features (default: 78).
        hidden_dim (int): Hidden representation size (default: 256).
        output_dim (int): Final output drug embedding size (default: 256).
        gnn_type (str): GNN architecture type: 'gcn', 'gat', or 'gin' (default: 'gcn').
        dropout (float): Dropout probability between layers (default: 0.2).
        pooling (str): Readout aggregation ('mean', 'max', 'sum'). Default: 'mean'.
    """

    def __init__(
        self,
        input_dim: int = 78,
        hidden_dim: int = 256,
        output_dim: int = 256,
        gnn_type: str = "gcn",
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gnn_type = gnn_type.lower()
        self.dropout = dropout
        self.pooling = pooling.lower()

        # Build GNN Layer 1
        self.conv1 = self._build_conv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Build GNN Layer 2
        self.conv2 = self._build_conv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Build GNN Layer 3
        self.conv3 = self._build_conv(hidden_dim, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)

        # Optional linear projection if output_dim differs from pooling
        self.proj = nn.Linear(output_dim, output_dim) if output_dim != hidden_dim else nn.Identity()

    def _build_conv(self, in_features: int, out_features: int) -> nn.Module:
        """Construct individual graph convolution layer based on gnn_type."""
        if self.gnn_type == "gcn":
            return GCNConv(in_features, out_features)
        elif self.gnn_type == "gat":
            # Multi-head attention with 4 heads, projecting back to out_features
            heads = 4
            assert out_features % heads == 0, "out_features must be divisible by heads for GAT"
            return GATConv(in_features, out_features // heads, heads=heads, concat=True)
        elif self.gnn_type == "gin":
            # GIN with a 2-layer MLP for isomorphic graph expressiveness
            gin_mlp = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Linear(out_features, out_features),
            )
            return GINConv(gin_mlp)
        else:
            raise ValueError(f"Unsupported GNN type: '{self.gnn_type}'. Expected 'gcn', 'gat', or 'gin'.")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Molecular GNN branch.

        Args:
            x (torch.Tensor): Atom feature matrix of shape [num_nodes, 78].
            edge_index (torch.Tensor): Graph adjacency tensor of shape [2, num_edges].
            batch (Optional[torch.Tensor]): Batch assignment vector of shape [num_nodes].
                                            If None, all nodes belong to a single graph.

        Returns:
            torch.Tensor: Aggregated graph representation tensor of shape [batch_size, output_dim].
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Handle empty graph edge case (e.g. single isolated atom)
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)

        # Layer 1: Conv -> BatchNorm -> ReLU -> Dropout
        h = self.conv1(x, edge_index)
        if h.size(0) > 1:
            h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2: Conv -> BatchNorm -> ReLU -> Dropout
        h = self.conv2(h, edge_index)
        if h.size(0) > 1:
            h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 3: Conv -> BatchNorm -> ReLU -> Dropout
        h = self.conv3(h, edge_index)
        if h.size(0) > 1:
            h = self.bn3(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Global Readout Pooling
        if self.pooling == "mean":
            out = global_mean_pool(h, batch)
        elif self.pooling == "max":
            out = global_max_pool(h, batch)
        elif self.pooling == "sum":
            out = global_add_pool(h, batch)
        else:
            out = global_mean_pool(h, batch)

        out = self.proj(out)
        return out
