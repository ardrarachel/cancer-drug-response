"""
Multimodal Drug Response Network fusing molecular graph representations
with cancer cell transcriptomic profiles for continuous ln(IC50) regression.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from src.models.gnn_drug import MolecularGNN
from src.models.mlp_genomics import GenomicMLP


class MultimodalDrugResponseNet(nn.Module):
    """
    Unified Multimodal Deep Learning Predictor.

    Fuses:
      1. Molecular Graph Branch (GCN/GAT/GIN) -> Drug representation (256-d)
      2. Cellular Genomics Branch (MLP) -> Cell representation (256-d)
    Fused representation: 512-d cross-modal concatenation
    Regression Head: 512 -> 256 -> 64 -> 1 with LayerNorm, LeakyReLU, and Dropout.

    Args:
        atom_feature_dim (int): Number of node/atom features (default: 78).
        gnn_hidden_dim (int): Hidden dimension for GNN (default: 256).
        gnn_output_dim (int): Drug embedding dimension (default: 256).
        gnn_type (str): Graph convolution type ('gcn', 'gat', 'gin').
        gnn_dropout (float): Dropout probability for GNN branch.
        num_genes (int): Number of L1000 landmark genes (default: 978).
        genomic_hidden (list): Hidden dimensions for genomics MLP (default: [512, 256]).
        genomic_dropout (float): Dropout probability for genomics MLP.
        fusion_dims (list): Layer dimensions for regression head (default: [512, 256, 64, 1]).
        dropout_rate (float): Dropout probability in the regression head.
    """

    def __init__(
        self,
        atom_feature_dim: int = 78,
        gnn_hidden_dim: int = 256,
        gnn_output_dim: int = 256,
        gnn_type: str = "gcn",
        gnn_dropout: float = 0.2,
        num_genes: int = 978,
        genomic_hidden: list = None,
        genomic_dropout: float = 0.3,
        fusion_dims: list = None,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        if genomic_hidden is None:
            genomic_hidden = [512, 256]
        if fusion_dims is None:
            fusion_dims = [512, 256, 64, 1]

        # 1. Molecular Graph Encoder Branch
        self.drug_encoder = MolecularGNN(
            input_dim=atom_feature_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            gnn_type=gnn_type,
            dropout=gnn_dropout,
        )

        # 2. Cellular Genomics Encoder Branch
        self.cell_encoder = GenomicMLP(
            input_dim=num_genes,
            hidden_dims=genomic_hidden,
            dropout=genomic_dropout,
        )

        cell_out_dim = genomic_hidden[-1]
        fusion_in_dim = gnn_output_dim + cell_out_dim  # 256 + 256 = 512

        # 3. Cross-Modal Regression Head
        head_layers = []
        current_dim = fusion_in_dim

        for next_dim in fusion_dims[1:-1]:  # e.g., 256, 64
            head_layers.append(nn.Linear(current_dim, next_dim))
            head_layers.append(nn.LayerNorm(next_dim))
            head_layers.append(nn.LeakyReLU(negative_slope=0.2))
            head_layers.append(nn.Dropout(p=dropout_rate))
            current_dim = next_dim

        # Final linear projection to scalar ln(IC50)
        head_layers.append(nn.Linear(current_dim, fusion_dims[-1]))
        self.regression_head = nn.Sequential(*head_layers)

    def _unpack_inputs(
        self,
        data_or_x: Union[Data, Batch, torch.Tensor],
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        genomics: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Normalize inputs from either a PyG Data/Batch container or distinct tensors."""
        if isinstance(data_or_x, (Data, Batch)):
            x = data_or_x.x
            edge_index = data_or_x.edge_index
            batch = getattr(data_or_x, "batch", None)
            genomics = data_or_x.genomics
        else:
            x = data_or_x
            assert edge_index is not None, "edge_index must be provided when passing raw tensors"
            assert genomics is not None, "genomics must be provided when passing raw tensors"
        return x, edge_index, batch, genomics

    def get_embeddings(
        self,
        data_or_x: Union[Data, Batch, torch.Tensor],
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        genomics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract fused cross-modal latent embeddings (512-dim) for t-SNE / UMAP visualization.

        Returns:
            torch.Tensor: Multimodal representation tensor [batch_size, 512].
        """
        x, edge_index, batch, genomics = self._unpack_inputs(data_or_x, edge_index, batch, genomics)
        drug_emb = self.drug_encoder(x, edge_index, batch)
        cell_emb = self.cell_encoder(genomics)
        fusion_emb = torch.cat([drug_emb, cell_emb], dim=-1)
        return fusion_emb

    def forward(
        self,
        data_or_x: Union[Data, Batch, torch.Tensor],
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        genomics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict continuous ln(IC50) response for (drug, cell line) pairs.

        Returns:
            torch.Tensor: Predicted values of shape [batch_size, 1].
        """
        fusion_emb = self.get_embeddings(data_or_x, edge_index, batch, genomics)
        output = self.regression_head(fusion_emb)
        return output
