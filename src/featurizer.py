"""
Featurizer utility module converting SMILES chemical representations
into PyTorch Geometric molecular graph tensors (x, edge_index).
"""

from typing import Tuple
import torch
from torch_geometric.data import Data
from src.dataset import smiles_to_graph, get_atom_features, ATOM_FEATURE_DIM


def smiles_to_graph_data(smiles: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a SMILES string to node features and edge index tensors.

    Args:
        smiles (str): Canonical chemical SMILES.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (x, edge_index)
            - x: [num_atoms, 78] FloatTensor
            - edge_index: [2, 2 * num_bonds] LongTensor
    """
    graph = smiles_to_graph(smiles)
    if graph is None:
        raise ValueError(f"Unable to parse molecular graph from SMILES: '{smiles}'")
    return graph.x, graph.edge_index


__all__ = ["smiles_to_graph_data", "smiles_to_graph", "get_atom_features", "ATOM_FEATURE_DIM"]
