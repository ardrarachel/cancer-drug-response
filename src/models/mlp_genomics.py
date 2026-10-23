"""
Deep Multilayer Perceptron (MLP) branch for cancer cell genomics featurization.
Processes 978 L1000 landmark gene expression profiles with BatchNorm, LeakyReLU,
and Dropout into a dense 256-dimensional cellular latent state.
"""

from typing import List
import torch
import torch.nn as nn


class GenomicMLP(nn.Module):
    """
    Genomics MLP branch processing transcriptomic profiles.

    Architecture:
      Input (num_genes, e.g. 978)
        -> Linear(978, 512) -> BatchNorm1d(512) -> LeakyReLU(0.2) -> Dropout(0.3)
        -> Linear(512, 256) -> BatchNorm1d(256) -> LeakyReLU(0.2) -> Dropout(0.3)
        -> Output (256)

    Args:
        input_dim (int): Number of gene expression features (default: 978).
        hidden_dims (List[int]): List of hidden dimensions (default: [512, 256]).
        dropout (float): Dropout probability (default: 0.3).
        negative_slope (float): Negative slope for LeakyReLU (default: 0.2).
    """

    def __init__(
        self,
        input_dim: int = 978,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.negative_slope = negative_slope

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim

        self.network = nn.ModuleList(layers)

    def forward(self, genomics: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for cell line genomics.

        Args:
            genomics (torch.Tensor): Tensor of shape [batch_size, num_genes] or [num_genes].

        Returns:
            torch.Tensor: Cellular latent embedding of shape [batch_size, output_dim].
        """
        # Ensure 2D tensor [batch_size, num_genes]
        if genomics.dim() == 1:
            genomics = genomics.unsqueeze(0)

        x = genomics
        # Iterate through linear, batchnorm, activation, dropout blocks
        for i in range(0, len(self.network), 4):
            linear = self.network[i]
            bn = self.network[i + 1]
            act = self.network[i + 2]
            drop = self.network[i + 3]

            x = linear(x)
            # Guard against batch_size=1 training crash in BatchNorm1d
            if x.size(0) > 1 or not self.training:
                x = bn(x)
            x = act(x)
            x = drop(x)

        return x
