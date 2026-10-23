"""
Unit tests for neural network architectures, dimensions, and gradient backpropagation.
"""

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.models.gnn_drug import MolecularGNN
from src.models.mlp_genomics import GenomicMLP
from src.models.multimodal_net import MultimodalDrugResponseNet


class TestMolecularGNN:
    """Test suite for the Graph Neural Network molecular branch."""

    @pytest.mark.parametrize("gnn_type", ["gcn", "gat", "gin"])
    def test_gnn_backbones_forward(self, gnn_type):
        """Verify forward pass on GCN, GAT, and GIN backbones."""
        model = MolecularGNN(
            input_dim=78,
            hidden_dim=256,
            output_dim=256,
            gnn_type=gnn_type,
            dropout=0.1,
        )
        x = torch.randn(15, 78)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        batch = torch.tensor([0] * 8 + [1] * 7, dtype=torch.long)

        out = model(x, edge_index, batch)
        assert out.shape == torch.Size([2, 256])
        assert not torch.isnan(out).any()

    def test_single_graph_no_batch(self):
        """Verify handling of single graph without explicit batch vector."""
        model = MolecularGNN(input_dim=78, hidden_dim=128, output_dim=128)
        x = torch.randn(10, 78)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        out = model(x, edge_index, batch=None)
        assert out.shape == torch.Size([1, 128])


class TestGenomicMLP:
    """Test suite for the transcriptomic genomic branch."""

    def test_genomics_forward_shape(self):
        """Verify output dimension [batch_size, 256] from 978-dim input."""
        mlp = GenomicMLP(input_dim=978, hidden_dims=[512, 256], dropout=0.2)
        genomics = torch.randn(8, 978)

        out = mlp(genomics)
        assert out.shape == torch.Size([8, 256])
        assert not torch.isnan(out).any()

    def test_genomics_single_sample_eval(self):
        """Verify forward pass for single cell line vector in eval mode."""
        mlp = GenomicMLP(input_dim=978, hidden_dims=[512, 256])
        mlp.eval()
        genomics = torch.randn(978)

        with torch.no_grad():
            out = mlp(genomics)
        assert out.shape == torch.Size([1, 256])


class TestMultimodalDrugResponseNet:
    """Test suite for the unified multimodal predictor."""

    @pytest.fixture
    def sample_batch(self):
        # Create 2 mock molecular graphs
        g1 = Data(
            x=torch.randn(12, 78),
            edge_index=torch.tensor([[0, 1, 2], [1, 0, 1]], dtype=torch.long),
            genomics=torch.randn(1, 978),
            y=torch.tensor([1.5], dtype=torch.float),
        )
        g2 = Data(
            x=torch.randn(8, 78),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            genomics=torch.randn(1, 978),
            y=torch.tensor([0.8], dtype=torch.float),
        )
        batch = Batch.from_data_list([g1, g2])
        return batch

    def test_multimodal_forward_and_embeddings(self, sample_batch):
        """Verify prediction shape and latent embedding extraction."""
        model = MultimodalDrugResponseNet(
            atom_feature_dim=78,
            gnn_hidden_dim=256,
            gnn_output_dim=256,
            num_genes=978,
            genomic_hidden=[512, 256],
            fusion_dims=[512, 256, 64, 1],
        )

        preds = model(sample_batch)
        assert preds.shape == torch.Size([2, 1])

        embeddings = model.get_embeddings(sample_batch)
        assert embeddings.shape == torch.Size([2, 512])

    def test_simultaneous_gradient_flow(self, sample_batch):
        """
        Critical verification: Ensure gradients flow backward through BOTH
        the molecular GNN branch and the genomic MLP branch simultaneously.
        """
        model = MultimodalDrugResponseNet(
            atom_feature_dim=78,
            gnn_hidden_dim=256,
            gnn_output_dim=256,
            num_genes=978,
            genomic_hidden=[512, 256],
            fusion_dims=[512, 256, 64, 1],
        )
        model.train()

        preds = model(sample_batch).view(-1)
        targets = sample_batch.y.view(-1)
        loss = torch.nn.functional.mse_loss(preds, targets)
        loss.backward()

        # Check gradients in GNN branch
        gnn_grads = [p.grad for p in model.drug_encoder.parameters() if p.requires_grad]
        assert len(gnn_grads) > 0, "No trainable parameters in drug_encoder"
        assert any(g is not None and torch.sum(torch.abs(g)) > 0 for g in gnn_grads), (
            "GNN branch received zero or None gradients!"
        )

        # Check gradients in Genomic branch
        cell_grads = [p.grad for p in model.cell_encoder.parameters() if p.requires_grad]
        assert len(cell_grads) > 0, "No trainable parameters in cell_encoder"
        assert any(g is not None and torch.sum(torch.abs(g)) > 0 for g in cell_grads), (
            "Genomic MLP branch received zero or None gradients!"
        )

        # Check gradients in Regression Head
        head_grads = [p.grad for p in model.regression_head.parameters() if p.requires_grad]
        assert len(head_grads) > 0, "No trainable parameters in regression_head"
        assert any(g is not None and torch.sum(torch.abs(g)) > 0 for g in head_grads), (
            "Regression head received zero or None gradients!"
        )
