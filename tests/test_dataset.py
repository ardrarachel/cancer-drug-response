"""
Unit tests for chemical featurization, PyG dataset, and synthetic benchmark generator.
"""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from src.dataset import (
    ATOM_FEATURE_DIM,
    DrugCellDataset,
    generate_synthetic_benchmark,
    get_atom_features,
    smiles_to_graph,
)
from rdkit import Chem


class TestChemicalFeaturization:
    """Test suite for RDKit molecular graph conversion."""

    def test_atom_feature_dimensions(self):
        """Verify that every atom generates an exact 78-dimensional feature vector."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
        assert mol is not None

        for atom in mol.GetAtoms():
            feats = get_atom_features(atom)
            assert len(feats) == ATOM_FEATURE_DIM
            assert all(isinstance(f, float) for f in feats)

    def test_valid_smiles_graph_conversion(self):
        """Verify graph construction for valid drug structures."""
        smiles = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc4nccc(n4)c5cccnc5"  # Imatinib
        graph = smiles_to_graph(smiles)

        assert graph is not None
        assert isinstance(graph, Data)
        assert graph.x.dim() == 2
        assert graph.x.size(1) == ATOM_FEATURE_DIM
        assert graph.edge_index.dim() == 2
        assert graph.edge_index.size(0) == 2
        assert graph.edge_index.dtype == torch.long

        # Check no NaN or Inf values
        assert not torch.isnan(graph.x).any()
        assert not torch.isinf(graph.x).any()

    def test_undirected_edge_connectivity(self):
        """Verify that molecular graph edges are bidirectional (undirected)."""
        smiles = "CCO"  # Ethanol
        mol = Chem.MolFromSmiles(smiles)
        graph = smiles_to_graph(smiles)

        assert graph is not None
        num_bonds = mol.GetNumBonds()
        # Each bond should have forward and backward directed edges
        assert graph.edge_index.size(1) == 2 * num_bonds

        # Check that for every (u, v) edge, (v, u) also exists
        edges = set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))
        for u, v in edges:
            assert (v, u) in edges

    def test_invalid_smiles_handling(self):
        """Verify graceful fallback for invalid or corrupted SMILES strings."""
        invalid_smiles_list = [
            "NOT_A_VALID_SMILES",
            "C1CC",  # Unclosed ring
            "",
            "   ",
            None,
        ]
        for s in invalid_smiles_list:
            graph = smiles_to_graph(s)
            assert graph is None

    def test_single_atom_edge_case(self):
        """Verify handling of single-atom compounds with zero bonds."""
        graph = smiles_to_graph("[Na+]")
        assert graph is not None
        assert graph.x.size(0) == 1
        assert graph.x.size(1) == ATOM_FEATURE_DIM
        assert graph.edge_index.size(0) == 2
        assert graph.edge_index.size(1) == 0


class TestDrugCellDataset:
    """Test suite for PyG Dataset container and DataLoaders."""

    @pytest.fixture
    def mock_samples(self):
        return [
            {
                "drug_name": f"Drug_{i}",
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "cell_id": f"Cell_{i % 3}",
                "genomics": np.random.randn(978).astype(np.float32),
                "ic50": float(1.2 + 0.1 * i),
            }
            for i in range(16)
        ]

    def test_dataset_len_and_get(self, mock_samples):
        """Verify dataset length and single sample retrieval."""
        dataset = DrugCellDataset(mock_samples, cache_graphs=True)
        assert len(dataset) == 16

        sample = dataset[0]
        assert isinstance(sample, Data)
        assert sample.x.size(1) == ATOM_FEATURE_DIM
        assert sample.genomics.shape == torch.Size([1, 978])
        assert sample.y.shape == torch.Size([1])

    def test_dataloader_batch_collation(self, mock_samples):
        """Verify batch tensor shapes when iterating through PyG DataLoader."""
        dataset = DrugCellDataset(mock_samples, cache_graphs=True)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        batch = next(iter(loader))
        assert isinstance(batch, Batch)
        assert batch.x.dim() == 2
        assert batch.x.size(1) == ATOM_FEATURE_DIM
        assert batch.edge_index.size(0) == 2
        assert batch.batch.dim() == 1
        assert batch.genomics.shape == torch.Size([4, 978])
        assert batch.y.shape == torch.Size([4])


class TestSyntheticBenchmarkGenerator:
    """Test suite for out-of-the-box benchmark generation."""

    def test_synthetic_generator_properties(self, tmp_path):
        """Verify synthetic benchmark creation, dimensions, and distribution."""
        dataset = generate_synthetic_benchmark(
            num_samples=100,
            num_drugs=10,
            num_cells=5,
            data_dir=tmp_path,
            seed=42,
        )
        assert len(dataset) == 100

        # Check saved CSVs
        assert (tmp_path / "synthetic_drugs.csv").exists()
        assert (tmp_path / "synthetic_genomics.csv").exists()
        assert (tmp_path / "synthetic_response_matrix.csv").exists()

        # Check response distribution
        ic50_values = [s["ic50"] for s in dataset.samples]
        mean_val = float(np.mean(ic50_values))
        std_val = float(np.std(ic50_values))

        assert -5.0 < mean_val < 5.0
        assert std_val > 0.5


class TestRealDrugCellDataset:
    """Test suite for RealDrugCellDataset loading GDSC benchmark pairs."""

    def test_real_drug_cell_dataset_loading(self, tmp_path):
        import pandas as pd
        from src.dataset import RealDrugCellDataset
        from src.featurizer import smiles_to_graph_data

        # Create temporary GDSC csv
        csv_file = tmp_path / "test_gdsc_pairs.csv"
        df = pd.DataFrame([
            {"drug_name": "Imatinib", "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5", "cell_line": "K562", "ln_ic50": -1.45},
            {"drug_name": "Cisplatin", "smiles": "N.N.Cl[Pt]Cl", "cell_line": "MCF7", "ln_ic50": 0.82},
        ])
        df.to_csv(csv_file, index=False)

        ds = RealDrugCellDataset(csv_path=csv_file, auto_download=False)
        assert len(ds) == 2

        sample = ds[0]
        assert sample.x.size(1) == ATOM_FEATURE_DIM
        assert sample.genomics.shape == torch.Size([1, 978])
        assert sample.y.shape == torch.Size([1])
        assert sample.drug_name == "Imatinib"
        assert sample.cell_line == "K562"

        # Verify smiles_to_graph_data
        x, edge_index = smiles_to_graph_data("CC(=O)Oc1ccccc1C(=O)O")
        assert x.size(1) == ATOM_FEATURE_DIM
        assert edge_index.size(0) == 2

