"""
Real-time Prediction Demo for Cancer Drug Response.
Loads the trained checkpoint and predicts ln(IC50) for candidate (drug, cell) pairs.
"""

from pathlib import Path
import numpy as np
import torch

from src.config import Config
from src.dataset import smiles_to_graph
from src.models.multimodal_net import MultimodalDrugResponseNet


def run_prediction():
    config = Config()
    device = config.device
    checkpoint_path = config.checkpoints_dir / "best_model.pt"

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}. Please run 'python main.py --mode demo' first.")
        return

    # 1. Initialize model architecture
    model = MultimodalDrugResponseNet(
        atom_feature_dim=config.atom_feature_dim,
        gnn_hidden_dim=config.gnn_hidden_dim,
        gnn_output_dim=config.gnn_output_dim,
        gnn_type=config.gnn_type,
        num_genes=config.num_genes,
        genomic_hidden=config.genomic_hidden,
        fusion_dims=config.fusion_dims,
    )

    # 2. Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Candidate drug SMILES (Imatinib) and cell line gene expression (MCF-7 simulation)
    drug_name = "Imatinib (Gleevec)"
    smiles = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc4nccc(n4)c5cccnc5"
    graph = smiles_to_graph(smiles)

    np.random.seed(42)
    sample_cell_genomics = np.random.randn(978).astype(np.float32)

    # 4. Format tensors
    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    genomics = torch.tensor(sample_cell_genomics, dtype=torch.float).unsqueeze(0).to(device)

    # 5. Predict response
    with torch.no_grad():
        predicted_val = model(x, edge_index, batch, genomics).item()

    print("\n" + "=" * 65)
    print("                 REAL-TIME INFERENCE RESULT                    ")
    print("=" * 65)
    print(f"Compound Name       : {drug_name}")
    print(f"SMILES String       : {smiles}")
    print(f"Predicted ln(IC50)  : {predicted_val:.4f}")
    if predicted_val < 0.0:
        verdict = "High Sensitivity (Drug is highly potent and effective)"
    elif predicted_val < 1.5:
        verdict = "Moderate Sensitivity (Effective within therapeutic window)"
    else:
        verdict = "Drug Resistance (High concentration required / Ineffective)"
    print(f"Biological Verdict  : {verdict}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_prediction()