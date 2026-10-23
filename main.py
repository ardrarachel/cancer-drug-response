"""
Main CLI entrypoint for Cancer Drug Response Predictor.
Orchestrates data preparation, model training, evaluation, and publication-grade figure generation.

Usage:
    python main.py --mode demo
    python main.py --mode train --epochs 50 --batch_size 32
    python main.py --mode eval
    python main.py --mode synthesize
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.config import Config, set_seed
from src.dataset import create_dataloaders, generate_synthetic_benchmark, RealDrugCellDataset
from src.evaluate import evaluate_model
from src.models.multimodal_net import MultimodalDrugResponseNet
from src.train import train_model
from src.visualize import generate_all_diagnostic_figures

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DrugResponse.Main")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cancer Drug Response Predictor using Molecular Graphs and Genomics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "train", "eval", "synthesize"],
        help="Execution mode: 'demo' (end-to-end demo), 'train', 'eval', or 'synthesize'.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 20 for demo, 100 for train).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size for training and inference.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="L2 regularization penalty.",
    )
    parser.add_argument(
        "--gnn_type",
        type=str,
        default="gcn",
        choices=["gcn", "gat", "gin"],
        help="Molecular graph convolution architecture.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Target compute device.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1200,
        help="Number of synthetic response pairs to generate.",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="real",
        choices=["real", "synthetic"],
        help="Dataset source: 'real' (GDSC benchmark from download_real_data.py) or 'synthetic'.",
    )
    return parser.parse_args()


def run_demo(config: Config) -> None:
    """
    Execute full end-to-end demonstration:
      1. Load dataset (real GDSC benchmark or synthetic)
      2. Split into train (80%), val (10%), test (10%) PyG DataLoaders
      3. Instantiate MultimodalDrugResponseNet
      4. Train model for epochs with Cosine Annealing and early stopping
      5. Evaluate on held-out test split
      6. Generate and export 4 publication-grade figures (300 DPI)
    """
    logger.info("=" * 70)
    logger.info("STARTING DEMO: Cancer Drug Response Prediction (GNN + Genomics)")
    logger.info("=" * 70)

    # 1. Load dataset
    if getattr(config, "data_source", "real") == "real":
        logger.info("Step 1/5: Loading real GDSC benchmark dataset (via RealDrugCellDataset)...")
        dataset = RealDrugCellDataset(
            csv_path=config.processed_data_dir / "gdsc_pairs.csv",
            auto_download=True,
        )
    else:
        logger.info("Step 1/5: Synthesizing real-world oncology benchmark dataset...")
        dataset = generate_synthetic_benchmark(
            num_samples=config.num_samples if hasattr(config, "num_samples") else 1200,
            num_drugs=60,
            num_cells=40,
            data_dir=config.synthetic_data_dir,
            seed=config.seed,
        )

    # 2. Construct DataLoaders
    logger.info("Step 2/5: Creating PyG DataLoaders (80% Train, 10% Val, 10% Test)...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset=dataset,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    # 3. Model instantiation
    logger.info(f"Step 3/5: Initializing MultimodalDrugResponseNet ({config.gnn_type.upper()} branch)...")
    model = MultimodalDrugResponseNet(
        atom_feature_dim=config.atom_feature_dim,
        gnn_hidden_dim=config.gnn_hidden_dim,
        gnn_output_dim=config.gnn_output_dim,
        gnn_type=config.gnn_type,
        gnn_dropout=config.gnn_dropout,
        num_genes=config.num_genes,
        genomic_hidden=config.genomic_hidden,
        genomic_dropout=config.genomic_dropout,
        fusion_dims=config.fusion_dims,
        dropout_rate=config.dropout_rate,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} trainable weights.")

    # 4. Training loop
    logger.info(f"Step 4/5: Training model for {config.max_epochs} epochs on {config.device}...")
    model, history, best_epoch, checkpoint_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        verbose=True,
    )

    # 5. Held-out test evaluation
    logger.info("Step 5/5: Evaluating model on held-out test split...")
    criterion = nn.MSELoss()
    metrics, y_true, y_pred, embeddings = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=config.device,
    )

    # 6. Generate publication figures
    logger.info("Generating publication-grade diagnostic plots (300 DPI)...")
    fig_paths = generate_all_diagnostic_figures(
        history=history,
        y_true=y_true,
        y_pred=y_pred,
        embeddings=embeddings,
        metrics=metrics,
        figures_dir=config.figures_dir,
        best_epoch=best_epoch,
    )

    # Print Closing Summary Report
    print("\n" + "=" * 70)
    print("                     EVALUATION RESULTS SUMMARY                        ")
    print("=" * 70)
    print(f"  Test Loss (MSE):                 {metrics['loss']:.4f}")
    print(f"  Root Mean Squared Error (RMSE):  {metrics['rmse']:.4f}")
    print(f"  Mean Absolute Error (MAE):       {metrics['mae']:.4f}")
    print(f"  Pearson Correlation (PCC):       {metrics['pcc']:.4f} (p = {metrics['pcc_pvalue']:.2e})")
    print(f"  Spearman Correlation (SCC):      {metrics['scc']:.4f} (p = {metrics['scc_pvalue']:.2e})")
    print(f"  Coefficient of Det. (R^2):       {metrics['r2']:.4f}")
    print("-" * 70)
    print(f"  Best Model Checkpoint:           {checkpoint_path.resolve()}")
    print("  Publication Figures Generated:")
    for name, path in fig_paths.items():
        print(f"    - [{name}]: {path.resolve()}")
    print("=" * 70 + "\n")


def main() -> None:
    """CLI orchestrator."""
    args = parse_args()
    set_seed(args.seed)

    config = Config()
    config.seed = args.seed
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.gnn_type = args.gnn_type

    # Configure device
    if args.device != "auto":
        config.device = torch.device(args.device)

    # Default epochs
    if args.epochs is not None:
        config.max_epochs = args.epochs
    else:
        config.max_epochs = 28
    config.patience = 8

    config.num_samples = args.num_samples
    config.data_source = args.data_source

    if args.mode == "demo":
        run_demo(config)
    elif args.mode == "synthesize":
        generate_synthetic_benchmark(
            num_samples=args.num_samples,
            num_drugs=60,
            num_cells=40,
            data_dir=config.synthetic_data_dir,
            seed=config.seed,
        )
    elif args.mode == "train":
        if config.data_source == "real":
            dataset = RealDrugCellDataset(
                csv_path=config.processed_data_dir / "gdsc_pairs.csv",
                auto_download=True,
            )
        else:
            dataset = generate_synthetic_benchmark(
                num_samples=args.num_samples,
                num_drugs=60,
                num_cells=40,
                data_dir=config.synthetic_data_dir,
                seed=config.seed,
            )
        train_loader, val_loader, _ = create_dataloaders(
            dataset=dataset,
            batch_size=config.batch_size,
            seed=config.seed,
        )
        model = MultimodalDrugResponseNet(
            atom_feature_dim=config.atom_feature_dim,
            gnn_hidden_dim=config.gnn_hidden_dim,
            gnn_output_dim=config.gnn_output_dim,
            gnn_type=config.gnn_type,
            gnn_dropout=config.gnn_dropout,
            num_genes=config.num_genes,
            genomic_hidden=config.genomic_hidden,
            genomic_dropout=config.genomic_dropout,
            fusion_dims=config.fusion_dims,
            dropout_rate=config.dropout_rate,
        )
        train_model(model, train_loader, val_loader, config)
    elif args.mode == "eval":
        if config.data_source == "real":
            dataset = RealDrugCellDataset(
                csv_path=config.processed_data_dir / "gdsc_pairs.csv",
                auto_download=True,
            )
        else:
            dataset = generate_synthetic_benchmark(
                num_samples=args.num_samples,
                num_drugs=60,
                num_cells=40,
                seed=config.seed,
            )
        _, _, test_loader = create_dataloaders(dataset=dataset, seed=config.seed)
        model = MultimodalDrugResponseNet(
            atom_feature_dim=config.atom_feature_dim,
            gnn_hidden_dim=config.gnn_hidden_dim,
            gnn_output_dim=config.gnn_output_dim,
            gnn_type=config.gnn_type,
            num_genes=config.num_genes,
        )
        checkpoint_path = config.checkpoints_dir / "best_model.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        model.to(config.device)
        evaluate_model(model, test_loader, nn.MSELoss(), config.device)


if __name__ == "__main__":
    main()
