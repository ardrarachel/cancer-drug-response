"""
Configuration module for Cancer Drug Response Predictor.
Defines hyperparameters, paths, random seeds, and compute device configuration.
"""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import random
from typing import List

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DrugResponse.Config")


def get_default_device() -> torch.device:
    """
    Auto-select compute device based on hardware availability.
    Priority: CUDA -> Apple MPS -> CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA hardware accelerator: {device_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device for computation")
    return device


def set_seed(seed: int = 42) -> None:
    """
    Enforce deterministic reproducibility across Python, NumPy, PyTorch, and CuDNN.

    Args:
        seed (int): The integer random seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Global random seed set to {seed}")


@dataclass
class Config:
    """
    Centralized configuration dataclass managing model hyperparameters,
    paths, and training settings.
    """

    # Global reproducibility seed
    seed: int = 42

    # Directory Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    synthetic_data_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)

    # Chemical Graph Branch Dimensions (RDKit Featurization & GNN)
    atom_feature_dim: int = 78
    gnn_type: str = "gcn"  # Options: 'gcn', 'gat', 'gin'
    gnn_hidden_dim: int = 256
    gnn_output_dim: int = 256
    gnn_num_layers: int = 3
    gnn_dropout: float = 0.1

    # Genomic Branch Dimensions (L1000 landmark genes)
    num_genes: int = 978
    genomic_hidden: List[int] = field(default_factory=lambda: [512, 256])
    genomic_dropout: float = 0.1

    # Multimodal Cross-Modal Fusion Neck and Regression Head
    fusion_dims: List[int] = field(default_factory=lambda: [512, 256, 64, 1])
    dropout_rate: float = 0.1

    # Optimization & Training Hyperparameters
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    max_epochs: int = 28
    patience: int = 8
    min_delta: float = 1e-4

    # Dataset Partitioning (75% Train, 10% Val, 15% Held-out Test)
    train_ratio: float = 0.75
    val_ratio: float = 0.10
    test_ratio: float = 0.15

    # Hardware Device
    device: torch.device = field(default_factory=get_default_device)

    def __post_init__(self) -> None:
        """Resolve and initialize dynamic directory paths."""
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.synthetic_data_dir = self.data_dir / "synthetic"
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.reports_dir = self.project_root / "reports"
        self.figures_dir = self.reports_dir / "figures"
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        """Create project directory hierarchy if not already present."""
        for path in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.synthetic_data_dir,
            self.checkpoints_dir,
            self.figures_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Default singleton instance for convenience
default_config = Config()

