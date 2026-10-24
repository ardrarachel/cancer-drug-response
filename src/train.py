"""
Training loop and checkpointing engine for Cancer Drug Response Predictor.
Integrates AdamW optimization, Cosine Annealing learning rate scheduling,
validation tracking, and early stopping.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.config import Config, default_config

logger = logging.getLogger("DrugResponse.Train")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Execute one training epoch across all batches.

    Args:
        model (nn.Module): Multimodal neural network.
        loader (DataLoader): PyG DataLoader for training split.
        optimizer (torch.optim.Optimizer): AdamW optimizer.
        criterion (nn.Module): MSE loss criterion.
        device (torch.device): Compute device.

    Returns:
        float: Average training loss across samples.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch).view(-1)
        targets = batch.y.view(-1)

        loss = criterion(predictions, targets)
        loss.backward()

        # Gradient clipping to stabilize deep multimodal training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate validation loss without computing gradients.

    Args:
        model (nn.Module): Multimodal neural network.
        loader (DataLoader): PyG DataLoader for validation split.
        criterion (nn.Module): MSE loss criterion.
        device (torch.device): Compute device.

    Returns:
        float: Average validation loss across samples.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch).view(-1)
            targets = batch.y.view(-1)

            loss = criterion(predictions, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[Config] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]], int, Path]:
    """
    Full training routine with AdamW, Cosine Annealing, and Early Stopping.

    Args:
        model (nn.Module): Initialized multimodal network.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        config (Optional[Config]): Configuration dataclass instance.
        checkpoint_path (Optional[Union[str, Path]]): Destination path for best model checkpoint.
        verbose (bool): Whether to log epoch progress.

    Returns:
        Tuple:
            - model (nn.Module): Model restored to best checkpoint weights.
            - history (Dict[str, List[float]]): Epoch loss and lr histories.
            - best_epoch (int): Epoch index that achieved minimum validation loss.
            - checkpoint_path (Path): Path to saved best_model.pt.
    """
    cfg = config or default_config
    device = cfg.device
    model.to(device)

    if checkpoint_path is None:
        checkpoint_path = cfg.checkpoints_dir / "best_model.pt"
    else:
        checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_epochs,
        eta_min=1e-6,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    patience_counter = 0

    logger.info(
        f"Starting training: {cfg.max_epochs} max epochs, patience={cfg.patience}, "
        f"lr={cfg.learning_rate}, weight_decay={cfg.weight_decay} on {device}"
    )

    epoch_iter = range(1, cfg.max_epochs + 1)
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc="Training Epochs")

    for epoch in epoch_iter:
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rates"].append(current_lr)

        # Check for improvement
        if val_loss < (best_val_loss - cfg.min_delta):
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Strictly save best model checkpoint with primitive data types
            config_dict = {
                k: v for k, v in vars(cfg).items()
                if isinstance(v, (int, float, str, bool, list, tuple, dict))
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "config": config_dict,
                },
                checkpoint_path,
            )
            improvement_flag = " [BEST SAVED]"
        else:
            patience_counter += 1
            improvement_flag = ""

        if verbose:
            msg = (
                f"Epoch {epoch:03d}/{cfg.max_epochs:03d} | "
                f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | "
                f"LR: {current_lr:.2e}{improvement_flag}"
            )
            if hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix({"train": f"{train_loss:.4f}", "val": f"{val_loss:.4f}"})
            else:
                logger.info(msg)

        # Early stopping condition
        if patience_counter >= cfg.patience:
            logger.info(
                f"Early stopping triggered at epoch {epoch}. "
                f"Validation loss did not improve for {cfg.patience} consecutive epochs."
            )
            break

    logger.info(f"Training completed. Best validation MSE: {best_val_loss:.4f} at epoch {best_epoch}.")

    # Restore best checkpoint weights
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Restored model weights from best checkpoint: {checkpoint_path.resolve()}")

    return model, history, best_epoch, checkpoint_path
