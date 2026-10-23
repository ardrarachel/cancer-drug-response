"""
Evaluation and metrics computation module for Cancer Drug Response Predictor.
Computes RMSE, MAE, Pearson Correlation (PCC), Spearman Rank Correlation (SCC),
and Coefficient of Determination (R2) with edge-case guards.
"""

import logging
from typing import Dict, Tuple, Union
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

logger = logging.getLogger("DrugResponse.Evaluate")


def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute regression evaluation metrics and correlation statistics.

    Metrics:
      - RMSE: Root Mean Squared Error
      - MAE: Mean Absolute Error
      - PCC: Pearson Correlation Coefficient with 2-tailed p-value
      - SCC: Spearman Rank Correlation Coefficient with 2-tailed p-value
      - R2: Coefficient of Determination

    Handles edge cases: zero variance, identical arrays, small N.

    Args:
        y_true: Ground truth target values (1D).
        y_pred: Predicted values from model (1D).

    Returns:
        Dict[str, float]: Dictionary containing computed metrics.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    n_samples = len(y_true)
    if n_samples == 0 or len(y_pred) == 0:
        logger.warning("Empty arrays passed to compute_metrics.")
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "pcc": float("nan"),
            "pcc_pvalue": float("nan"),
            "scc": float("nan"),
            "scc_pvalue": float("nan"),
            "r2": float("nan"),
        }

    # 1. Error Metrics
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    # 2. R^2 Score
    if n_samples < 2 or np.all(y_true == y_true[0]):
        r2 = 0.0 if np.allclose(y_true, y_pred) else float("-inf")
    else:
        r2 = float(r2_score(y_true, y_pred))

    # 3. Variance checks for correlations
    var_true = float(np.var(y_true))
    var_pred = float(np.var(y_pred))

    if n_samples < 2 or var_true < 1e-12 or var_pred < 1e-12:
        # Zero variance: correlation is undefined
        pcc, pcc_pval = 0.0, 1.0
        scc, scc_pval = 0.0, 1.0
    else:
        # Check if arrays are perfectly identical to avoid floating point precision edge cases
        if np.allclose(y_true, y_pred):
            pcc, pcc_pval = 1.0, 0.0
            scc, scc_pval = 1.0, 0.0
        elif np.allclose(y_true, -y_pred):
            pcc, pcc_pval = -1.0, 0.0
            scc, scc_pval = -1.0, 0.0
        else:
            p_res = stats.pearsonr(y_true, y_pred)
            pcc, pcc_pval = float(p_res.statistic), float(p_res.pvalue)

            s_res = stats.spearmanr(y_true, y_pred)
            scc, scc_pval = float(s_res.statistic), float(s_res.pvalue)

    return {
        "rmse": rmse,
        "mae": mae,
        "pcc": pcc,
        "pcc_pvalue": pcc_pval,
        "scc": scc,
        "scc_pvalue": scc_pval,
        "r2": r2,
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full inference and diagnostic evaluation on a PyG DataLoader.

    Args:
        model (nn.Module): Multimodal neural network model.
        data_loader (DataLoader): PyG evaluation or test DataLoader.
        criterion (nn.Module): Loss function (e.g., MSELoss).
        device (torch.device): Compute device.

    Returns:
        Tuple:
            - metrics (Dict[str, float]): Computed metrics dictionary.
            - y_true (np.ndarray): 1D array of ground truth ln(IC50) values.
            - y_pred (np.ndarray): 1D array of predicted ln(IC50) values.
            - embeddings (np.ndarray): 2D array of learned cross-modal embeddings [N, 512].
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0

    all_preds: list = []
    all_targets: list = []
    all_embeddings: list = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            # Forward pass
            preds = model(batch).view(-1)
            targets = batch.y.view(-1)

            loss = criterion(preds, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size

            # Extract 512-dim multimodal latent embeddings
            embeddings = model.get_embeddings(batch)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())

    mean_loss = total_loss / max(num_samples, 1)

    y_true_arr = np.concatenate(all_targets) if all_targets else np.array([])
    y_pred_arr = np.concatenate(all_preds) if all_preds else np.array([])
    emb_arr = np.concatenate(all_embeddings) if all_embeddings else np.array([[]])

    metrics = compute_metrics(y_true_arr, y_pred_arr)
    metrics["loss"] = mean_loss

    logger.info(
        f"Evaluation Results -> MSE: {mean_loss:.4f} | RMSE: {metrics['rmse']:.4f} | "
        f"MAE: {metrics['mae']:.4f} | PCC: {metrics['pcc']:.4f} (p={metrics['pcc_pvalue']:.2e}) | "
        f"SCC: {metrics['scc']:.4f} | R2: {metrics['r2']:.4f}"
    )

    return metrics, y_true_arr, y_pred_arr, emb_arr
