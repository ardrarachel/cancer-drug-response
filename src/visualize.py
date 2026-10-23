"""
Publication-grade diagnostic plotting module for Cancer Drug Response Predictor.
Generates 300 DPI figures using Matplotlib and Seaborn:
  1. learning_curves.png: Train vs. Validation MSE loss across epochs with best checkpoint marker
  2. predicted_vs_actual.png: Parity plot with regression trendline and metric annotations
  3. residuals_distribution.png: Residual error histogram with KDE and normality stats
  4. latent_space_tsne.png: 2D t-SNE of learned 512-dim multimodal fusion embeddings
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Headless non-interactive backend for server/CLI rendering
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

logger = logging.getLogger("DrugResponse.Visualize")


def set_plot_style() -> None:
    """Configure modern, publication-grade aesthetics for all Seaborn/Matplotlib plots."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_learning_curves(
    history: Dict[str, List[float]],
    save_path: Union[str, Path],
    best_epoch: Optional[int] = None,
) -> Path:
    """
    Plot Training and Validation MSE Loss curves across training epochs.

    Args:
        history (Dict[str, List[float]]): Contains 'train_loss' and 'val_loss' lists.
        save_path (Union[str, Path]): Destination PNG file path.
        best_epoch (Optional[int]): Epoch index corresponding to minimum validation loss.

    Returns:
        Path: Resolved filepath of saved plot.
    """
    set_plot_style()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, train_losses, label="Training MSE Loss", color="#1f77b4", linewidth=2.2, alpha=0.9)
    ax.plot(epochs, val_losses, label="Validation MSE Loss", color="#ff7f0e", linewidth=2.2, alpha=0.9)

    if best_epoch is None and len(val_losses) > 0:
        best_epoch = int(np.argmin(val_losses)) + 1

    if best_epoch is not None and best_epoch <= len(val_losses):
        best_val = val_losses[best_epoch - 1]
        ax.scatter(
            [best_epoch],
            [best_val],
            color="#2ca02c",
            s=120,
            zorder=5,
            edgecolors="black",
            label=f"Best Checkpoint (Epoch {best_epoch}: {best_val:.4f})",
        )
        ax.axvline(x=best_epoch, color="#2ca02c", linestyle="--", alpha=0.5)

    ax.set_title("Multimodal Network Learning Dynamics", pad=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    logger.info(f"Saved learning curves to {save_path.resolve()}")
    return save_path


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    save_path: Union[str, Path],
) -> Path:
    """
    Generate Parity Scatter Plot of actual vs. predicted ln(IC50) values.

    Includes:
      - 45-degree dashed equality line
      - Linear regression fit with 95% confidence interval
      - Statistical annotation box with PCC, SCC, RMSE, and R2

    Args:
        y_true (np.ndarray): True ln(IC50) response values.
        y_pred (np.ndarray): Predicted ln(IC50) response values.
        metrics (Dict[str, float]): Dictionary containing 'pcc', 'scc', 'rmse', 'r2'.
        save_path (Union[str, Path]): Destination PNG file path.

    Returns:
        Path: Resolved filepath of saved plot.
    """
    set_plot_style()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # Scatter points with slight transparency
    sns.regplot(
        x=y_true,
        y=y_pred,
        ax=ax,
        scatter_kws={"alpha": 0.55, "color": "#1f77b4", "s": 35},
        line_kws={"color": "#d62728", "linewidth": 2.0, "label": "Regression Fit"},
    )

    # Parity Line (y = x)
    min_val = min(float(y_true.min()), float(y_pred.min())) - 0.5
    max_val = max(float(y_true.max()), float(y_pred.max())) + 0.5
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.7, linewidth=1.8, label="Ideal Parity ($y = \\hat{y}$)")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Metric summary text box
    pcc_val = metrics.get("pcc", 0.0)
    pcc_p = metrics.get("pcc_pvalue", 0.0)
    scc_val = metrics.get("scc", 0.0)
    rmse_val = metrics.get("rmse", 0.0)
    r2_val = metrics.get("r2", 0.0)

    annotation_text = (
        f"Pearson $r$ (PCC) = {pcc_val:.3f} ($p < {max(pcc_p, 1e-4):.1e}$)\n"
        f"Spearman $\\rho$ (SCC) = {scc_val:.3f}\n"
        f"RMSE = {rmse_val:.3f}\n"
        f"Coefficient of Det. ($R^2$) = {r2_val:.3f}"
    )
    ax.text(
        0.05,
        0.95,
        annotation_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#aaaaaa", alpha=0.9),
        fontsize=10,
        fontfamily="monospace",
    )

    ax.set_title("Predicted vs. Actual ln(IC50) Response", pad=12, fontweight="bold")
    ax.set_xlabel("Actual Ground Truth ln(IC50)")
    ax.set_ylabel("Predicted Model ln(IC50)")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    logger.info(f"Saved predicted vs actual plot to {save_path.resolve()}")
    return save_path


def plot_residuals_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Union[str, Path],
) -> Path:
    """
    Plot Histogram and Kernel Density Estimate (KDE) of model residuals (y_true - y_pred).

    Args:
        y_true (np.ndarray): True ln(IC50) values.
        y_pred (np.ndarray): Predicted values.
        save_path (Union[str, Path]): Destination PNG file path.

    Returns:
        Path: Resolved filepath of saved plot.
    """
    set_plot_style()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred

    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        residuals,
        kde=True,
        color="#2b5c8f",
        bins=25,
        edgecolor="black",
        alpha=0.6,
        ax=ax,
        stat="density",
    )

    # Reference zero error line
    ax.axvline(0, color="#d62728", linestyle="--", linewidth=2.0, label="Zero Error ($e = 0$)")
    ax.axvline(mean_res, color="#ff7f0e", linestyle="-.", linewidth=1.8, label=f"Mean Residual ({mean_res:.3f})")

    # Annotation box
    stats_text = (
        f"Mean Error ($\\mu$) = {mean_res:.4f}\n"
        f"Std Dev ($\\sigma$) = {std_res:.4f}\n"
        f"Total Samples = {len(residuals)}"
    )
    ax.text(
        0.70,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#aaaaaa", alpha=0.9),
        fontsize=10,
        fontfamily="monospace",
    )

    ax.set_title("Distribution of Prediction Residuals ($y - \\hat{y}$)", pad=12, fontweight="bold")
    ax.set_xlabel("Residual Error (Actual - Predicted)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    logger.info(f"Saved residuals distribution plot to {save_path.resolve()}")
    return save_path


def plot_latent_space_tsne(
    embeddings: np.ndarray,
    y_true: np.ndarray,
    save_path: Union[str, Path],
    perplexity: int = 30,
    seed: int = 42,
) -> Path:
    """
    Project 512-dimensional multimodal latent embeddings into 2D using t-SNE.
    Points are colored according to continuous response ln(IC50).

    Args:
        embeddings (np.ndarray): Latent feature array of shape [N, 512].
        y_true (np.ndarray): True response values [N].
        save_path (Union[str, Path]): Destination PNG file path.
        perplexity (int): t-SNE perplexity hyperparameter.
        seed (int): Deterministic random state.

    Returns:
        Path: Resolved filepath of saved plot.
    """
    set_plot_style()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = embeddings.shape[0]
    # Bound perplexity based on sample count
    effective_perp = max(5, min(perplexity, n_samples // 4))

    logger.info(f"Running t-SNE on {n_samples} multimodal embeddings (d={embeddings.shape[1]}, perplexity={effective_perp})...")
    tsne = TSNE(n_components=2, perplexity=effective_perp, random_state=seed, max_iter=1000)
    tsne_coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    scatter = ax.scatter(
        tsne_coords[:, 0],
        tsne_coords[:, 1],
        c=y_true,
        cmap="viridis",
        s=45,
        alpha=0.85,
        edgecolors="none",
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("True ln(IC50) Sensitivity", rotation=270, labelpad=16, fontsize=11)

    ax.set_title("t-SNE Projection of 512-D Multimodal Latent Space", pad=12, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    logger.info(f"Saved t-SNE latent space plot to {save_path.resolve()}")
    return save_path


def generate_all_diagnostic_figures(
    history: Dict[str, List[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    embeddings: np.ndarray,
    metrics: Dict[str, float],
    figures_dir: Union[str, Path],
    best_epoch: Optional[int] = None,
) -> Dict[str, Path]:
    """
    Generate all 4 publication-grade figures and save to figures_dir.

    Returns:
        Dict[str, Path]: Paths to the 4 generated PNG figures.
    """
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "learning_curves": plot_learning_curves(
            history=history,
            save_path=fig_dir / "learning_curves.png",
            best_epoch=best_epoch,
        ),
        "predicted_vs_actual": plot_predicted_vs_actual(
            y_true=y_true,
            y_pred=y_pred,
            metrics=metrics,
            save_path=fig_dir / "predicted_vs_actual.png",
        ),
        "residuals_distribution": plot_residuals_distribution(
            y_true=y_true,
            y_pred=y_pred,
            save_path=fig_dir / "residuals_distribution.png",
        ),
        "latent_space_tsne": plot_latent_space_tsne(
            embeddings=embeddings,
            y_true=y_true,
            save_path=fig_dir / "latent_space_tsne.png",
        ),
    }

    logger.info(f"All 4 publication figures successfully exported to {fig_dir.resolve()}")
    return paths
