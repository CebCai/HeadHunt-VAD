"""t-SNE visualizations for feature analysis."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from headhunt_vad.utils.logging import get_logger

logger = get_logger(__name__)

# Style settings
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 12


def setup_style():
    """Set up matplotlib style."""
    sns.set_context("paper")
    plt.style.use("seaborn-v0_8-whitegrid")


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE Visualization",
    class_names: Optional[List[str]] = None,
    perplexity: int = 30,
    random_state: int = 42,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Create a t-SNE visualization of features.

    Args:
        features: Feature matrix of shape (n_samples, n_features).
        labels: Binary labels (0 or 1).
        title: Plot title.
        class_names: Optional list of class names [class_0_name, class_1_name].
        perplexity: t-SNE perplexity parameter.
        random_state: Random state for reproducibility.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    setup_style()

    if len(features) < 2:
        logger.warning("Insufficient data for t-SNE")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Insufficient Data", ha="center", va="center", fontsize=14)
        return fig

    # Adjust perplexity if needed
    perplexity = min(perplexity, len(features) - 1)
    if perplexity <= 0:
        perplexity = 5

    # Perform t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        init="pca",
    )

    try:
        embeddings = tsne.fit_transform(features)
    except Exception as e:
        logger.error(f"t-SNE failed: {e}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "t-SNE Failed", ha="center", va="center", fontsize=14)
        return fig

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Default class names
    if class_names is None:
        class_names = ["Normal", "Anomaly"]

    # Plot each class
    colors = ["#1f77b4", "#d62728"]  # Blue for normal, red for anomaly

    for label_val, (name, color) in enumerate(zip(class_names, colors)):
        mask = labels == label_val
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=color,
            label=name,
            alpha=0.7,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=15)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved t-SNE plot to {save_path}")

    return fig


def plot_head_comparison(
    head1_features: Tuple[np.ndarray, np.ndarray],
    head2_features: Tuple[np.ndarray, np.ndarray],
    head1_info: Tuple[int, int, float],
    head2_info: Tuple[int, int, float],
    title: str = "Head Comparison",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 7),
) -> plt.Figure:
    """
    Plot t-SNE comparison between two attention heads.

    Args:
        head1_features: Tuple of (pos_data, neg_data) for first head.
        head2_features: Tuple of (pos_data, neg_data) for second head.
        head1_info: Tuple of (layer, head, score) for first head.
        head2_info: Tuple of (layer, head, score) for second head.
        title: Overall plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, (pos_data, neg_data), (layer, head, score) in zip(
        axes,
        [head1_features, head2_features],
        [head1_info, head2_info],
    ):
        if pos_data is None or neg_data is None or len(pos_data) < 2 or len(neg_data) < 2:
            ax.text(0.5, 0.5, "Insufficient Data", ha="center", va="center", fontsize=14)
            ax.set_title(f"Layer {layer}, Head {head}\nScore: {score:.3f}", fontsize=14)
            continue

        # Combine data
        features = np.vstack([pos_data, neg_data])
        labels = np.array([0] * len(pos_data) + [1] * len(neg_data))

        # t-SNE
        perplexity = min(30, len(features) - 1)
        if perplexity > 0:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
            try:
                embeddings = tsne.fit_transform(features)

                # Plot
                for label_val, (name, color) in enumerate(
                    zip(["Normal", "Anomaly"], ["#1f77b4", "#d62728"])
                ):
                    mask = labels == label_val
                    ax.scatter(
                        embeddings[mask, 0],
                        embeddings[mask, 1],
                        c=color,
                        label=name,
                        alpha=0.7,
                        s=50,
                    )
            except Exception as e:
                ax.text(0.5, 0.5, "t-SNE Failed", ha="center", va="center", fontsize=14)

        ax.set_title(f"Layer {layer}, Head {head}\nScore: {score:.3f}", fontsize=14)
        ax.set_xlabel("t-SNE Dim 1", fontsize=12)
        ax.set_ylabel("t-SNE Dim 2", fontsize=12)
        ax.legend(fontsize=10)

    fig.suptitle(title, fontsize=TITLE_FONTSIZE, y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved head comparison plot to {save_path}")

    return fig


def plot_layer_tsne_grid(
    layer_data: List[Tuple[np.ndarray, np.ndarray, int]],
    num_cols: int = 4,
    title: str = "t-SNE Across Layers",
    save_path: Optional[Union[str, Path]] = None,
    figsize_per_plot: Tuple[int, int] = (4, 4),
) -> plt.Figure:
    """
    Plot t-SNE visualizations for multiple layers in a grid.

    Args:
        layer_data: List of (pos_data, neg_data, layer_idx) tuples.
        num_cols: Number of columns in the grid.
        title: Overall plot title.
        save_path: Optional path to save the figure.
        figsize_per_plot: Size of each individual plot.

    Returns:
        Matplotlib figure object.
    """
    setup_style()

    num_layers = len(layer_data)
    num_rows = (num_layers + num_cols - 1) // num_cols

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(figsize_per_plot[0] * num_cols, figsize_per_plot[1] * num_rows),
    )
    axes = np.array(axes).flatten()

    for idx, (pos_data, neg_data, layer_idx) in enumerate(layer_data):
        ax = axes[idx]

        if pos_data is None or neg_data is None or len(pos_data) < 2 or len(neg_data) < 2:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=10)
            ax.set_title(f"Layer {layer_idx}", fontsize=11)
            ax.axis("off")
            continue

        features = np.vstack([pos_data, neg_data])
        labels = np.array([0] * len(pos_data) + [1] * len(neg_data))

        perplexity = min(30, len(features) - 1)
        if perplexity > 0:
            try:
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=500)
                embeddings = tsne.fit_transform(features)

                ax.scatter(
                    embeddings[labels == 0, 0],
                    embeddings[labels == 0, 1],
                    c="#1f77b4",
                    alpha=0.6,
                    s=20,
                )
                ax.scatter(
                    embeddings[labels == 1, 0],
                    embeddings[labels == 1, 1],
                    c="#d62728",
                    alpha=0.6,
                    s=20,
                )
            except Exception:
                ax.text(0.5, 0.5, "Error", ha="center", va="center", fontsize=10)

        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(num_layers, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=TITLE_FONTSIZE, y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved layer t-SNE grid to {save_path}")

    return fig
