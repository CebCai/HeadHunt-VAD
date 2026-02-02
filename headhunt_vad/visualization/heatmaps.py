"""Heatmap visualizations for attention head analysis."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from headhunt_vad.utils.logging import get_logger

logger = get_logger(__name__)

# Default style settings
TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 12
ANNOT_FONTSIZE = 8


def setup_style():
    """Set up matplotlib style."""
    sns.set_context("paper")
    plt.style.use("seaborn-v0_8-whitegrid")


def plot_saliency_heatmap(
    scores: np.ndarray,
    title: str = "Saliency Score Heatmap",
    cmap: str = "viridis",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_annotations: bool = False,
) -> plt.Figure:
    """
    Plot a heatmap of saliency scores.

    Args:
        scores: 2D array of shape (num_layers, num_heads).
        title: Plot title.
        cmap: Colormap name.
        save_path: Optional path to save the figure.
        figsize: Figure size.
        show_annotations: Whether to show value annotations.

    Returns:
        Matplotlib figure object.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        scores,
        annot=show_annotations,
        cmap=cmap,
        robust=True,
        ax=ax,
        annot_kws={"size": ANNOT_FONTSIZE} if show_annotations else None,
    )

    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=20)
    ax.set_xlabel("Head Index", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Layer Index", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved heatmap to {save_path}")

    return fig


def plot_robustness_heatmap(
    robustness_scores: np.ndarray,
    selected_heads: Optional[List[Tuple[int, int]]] = None,
    title: str = "Robust Saliency Score (RSS)",
    lambda_penalty: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Plot a heatmap of robustness scores with optional head highlighting.

    Args:
        robustness_scores: 2D array of RSS values.
        selected_heads: Optional list of (layer, head) tuples to highlight.
        title: Plot title.
        lambda_penalty: Lambda value for title annotation.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        robustness_scores,
        annot=False,
        cmap="rocket_r",
        robust=True,
        ax=ax,
    )

    # Highlight selected heads
    if selected_heads:
        for layer, head in selected_heads:
            rect = plt.Rectangle(
                (head, layer),
                1,
                1,
                fill=False,
                edgecolor="lime",
                linewidth=3,
            )
            ax.add_patch(rect)

    full_title = title
    if lambda_penalty is not None:
        full_title += f" (λ={lambda_penalty})"

    ax.set_title(full_title, fontsize=TITLE_FONTSIZE, pad=20)
    ax.set_xlabel("Head Index", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Layer Index", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved robustness heatmap to {save_path}")

    return fig


def plot_metric_comparison(
    metrics: Dict[str, np.ndarray],
    save_dir: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (20, 5),
) -> plt.Figure:
    """
    Plot comparison of multiple metrics as heatmaps.

    Args:
        metrics: Dictionary mapping metric names to 2D arrays.
        save_dir: Optional directory to save individual and combined figures.
        figsize: Figure size for individual plots.

    Returns:
        Combined matplotlib figure object.
    """
    setup_style()

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(figsize[0], figsize[1]))

    if num_metrics == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, metrics.items()):
        sns.heatmap(data, annot=False, cmap="viridis", robust=True, ax=ax)
        ax.set_title(name.upper(), fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Head Index", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Layer Index", fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "metric_comparison.png", dpi=300, bbox_inches="tight")
        logger.info(f"Saved metric comparison to {save_dir}")

    return fig


def plot_disagreement_heatmap(
    metrics: Dict[str, np.ndarray],
    title: str = "Metric Disagreement (Std Dev)",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Plot a heatmap showing disagreement between metrics.

    Args:
        metrics: Dictionary mapping metric names to normalized 2D arrays.
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    setup_style()

    # Stack metrics and compute std
    stacked = np.stack(list(metrics.values()), axis=0)
    disagreement = np.nanstd(stacked, axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(disagreement, annot=False, cmap="magma", robust=True, ax=ax)

    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=20)
    ax.set_xlabel("Head Index", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Layer Index", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved disagreement heatmap to {save_path}")

    return fig


def plot_head_importance(
    layer_scores: np.ndarray,
    layer_idx: int,
    highlight_top_k: int = 5,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Plot head importance bar chart for a single layer.

    Args:
        layer_scores: 1D array of head scores for the layer.
        layer_idx: Layer index for title.
        highlight_top_k: Number of top heads to highlight.
        title: Optional custom title.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    num_heads = len(layer_scores)
    colors = sns.color_palette("viridis", num_heads)

    # Highlight top-k heads
    top_k_indices = np.argsort(layer_scores)[::-1][:highlight_top_k]
    bar_colors = [
        "coral" if i in top_k_indices else colors[i] for i in range(num_heads)
    ]

    bars = ax.bar(range(num_heads), layer_scores, color=bar_colors)

    if title is None:
        title = f"Head Importance at Layer {layer_idx}"

    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=15)
    ax.set_xlabel("Head Index", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Saliency Score", fontsize=LABEL_FONTSIZE)
    ax.set_xticks(range(num_heads))
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved head importance plot to {save_path}")

    return fig
