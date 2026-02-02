"""Temporal visualization for anomaly detection results."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from headhunt_vad.utils.logging import get_logger

logger = get_logger(__name__)

# Style settings
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 11


def setup_style():
    """Set up matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")


def plot_temporal_scores(
    scores: np.ndarray,
    fps: float = 30.0,
    title: str = "Anomaly Score Over Time",
    threshold: Optional[float] = None,
    ground_truth: Optional[List[Tuple[float, float]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot temporal anomaly scores.

    Args:
        scores: Array of frame-level or segment-level scores.
        fps: Frame rate for time axis.
        title: Plot title.
        threshold: Optional threshold line to draw.
        ground_truth: Optional list of (start_time, end_time) tuples for anomaly events.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Time axis
    time = np.arange(len(scores)) / fps

    # Plot scores
    ax.plot(time, scores, color="steelblue", linewidth=2, label="Anomaly Score")

    # Add threshold line
    if threshold is not None:
        ax.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({threshold:.2f})",
        )

    # Shade ground truth regions
    if ground_truth:
        for start, end in ground_truth:
            ax.axvspan(start, end, alpha=0.3, color="red", label="Ground Truth Anomaly")

    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=15)
    ax.set_xlabel("Time (seconds)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Anomaly Score", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved temporal plot to {save_path}")

    return fig


def plot_anomaly_events(
    scores: np.ndarray,
    detected_events: List[Tuple[float, float]],
    ground_truth_events: Optional[List[Tuple[float, float]]] = None,
    fps: float = 30.0,
    title: str = "Anomaly Event Detection",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Plot detected anomaly events with ground truth comparison.

    Args:
        scores: Array of anomaly scores.
        detected_events: List of (start_time, end_time) tuples for detected events.
        ground_truth_events: Optional list of ground truth events.
        fps: Frame rate for time axis.
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    setup_style()

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    time = np.arange(len(scores)) / fps

    # Top plot: Anomaly scores
    axes[0].plot(time, scores, color="steelblue", linewidth=1.5)
    axes[0].set_ylabel("Anomaly Score", fontsize=LABEL_FONTSIZE)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title, fontsize=TITLE_FONTSIZE, pad=10)

    # Bottom plot: Event timeline
    axes[1].set_ylabel("Events", fontsize=LABEL_FONTSIZE)
    axes[1].set_xlabel("Time (seconds)", fontsize=LABEL_FONTSIZE)
    axes[1].set_yticks([0.25, 0.75])
    axes[1].set_yticklabels(["Detected", "Ground Truth"])
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # Draw detected events
    for start, end in detected_events:
        rect = mpatches.Rectangle(
            (start, 0.1),
            end - start,
            0.3,
            facecolor="orange",
            edgecolor="darkorange",
            linewidth=2,
        )
        axes[1].add_patch(rect)

    # Draw ground truth events
    if ground_truth_events:
        for start, end in ground_truth_events:
            rect = mpatches.Rectangle(
                (start, 0.6),
                end - start,
                0.3,
                facecolor="red",
                edgecolor="darkred",
                linewidth=2,
            )
            axes[1].add_patch(rect)

    # Legend
    detected_patch = mpatches.Patch(color="orange", label="Detected")
    gt_patch = mpatches.Patch(color="red", label="Ground Truth")
    axes[1].legend(
        handles=[detected_patch, gt_patch],
        loc="upper right",
        fontsize=10,
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved event plot to {save_path}")

    return fig


def plot_comparison(
    raw_scores: np.ndarray,
    smoothed_scores: np.ndarray,
    fps: float = 30.0,
    threshold: Optional[float] = None,
    title: str = "Raw vs Smoothed Scores",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot comparison of raw and smoothed scores.

    Args:
        raw_scores: Raw anomaly scores.
        smoothed_scores: Smoothed anomaly scores.
        fps: Frame rate for time axis.
        threshold: Optional threshold line.
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    time = np.arange(len(raw_scores)) / fps

    ax.plot(
        time,
        raw_scores,
        color="lightblue",
        linewidth=1,
        alpha=0.7,
        label="Raw Scores",
    )
    ax.plot(
        time,
        smoothed_scores,
        color="steelblue",
        linewidth=2,
        label="Smoothed Scores",
    )

    if threshold is not None:
        ax.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({threshold:.2f})",
        )

    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=15)
    ax.set_xlabel("Time (seconds)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Anomaly Score", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {save_path}")

    return fig


def plot_multi_video_summary(
    video_results: List[Dict],
    metric: str = "roc_auc",
    title: str = "Per-Video Performance",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Plot summary of per-video performance.

    Args:
        video_results: List of per-video result dictionaries.
        metric: Metric to plot ("roc_auc", "ap", "mean_score").
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    setup_style()

    # Filter videos with the metric
    valid_results = [r for r in video_results if metric in r]

    if not valid_results:
        logger.warning(f"No videos with metric {metric}")
        return plt.figure()

    # Sort by metric value
    valid_results.sort(key=lambda x: x[metric], reverse=True)

    video_names = [r["video_name"][:20] for r in valid_results]
    values = [r[metric] for r in valid_results]
    is_anomaly = [r.get("is_anomaly_video", False) for r in valid_results]

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["coral" if a else "steelblue" for a in is_anomaly]
    bars = ax.barh(range(len(video_names)), values, color=colors)

    ax.set_yticks(range(len(video_names)))
    ax.set_yticklabels(video_names, fontsize=8)
    ax.set_xlabel(metric.upper(), fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=15)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # Legend
    anomaly_patch = mpatches.Patch(color="coral", label="Anomaly Video")
    normal_patch = mpatches.Patch(color="steelblue", label="Normal Video")
    ax.legend(handles=[anomaly_patch, normal_patch], loc="lower right", fontsize=10)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved summary plot to {save_path}")

    return fig
