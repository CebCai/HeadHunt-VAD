"""Video-level evaluation for video anomaly detection."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from headhunt_vad.utils.io import load_json, list_files
from headhunt_vad.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_video_level_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate video-level evaluation metrics.

    Args:
        y_true: Binary ground truth labels (0=normal, 1=anomaly).
        y_scores: Predicted anomaly scores.
        threshold: Classification threshold.

    Returns:
        Dictionary containing evaluation metrics.
    """
    y_pred = (y_scores >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
        metrics["average_precision"] = average_precision_score(y_true, y_scores)

    return metrics


class VideoLevelEvaluator:
    """
    Video-level evaluator for anomaly detection.

    This evaluator computes metrics at the video level, where each video
    gets a single anomaly score based on aggregation of frame/segment scores.
    """

    def __init__(
        self,
        aggregation: str = "max",
        threshold: float = 0.5,
    ):
        """
        Initialize the evaluator.

        Args:
            aggregation: Score aggregation method ("max", "mean", "percentile_90").
            threshold: Classification threshold for binary predictions.
        """
        self.aggregation = aggregation
        self.threshold = threshold

    def aggregate_scores(self, segment_scores: List[float]) -> float:
        """
        Aggregate segment scores to video-level score.

        Args:
            segment_scores: List of segment-level scores.

        Returns:
            Aggregated video-level score.
        """
        if not segment_scores:
            return 0.0

        scores = np.array(segment_scores)

        if self.aggregation == "max":
            return float(np.max(scores))
        elif self.aggregation == "mean":
            return float(np.mean(scores))
        elif self.aggregation == "percentile_90":
            return float(np.percentile(scores, 90))
        else:
            return float(np.max(scores))

    def evaluate(
        self,
        predictions: Dict[str, float],
        ground_truth: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Evaluate video-level predictions.

        Args:
            predictions: Dictionary mapping video names to scores.
            ground_truth: Dictionary mapping video names to labels.

        Returns:
            Evaluation metrics dictionary.
        """
        # Align predictions with ground truth
        common_videos = set(predictions.keys()) & set(ground_truth.keys())

        if not common_videos:
            logger.error("No common videos between predictions and ground truth")
            return {"error": "No common videos"}

        y_true = []
        y_scores = []
        video_names = []

        for video_name in sorted(common_videos):
            y_true.append(ground_truth[video_name])
            y_scores.append(predictions[video_name])
            video_names.append(video_name)

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        # Calculate metrics
        metrics = calculate_video_level_metrics(y_true, y_scores, self.threshold)

        # Add statistics
        metrics["num_videos"] = len(common_videos)
        metrics["num_normal"] = int(np.sum(y_true == 0))
        metrics["num_anomaly"] = int(np.sum(y_true == 1))

        # Confusion matrix
        y_pred = (y_scores >= self.threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=["Normal", "Anomaly"],
            output_dict=True,
        )
        metrics["classification_report"] = report

        logger.info(f"Video-level ROC-AUC: {metrics.get('roc_auc', 0) * 100:.2f}%")
        logger.info(f"Video-level Accuracy: {metrics['accuracy'] * 100:.2f}%")
        logger.info(f"Video-level F1: {metrics['f1_score'] * 100:.2f}%")

        return metrics

    def evaluate_from_results(
        self,
        results_dir: Union[str, Path],
        ground_truth: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Evaluate from result files.

        Args:
            results_dir: Directory containing result JSON files.
            ground_truth: Dictionary mapping video names to labels.

        Returns:
            Evaluation metrics dictionary.
        """
        results_dir = Path(results_dir)
        predictions = {}

        # Load all result files
        json_files = list_files(results_dir, extensions=[".json"])

        for json_path in json_files:
            if json_path.name == "all_results.json":
                continue

            try:
                data = load_json(json_path)
                if isinstance(data, dict) and "video_name" in data:
                    video_name = data["video_name"].split(".")[0]

                    # Try to get video-level probability first
                    if "anomaly_probability" in data:
                        predictions[video_name] = data["anomaly_probability"]
                    # Fall back to aggregating segment scores
                    elif "segments" in data:
                        segments = data["segments"]
                        if segments:
                            scores = [s.get("anomaly_probability") or s.get("smoothed_score") or s.get("raw_score", 0.5) for s in segments]
                            predictions[video_name] = self.aggregate_scores(scores)

            except Exception as e:
                logger.warning(f"Error loading {json_path}: {e}")

        logger.info(f"Loaded {len(predictions)} video predictions")

        return self.evaluate(predictions, ground_truth)

    def find_optimal_threshold(
        self,
        predictions: Dict[str, float],
        ground_truth: Dict[str, int],
        metric: str = "f1_score",
    ) -> Tuple[float, float]:
        """
        Find the optimal classification threshold.

        Args:
            predictions: Dictionary mapping video names to scores.
            ground_truth: Dictionary mapping video names to labels.
            metric: Metric to optimize ("f1_score", "accuracy", etc.).

        Returns:
            Tuple of (optimal_threshold, best_metric_value).
        """
        common_videos = set(predictions.keys()) & set(ground_truth.keys())

        y_true = np.array([ground_truth[v] for v in sorted(common_videos)])
        y_scores = np.array([predictions[v] for v in sorted(common_videos)])

        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = self.threshold
        best_value = 0.0

        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)

            if metric == "f1_score":
                value = f1_score(y_true, y_pred)
            elif metric == "accuracy":
                value = accuracy_score(y_true, y_pred)
            else:
                value = f1_score(y_true, y_pred)

            if value > best_value:
                best_value = value
                best_threshold = thresh

        return best_threshold, best_value
