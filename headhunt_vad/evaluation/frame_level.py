"""Frame-level evaluation for video anomaly detection."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from headhunt_vad.utils.io import load_json, list_files
from headhunt_vad.utils.logging import get_logger
from headhunt_vad.core.temporal_locator import TemporalLocator

logger = get_logger(__name__)


def parse_ground_truth(
    annotation_file: Union[str, Path],
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parse ground truth annotation file.

    Expected format (UCF-Crime style):
    video_name.mp4 Label start_frame end_frame [start_frame2 end_frame2]
    - Normal videos have "Normal -1 -1"
    - Anomaly videos have "Anomaly start end" (optionally with second event)

    Args:
        annotation_file: Path to the annotation file.

    Returns:
        Dictionary mapping video names (without extension) to list of
        (start_frame, end_frame) tuples for anomaly events.
    """
    annotation_file = Path(annotation_file)
    ground_truth = {}

    with open(annotation_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            video_name = parts[0].split(".")[0]  # Remove extension
            anomalies = []

            # Check if normal video
            if parts[1].lower() == "normal" and int(parts[2]) == -1:
                anomalies = []
            else:
                # First anomaly event
                if int(parts[2]) != -1 and int(parts[3]) != -1:
                    anomalies.append((int(parts[2]), int(parts[3])))

                # Second anomaly event (if present)
                if len(parts) > 5 and int(parts[4]) != -1 and int(parts[5]) != -1:
                    anomalies.append((int(parts[4]), int(parts[5])))

            ground_truth[video_name] = anomalies

    return ground_truth


def calculate_frame_level_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate frame-level evaluation metrics.

    Args:
        y_true: Binary ground truth labels.
        y_scores: Predicted anomaly scores.

    Returns:
        Dictionary containing AUC, AP, and other metrics.
    """
    # Ensure we have both classes
    if len(np.unique(y_true)) < 2:
        logger.warning("Ground truth contains only one class")
        return {"roc_auc": 0.0, "average_precision": 0.0}

    roc_auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    # Calculate optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    return {
        "roc_auc": roc_auc,
        "average_precision": ap,
        "optimal_threshold": optimal_threshold,
        "max_f1": f1_scores[optimal_idx],
    }


class FrameLevelEvaluator:
    """
    Frame-level evaluator for video anomaly detection.

    This class handles loading results, parsing ground truth, and
    computing frame-level evaluation metrics.
    """

    def __init__(
        self,
        fps: float = 30.0,
        temporal_locator: Optional[TemporalLocator] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            fps: Frame rate for temporal calculations.
            temporal_locator: Optional temporal locator for smoothing.
        """
        self.fps = fps
        self.locator = temporal_locator or TemporalLocator()

    def load_results(
        self,
        results_dir: Union[str, Path],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load prediction results from JSON files.

        Args:
            results_dir: Directory containing result JSON files.

        Returns:
            Dictionary mapping video names to result dictionaries.
        """
        results_dir = Path(results_dir)
        results = {}

        json_files = list_files(results_dir, extensions=[".json"])
        for json_path in json_files:
            if json_path.name == "all_results.json":
                continue

            try:
                data = load_json(json_path)
                if isinstance(data, dict) and "video_name" in data:
                    video_name = data["video_name"].split(".")[0]
                    results[video_name] = data
            except Exception as e:
                logger.warning(f"Error loading {json_path}: {e}")

        logger.info(f"Loaded {len(results)} result files from {results_dir}")
        return results

    def evaluate(
        self,
        results_dir: Union[str, Path],
        ground_truth_file: Union[str, Path],
        apply_smoothing: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth.

        Args:
            results_dir: Directory containing result JSON files.
            ground_truth_file: Path to ground truth annotation file.
            apply_smoothing: Whether to apply temporal smoothing.

        Returns:
            Dictionary containing evaluation metrics.
        """
        # Load data
        results = self.load_results(results_dir)
        ground_truth = parse_ground_truth(ground_truth_file)

        logger.info(f"Loaded {len(ground_truth)} ground truth annotations")

        # Collect all predictions and labels
        all_y_true = []
        all_y_scores = []
        normal_scores = []
        anomaly_y_true = []
        anomaly_y_scores = []

        for video_name, result in results.items():
            if video_name not in ground_truth:
                logger.warning(f"No ground truth for {video_name}")
                continue

            # Get segments
            segments = result.get("segments", [])
            if not segments:
                continue

            # Estimate video duration and frame count
            total_duration = segments[-1]["end_time"]
            total_frames = int(np.ceil(total_duration * self.fps))
            if total_frames == 0:
                continue

            # Create frame-level arrays
            y_true = np.zeros(total_frames, dtype=int)
            y_scores = np.zeros(total_frames, dtype=float)

            # Fill ground truth
            for start_frame, end_frame in ground_truth[video_name]:
                start = min(start_frame, total_frames - 1)
                end = min(end_frame, total_frames)
                y_true[start:end] = 1

            # Collect segment scores
            segment_scores = []
            segment_starts = []
            segment_ends = []

            for segment in segments:
                # Support multiple key names for compatibility
                score = segment.get("anomaly_probability") or segment.get("smoothed_score") or segment.get("raw_score", 0.5)
                segment_scores.append(score)
                segment_starts.append(int(segment["start_time"] * self.fps))
                segment_ends.append(int(np.ceil(segment["end_time"] * self.fps)))

            # Apply smoothing if requested
            if apply_smoothing and len(segment_scores) > 1:
                segment_scores = self.locator.smooth(np.array(segment_scores))

            # Assign scores to frames
            for i, score in enumerate(segment_scores):
                start = min(segment_starts[i], total_frames - 1)
                end = min(segment_ends[i], total_frames)
                y_scores[start:end] = np.maximum(y_scores[start:end], score)

            # Apply sigmoid normalization
            y_scores = 1 / (1 + np.exp(-y_scores))

            # Collect data
            all_y_true.append(y_true)
            all_y_scores.append(y_scores)

            # Separate by video type
            if np.sum(y_true) > 0:
                anomaly_y_true.append(y_true)
                anomaly_y_scores.append(y_scores)
            else:
                normal_scores.append(y_scores)

        # Calculate overall metrics
        final_y_true = np.concatenate(all_y_true)
        final_y_scores = np.concatenate(all_y_scores)

        if len(np.unique(final_y_true)) < 2:
            logger.error("Ground truth contains only one class")
            return {"error": "Insufficient class diversity"}

        metrics = calculate_frame_level_metrics(final_y_true, final_y_scores)

        # Add subset metrics
        if anomaly_y_true and len(np.unique(np.concatenate(anomaly_y_true))) > 1:
            anomaly_true = np.concatenate(anomaly_y_true)
            anomaly_scores = np.concatenate(anomaly_y_scores)
            metrics["anomaly_roc_auc"] = roc_auc_score(anomaly_true, anomaly_scores)
            metrics["anomaly_ap"] = average_precision_score(anomaly_true, anomaly_scores)

        if normal_scores:
            normal_all = np.concatenate(normal_scores)
            metrics["normal_mean_score"] = float(np.mean(normal_all))
            metrics["normal_std_score"] = float(np.std(normal_all))

        # Statistics
        metrics["num_videos"] = len(results)
        metrics["num_frames"] = len(final_y_true)
        metrics["num_anomaly_frames"] = int(np.sum(final_y_true))

        logger.info(f"ROC-AUC: {metrics['roc_auc'] * 100:.2f}%")
        logger.info(f"Average Precision: {metrics['average_precision'] * 100:.2f}%")

        return metrics

    def evaluate_per_video(
        self,
        results_dir: Union[str, Path],
        ground_truth_file: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        """
        Evaluate predictions per video.

        Args:
            results_dir: Directory containing result JSON files.
            ground_truth_file: Path to ground truth annotation file.

        Returns:
            List of per-video evaluation results.
        """
        results = self.load_results(results_dir)
        ground_truth = parse_ground_truth(ground_truth_file)

        per_video_results = []

        for video_name, result in results.items():
            if video_name not in ground_truth:
                continue

            segments = result.get("segments", [])
            if not segments:
                continue

            total_duration = segments[-1]["end_time"]
            total_frames = int(np.ceil(total_duration * self.fps))
            if total_frames == 0:
                continue

            y_true = np.zeros(total_frames, dtype=int)
            y_scores = np.zeros(total_frames, dtype=float)

            for start_frame, end_frame in ground_truth[video_name]:
                start = min(start_frame, total_frames - 1)
                end = min(end_frame, total_frames)
                y_true[start:end] = 1

            segment_scores = [s.get("anomaly_probability") or s.get("smoothed_score") or s.get("raw_score", 0.5) for s in segments]
            if len(segment_scores) > 1:
                segment_scores = self.locator.smooth(np.array(segment_scores))

            for i, segment in enumerate(segments):
                start = int(segment["start_time"] * self.fps)
                end = int(np.ceil(segment["end_time"] * self.fps))
                start = min(start, total_frames - 1)
                end = min(end, total_frames)
                y_scores[start:end] = np.maximum(y_scores[start:end], segment_scores[i])

            y_scores = 1 / (1 + np.exp(-y_scores))

            video_result = {
                "video_name": video_name,
                "num_frames": total_frames,
                "num_anomaly_frames": int(np.sum(y_true)),
                "mean_score": float(np.mean(y_scores)),
                "max_score": float(np.max(y_scores)),
                "is_anomaly_video": len(ground_truth[video_name]) > 0,
            }

            # Calculate metrics if both classes present
            if len(np.unique(y_true)) > 1:
                video_result["roc_auc"] = roc_auc_score(y_true, y_scores)
                video_result["ap"] = average_precision_score(y_true, y_scores)

            per_video_results.append(video_result)

        return per_video_results
