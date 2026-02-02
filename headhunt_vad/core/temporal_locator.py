"""Temporal localization for video anomaly detection."""

from typing import List, Optional, Tuple, Union

import numpy as np


class TemporalLocator:
    """
    Temporal localization for anomaly detection.

    Provides methods for smoothing scores and localizing anomalies using a
    Gaussian kernel and calibrated threshold.
    """

    def __init__(
        self,
        gaussian_sigma: float = 1.5,
        threshold: float = 0.65,
    ):
        """
        Initialize the temporal locator.

        Args:
            gaussian_sigma: Sigma for Gaussian smoothing (σ_g).
            threshold: Threshold for anomaly detection (τ).
        """
        self.gaussian_sigma = gaussian_sigma
        self.threshold = threshold

    @staticmethod
    def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        """
        Generate a normalized 1D Gaussian kernel.

        Args:
            size: Size of the kernel (should be odd).
            sigma: Standard deviation.

        Returns:
            Normalized Gaussian kernel.
        """
        x = np.linspace(-(size // 2), size // 2, size)
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        return kernel / kernel.sum()

    def smooth(
        self,
        scores: np.ndarray,
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Applies 1D Gaussian smoothing to the anomaly score sequence.

        Args:
            scores: Raw anomaly probability sequence.
            sigma: Override sigma for smoothing. Uses self.gaussian_sigma if None.

        Returns:
            Smoothed scores.
        """
        if len(scores) <= 1:
            return scores.copy()

        sigma = sigma if sigma is not None else self.gaussian_sigma

        # Determine kernel size based on sigma (6σ rule, but capped)
        kernel_size = min(int(6 * sigma) + 1, len(scores))
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)

        kernel = self.gaussian_kernel(kernel_size, sigma)
        smoothed = np.convolve(scores, kernel, mode="same")

        return smoothed

    def binarize(
        self,
        scores: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Binarizes scores using the specified threshold.

        Args:
            scores: Smoothed anomaly scores.
            threshold: Override threshold. Uses self.threshold if None.

        Returns:
            Binary predictions (0 or 1).
        """
        threshold = threshold if threshold is not None else self.threshold
        return (scores > threshold).astype(int)

    def calibrate(
        self,
        val_scores: List[np.ndarray],
        val_labels: List[np.ndarray],
        sigma_range: Optional[Tuple[float, float, int]] = None,
        threshold_range: Optional[Tuple[float, float, int]] = None,
        metric: str = "f1",
    ) -> Tuple[float, float]:
        """
        Calibrates smoothing parameter (sigma) and threshold using grid search on validation data.

        Args:
            val_scores: List of score arrays for validation videos.
            val_labels: List of label arrays for validation videos.
            sigma_range: (min, max, num_steps) for sigma grid search.
            threshold_range: (min, max, num_steps) for threshold grid search.
            metric: Metric to optimize ("f1", "precision", "recall").

        Returns:
            Tuple of (optimal_sigma, optimal_threshold).
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        # Default ranges
        if sigma_range is None:
            sigma_range = (0.5, 3.0, 11)
        if threshold_range is None:
            threshold_range = (0.1, 0.9, 81)

        sigmas = np.linspace(sigma_range[0], sigma_range[1], sigma_range[2])
        thresholds = np.linspace(threshold_range[0], threshold_range[1], threshold_range[2])

        metric_fn = {
            "f1": f1_score,
            "precision": precision_score,
            "recall": recall_score,
        }.get(metric, f1_score)

        best_sigma = self.gaussian_sigma
        best_threshold = self.threshold
        best_metric = 0.0

        for sigma in sigmas:
            # Apply smoothing to all validation videos
            smoothed_videos = [self.smooth(s, sigma=sigma) for s in val_scores]

            for thresh in thresholds:
                all_preds = []
                all_labels = []

                for smoothed, labels in zip(smoothed_videos, val_labels):
                    preds = (smoothed > thresh).astype(int)
                    all_preds.append(preds)
                    all_labels.append(labels)

                # Concatenate all videos
                concat_preds = np.concatenate(all_preds)
                concat_labels = np.concatenate(all_labels)

                try:
                    score = metric_fn(concat_labels, concat_preds, zero_division=0)
                    if score > best_metric:
                        best_metric = score
                        best_sigma = sigma
                        best_threshold = thresh
                except Exception:
                    continue

        # Update instance parameters
        self.gaussian_sigma = best_sigma
        self.threshold = best_threshold

        return best_sigma, best_threshold

    def localize(
        self,
        scores: np.ndarray,
        fps: float = 30.0,
        min_duration: float = 0.0,
        apply_smoothing: bool = True,
    ) -> List[Tuple[float, float]]:
        """
        Localize anomaly events in a video.

        Args:
            scores: Anomaly scores (segment or frame-level).
            fps: Frame rate or segment rate for time conversion.
            min_duration: Minimum event duration in seconds.
            apply_smoothing: Whether to apply Gaussian smoothing.

        Returns:
            List of (start_time, end_time) tuples for detected events.
        """
        if len(scores) == 0:
            return []

        # Apply smoothing if requested
        if apply_smoothing:
            scores = self.smooth(scores)

        # Binarize using threshold
        binary = self.binarize(scores)

        # Find event boundaries
        events = []
        in_event = False
        start_idx = 0

        for i, val in enumerate(binary):
            if val == 1 and not in_event:
                # Start of event
                in_event = True
                start_idx = i
            elif val == 0 and in_event:
                # End of event
                in_event = False
                duration_sec = (i - start_idx) / fps

                if duration_sec >= min_duration:
                    start_time = start_idx / fps
                    end_time = i / fps
                    events.append((start_time, end_time))

        # Handle event that extends to end of video
        if in_event:
            duration_sec = (len(binary) - start_idx) / fps

            if duration_sec >= min_duration:
                start_time = start_idx / fps
                end_time = len(binary) / fps
                events.append((start_time, end_time))

        return events

    def process_video(
        self,
        segment_scores: np.ndarray,
        fps: float = 30.0,
        apply_smoothing: bool = True,
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Process a video's segment scores and localize anomalies.

        Args:
            segment_scores: Array of segment-level anomaly scores.
            fps: Video frame rate (or segment rate).
            apply_smoothing: Whether to apply smoothing.

        Returns:
            Tuple of (processed_scores, events).
        """
        # Apply smoothing
        if apply_smoothing:
            processed = self.smooth(segment_scores)
        else:
            processed = segment_scores.copy()

        # Localize events
        events = self.localize(segment_scores, fps, apply_smoothing=apply_smoothing)

        return processed, events
