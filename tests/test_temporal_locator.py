"""Tests for temporal locator."""

import numpy as np
import pytest

from headhunt_vad.core.temporal_locator import TemporalLocator


class TestTemporalLocator:
    """Tests for the TemporalLocator class."""

    @pytest.fixture
    def locator(self):
        """Create a default locator instance."""
        return TemporalLocator(
            gaussian_sigma=1.5,
            threshold=0.5,
        )

    def test_initialization(self, locator):
        """Test locator initialization."""
        assert locator.gaussian_sigma == 1.5
        assert locator.threshold == 0.5

    def test_gaussian_kernel(self, locator):
        """Test Gaussian kernel generation."""
        kernel = locator.gaussian_kernel(5, 1.0)

        assert len(kernel) == 5
        assert abs(np.sum(kernel) - 1.0) < 1e-6  # Normalized
        assert kernel[2] == max(kernel)  # Center is max

    def test_gaussian_smooth(self, locator):
        """Test Gaussian smoothing."""
        data = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        smoothed = locator.smooth(data, sigma=1.0)

        assert len(smoothed) == len(data)
        assert smoothed[3] > smoothed[0]  # Center region higher

    def test_smooth_empty(self):
        """Test smoothing with empty input."""
        locator = TemporalLocator()
        scores = np.array([])
        smoothed = locator.smooth(scores)
        assert len(smoothed) == 0

    def test_smooth_single(self):
        """Test smoothing with single value."""
        locator = TemporalLocator()
        scores = np.array([0.5])
        smoothed = locator.smooth(scores)
        assert len(smoothed) == 1

    def test_binarize(self, locator):
        """Test binarization."""
        scores = np.array([0.2, 0.6, 0.8, 0.3, 0.7])
        binary = locator.binarize(scores, threshold=0.5)

        assert len(binary) == len(scores)
        np.testing.assert_array_equal(binary, [0, 1, 1, 0, 1])

    def test_localize_single_event(self, locator):
        """Test localization of a single event."""
        # Create scores with a clear anomaly region
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.85, 0.3, 0.1])
        locator.threshold = 0.5

        events = locator.localize(scores, fps=4, apply_smoothing=False)

        assert len(events) >= 1
        # Event should be in the high-score region
        start, end = events[0]
        assert start >= 0
        assert end > start

    def test_localize_no_events(self, locator):
        """Test localization with no anomalies."""
        scores = np.array([0.1, 0.2, 0.1, 0.2, 0.1])
        locator.threshold = 0.5

        events = locator.localize(scores, fps=4, apply_smoothing=False)

        assert len(events) == 0

    def test_localize_multiple_events(self, locator):
        """Test localization of multiple events."""
        # Create scores with two anomaly regions
        scores = np.array([0.1, 0.8, 0.9, 0.2, 0.1, 0.9, 0.85, 0.1])
        locator.threshold = 0.5

        events = locator.localize(scores, fps=4, apply_smoothing=False)

        # Should detect two events
        assert len(events) == 2

    def test_process_video(self, locator):
        """Test full video processing."""
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3, 0.1])

        processed, events = locator.process_video(
            scores,
            fps=4,
            apply_smoothing=True,
        )

        assert len(processed) == len(scores)
        assert isinstance(events, list)

    def test_calibrate(self, locator):
        """Test threshold calibration."""
        val_scores = [
            np.array([0.1, 0.2, 0.8, 0.9]),
            np.array([0.1, 0.1, 0.2, 0.1]),
        ]
        val_labels = [
            np.array([0, 0, 1, 1]),
            np.array([0, 0, 0, 0]),
        ]

        optimal_sigma, optimal_threshold = locator.calibrate(
            val_scores, val_labels, metric="f1"
        )

        assert 0.1 <= optimal_threshold <= 0.9
        assert 0.5 <= optimal_sigma <= 3.0


class TestTemporalLocatorEdgeCases:
    """Edge case tests for TemporalLocator."""

    def test_all_zeros(self):
        """Test with all zeros."""
        locator = TemporalLocator(threshold=0.5)
        scores = np.zeros(10)

        events = locator.localize(scores, apply_smoothing=False)
        assert len(events) == 0

    def test_all_ones(self):
        """Test with all ones."""
        locator = TemporalLocator(threshold=0.5)
        scores = np.ones(10)

        events = locator.localize(scores, fps=10, apply_smoothing=False)
        assert len(events) == 1
        start, end = events[0]
        assert start == 0
        assert abs(end - 1.0) < 0.1  # ~1 second
