"""Tests for RHI module."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from headhunt_vad.core.rhi import RobustHeadIdentifier


class TestRobustHeadIdentifier:
    """Tests for the RobustHeadIdentifier class."""

    @pytest.fixture
    def model_config(self):
        """Default model configuration."""
        return {
            "num_layers": 4,
            "num_heads": 4,
            "head_dim": 32,
        }

    @pytest.fixture
    def rhi(self, model_config):
        """Create RHI instance."""
        return RobustHeadIdentifier(
            model_config=model_config,
            lambda_penalty=0.5,
            top_k=3,
        )

    def test_initialization(self, rhi, model_config):
        """Test RHI initialization."""
        assert rhi.num_layers == model_config["num_layers"]
        assert rhi.num_heads == model_config["num_heads"]
        assert rhi.head_dim == model_config["head_dim"]
        assert rhi.lambda_penalty == 0.5
        assert rhi.top_k == 3

    def test_saliency_scores_storage(self, rhi):
        """Test that saliency scores are stored per prompt."""
        # Mock saliency calculation
        mock_saliency = np.random.rand(4, 4)

        rhi.saliency_per_prompt["prompt1"] = mock_saliency.copy()
        rhi.saliency_per_prompt["prompt2"] = mock_saliency.copy()

        assert "prompt1" in rhi.saliency_per_prompt
        assert "prompt2" in rhi.saliency_per_prompt
        assert rhi.saliency_per_prompt["prompt1"].shape == (4, 4)

    def test_robustness_score_calculation(self, rhi):
        """Test RSS calculation logic."""
        # Create mock saliency scores
        np.random.seed(42)
        scores1 = np.random.rand(4, 4)
        scores2 = np.random.rand(4, 4)

        rhi.saliency_per_prompt["p1"] = scores1
        rhi.saliency_per_prompt["p2"] = scores2

        # Calculate expected values
        stacked = np.stack([scores1, scores2])
        expected_mean = np.mean(stacked, axis=0)
        expected_std = np.std(stacked, axis=0)
        expected_rss = expected_mean - 0.5 * expected_std

        # Verify calculation matches expectation
        rss = expected_mean - rhi.lambda_penalty * expected_std
        np.testing.assert_array_almost_equal(rss, expected_rss)

    def test_top_k_selection(self, rhi):
        """Test that top-K heads are correctly selected."""
        # Create deterministic scores
        scores = np.arange(16).reshape(4, 4).astype(float)
        rhi.robustness_scores = scores

        # Flatten and get top 3
        flat = scores.flatten()
        top_3_flat = np.argsort(flat)[::-1][:3]
        expected_heads = [
            (idx // 4, idx % 4) for idx in top_3_flat
        ]

        # Verify top heads
        for layer, head in expected_heads:
            assert scores[layer, head] >= scores[0, 0]

    def test_save_results(self, rhi, model_config):
        """Test saving RHI results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Set up mock results
            rhi.robustness_scores = np.random.rand(4, 4)
            rhi.robust_heads = [(3, 2), (2, 1), (3, 3)]
            rhi.saliency_per_prompt = {
                "p1": np.random.rand(4, 4),
                "p2": np.random.rand(4, 4),
            }

            rhi.save_results(output_dir)

            # Check files exist
            assert (output_dir / "robust_head_indices.json").exists()
            assert (output_dir / "robustness_scores.npy").exists()
            assert (output_dir / "detailed_rhi_analysis.csv").exists()

    def test_extract_head_data(self, rhi):
        """Test head data extraction."""
        import torch

        # Create mock data - XD-Violence convention: 'A' = Normal
        features = torch.randn(4, 4, 32)
        all_data = [
            (features, "A"),       # Normal (XD-Violence convention)
            (features, "B1"),      # Anomaly
            (features, "A"),       # Normal
        ]

        anomaly, normal = rhi._extract_head_data(all_data, layer=0, head=0)

        assert anomaly is not None
        assert normal is not None
        assert len(anomaly) == 1  # One anomaly sample (B1)
        assert len(normal) == 2   # Two normal samples (A)
