"""Tests for saliency metrics."""

import numpy as np
import pytest

from headhunt_vad.core.metrics import (
    calculate_lda_score,
    calculate_kl_divergence,
    calculate_mmd,
    calculate_nmi,
    compute_all_metrics,
)


class TestLDAScore:
    """Tests for LDA score calculation."""

    def test_separable_data(self):
        """Test LDA on well-separated classes."""
        np.random.seed(42)
        pos_data = np.random.randn(50, 10) + 5
        neg_data = np.random.randn(50, 10) - 5

        score = calculate_lda_score(pos_data, neg_data)

        assert score > 0
        assert np.isfinite(score)

    def test_overlapping_data(self):
        """Test LDA on overlapping classes."""
        np.random.seed(42)
        pos_data = np.random.randn(50, 10)
        neg_data = np.random.randn(50, 10)

        score = calculate_lda_score(pos_data, neg_data)

        assert score >= 0
        assert np.isfinite(score)

    def test_insufficient_samples(self):
        """Test LDA with insufficient samples."""
        pos_data = np.random.randn(1, 10)
        neg_data = np.random.randn(1, 10)

        score = calculate_lda_score(pos_data, neg_data)

        assert score == 0.0


class TestKLDivergence:
    """Tests for KL divergence calculation."""

    def test_identical_distributions(self):
        """Test KL on identical distributions."""
        np.random.seed(42)
        data = np.random.randn(100, 10)

        score = calculate_kl_divergence(data, data)

        # Should be close to 0 for identical distributions
        assert score >= 0
        assert np.isfinite(score)

    def test_different_distributions(self):
        """Test KL on different distributions."""
        np.random.seed(42)
        pos_data = np.random.randn(50, 10) + 3
        neg_data = np.random.randn(50, 10) - 3

        score = calculate_kl_divergence(pos_data, neg_data)

        assert score > 0
        assert np.isfinite(score)

    def test_insufficient_samples(self):
        """Test KL with insufficient samples."""
        pos_data = np.random.randn(1, 10)
        neg_data = np.random.randn(1, 10)

        score = calculate_kl_divergence(pos_data, neg_data)

        assert score == 0.0


class TestMMD:
    """Tests for MMD calculation."""

    def test_identical_distributions(self):
        """Test MMD on identical distributions."""
        np.random.seed(42)
        data = np.random.randn(50, 10)

        score = calculate_mmd(data, data)

        # Should be close to 0 for identical data
        assert score >= 0
        assert score < 0.1

    def test_different_distributions(self):
        """Test MMD on different distributions."""
        np.random.seed(42)
        pos_data = np.random.randn(50, 10) + 5
        neg_data = np.random.randn(50, 10) - 5

        score = calculate_mmd(pos_data, neg_data)

        assert score > 0
        assert np.isfinite(score)


class TestNMI:
    """Tests for NMI calculation."""

    def test_separable_data(self):
        """Test NMI on well-separated classes."""
        np.random.seed(42)
        pos_data = np.random.randn(50, 10) + 10
        neg_data = np.random.randn(50, 10) - 10

        score = calculate_nmi(pos_data, neg_data)

        assert 0 <= score <= 1
        # Well-separated data should have high NMI
        assert score > 0.5

    def test_overlapping_data(self):
        """Test NMI on overlapping classes."""
        np.random.seed(42)
        pos_data = np.random.randn(50, 10)
        neg_data = np.random.randn(50, 10)

        score = calculate_nmi(pos_data, neg_data)

        assert 0 <= score <= 1


class TestComputeAllMetrics:
    """Tests for combined metric computation."""

    def test_returns_all_four_metrics(self):
        """Test that all four metrics are returned."""
        np.random.seed(42)
        pos_data = np.random.randn(50, 10)
        neg_data = np.random.randn(50, 10)

        lda, kl, mmd, nmi = compute_all_metrics(pos_data, neg_data)

        assert np.isfinite(lda)
        assert np.isfinite(kl)
        assert np.isfinite(mmd)
        assert 0 <= nmi <= 1

    def test_consistency_with_individual_functions(self):
        """Test that combined function matches individual functions."""
        np.random.seed(42)
        pos_data = np.random.randn(50, 10)
        neg_data = np.random.randn(50, 10)

        lda, kl, mmd, nmi = compute_all_metrics(pos_data, neg_data)

        assert lda == calculate_lda_score(pos_data, neg_data)
        assert kl == calculate_kl_divergence(pos_data, neg_data)
        assert mmd == calculate_mmd(pos_data, neg_data)
        assert nmi == calculate_nmi(pos_data, neg_data)
