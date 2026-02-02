"""Tests for feature extractors."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from headhunt_vad.models.base_extractor import BaseAttentionExtractor
from headhunt_vad.models.factory import create_extractor, get_available_models


class TestBaseAttentionExtractor:
    """Tests for the base extractor class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that base class cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseAttentionExtractor(model_path="/fake/path")

    def test_hook_creation(self):
        """Test that hook function is created correctly."""
        # Create a concrete subclass for testing
        class TestExtractor(BaseAttentionExtractor):
            def get_model_config(self):
                return {"num_layers": 4, "num_heads": 4, "head_dim": 32}

            def load_model(self):
                pass

            def _get_attention_layers(self):
                return None

            def _get_oproj_module(self, layer_idx):
                return MagicMock()

            def _get_final_layer(self):
                return MagicMock()

            def extract_features(self, video_path, prompt, num_segments=16):
                return None

        extractor = TestExtractor(model_path="/fake/path")

        # Test hook creation
        hook = extractor._create_oproj_hook(0)
        assert callable(hook)


class TestExtractorFactory:
    """Tests for the extractor factory."""

    def test_get_available_models(self):
        """Test that available models are returned."""
        models = get_available_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert "internvl3" in models or any("internvl" in m.lower() for m in models)

    def test_create_extractor_invalid_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_extractor(
                model_type="nonexistent_model",
                model_path="/fake/path",
            )

    def test_create_extractor_valid_type(self):
        """Test that valid model type creates extractor."""
        # Don't actually load the model
        with patch.object(
            __import__("headhunt_vad.models.internvl3_extractor", fromlist=["InternVL3Extractor"]).InternVL3Extractor,
            "load_model",
            return_value=None,
        ):
            extractor = create_extractor(
                model_type="internvl3",
                model_path="/fake/path",
            )

            assert extractor is not None
            assert hasattr(extractor, "extract_features")
            assert hasattr(extractor, "get_model_config")


class TestInternVL3Extractor:
    """Tests for InternVL3 extractor (without loading actual model)."""

    def test_default_config(self):
        """Test default configuration."""
        from headhunt_vad.models.internvl3_extractor import InternVL3Extractor

        extractor = InternVL3Extractor(model_path="/fake/path")
        config = extractor.get_model_config()

        assert config["num_layers"] == 28
        assert config["num_heads"] == 28
        assert config["head_dim"] == 128

    def test_config_override(self):
        """Test configuration override."""
        from headhunt_vad.models.internvl3_extractor import InternVL3Extractor

        extractor = InternVL3Extractor(
            model_path="/fake/path",
            config_override={"num_layers": 16},
        )
        config = extractor.get_model_config()

        assert config["num_layers"] == 16
        assert config["num_heads"] == 28  # Not overridden

    def test_hook_management(self):
        """Test hook registration and removal."""
        from headhunt_vad.models.internvl3_extractor import InternVL3Extractor

        extractor = InternVL3Extractor(model_path="/fake/path")

        # Initially no handles
        assert len(extractor.handles) == 0

        # After remove (should not error even with no handles)
        extractor.remove_hooks()
        assert len(extractor.handles) == 0


class TestAnomalyScorer:
    """Tests for anomaly scorer."""

    def test_scorer_initialization(self):
        """Test scorer initialization."""
        from headhunt_vad.models.anomaly_scorer import AnomalyScorer

        head_indices = [(0, 0), (1, 1), (2, 2)]
        scorer = AnomalyScorer(head_indices=head_indices, head_dim=128)

        assert scorer.head_indices == head_indices
        assert scorer.head_dim == 128
        assert scorer.feature_dim == 3 * 128

    def test_scorer_fit_predict(self):
        """Test scorer training and prediction."""
        import numpy as np
        from headhunt_vad.models.anomaly_scorer import AnomalyScorer

        np.random.seed(42)

        head_indices = [(0, 0), (1, 1)]
        scorer = AnomalyScorer(head_indices=head_indices, head_dim=64)

        # Create training data
        X = np.random.randn(100, 128)
        y = np.random.randint(0, 2, 100)

        # Fit
        scorer.fit(X, y)

        # Predict
        preds = scorer.predict(X)
        proba = scorer.predict_proba(X)

        assert preds.shape == (100,)
        assert proba.shape == (100,)
        assert all(p in [0, 1] for p in preds)
        assert all(0 <= p <= 1 for p in proba)

    def test_scorer_save_load(self):
        """Test scorer save and load."""
        import tempfile
        import numpy as np
        from pathlib import Path
        from headhunt_vad.models.anomaly_scorer import AnomalyScorer

        np.random.seed(42)

        head_indices = [(0, 0), (1, 1)]
        scorer = AnomalyScorer(head_indices=head_indices, head_dim=64)

        X = np.random.randn(50, 128)
        y = np.random.randint(0, 2, 50)
        scorer.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            scorer.save(path)

            # Load
            new_scorer = AnomalyScorer(head_indices=[], head_dim=64)
            new_scorer.load(path)

            assert new_scorer.head_indices == head_indices
            assert new_scorer.head_dim == 64

            # Predictions should match
            orig_preds = scorer.predict_proba(X)
            loaded_preds = new_scorer.predict_proba(X)
            np.testing.assert_array_almost_equal(orig_preds, loaded_preds)
