"""Anomaly scoring model for HeadHunt-VAD."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import numpy as np

from headhunt_vad.utils.logging import get_logger

logger = get_logger(__name__)


class AnomalyScorer:
    """
    Implements a Logistic Regression based anomaly scorer.

    Concatenates features from selected expert heads and predicts anomaly
    probability using a sigmoid activation function.
    """

    def __init__(
        self,
        head_indices: List[Tuple[int, int]],
        head_dim: int = 128,
        max_iter: int = 2000,
        solver: str = "liblinear",
        random_state: int = 42,
        C: float = 1.0,
    ):
        """
        Initialize the anomaly scorer.

        Args:
            head_indices: List of (layer, head) indices for expert heads.
            head_dim: Dimension of each attention head (d_h).
            max_iter: Maximum iterations for logistic regression.
            solver: Solver for logistic regression.
            random_state: Random state for reproducibility.
            C: Inverse of regularization strength.
        """
        from sklearn.linear_model import LogisticRegression

        self.head_indices = head_indices
        self.head_dim = head_dim
        self.feature_dim = len(head_indices) * head_dim  # K * d_h

        self.model = LogisticRegression(
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
            C=C,
        )

        self._is_fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "AnomalyScorer":
        """
        Train the anomaly scorer.

        Args:
            features: Feature matrix of shape (n_samples, K * d_h).
                     Concatenated features from K expert heads.
            labels: Binary labels (0=normal, 1=anomaly).

        Returns:
            Self for method chaining.
        """
        logger.info(f"Training anomaly scorer on {len(features)} samples")
        logger.info(f"Feature dimension: {features.shape[1]}")
        logger.info(f"Class distribution: {np.bincount(labels.astype(int))}")

        self.model.fit(features, labels)
        self._is_fitted = True

        logger.info("Anomaly scorer trained successfully")
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Args:
            features: Feature matrix of shape (n_samples, K * d_h).

        Returns:
            Binary predictions (0 or 1).
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict anomaly probabilities.

        p_i = σ(w^T z_i + b)

        Args:
            features: Feature matrix of shape (n_samples, K * d_h).

        Returns:
            Anomaly probabilities (probability of class 1).
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(features)[:, 1]

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute accuracy score on test data.

        Args:
            features: Feature matrix.
            labels: True labels.

        Returns:
            Accuracy score.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.score(features, labels)

    @property
    def weights(self) -> Optional[np.ndarray]:
        """Get the learned weight vector w."""
        if not self._is_fitted:
            return None
        return self.model.coef_[0]

    @property
    def bias(self) -> Optional[float]:
        """Get the learned bias b."""
        if not self._is_fitted:
            return None
        return self.model.intercept_[0]

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to a file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "head_indices": self.head_indices,
            "head_dim": self.head_dim,
            "feature_dim": self.feature_dim,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> "AnomalyScorer":
        """
        Load a model from a file.

        Args:
            path: Path to the saved model.

        Returns:
            Self for method chaining.
        """
        path = Path(path)
        data = joblib.load(path)
        self.model = data["model"]
        self.head_indices = data["head_indices"]
        self.head_dim = data["head_dim"]
        self.feature_dim = data.get("feature_dim", len(self.head_indices) * self.head_dim)
        self._is_fitted = True
        logger.info(f"Model loaded from {path}")
        return self

    @staticmethod
    def extract_expert_features(
        all_head_features: np.ndarray,
        head_indices: List[Tuple[int, int]],
        num_heads: int,
        head_dim: int,
    ) -> np.ndarray:
        """
        Extract and concatenate features from expert heads.

        Args:
            all_head_features: Features of shape (num_layers, num_heads * head_dim)
                              or (batch, num_layers, num_heads * head_dim).
            head_indices: List of (layer, head) tuples for expert heads.
            num_heads: Number of heads per layer.
            head_dim: Dimension per head.

        Returns:
            Concatenated features of shape (K * head_dim,) or (batch, K * head_dim).
        """
        # Handle batch dimension
        if all_head_features.ndim == 2:
            # Single sample: (num_layers, num_heads * head_dim)
            batched = False
            all_head_features = all_head_features[np.newaxis, ...]
        else:
            batched = True

        batch_size, num_layers, hidden_size = all_head_features.shape

        # Reshape to (batch, num_layers, num_heads, head_dim)
        reshaped = all_head_features.reshape(batch_size, num_layers, num_heads, head_dim)

        # Extract specified heads
        selected = []
        for layer_idx, head_idx in head_indices:
            selected.append(reshaped[:, layer_idx, head_idx, :])

        # Stack and flatten: (batch, K, head_dim) -> (batch, K * head_dim)
        expert_features = np.stack(selected, axis=1).reshape(batch_size, -1)

        if not batched:
            expert_features = expert_features[0]

        return expert_features
