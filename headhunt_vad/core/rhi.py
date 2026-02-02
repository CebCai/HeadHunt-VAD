"""Robust Head Identification (RHI) module."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from headhunt_vad.core.metrics import compute_all_metrics
from headhunt_vad.utils.io import load_pickle, list_files, save_json
from headhunt_vad.utils.logging import get_logger

logger = get_logger(__name__)


class RobustHeadIdentifier:
    """
    Implements the Robust Head Identification (RHI) algorithm.

    Identifies attention heads that exhibit consistent discriminative power
    across multiple textual prompts. Computes a Robust Saliency Score (RSS)
    balancing mean performance and stability.
    """

    # Label conventions for different datasets
    LABEL_CONVENTIONS = {
        "xd_violence": {
            "normal_labels": ["A"],  # XD-Violence: 'A' = Normal
            "description": "XD-Violence: label 'A' indicates normal samples",
        },
        "ucf_crime": {
            "normal_labels": ["Normal", "Testing_Normal_Videos_Anomaly"],
            "description": "UCF-Crime: labels containing 'Normal' indicate normal samples",
        },
        "auto": {
            "normal_labels": ["A", "Normal", "normal", "Testing_Normal_Videos_Anomaly"],
            "description": "Auto-detect: checks multiple common normal labels",
        },
    }

    def __init__(
        self,
        model_config: Dict[str, int],
        lambda_penalty: float = 0.5,
        top_k: int = 5,
        dataset: str = "auto",
    ):
        """
        Initialize the RHI module.

        Args:
            model_config: Dictionary with num_layers, num_heads, head_dim.
            lambda_penalty: Weight for instability penalty (λ).
            top_k: Number of robust heads to select (K).
            dataset: Dataset name for label convention ("xd_violence", "ucf_crime", or "auto").
        """
        self.model_config = model_config
        self.lambda_penalty = lambda_penalty
        self.top_k = top_k
        self.dataset = dataset

        self.num_layers = model_config["num_layers"]
        self.num_heads = model_config["num_heads"]
        self.head_dim = model_config["head_dim"]

        # Storage for results
        self.saliency_per_prompt = {}
        self.robustness_scores = None
        self.robust_heads = None

    def _load_prompt_data(
        self,
        data_path: Union[str, Path],
        feature_key: str = "self_attns",
        max_files: Optional[int] = None,
    ) -> List[Tuple[torch.Tensor, str]]:
        """
        Load feature data for a prompt.

        Args:
            data_path: Path to the data directory.
            feature_key: Key for features in pickle files.
                        Supports "self_attns" (shape: [num_layers, num_heads*head_dim])
                        and "oproj_inputs_last_token" (shape: [1, num_layers, num_heads*head_dim]).
            max_files: Maximum number of files to load.

        Returns:
            List of (features, label) tuples.
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        files = list_files(data_path, extensions=[".pkl"])
        if max_files is not None:
            files = files[:max_files]

        all_data = []
        for file_path in files:
            try:
                item = load_pickle(file_path)

                # Try different feature keys for compatibility
                features_raw = item.get(feature_key)
                if features_raw is None:
                    # Try alternative keys
                    for alt_key in ["self_attns", "oproj_inputs_last_token", "oproj_inputs"]:
                        features_raw = item.get(alt_key)
                        if features_raw is not None:
                            break

                label = item.get("label")

                if features_raw is None or label is None:
                    continue

                # Handle different feature shapes
                # Expected final shape: (num_layers, num_heads, head_dim)
                if features_raw.dim() == 2:
                    # Shape: [num_layers, num_heads * head_dim]
                    features = features_raw.view(
                        self.num_layers, self.num_heads, self.head_dim
                    )
                elif features_raw.dim() == 3:
                    # Shape: [1, num_layers, num_heads * head_dim] or similar
                    features = features_raw.squeeze(0).view(
                        self.num_layers, self.num_heads, self.head_dim
                    )
                else:
                    # Try to squeeze and reshape
                    features = features_raw.squeeze().view(
                        self.num_layers, self.num_heads, self.head_dim
                    )

                all_data.append((features, label))

            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")

        pos_count = sum(1 for _, label in all_data if not self._is_normal(label))
        neg_count = len(all_data) - pos_count
        logger.info(f"Loaded {len(all_data)} samples ({pos_count} anomaly, {neg_count} normal)")

        return all_data

    def _is_normal(self, label: str) -> bool:
        """
        Determine if a label indicates a normal sample.

        Uses the label convention specified by self.dataset:
        - XD-Violence: label 'A' = Normal
        - UCF-Crime: label contains 'Normal' = Normal
        - Auto: checks both conventions

        Args:
            label: The sample label string.

        Returns:
            True if the sample is normal, False if anomaly.
        """
        label_str = str(label)

        if self.dataset == "xd_violence":
            return label_str == "A"
        elif self.dataset == "ucf_crime":
            return "Normal" in label_str
        else:  # "auto" mode - try to detect
            # Check XD-Violence convention
            if label_str == "A":
                return True
            # Check UCF-Crime convention
            if "Normal" in label_str or "normal" in label_str.lower():
                return True
            # Check for common normal directory names
            if label_str in ["Testing_Normal_Videos_Anomaly", "Normal_Videos_event"]:
                return True
            return False

    def _extract_head_data(
        self,
        all_data: List[Tuple[torch.Tensor, str]],
        layer: int,
        head: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract anomaly and normal data for a specific head.

        Args:
            all_data: List of (features, label) tuples.
            layer: Layer index.
            head: Head index.

        Returns:
            Tuple of (anomaly_data, normal_data) arrays.
        """
        anomaly_data = []
        normal_data = []

        for features, label in all_data:
            head_features = features[layer, head, :].float().numpy()
            if self._is_normal(label):
                normal_data.append(head_features)
            else:
                anomaly_data.append(head_features)

        if not anomaly_data or not normal_data:
            return None, None

        return np.array(anomaly_data), np.array(normal_data)

    def _calculate_saliency_for_prompt(
        self,
        prompt_name: str,
        data_path: Union[str, Path],
    ) -> Optional[np.ndarray]:
        """
        Calculate combined saliency scores for a single prompt.

        Args:
            prompt_name: Name of the prompt.
            data_path: Path to the data directory.

        Returns:
            Combined saliency matrix of shape (num_layers, num_heads).
        """
        logger.info(f"Calculating saliency for prompt: {prompt_name}")

        # Load data
        all_data = self._load_prompt_data(data_path)
        if not all_data:
            logger.warning(f"No data loaded for prompt '{prompt_name}'")
            return None

        # Initialize result matrices
        results = {
            "lda": np.full((self.num_layers, self.num_heads), np.nan),
            "kl": np.full((self.num_layers, self.num_heads), np.nan),
            "mmd": np.full((self.num_layers, self.num_heads), np.nan),
            "nmi": np.full((self.num_layers, self.num_heads), np.nan),
        }

        # Calculate metrics for each head
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                pos_data, neg_data = self._extract_head_data(all_data, layer, head)
                pos_data, neg_data = self._extract_head_data(all_data, layer, head)

                if pos_data is not None and neg_data is not None:
                    lda, kl, mmd, nmi = compute_all_metrics(pos_data, neg_data)
                    results["lda"][layer, head] = lda
                    results["kl"][layer, head] = kl
                    results["mmd"][layer, head] = mmd
                    results["nmi"][layer, head] = nmi

            logger.debug(f"Processed layer {layer + 1}/{self.num_layers}")

        # Normalize each metric to [0, 1]
        normalized = {}
        for name, data in results.items():
            valid_mask = ~np.isnan(data)
            if valid_mask.sum() > 0:
                valid_data = data[valid_mask].reshape(-1, 1)
                scaled = MinMaxScaler().fit_transform(valid_data)
                new_data = np.full_like(data, np.nan)
                new_data[valid_mask] = scaled.flatten()
                normalized[name] = new_data
            else:
                normalized[name] = data

        # Combine scores (average of normalized metrics)
        combined = np.nanmean(list(normalized.values()), axis=0)

        return combined

    def run_identification(
        self,
        prompt_data_paths: Dict[str, Union[str, Path]],
    ) -> List[Tuple[int, int]]:
        """
        Run the full RHI pipeline.

        Args:
            prompt_data_paths: Dictionary mapping prompt names to data paths.

        Returns:
            List of (layer, head) tuples for selected robust heads.
        """
        logger.info("=" * 60)
        logger.info("Starting Robust Head Identification (RHI)")
        logger.info(f"Lambda penalty (λ): {self.lambda_penalty}")
        logger.info(f"Top-K selection: {self.top_k}")
        logger.info(f"Prompts: {list(prompt_data_paths.keys())}")
        logger.info("=" * 60)

        # Step 1: Calculate saliency for each prompt
        for name, path in prompt_data_paths.items():
            saliency = self._calculate_saliency_for_prompt(name, path)
            if saliency is not None:
                self.saliency_per_prompt[name] = saliency

        if len(self.saliency_per_prompt) < 2:
            logger.error("RHI requires at least two prompts with valid data")
            return []

        # Step 2: Calculate robustness scores
        logger.info("Calculating robustness scores...")
        stacked = np.stack(list(self.saliency_per_prompt.values()), axis=0)

        # Mean saliency (μ)
        mean_saliency = np.nanmean(stacked, axis=0)

        # Instability (σ) - standard deviation across prompts
        instability = np.nanstd(stacked, axis=0)

        # Robust Saliency Score: RSS = μ - λ * σ
        self.robustness_scores = mean_saliency - self.lambda_penalty * instability

        # Step 3: Select top-K heads
        flat_rss = self.robustness_scores.flatten()
        valid_mask = ~np.isnan(flat_rss)
        valid_indices = np.where(valid_mask)[0]
        valid_scores = flat_rss[valid_mask]

        # Get top-K indices
        top_k_local_indices = np.argsort(valid_scores)[::-1][:self.top_k]
        top_k_flat_indices = valid_indices[top_k_local_indices]

        # Convert to (layer, head) coordinates
        self.robust_heads = [
            (int(idx // self.num_heads), int(idx % self.num_heads))
            for idx in top_k_flat_indices
        ]

        # Log results
        logger.info(f"\nTop {self.top_k} Robust Heads:")
        for i, (layer, head) in enumerate(self.robust_heads):
            score = self.robustness_scores[layer, head]
            logger.info(f"  #{i + 1}: Layer {layer}, Head {head} (RSS: {score:.4f})")

        return self.robust_heads

    def save_results(self, output_dir: Union[str, Path]) -> None:
        """
        Save RHI results to files.

        Args:
            output_dir: Output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save robust head indices
        head_indices_path = output_dir / "robust_head_indices.json"
        save_json(self.robust_heads, head_indices_path)
        logger.info(f"Saved head indices to {head_indices_path}")

        # Save robustness scores
        if self.robustness_scores is not None:
            scores_path = output_dir / "robustness_scores.npy"
            np.save(scores_path, self.robustness_scores)
            logger.info(f"Saved robustness scores to {scores_path}")

        # Save detailed analysis
        self._save_detailed_analysis(output_dir)

    def _save_detailed_analysis(self, output_dir: Path) -> None:
        """Save detailed analysis to CSV."""
        import pandas as pd

        if self.robustness_scores is None:
            return

        # Build detailed dataframe
        data = []
        stacked = np.stack(list(self.saliency_per_prompt.values()), axis=0)
        mean_saliency = np.nanmean(stacked, axis=0)
        instability = np.nanstd(stacked, axis=0)

        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                entry = {
                    "layer": layer,
                    "head": head,
                    "mean_saliency": mean_saliency[layer, head],
                    "instability": instability[layer, head],
                    "rss": self.robustness_scores[layer, head],
                }
                # Add per-prompt scores
                for name, scores in self.saliency_per_prompt.items():
                    entry[f"saliency_{name}"] = scores[layer, head]
                data.append(entry)

        df = pd.DataFrame(data)
        df = df.sort_values("rss", ascending=False).reset_index(drop=True)

        csv_path = output_dir / "detailed_rhi_analysis.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed analysis to {csv_path}")
