"""Dataset classes for HeadHunt-VAD."""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from headhunt_vad.utils.io import load_pickle, list_files, get_video_files
from headhunt_vad.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureDataset(Dataset):
    """
    Dataset for pre-extracted attention head features.

    This dataset loads pre-computed features from pickle files for training
    the anomaly scorer.
    """

    # Common normal label patterns for different datasets
    NORMAL_LABELS = {
        "xd_violence": ["A"],
        "ucf_crime": ["Normal", "Testing_Normal_Videos_Anomaly", "Normal_Videos_event"],
        "auto": ["A", "Normal", "normal", "Testing_Normal_Videos_Anomaly", "Normal_Videos_event"],
    }

    def __init__(
        self,
        data_dir: Union[str, Path],
        head_indices: List[Tuple[int, int]],
        model_config: Dict[str, int],
        feature_key: str = "oproj_inputs_last_token",
        label_key: str = "label",
        normal_label: Optional[str] = None,
        dataset_type: str = "auto",
        transform: Optional[Callable] = None,
        max_files: Optional[int] = None,
    ):
        """
        Initialize the feature dataset.

        Args:
            data_dir: Directory containing pickle files.
            head_indices: List of (layer, head) indices to use.
            model_config: Model configuration with num_layers, num_heads, head_dim.
            feature_key: Key for features in pickle files.
            label_key: Key for labels in pickle files.
            normal_label: Specific label string indicating normal videos (optional).
            dataset_type: Dataset type for auto-detection ("xd_violence", "ucf_crime", or "auto").
            transform: Optional transform to apply to features.
            max_files: Maximum number of files to load (for debugging).
        """
        self.data_dir = Path(data_dir)
        self.head_indices = head_indices
        self.model_config = model_config
        self.feature_key = feature_key
        self.label_key = label_key
        self.normal_label = normal_label
        self.dataset_type = dataset_type
        self.transform = transform

        # Find all pickle files
        self.files = list_files(self.data_dir, extensions=[".pkl"])
        if max_files is not None:
            self.files = self.files[:max_files]

        # Load all data
        self.data = []
        self._load_data()

        logger.info(f"Loaded {len(self.data)} samples from {self.data_dir}")

    def _is_normal_label(self, label: str) -> bool:
        """
        Check if a label indicates a normal sample.

        Args:
            label: The sample label string.

        Returns:
            True if the label indicates a normal sample.
        """
        label_str = str(label)

        # If specific normal_label is provided, use it
        if self.normal_label is not None:
            return label_str == self.normal_label

        # Otherwise use dataset-specific or auto-detection
        normal_labels = self.NORMAL_LABELS.get(self.dataset_type, self.NORMAL_LABELS["auto"])

        # Direct match
        if label_str in normal_labels:
            return True

        # Substring match for flexibility
        label_lower = label_str.lower()
        if "normal" in label_lower:
            return True

        return False

    def _load_data(self) -> None:
        """Load all data from pickle files."""
        num_layers = self.model_config["num_layers"]
        num_heads = self.model_config["num_heads"]
        head_dim = self.model_config["head_dim"]

        for file_path in self.files:
            try:
                item = load_pickle(file_path)

                # Try different feature keys for compatibility
                features_raw = item.get(self.feature_key)
                if features_raw is None:
                    # Try alternative keys
                    for alt_key in ["self_attns", "oproj_inputs_last_token", "oproj_inputs"]:
                        features_raw = item.get(alt_key)
                        if features_raw is not None:
                            break

                label = item.get(self.label_key)

                if features_raw is None or label is None:
                    continue

                # Handle different feature shapes
                # Expected final shape: (num_layers, num_heads, head_dim)
                if features_raw.dim() == 2:
                    # Shape: [num_layers, num_heads * head_dim]
                    features = features_raw.view(num_layers, num_heads, head_dim)
                elif features_raw.dim() == 3:
                    # Shape: [1, num_layers, num_heads * head_dim] or similar
                    features = features_raw.squeeze(0).view(num_layers, num_heads, head_dim)
                else:
                    # Try to squeeze and reshape
                    features = features_raw.squeeze().view(num_layers, num_heads, head_dim)

                # Extract selected heads and concatenate
                selected = [features[l, h, :] for l, h in self.head_indices]
                concatenated = torch.cat(selected).float()

                # Convert label to binary using the improved method
                label_binary = 0 if self._is_normal_label(label) else 1

                self.data.append({
                    "features": concatenated,
                    "label": label_binary,
                    "label_str": label,
                    "filename": file_path.name,
                })

            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with features, label, label_str, and filename.
        """
        item = self.data[idx].copy()

        if self.transform is not None:
            item["features"] = self.transform(item["features"])

        return item

    def get_features_and_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all features and labels as tensors.

        Returns:
            Tuple of (features, labels) tensors.
        """
        features = torch.stack([d["features"] for d in self.data])
        labels = torch.tensor([d["label"] for d in self.data])
        return features, labels


class VideoDataset(Dataset):
    """
    Dataset for video files.

    This dataset provides video file paths and labels for feature extraction.
    """

    def __init__(
        self,
        video_dir: Union[str, Path],
        label_mode: str = "directory",
        normal_label: str = "Testing_Normal_Videos_Anomaly",
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ):
        """
        Initialize the video dataset.

        Args:
            video_dir: Directory containing video files.
            label_mode: How to extract labels:
                - "directory": Use parent directory name as label
                - "filename": Extract label from filename pattern
            normal_label: Directory/label name indicating normal videos.
            recursive: Whether to search recursively.
            extensions: Video file extensions to include.
        """
        self.video_dir = Path(video_dir)
        self.label_mode = label_mode
        self.normal_label = normal_label
        self.recursive = recursive

        if extensions is None:
            extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]

        # Find all video files
        self.video_files = get_video_files(self.video_dir, recursive=recursive)

        logger.info(f"Found {len(self.video_files)} videos in {self.video_dir}")

    def __len__(self) -> int:
        """Return number of videos."""
        return len(self.video_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a video sample by index.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with video_path, label, and label_str.
        """
        video_path = self.video_files[idx]

        # Extract label
        label_str = self._extract_label(video_path)
        label = 0 if label_str == self.normal_label else 1

        return {
            "video_path": str(video_path),
            "label": label,
            "label_str": label_str,
        }

    def _extract_label(self, video_path: Path) -> str:
        """
        Extract label from video path.

        Args:
            video_path: Path to the video file.

        Returns:
            Label string.
        """
        if self.label_mode == "directory":
            # Use parent directory name
            return video_path.parent.name
        elif self.label_mode == "filename":
            # Extract from filename (e.g., "video_label_Fighting.mp4")
            import re
            match = re.search(r"label_([A-Za-z]+)", video_path.stem)
            if match:
                return match.group(1)
            return "unknown"
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

    def get_all_paths(self) -> List[Path]:
        """Get all video paths."""
        return self.video_files.copy()
