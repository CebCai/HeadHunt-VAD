"""Abstract base class for attention head feature extractors."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


class BaseAttentionExtractor(ABC):
    """
    Abstract base class for extracting attention head features from MLLMs.

    This class defines the interface for extracting features from the attention
    heads of multi-modal large language models. Implementations should handle
    model loading, hook registration, and feature extraction.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Union[str, torch.device] = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the extractor.

        Args:
            model_path: Path to the pre-trained model.
            device: Device to load the model on.
            dtype: Data type for model computations.
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.dtype = dtype

        # To be initialized by subclasses
        self.model = None
        self.tokenizer = None
        self.handles = []
        self.layer_outputs = {}

    @abstractmethod
    def get_model_config(self) -> Dict[str, int]:
        """
        Get the model configuration.

        Returns:
            Dictionary with keys:
            - num_layers: Number of transformer layers
            - num_heads: Number of attention heads per layer
            - head_dim: Dimension of each attention head
            - hidden_size: Hidden dimension of the model
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model and tokenizer.

        This method should initialize self.model and self.tokenizer.
        """
        pass

    @abstractmethod
    def _get_attention_layers(self) -> Any:
        """
        Get the attention layers from the model.

        Returns:
            The module containing the attention layers.
        """
        pass

    @abstractmethod
    def _get_oproj_module(self, layer_idx: int) -> torch.nn.Module:
        """
        Get the o_proj module for a specific layer.

        Args:
            layer_idx: Index of the transformer layer.

        Returns:
            The o_proj module.
        """
        pass

    @abstractmethod
    def _get_final_layer(self) -> torch.nn.Module:
        """
        Get the final transformer layer.

        Returns:
            The final layer module.
        """
        pass

    def _create_oproj_hook(self, layer_id: int):
        """
        Create a forward pre-hook for capturing o_proj inputs.

        This hook captures the input to the output projection layer,
        which contains the concatenated attention head outputs before
        projection.

        Args:
            layer_id: Index of the layer.

        Returns:
            Hook function.
        """
        def hook(module, input_args):
            if layer_id not in self.layer_outputs:
                self.layer_outputs[layer_id] = []

            # input_args[0] shape: (batch_size, seq_len, num_heads * head_dim)
            input_tensor = input_args[0]

            if input_tensor.ndim == 3 and input_tensor.shape[1] > 0:
                # Capture only the last token
                last_token_features = input_tensor[:, -1, :].detach().cpu()
                self.layer_outputs[layer_id].append(last_token_features)

            return None

        return hook

    def register_hooks(self, layer_indices: Optional[List[int]] = None) -> None:
        """
        Register hooks on specified layers.

        Args:
            layer_indices: List of layer indices to hook. If None, hooks all layers.
        """
        self.remove_hooks()
        self.layer_outputs.clear()

        config = self.get_model_config()
        num_layers = config["num_layers"]

        if layer_indices is None:
            layer_indices = list(range(num_layers))

        for layer_idx in layer_indices:
            if 0 <= layer_idx < num_layers:
                oproj_module = self._get_oproj_module(layer_idx)
                handle = oproj_module.register_forward_pre_hook(
                    self._create_oproj_hook(layer_idx)
                )
                self.handles.append(handle)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @abstractmethod
    def extract_features(
        self,
        video_path: Union[str, Path],
        prompt: str,
        num_segments: int = 16,
        bound: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention head features from a video.

        Args:
            video_path: Path to the video file.
            prompt: Text prompt for the model.
            num_segments: Number of frames to sample.
            bound: Optional (start_time, end_time) in seconds to limit extraction.

        Returns:
            Dictionary containing:
            - oproj_inputs: Tensor of shape (num_layers, num_heads * head_dim)
            - label: Label extracted from video path (if available)
            - video_name: Name of the video file
        """
        pass

    def extract_head_features(
        self,
        video_path: Union[str, Path],
        prompt: str,
        head_indices: List[Tuple[int, int]],
        num_segments: int = 16,
        bound: Optional[Tuple[float, float]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Extract features from specific attention heads.

        Args:
            video_path: Path to the video file.
            prompt: Text prompt for the model.
            head_indices: List of (layer, head) indices.
            num_segments: Number of frames to sample.
            bound: Optional (start_time, end_time) in seconds to limit extraction.

        Returns:
            expert_head_features: Tensor of shape (1, num_heads, head_dim)
        """
        # Extract all features
        features = self.extract_features(video_path, prompt, num_segments, bound=bound)

        if features is None or "oproj_inputs" not in features:
            return None

        oproj_inputs = features["oproj_inputs"]  # (num_layers, num_heads * head_dim)
        config = self.get_model_config()
        num_heads = config["num_heads"]
        head_dim = config["head_dim"]

        # Reshape to (num_layers, num_heads, head_dim)
        reshaped = oproj_inputs.view(config["num_layers"], num_heads, head_dim)

        # Extract specified heads
        selected_heads = []
        for layer_idx, head_idx in head_indices:
            selected_heads.append(reshaped[layer_idx, head_idx, :])

        # Stack: (num_selected_heads, head_dim)
        expert_head_features = torch.stack(selected_heads, dim=0).unsqueeze(0)

        return expert_head_features

    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()
