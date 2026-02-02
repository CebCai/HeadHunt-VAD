"""InternVL3 attention head feature extractor."""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModel, AutoTokenizer

from headhunt_vad.models.base_extractor import BaseAttentionExtractor
from headhunt_vad.data.video_loader import load_video
from headhunt_vad.utils.logging import get_logger

logger = get_logger(__name__)


class InternVL3Extractor(BaseAttentionExtractor):
    """
    Feature extractor for InternVL3 models.

    This extractor captures attention head outputs from InternVL3's
    language model component by hooking the o_proj modules.
    """

    # Default configuration for InternVL3-8B
    DEFAULT_CONFIG = {
        "num_layers": 28,
        "num_heads": 28,
        "head_dim": 128,
        "hidden_size": 4096,
    }

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Union[str, torch.device] = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        config_override: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the InternVL3 extractor.

        Args:
            model_path: Path to the InternVL3 model.
            device: Device to load the model on.
            dtype: Data type for computations.
            config_override: Optional dictionary to override default config.
        """
        super().__init__(model_path, device, dtype)

        # Apply config overrides
        self._config = self.DEFAULT_CONFIG.copy()
        if config_override:
            self._config.update(config_override)

    def get_model_config(self) -> Dict[str, int]:
        """Get the model configuration."""
        return self._config.copy()

    def load_model(self) -> None:
        """Load the InternVL3 model and tokenizer."""
        logger.info(f"Loading InternVL3 model from {self.model_path}")

        self.model = AutoModel.from_pretrained(
            str(self.model_path),
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            use_fast=False,
        )

        # Attach tokenizer to model if needed
        if not hasattr(self.model, "tokenizer"):
            self.model.tokenizer = self.tokenizer

        # Update config from model if available
        if hasattr(self.model, "language_model"):
            lm_config = self.model.language_model.config
            if hasattr(lm_config, "num_hidden_layers"):
                self._config["num_layers"] = lm_config.num_hidden_layers
            if hasattr(lm_config, "num_attention_heads"):
                self._config["num_heads"] = lm_config.num_attention_heads

        logger.info(f"Model loaded. Config: {self._config}")

    def _get_attention_layers(self) -> Any:
        """Get the attention layers from InternVL3."""
        return self.model.language_model.model.layers

    def _get_oproj_module(self, layer_idx: int) -> torch.nn.Module:
        """Get the o_proj module for a specific layer."""
        layers = self._get_attention_layers()
        return layers[layer_idx].self_attn.o_proj

    def _get_final_layer(self) -> torch.nn.Module:
        """Get the final transformer layer."""
        layers = self._get_attention_layers()
        return layers[-1]

    def extract_features(
        self,
        video_path: Union[str, Path],
        prompt: str,
        num_segments: int = 16,
        bound: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extract attention head features from a video.

        Args:
            video_path: Path to the video file.
            prompt: Text prompt for the model.
            num_segments: Number of frames to sample.
            bound: Optional (start_time, end_time) in seconds to limit extraction.

        Returns:
            Dictionary with features or None if extraction failed.
        """
        if self.model is None:
            self.load_model()

        video_path = Path(video_path)

        # Clear previous outputs
        self.layer_outputs.clear()

        # Load video
        try:
            pixel_values, num_patches_list, fps = load_video(
                video_path,
                num_segments=num_segments,
                input_size=448,
                max_num=1,
                bound=bound,
            )
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return None

        pixel_values = pixel_values.to(self.dtype).to(self.device)

        # Build prompt with frame placeholders
        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
        question = video_prefix + prompt

        # Register hooks
        self.register_hooks()

        try:
            with torch.inference_mode():
                # Run forward pass
                self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config={"max_new_tokens": 1, "do_sample": False},
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=False,
                )
        except Exception as e:
            logger.error(f"Error during forward pass for {video_path}: {e}")
            return None
        finally:
            self.remove_hooks()

        # Process captured outputs
        if not self.layer_outputs:
            logger.warning(f"No features captured for {video_path}")
            return None

        # Stack layer outputs: (num_layers, num_heads * head_dim)
        valid_layer_ids = sorted(self.layer_outputs.keys())
        stacked_layers = []
        for layer_id in valid_layer_ids:
            if self.layer_outputs[layer_id]:
                # Take the last captured output
                stacked_layers.append(self.layer_outputs[layer_id][-1])

        if not stacked_layers:
            return None

        oproj_inputs = torch.stack(stacked_layers)
        if oproj_inputs.shape[1] == 1:
            oproj_inputs = oproj_inputs.squeeze(1)

        # Extract label from path
        label = self._extract_label(video_path)

        return {
            "oproj_inputs": oproj_inputs,
            "label": label,
            "video_name": video_path.name,
        }

    def _extract_label(self, video_path: Path) -> str:
        """
        Extract label from video path.

        Supports multiple label formats:
        1. XD-Violence: filename contains 'label_XXX' (e.g., video_label_A.mp4 -> A)
        2. UCF-Crime: directory structure Videos/Category/video.mp4 -> Category
        3. Fallback: parent directory name

        Args:
            video_path: Path to the video file.

        Returns:
            Extracted label string.
        """
        import re

        # Try to extract from filename pattern (XD-Violence format)
        # Pattern: label_XXX where XXX can be A, B1-0-0, etc.
        match = re.search(r"label_([A-Za-z0-9\-]+)", video_path.stem)
        if match:
            return match.group(1)

        # Try UCF-Crime directory structure
        parts = video_path.parts
        try:
            videos_idx = parts.index("Videos")
            if videos_idx + 1 < len(parts):
                return parts[videos_idx + 1]
        except ValueError:
            pass

        # Fallback to parent directory name
        return video_path.parent.name if video_path.parent.name else "unknown"

    def extract_expert_heads(
        self,
        video_path: Union[str, Path],
        prompt: str,
        head_indices: List[Tuple[int, int]],
        num_segments: int = 16,
    ) -> Optional[torch.Tensor]:
        """
        Extract features from specific expert heads.

        This is a convenience method that extracts features from only
        the specified attention heads.

        Args:
            video_path: Path to the video file.
            prompt: Text prompt for the model.
            head_indices: List of (layer, head) tuples.
            num_segments: Number of frames to sample.

        Returns:
            expert_head_features
        """
        return self.extract_head_features(
            video_path, prompt, head_indices, num_segments
        )
