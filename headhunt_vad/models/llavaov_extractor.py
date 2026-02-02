"""LLaVA-OV attention head feature extractor."""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModel, AutoTokenizer

from headhunt_vad.models.base_extractor import BaseAttentionExtractor
from headhunt_vad.data.video_loader import load_video
from headhunt_vad.utils.logging import get_logger

logger = get_logger(__name__)


class LLaVAOVExtractor(BaseAttentionExtractor):
    """
    Feature extractor for LLaVA-OV models.

    This extractor captures attention head outputs from LLaVA-OV's
    language model component by hooking the o_proj modules.
    """

    # Default configuration for LLaVA-OV
    DEFAULT_CONFIG = {
        "num_layers": 32,
        "num_heads": 32,
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
        Initialize the LLaVA-OV extractor.

        Args:
            model_path: Path to the LLaVA-OV model.
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
        """Load the LLaVA-OV model and tokenizer."""
        logger.info(f"Loading LLaVA-OV model from {self.model_path}")

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

        # Update config from model if available
        if hasattr(self.model, "config"):
            config = self.model.config
            if hasattr(config, "num_hidden_layers"):
                self._config["num_layers"] = config.num_hidden_layers
            if hasattr(config, "num_attention_heads"):
                self._config["num_heads"] = config.num_attention_heads

        logger.info(f"Model loaded. Config: {self._config}")

    def _get_attention_layers(self) -> Any:
        """Get the attention layers from LLaVA-OV."""
        # LLaVA-OV typically has model.layers structure
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "layers"):
            return self.model.layers
        else:
            raise AttributeError("Could not find attention layers in LLaVA-OV model")

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
            bound: Optional (start_time, end_time) in seconds.

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

        # Build prompt with frame placeholders (LLaVA-OV style)
        video_prefix = "".join([f"<image>\n" for _ in range(len(num_patches_list))])
        full_prompt = video_prefix + prompt

        # Register hooks
        self.register_hooks()

        try:
            with torch.inference_mode():
                # Tokenize input
                inputs = self.tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                # Run forward pass (implementation depends on specific LLaVA-OV version)
                if hasattr(self.model, "generate"):
                    self.model.generate(
                        **inputs,
                        pixel_values=pixel_values,
                        max_new_tokens=1,
                        do_sample=False,
                    )
                else:
                    # Fallback to forward pass
                    self.model(
                        **inputs,
                        pixel_values=pixel_values,
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

        # Stack layer outputs
        valid_layer_ids = sorted(self.layer_outputs.keys())
        stacked_layers = []
        for layer_id in valid_layer_ids:
            if self.layer_outputs[layer_id]:
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
        """Extract label from video path."""
        import re
        # Try to extract from filename pattern
        match = re.search(r"label_([A-Za-z]+)", video_path.stem)
        if match:
            return match.group(1)
        return video_path.parent.name if video_path.parent.name else "unknown"

    def extract_expert_heads(
        self,
        video_path: Union[str, Path],
        prompt: str,
        head_indices: List[Tuple[int, int]],
        num_segments: int = 16,
    ) -> Optional[torch.Tensor]:
        """Extract features from specific expert heads."""
        return self.extract_head_features(
            video_path, prompt, head_indices, num_segments
        )
