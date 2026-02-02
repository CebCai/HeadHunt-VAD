"""QwenVL attention head feature extractor."""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

from headhunt_vad.models.base_extractor import BaseAttentionExtractor
from headhunt_vad.data.video_loader import load_video
from headhunt_vad.utils.logging import get_logger

logger = get_logger(__name__)


class QwenVLExtractor(BaseAttentionExtractor):
    """
    Feature extractor for QwenVL models.

    This extractor captures attention head outputs from QwenVL's
    language model component by hooking the o_proj modules.
    """

    # Default configuration for QwenVL
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
        Initialize the QwenVL extractor.

        Args:
            model_path: Path to the QwenVL model.
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
        """Load the QwenVL model and tokenizer."""
        logger.info(f"Loading QwenVL model from {self.model_path}")

        # Check if it's likely a Qwen2.5-VL model
        is_qwen2_5 = "Qwen2.5" in str(self.model_path) or "qwen2.5" in str(self.model_path).lower()

        if is_qwen2_5 and Qwen2_5_VLForConditionalGeneration is not None:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(self.model_path),
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval().to(self.device)
            
            try:
                self.processor = AutoProcessor.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=True,
                )
                self.tokenizer = self.processor.tokenizer
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=True,
                )
                self.processor = None
        else:
            self.model = AutoModel.from_pretrained(
                str(self.model_path),
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval().to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )
            self.processor = None

        # Update config from model if available
        if hasattr(self.model, "config"):
            config = self.model.config
            if hasattr(config, "num_hidden_layers"):
                self._config["num_layers"] = config.num_hidden_layers
            if hasattr(config, "num_attention_heads"):
                self._config["num_heads"] = config.num_attention_heads

        logger.info(f"Model loaded. Config: {self._config}")

    def _get_attention_layers(self) -> Any:
        """Get the attention layers from QwenVL."""
        # QwenVL typically has model.layers structure
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        elif hasattr(self.model, "layers"):
            return self.model.layers
        # Qwen2.5-VL specific path
        elif hasattr(self.model, "visual"):  # Qwen2.5-VL has a visual encoder
             if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                 return self.model.model.layers
             elif hasattr(self.model, "layers"):
                 return self.model.layers
        
        # Generic search
        candidates = ["model.layers", "model.decoder.layers", "transformer.h", "visual.transformer.h"]
        for path in candidates:
            obj = self.model
            for part in path.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    obj = None
                    break
            if obj is not None:
                return obj

        raise AttributeError("Could not find attention layers in QwenVL model")

    def _get_oproj_module(self, layer_idx: int) -> torch.nn.Module:
        """Get the o_proj module for a specific layer."""
        layers = self._get_attention_layers()
        layer = layers[layer_idx]

        # Try different attribute names
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            return layer.self_attn.o_proj
        elif hasattr(layer, "attn") and hasattr(layer.attn, "c_proj"):
            return layer.attn.c_proj
        else:
            raise AttributeError(f"Could not find o_proj in layer {layer_idx}")

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

        # Build prompt with frame placeholders (QwenVL style)
        if self.processor is not None and process_vision_info is not None:
            # Qwen2.5-VL path
            from PIL import Image
            import numpy as np
            
            # Convert pixel values back to PIL images if needed (or assume load_video returns compatible format)
            # But load_video returns pixel_values tensor. For Qwen2.5-VL we typically need PIL images if using processor
            # However, headhunt's load_video returns tensors designed for internvl/llava.
            # We might need to reload as PIL for Qwen2.5-VL if we want to use the official processor properly.
            # For efficiency, let's try to adapt the tensor or re-read frames.
            # Re-reading frames is safer given the complexity of process_vision_info
            
            from headhunt_vad.data.video_loader import get_frame_indices
            
            try:
                from decord import VideoReader, cpu
                vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
                total_frames = len(vr)
                fps = float(vr.get_avg_fps()) or 30.0
                
                # Get frame indices
                frame_indices = get_frame_indices(total_frames, num_segments, fps, bound)
                
                # Read frames
                video_frames = []
                for idx in frame_indices:
                    video_frames.append(Image.fromarray(vr[idx].asnumpy()).convert("RGB"))
                    
            except Exception as e:
                logger.error(f"Error reading video frames for Qwen2.5-VL: {e}")
                return None
            
            # Construct messages with PIL images
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": frame} for frame in video_frames],
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Register hooks
            self.register_hooks()
            
            try:
                with torch.inference_mode():
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)
                    
                    self.model.generate(**inputs, max_new_tokens=1)
            except Exception as e:
                logger.error(f"Error during forward pass for {video_path}: {e}")
                return None
            finally:
                self.remove_hooks()

        else:
            # Legacy Qwen-VL path
            # QwenVL uses special tokens for images
            video_prefix = "".join([f"<img></img>\n" for _ in range(len(num_patches_list))])
            full_prompt = video_prefix + prompt

            # Register hooks
            self.register_hooks()

            try:
                with torch.inference_mode():
                    # QwenVL-specific inference
                    if hasattr(self.model, "chat"):
                        self.model.chat(
                            tokenizer=self.tokenizer,
                            query=full_prompt,
                            images=pixel_values,
                            max_new_tokens=1,
                        )
                    elif hasattr(self.model, "generate"):
                        inputs = self.tokenizer(
                            full_prompt,
                            return_tensors="pt",
                        ).to(self.device)
                        self.model.generate(
                            **inputs,
                            images=pixel_values,
                            max_new_tokens=1,
                            do_sample=False,
                        )
                    else:
                        # Fallback
                        inputs = self.tokenizer(
                            full_prompt,
                            return_tensors="pt",
                        ).to(self.device)
                        self.model(**inputs, images=pixel_values)
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
