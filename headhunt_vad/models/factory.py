"""Factory function for creating feature extractors."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from headhunt_vad.models.base_extractor import BaseAttentionExtractor
from headhunt_vad.models.internvl3_extractor import InternVL3Extractor
from headhunt_vad.models.llavaov_extractor import LLaVAOVExtractor
from headhunt_vad.models.qwenvl_extractor import QwenVLExtractor


# Registry of available extractors
_EXTRACTORS = {
    "internvl3": InternVL3Extractor,
    "internvl3_8b": InternVL3Extractor,
    "internvl3-8b": InternVL3Extractor,
    "internvl": InternVL3Extractor,
    "llavaov": LLaVAOVExtractor,
    "llava-ov": LLaVAOVExtractor,
    "llava_ov": LLaVAOVExtractor,
    "qwenvl": QwenVLExtractor,
    "qwen-vl": QwenVLExtractor,
    "qwen_vl": QwenVLExtractor,
}


def get_available_models() -> List[str]:
    """
    Get a list of available model types.

    Returns:
        List of model type names.
    """
    # Return unique model names (remove aliases)
    unique_models = []
    seen = set()
    for name, cls in _EXTRACTORS.items():
        if cls not in seen:
            unique_models.append(name)
            seen.add(cls)
    return unique_models


def create_extractor(
    model_type: str,
    model_path: Union[str, Path],
    device: Union[str, torch.device] = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    config_override: Optional[Dict[str, int]] = None,
) -> BaseAttentionExtractor:
    """
    Create a feature extractor for the specified model type.

    Args:
        model_type: Type of model (e.g., "internvl3", "llavaov", "qwenvl").
        model_path: Path to the pre-trained model.
        device: Device to load the model on.
        dtype: Data type for computations.
        config_override: Optional dictionary to override model config.

    Returns:
        Feature extractor instance.

    Raises:
        ValueError: If the model type is not supported.
    """
    model_type = model_type.lower()

    if model_type not in _EXTRACTORS:
        available = ", ".join(get_available_models())
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available models: {available}"
        )

    extractor_cls = _EXTRACTORS[model_type]
    return extractor_cls(
        model_path=model_path,
        device=device,
        dtype=dtype,
        config_override=config_override,
    )


def register_extractor(name: str, extractor_cls: type) -> None:
    """
    Register a custom extractor class.

    Args:
        name: Name for the extractor.
        extractor_cls: Extractor class (must inherit from BaseAttentionExtractor).

    Raises:
        ValueError: If the class doesn't inherit from BaseAttentionExtractor.
    """
    if not issubclass(extractor_cls, BaseAttentionExtractor):
        raise ValueError(
            f"Extractor class must inherit from BaseAttentionExtractor, "
            f"got {extractor_cls}"
        )
    _EXTRACTORS[name.lower()] = extractor_cls
