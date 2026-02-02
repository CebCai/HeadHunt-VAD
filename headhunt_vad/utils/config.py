"""Configuration management for HeadHunt-VAD."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.

    Later configurations override earlier ones. Nested dictionaries are
    merged recursively.

    Args:
        *configs: Configuration dictionaries to merge.

    Returns:
        Merged configuration dictionary.
    """
    result = {}
    for config in configs:
        result = _deep_merge(result, config)
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary.
        override: Dictionary with overriding values.

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_model_config(model_type: str, config_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load model-specific configuration.

    Args:
        model_type: Type of model (e.g., "internvl3_8b", "llavaov", "qwenvl").
        config_dir: Directory containing model configuration files.
                   Defaults to the package's configs/models directory.

    Returns:
        Model configuration dictionary.

    Raises:
        FileNotFoundError: If the model configuration file does not exist.
    """
    if config_dir is None:
        # Default to package config directory
        config_dir = Path(__file__).parent.parent.parent / "configs" / "models"
    else:
        config_dir = Path(config_dir)

    config_path = config_dir / f"{model_type}.yaml"
    return load_config(config_path)


def parse_cli_overrides(overrides: list) -> Dict[str, Any]:
    """
    Parse command-line configuration overrides.

    Supports dotted notation for nested keys:
        model.path=/path/to/model -> {"model": {"path": "/path/to/model"}}

    Args:
        overrides: List of key=value strings.

    Returns:
        Dictionary of configuration overrides.
    """
    result = {}
    for override in overrides:
        if "=" not in override:
            continue

        key, value = override.split("=", 1)
        keys = key.split(".")

        # Try to parse value as YAML for proper typing
        try:
            value = yaml.safe_load(value)
        except yaml.YAMLError:
            pass  # Keep as string

        # Build nested dictionary
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    return result


class Config:
    """
    Configuration container with attribute access.

    Provides both dictionary-style and attribute-style access to configuration
    values. Nested dictionaries are automatically wrapped as Config objects.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the configuration.

        Args:
            config_dict: Initial configuration dictionary.
        """
        if config_dict is None:
            config_dict = {}

        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value by key."""
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with default.

        Args:
            key: Configuration key.
            default: Default value if key does not exist.

        Returns:
            Configuration value or default.
        """
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self.to_dict()})"
