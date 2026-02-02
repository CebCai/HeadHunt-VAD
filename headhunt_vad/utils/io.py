"""File I/O utilities for HeadHunt-VAD."""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


def save_pickle(data: Any, path: Union[str, Path]) -> None:
    """
    Save data to a pickle file.

    Args:
        data: Data to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: Union[str, Path]) -> Any:
    """
    Load data from a pickle file.

    Args:
        path: Path to the pickle file.

    Returns:
        Loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save (must be JSON-serializable).
        path: Output file path.
        indent: JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types
    data = _convert_numpy_types(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Object to convert.

    Returns:
        Converted object.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    return obj


def list_files(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
    recursive: bool = False,
) -> List[Path]:
    """
    List files in a directory.

    Args:
        directory: Directory to search.
        extensions: List of file extensions to include (e.g., [".mp4", ".avi"]).
                   If None, includes all files.
        recursive: Whether to search recursively.

    Returns:
        List of file paths.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    if extensions is not None:
        extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions]

    files = []
    if recursive:
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                filepath = Path(root) / filename
                if extensions is None or filepath.suffix.lower() in extensions:
                    files.append(filepath)
    else:
        for filepath in directory.iterdir():
            if filepath.is_file():
                if extensions is None or filepath.suffix.lower() in extensions:
                    files.append(filepath)

    return sorted(files)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object of the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_video_files(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    Get all video files in a directory.

    Args:
        directory: Directory to search.
        recursive: Whether to search recursively.

    Returns:
        List of video file paths.
    """
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    return list_files(directory, extensions=video_extensions, recursive=recursive)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by replacing invalid characters.

    Args:
        filename: Original filename.

    Returns:
        Sanitized filename.
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename
