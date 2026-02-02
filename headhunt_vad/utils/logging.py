"""Logging utilities for HeadHunt-VAD."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


# Global logger cache
_loggers = {}


def setup_logger(
    name: str = "headhunt_vad",
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name.
        level: Logging level (e.g., logging.INFO or "INFO").
        log_file: Optional path to log file.
        format_string: Optional custom format string.

    Returns:
        Configured logger.
    """
    if name in _loggers:
        return _loggers[name]

    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear any existing handlers

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    # Cache the logger
    _loggers[name] = logger

    return logger


def get_logger(name: str = "headhunt_vad") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


class LoggerMixin:
    """
    Mixin class that provides a logger property.

    Classes that inherit from this mixin will have access to a logger
    named after the class.
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
