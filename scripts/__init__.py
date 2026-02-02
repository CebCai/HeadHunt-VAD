"""HeadHunt-VAD command-line scripts."""

from scripts.extract_features import main as extract_main
from scripts.run_rhi import main as rhi_main
from scripts.train_scorer import main as train_main
from scripts.inference import main as inference_main
from scripts.evaluate import main as evaluate_main

__all__ = [
    "extract_main",
    "rhi_main",
    "train_main",
    "inference_main",
    "evaluate_main",
]
