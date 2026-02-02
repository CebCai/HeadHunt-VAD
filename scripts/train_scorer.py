#!/usr/bin/env python
"""Train anomaly scorer script for HeadHunt-VAD.

Trains a logistic regression model on pre-extracted expert head features.
"""

import argparse
import ast
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from headhunt_vad.data.dataset import FeatureDataset
from headhunt_vad.models.anomaly_scorer import AnomalyScorer
from headhunt_vad.utils.config import load_config, merge_configs, parse_cli_overrides, get_model_config
from headhunt_vad.utils.io import ensure_dir, load_json
from headhunt_vad.utils.logging import setup_logger


# Default model configurations for supported models
DEFAULT_MODEL_CONFIGS = {
    "internvl3": {"num_layers": 28, "num_heads": 28, "head_dim": 128},
    "internvl3_8b": {"num_layers": 28, "num_heads": 28, "head_dim": 128},
    "internvl3_14b": {"num_layers": 40, "num_heads": 40, "head_dim": 128},
    "llavaov": {"num_layers": 32, "num_heads": 32, "head_dim": 128},
    "llava-ov": {"num_layers": 32, "num_heads": 32, "head_dim": 128},
    "qwenvl": {"num_layers": 32, "num_heads": 32, "head_dim": 128},
    "qwen-vl": {"num_layers": 32, "num_heads": 32, "head_dim": 128},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an anomaly scorer using attention head features."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Feature directory.")
    parser.add_argument(
        "--head_indices",
        type=str,
        help="Head indices as list of tuples, e.g., '[(18,4),(19,21)]'.",
    )
    parser.add_argument(
        "--head_indices_file",
        type=str,
        help="Path to JSON file containing head indices (from RHI output).",
    )
    parser.add_argument("--output_dir", type=str, default="./models", help="Model output directory.")
    parser.add_argument("--test_size", type=float, help="Test set fraction.")
    parser.add_argument("--random_state", type=int, help="Random seed.")
    parser.add_argument("--model_type", type=str, help="Model type for config lookup.")
    parser.add_argument("--dataset", type=str, choices=["ucf_crime", "xd_violence"],
                        help="Dataset type for label parsing.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional config overrides.",
    )
    return parser.parse_args()


def get_model_configuration(model_type: str, config: dict) -> dict:
    """
    Get model configuration from config file or defaults.

    Args:
        model_type: Model type string.
        config: Loaded configuration dictionary.

    Returns:
        Model configuration dictionary.
    """
    # Try to load from YAML config first
    try:
        yaml_cfg = get_model_config(model_type)
        if yaml_cfg and "model" in yaml_cfg:
            return yaml_cfg["model"]
    except FileNotFoundError:
        pass

    # Fall back to defaults
    model_type_lower = model_type.lower()
    if model_type_lower in DEFAULT_MODEL_CONFIGS:
        return DEFAULT_MODEL_CONFIGS[model_type_lower]

    # Last resort: check main config
    if "model" in config:
        model_cfg = config["model"]
        if all(k in model_cfg for k in ["num_layers", "num_heads", "head_dim"]):
            return {
                "num_layers": model_cfg["num_layers"],
                "num_heads": model_cfg["num_heads"],
                "head_dim": model_cfg["head_dim"],
            }

    # Default to InternVL3-8B
    return DEFAULT_MODEL_CONFIGS["internvl3_8b"]


def get_normal_label(dataset: str) -> str:
    """Get the normal label string for a dataset."""
    if dataset == "xd_violence":
        return "A"
    elif dataset == "ucf_crime":
        return "Testing_Normal_Videos_Anomaly"
    else:
        return "A"  # Default


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    cli_overrides = parse_cli_overrides(args.overrides) if args.overrides else {}
    config = merge_configs(config, cli_overrides)

    # Setup
    logger = setup_logger("train_scorer")
    ensure_dir(args.output_dir)

    # Get head indices from argument or file
    if args.head_indices:
        head_indices = ast.literal_eval(args.head_indices)
    elif args.head_indices_file:
        head_indices = load_json(args.head_indices_file)
        # Convert from list of lists to list of tuples if needed
        head_indices = [tuple(h) if isinstance(h, list) else h for h in head_indices]
    else:
        logger.error("Either --head_indices or --head_indices_file is required")
        return

    logger.info(f"Head indices: {head_indices}")

    # Get parameters
    scorer_config = config.get("scorer", {})
    test_size = args.test_size or scorer_config.get("test_size", 0.4)
    random_state = args.random_state or scorer_config.get("random_state", 42)

    # Get model configuration
    model_type = args.model_type or config.get("model", {}).get("type", "internvl3_8b")
    model_config = get_model_configuration(model_type, config)

    logger.info(f"Model type: {model_type}")
    logger.info(f"Model config: {model_config}")

    # Determine dataset type and normal label
    dataset = args.dataset
    if not dataset:
        # Try to infer from config
        if "ucf" in args.data_dir.lower() or "ucf" in config.get("data", {}).get("dataset", "").lower():
            dataset = "ucf_crime"
        else:
            dataset = "xd_violence"

    normal_label = get_normal_label(dataset)
    logger.info(f"Dataset: {dataset}, Normal label: {normal_label}")

    # Load dataset
    logger.info(f"Loading features from {args.data_dir}")
    dataset_obj = FeatureDataset(
        data_dir=args.data_dir,
        head_indices=head_indices,
        model_config=model_config,
        normal_label=normal_label,
    )

    features, labels = dataset_obj.get_features_and_labels()
    X = features.numpy()
    y = labels.numpy()

    logger.info(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {np.sum(y == 0)} normal, {np.sum(y == 1)} anomaly")

    # Validate we have both classes
    if len(np.unique(y)) < 2:
        logger.error("Dataset must contain both normal and anomaly samples")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train scorer
    scorer = AnomalyScorer(
        head_indices=head_indices,
        head_dim=model_config["head_dim"],
        random_state=random_state,
    )
    scorer.fit(X_train, y_train)

    # Evaluate
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score, average_precision_score,
        classification_report,
    )

    y_pred = scorer.predict(X_test)
    y_proba = scorer.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"ROC-AUC: {auc:.4f}")
    logger.info(f"Average Precision: {ap:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    # Save model
    heads_str = "_".join([f"L{l}H{h}" for l, h in head_indices])
    model_path = Path(args.output_dir) / f"anomaly_scorer_{heads_str}.joblib"
    scorer.save(model_path)

    # Also save a default-named version for easy loading
    default_path = Path(args.output_dir) / "anomaly_scorer.joblib"
    scorer.save(default_path)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Default model saved to {default_path}")

    # Save training metrics
    from headhunt_vad.utils.io import save_json
    metrics = {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "average_precision": float(ap),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "head_indices": head_indices,
        "model_type": model_type,
        "dataset": dataset,
    }
    save_json(metrics, Path(args.output_dir) / "training_metrics.json")


if __name__ == "__main__":
    main()
