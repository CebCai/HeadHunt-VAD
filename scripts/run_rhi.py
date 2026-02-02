#!/usr/bin/env python
"""RHI (Robust Head Identification) analysis script."""

import argparse
from pathlib import Path

from headhunt_vad.core.rhi import RobustHeadIdentifier
from headhunt_vad.utils.config import load_config, merge_configs, parse_cli_overrides
from headhunt_vad.utils.io import ensure_dir
from headhunt_vad.utils.logging import setup_logger
from headhunt_vad.visualization.heatmaps import plot_robustness_heatmap


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Robust Head Identification (RHI) analysis."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--prompt_dirs",
        nargs="+",
        required=True,
        help="Prompt data directories in format: name=/path/to/features (at least 2).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/rhi",
        help="Directory to save RHI results.",
    )
    parser.add_argument("--lambda_penalty", type=float, help="Instability penalty weight.")
    parser.add_argument("--top_k", type=int, help="Number of heads to select.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["xd_violence", "ucf_crime", "auto"],
        default="auto",
        help="Dataset type for label parsing (default: auto-detect).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional config overrides.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    cli_overrides = parse_cli_overrides(args.overrides) if args.overrides else {}
    config = merge_configs(config, cli_overrides)

    # Setup
    logger = setup_logger("rhi_analysis")

    # Parse prompt directories
    prompt_data_paths = {}
    for item in args.prompt_dirs:
        if "=" in item:
            name, path = item.split("=", 1)
            prompt_data_paths[name] = path
        else:
            name = Path(item).name
            prompt_data_paths[name] = item

    if len(prompt_data_paths) < 2:
        logger.error("At least 2 prompt data directories are required")
        return

    # Get RHI parameters
    rhi_config = config.get("rhi", {})
    lambda_penalty = args.lambda_penalty or rhi_config.get("lambda_penalty", 0.5)
    top_k = args.top_k or rhi_config.get("top_k", 5)

    # Get model config
    model_config_name = config.get("model", {}).get("type", "internvl3_8b")
    try:
        from headhunt_vad.utils.config import get_model_config
        model_cfg_yaml = get_model_config(model_config_name)
        model_config = model_cfg_yaml.get("model", {})
    except FileNotFoundError:
        # Fallback to default InternVL3 config
        model_config = {
            "num_layers": 28,
            "num_heads": 28,
            "head_dim": 128,
        }

    # Ensure required keys
    if "num_layers" not in model_config:
        model_config["num_layers"] = 28
    if "num_heads" not in model_config:
        model_config["num_heads"] = 28
    if "head_dim" not in model_config:
        model_config["head_dim"] = 128

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    logger.info(f"Model config: {model_config}")
    logger.info(f"Prompt data paths: {prompt_data_paths}")
    logger.info(f"Lambda penalty: {lambda_penalty}")
    logger.info(f"Top-K: {top_k}")

    # Run RHI
    rhi = RobustHeadIdentifier(
        model_config=model_config,
        lambda_penalty=lambda_penalty,
        top_k=top_k,
        dataset=args.dataset,
    )

    robust_heads = rhi.run_identification(prompt_data_paths)

    if robust_heads:
        # Save results
        rhi.save_results(output_dir)

        # Visualize
        if rhi.robustness_scores is not None:
            plot_robustness_heatmap(
                rhi.robustness_scores,
                selected_heads=robust_heads,
                lambda_penalty=lambda_penalty,
                save_path=output_dir / "robustness_heatmap.png",
            )

        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Selected heads: {robust_heads}")
    else:
        logger.error("RHI failed to identify heads")


if __name__ == "__main__":
    main()
