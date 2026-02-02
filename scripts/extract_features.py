#!/usr/bin/env python
"""Feature extraction script for HeadHunt-VAD."""

import argparse
import os
from pathlib import Path

from tqdm import tqdm

from headhunt_vad.models.factory import create_extractor, get_available_models
from headhunt_vad.utils.config import load_config, merge_configs, parse_cli_overrides
from headhunt_vad.utils.io import save_pickle, get_video_files, ensure_dir, sanitize_filename
from headhunt_vad.utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract attention head features from videos using MLLMs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--model.type",
        dest="model_type",
        type=str,
        help="Model type (e.g., internvl3, llavaov, qwenvl).",
    )
    parser.add_argument(
        "--model.path",
        dest="model_path",
        type=str,
        help="Path to the pre-trained model.",
    )
    parser.add_argument(
        "--data.video_dir",
        dest="video_dir",
        type=str,
        help="Directory containing videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save extracted features.",
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        help="Number of frames to sample per video.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt text for the model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., cuda:0).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional config overrides (key=value format).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply CLI overrides
    cli_overrides = parse_cli_overrides(args.overrides) if args.overrides else {}
    config = merge_configs(config, cli_overrides)

    # Override with explicit args
    model_type = args.model_type or config.get("model", {}).get("type", "internvl3")
    model_path = args.model_path or config.get("model", {}).get("path")
    video_dir = args.video_dir or config.get("data", {}).get("video_dir")
    output_dir = args.output_dir or config.get("output", {}).get("features_dir", "./features")
    num_segments = args.num_segments or config.get("data", {}).get("num_segments", 16)
    device = args.device or config.get("model", {}).get("device", "cuda:0")
    
    # Get prompt from args or config (default to coarse prompt)
    prompt = args.prompt or config.get("prompts", {}).get("coarse", "Identify any abnormal events in this video.")

    if not model_path:
        print("Error: model path is required. Use --model.path or set it in config.")
        return

    if not video_dir:
        print("Error: video directory is required. Use --data.video_dir or set it in config.")
        return

    # Setup
    logger = setup_logger("extract_features")
    ensure_dir(output_dir)

    logger.info(f"Model type: {model_type}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Video directory: {video_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Num segments: {num_segments}")
    logger.info(f"Device: {device}")

    # Create extractor
    extractor = create_extractor(
        model_type=model_type,
        model_path=model_path,
        device=device,
    )
    extractor.load_model()

    # Find video files
    video_files = get_video_files(video_dir)
    logger.info(f"Found {len(video_files)} video files")

    # Extract features
    for video_path in tqdm(video_files, desc="Extracting features"):
        try:
            features = extractor.extract_features(
                video_path=video_path,
                prompt=prompt,
                num_segments=num_segments,
            )

            if features is not None:
                # Save features
                video_name = sanitize_filename(video_path.stem)
                output_file = Path(output_dir) / f"{video_name}_oproj_input.pkl"
                save_pickle(features, output_file)
            else:
                logger.warning(f"No features extracted for {video_path}")

        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")

    logger.info("Feature extraction complete!")


if __name__ == "__main__":
    main()
