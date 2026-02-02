#!/usr/bin/env python
"""Evaluation script for HeadHunt-VAD."""

import argparse
from pathlib import Path

from headhunt_vad.evaluation.frame_level import FrameLevelEvaluator
from headhunt_vad.evaluation.video_level import VideoLevelEvaluator
from headhunt_vad.core.temporal_locator import TemporalLocator
from headhunt_vad.utils.config import load_config, merge_configs, parse_cli_overrides
from headhunt_vad.utils.io import save_json
from headhunt_vad.utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate anomaly detection results."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory with result JSON files.")
    parser.add_argument("--ground_truth", type=str, required=True, help="Ground truth annotation file.")
    parser.add_argument("--output_dir", type=str, default="./results/evaluation")
    parser.add_argument("--fps", type=float, help="Frame rate for evaluation.")
    parser.add_argument("--mode", type=str, default="frame", choices=["frame", "video", "both"])
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    cli_overrides = parse_cli_overrides(args.overrides) if args.overrides else {}
    config = merge_configs(config, cli_overrides)

    # Setup
    logger = setup_logger("evaluate")

    fps = args.fps or config.get("evaluation", {}).get("fps", 30)

    # Create temporal locator
    locator_config = config.get("locator", {})
    locator = TemporalLocator(
        gaussian_sigma=locator_config.get("gaussian_sigma", 1.5),
        threshold=locator_config.get("threshold", 0.65),
    )

    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Ground truth: {args.ground_truth}")
    logger.info(f"FPS: {fps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Frame-level evaluation
    if args.mode in ["frame", "both"]:
        logger.info("\n--- Frame-Level Evaluation ---")
        frame_evaluator = FrameLevelEvaluator(
            fps=fps,
            temporal_locator=locator,
        )

        frame_metrics = frame_evaluator.evaluate(
            results_dir=args.results_dir,
            ground_truth_file=args.ground_truth,
        )

        # Print results
        print("\n=== Frame-Level Results ===")
        print(f"ROC-AUC: {frame_metrics.get('roc_auc', 0) * 100:.2f}%")
        print(f"Average Precision: {frame_metrics.get('average_precision', 0) * 100:.2f}%")
        print(f"Optimal F1: {frame_metrics.get('max_f1', 0) * 100:.2f}%")
        print(f"Num Videos: {frame_metrics.get('num_videos', 0)}")

        # Save metrics
        save_json(frame_metrics, output_dir / "frame_level_metrics.json")

        # Per-video evaluation
        per_video = frame_evaluator.evaluate_per_video(
            results_dir=args.results_dir,
            ground_truth_file=args.ground_truth,
        )
        save_json(per_video, output_dir / "per_video_metrics.json")

    # Video-level evaluation
    if args.mode in ["video", "both"]:
        logger.info("\n--- Video-Level Evaluation ---")

        from headhunt_vad.evaluation.frame_level import parse_ground_truth
        gt = parse_ground_truth(args.ground_truth)
        gt_labels = {name: (1 if events else 0) for name, events in gt.items()}

        video_evaluator = VideoLevelEvaluator(aggregation="max")

        video_metrics = video_evaluator.evaluate_from_results(
            results_dir=args.results_dir,
            ground_truth=gt_labels,
        )

        # Print results
        print("\n=== Video-Level Results ===")
        print(f"ROC-AUC: {video_metrics.get('roc_auc', 0) * 100:.2f}%")
        print(f"Accuracy: {video_metrics.get('accuracy', 0) * 100:.2f}%")
        print(f"F1-Score: {video_metrics.get('f1_score', 0) * 100:.2f}%")

        # Save metrics
        save_json(video_metrics, output_dir / "video_level_metrics.json")

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
