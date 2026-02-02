#!/usr/bin/env python
"""Inference script for HeadHunt-VAD.

This script implements the complete online inference pipeline as described in the paper:
1. Split video into segments (default: 48 frames per segment)
2. Sample F frames per segment (default: F=16)
3. Extract features from expert heads via single forward pass
4. Score with anomaly scorer: p = σ(w^T z + b)
5. Apply Gaussian smoothing and threshold: ŷ_t = [p'_t > τ*]
6. Localize anomaly events with temporal boundaries
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from headhunt_vad.models.factory import create_extractor
from headhunt_vad.models.anomaly_scorer import AnomalyScorer
from headhunt_vad.core.temporal_locator import TemporalLocator
from headhunt_vad.utils.config import load_config, merge_configs, parse_cli_overrides
from headhunt_vad.utils.io import get_video_files, save_json, ensure_dir
from headhunt_vad.utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run anomaly detection inference on videos."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_path", type=str, required=True, help="Path to MLLM model.")
    parser.add_argument("--scorer_path", type=str, required=True, help="Path to trained scorer.")
    parser.add_argument("--video_dir", type=str, help="Directory with videos.")
    parser.add_argument("--video", type=str, help="Single video path.")
    parser.add_argument("--output_dir", type=str, default="./results/inference")
    parser.add_argument("--model_type", type=str, default="internvl3")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frames_per_segment", type=int, default=16, help="Frames to sample per segment (F).")
    parser.add_argument("--segment_interval", type=int, default=48, help="Frame interval between segments.")
    parser.add_argument("--gaussian_sigma", type=float, help="Gaussian smoothing sigma (σ_g).")
    parser.add_argument("--threshold", type=float, help="Detection threshold (τ).")
    parser.add_argument("--prompt", type=str, help="Custom prompt for model.")
    parser.add_argument("--save_segment_scores", action="store_true", help="Save per-segment scores.")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def get_video_info(video_path: Path) -> Tuple[int, float]:
    """Get video frame count and FPS."""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        fps = float(vr.get_avg_fps()) or 30.0
        return total_frames, fps
    except Exception:
        return 0, 30.0


def process_video_segments(
    video_path: Path,
    extractor,
    scorer: AnomalyScorer,
    prompt: str,
    segment_interval: int = 48,
    frames_per_segment: int = 16,
    logger=None,
) -> Tuple[List[float], float, int]:
    """
    Process video in segments and return anomaly scores.

    As described in the paper (Section 3.4):
    "An incoming video is first divided into a sequence of non-overlapping
    temporal segments {S_1, S_2, ..., S_T}. For each segment S_t, we uniformly
    sample F frames."

    Args:
        video_path: Path to the video file.
        extractor: Feature extractor.
        scorer: Trained anomaly scorer.
        prompt: Text prompt for the model.
        segment_interval: Frames per segment (default 48).
        frames_per_segment: Frames to sample per segment (F=16).
        logger: Optional logger.

    Returns:
        Tuple of (segment_scores, fps, total_frames).
    """
    total_frames, fps = get_video_info(video_path)

    if total_frames == 0:
        if logger:
            logger.warning(f"Could not read video: {video_path}")
        return [], fps, 0

    # Calculate number of segments
    num_segments = max(1, total_frames // segment_interval)

    segment_scores = []

    for seg_idx in range(num_segments):
        # Calculate segment boundaries
        start_frame = seg_idx * segment_interval
        end_frame = min((seg_idx + 1) * segment_interval, total_frames)

        # Calculate time bounds for this segment
        start_time = start_frame / fps
        end_time = end_frame / fps

        try:
            # Extract features for this segment
            expert_features = extractor.extract_head_features(
                video_path=video_path,
                prompt=prompt,
                head_indices=scorer.head_indices,
                num_segments=frames_per_segment,
                bound=(start_time, end_time),
            )

            if expert_features is None:
                # Use default score for failed segments
                segment_scores.append(0.5)
                continue

            # Prepare features for scorer: (1, K * head_dim)
            flat_features = expert_features.view(1, -1).float().numpy()

            # Get anomaly probability: p_t = σ(w^T f_t + b)
            proba = float(scorer.predict_proba(flat_features)[0])
            segment_scores.append(proba)

        except Exception as e:
            if logger:
                logger.warning(f"Error processing segment {seg_idx} of {video_path}: {e}")
            segment_scores.append(0.5)

    return segment_scores, fps, total_frames


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    cli_overrides = parse_cli_overrides(args.overrides) if args.overrides else {}
    config = merge_configs(config, cli_overrides)

    # Setup
    logger = setup_logger("inference")
    ensure_dir(args.output_dir)

    # Load scorer
    scorer = AnomalyScorer(head_indices=[], head_dim=128)
    scorer.load(args.scorer_path)
    logger.info(f"Loaded scorer from {args.scorer_path}")
    logger.info(f"Expert heads: {scorer.head_indices}")

    # Create extractor
    extractor = create_extractor(
        model_type=args.model_type,
        model_path=args.model_path,
        device=args.device,
    )
    extractor.load_model()

    # Get parameters
    data_config = config.get("data", {})
    locator_config = config.get("locator", {})

    segment_interval = args.segment_interval or data_config.get("segment_interval", 48)
    frames_per_segment = args.frames_per_segment or data_config.get("frames_per_segment", 16)
    gaussian_sigma = args.gaussian_sigma or locator_config.get("gaussian_sigma", 1.5)
    threshold = args.threshold or locator_config.get("threshold", 0.65)

    # Get prompt
    prompt = args.prompt or config.get("prompts", {}).get(
        "coarse",
        "Identify any abnormal events in this video.",
    )

    # Create temporal locator with calibrated parameters
    locator = TemporalLocator(
        gaussian_sigma=gaussian_sigma,
        threshold=threshold,
    )

    logger.info(f"Segment interval: {segment_interval} frames")
    logger.info(f"Frames per segment: {frames_per_segment}")
    logger.info(f"Gaussian sigma (σ_g): {gaussian_sigma}")
    logger.info(f"Threshold (τ): {threshold}")

    # Find videos
    if args.video:
        video_files = [Path(args.video)]
    elif args.video_dir:
        video_files = get_video_files(args.video_dir)
    else:
        logger.error("Either --video or --video_dir is required")
        return

    logger.info(f"Processing {len(video_files)} videos")

    # Process videos
    all_results = []

    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            # Process video in segments
            segment_scores, fps, total_frames = process_video_segments(
                video_path=video_path,
                extractor=extractor,
                scorer=scorer,
                prompt=prompt,
                segment_interval=segment_interval,
                frames_per_segment=frames_per_segment,
                logger=logger,
            )

            if not segment_scores:
                logger.warning(f"Skipping {video_path}: no segments processed")
                continue

            segment_scores_array = np.array(segment_scores)

            # Apply Gaussian smoothing: p'_t = (p * G_{σ_g})_t
            smoothed_scores = locator.smooth(segment_scores_array)

            # Binarize and localize events: ŷ_t = [p'_t > τ*]
            # Calculate segment duration for time conversion
            segment_duration = segment_interval / fps
            events = locator.localize(
                segment_scores_array,
                fps=1.0 / segment_duration,  # segments per second
                apply_smoothing=True,
            )

            # Calculate video-level anomaly probability
            video_anomaly_prob = float(np.max(smoothed_scores))
            is_anomaly = video_anomaly_prob > threshold

            # Build detailed result
            result = {
                "video_name": video_path.name,
                "video_path": str(video_path),
                "total_frames": total_frames,
                "fps": fps,
                "num_segments": len(segment_scores),
                "segment_interval": segment_interval,
                "anomaly_probability": video_anomaly_prob,
                "is_anomaly": is_anomaly,
                "threshold": threshold,
                "detected_events": [
                    {
                        "start_time": float(start),
                        "end_time": float(end),
                        "duration": float(end - start),
                    }
                    for start, end in events
                ],
                "segments": [
                    {
                        "segment_idx": i,
                        "start_frame": i * segment_interval,
                        "end_frame": min((i + 1) * segment_interval, total_frames),
                        "start_time": (i * segment_interval) / fps,
                        "end_time": min((i + 1) * segment_interval, total_frames) / fps,
                        "raw_score": float(segment_scores[i]),
                        "smoothed_score": float(smoothed_scores[i]),
                        "is_anomaly": float(smoothed_scores[i]) > threshold,
                    }
                    for i in range(len(segment_scores))
                ],
            }

            # Optionally remove detailed segment info for smaller output
            if not args.save_segment_scores:
                result.pop("segments")

            all_results.append(result)

            # Save individual result
            result_path = Path(args.output_dir) / f"{video_path.stem}.json"
            save_json(result, result_path)

        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary results
    summary = {
        "total_videos": len(all_results),
        "anomaly_videos": sum(1 for r in all_results if r.get("is_anomaly", False)),
        "normal_videos": sum(1 for r in all_results if not r.get("is_anomaly", False)),
        "config": {
            "segment_interval": segment_interval,
            "frames_per_segment": frames_per_segment,
            "gaussian_sigma": gaussian_sigma,
            "threshold": threshold,
            "expert_heads": scorer.head_indices,
        },
        "results": all_results,
    }
    save_json(summary, Path(args.output_dir) / "all_results.json")

    logger.info(f"Processed {len(all_results)} videos. Results saved to {args.output_dir}")
    logger.info(f"Detected {summary['anomaly_videos']} anomaly videos, {summary['normal_videos']} normal videos")


if __name__ == "__main__":
    main()