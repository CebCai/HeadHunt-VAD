#!/usr/bin/env python
"""
Demo script showing HeadHunt-VAD inference on a single video.

Usage:
    python demo_video.py --video /path/to/video.mp4 --model_path /path/to/internvl3-8b
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Demo: Analyze a single video for anomalies")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to MLLM model")
    parser.add_argument("--scorer_path", type=str, help="Path to trained scorer (optional)")
    parser.add_argument("--model_type", type=str, default="internvl3")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_segments", type=int, default=16)
    args = parser.parse_args()

    print("=" * 60)
    print("HeadHunt-VAD Demo: Single Video Analysis")
    print("=" * 60)

    from headhunt_vad.models.factory import create_extractor

    # Create and load extractor
    print(f"\nLoading {args.model_type} model from {args.model_path}...")
    extractor = create_extractor(
        model_type=args.model_type,
        model_path=args.model_path,
        device=args.device,
    )
    extractor.load_model()
    print("Model loaded successfully!")

    # Define prompt
    prompt = "Identify any abnormal events or safety hazards in this video."

    # Extract features
    print(f"\nProcessing video: {args.video}")
    print(f"Sampling {args.num_segments} frames...")

    features = extractor.extract_features(
        video_path=args.video,
        prompt=prompt,
        num_segments=args.num_segments,
    )

    if features is None:
        print("Error: Failed to extract features")
        return

    print(f"\nFeature extraction successful!")
    print(f"  - oproj_inputs shape: {features['oproj_inputs'].shape}")
    print(f"  - Video label: {features.get('label', 'unknown')}")

    # If scorer is provided, get anomaly prediction
    if args.scorer_path:
        from headhunt_vad.models.anomaly_scorer import AnomalyScorer

        print(f"\nLoading scorer from {args.scorer_path}...")
        scorer = AnomalyScorer(head_indices=[], head_dim=128)
        scorer.load(args.scorer_path)

        # Extract expert head features
        expert_features = extractor.extract_head_features(
            video_path=args.video,
            prompt=prompt,
            head_indices=scorer.head_indices,
            num_segments=args.num_segments,
        )

        if expert_features is not None:
            flat_features = expert_features.view(1, -1).float().numpy()
            proba = scorer.predict_proba(flat_features)[0]

            print("\n" + "=" * 40)
            print("ANOMALY DETECTION RESULT")
            print("=" * 40)
            print(f"Anomaly Probability: {proba:.2%}")
            print(f"Prediction: {'ANOMALY' if proba > 0.5 else 'NORMAL'}")

    else:
        print("\nNote: No scorer provided. Pass --scorer_path to get anomaly predictions.")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
