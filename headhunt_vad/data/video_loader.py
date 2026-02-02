"""Video loading utilities for HeadHunt-VAD."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from headhunt_vad.data.transforms import build_transform, dynamic_preprocess
from headhunt_vad.utils.logging import get_logger

logger = get_logger(__name__)


def get_frame_indices(
    total_frames: int,
    num_segments: int,
    fps: float = 30.0,
    bound: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Calculate frame indices for uniform sampling.

    Args:
        total_frames: Total number of frames in the video.
        num_segments: Number of segments to sample.
        fps: Video frame rate (used when bound is specified in seconds).
        bound: Optional tuple of (start_time, end_time) in seconds.

    Returns:
        Array of frame indices to sample.
    """
    if total_frames <= 0:
        return np.zeros(num_segments, dtype=int)

    if bound is not None:
        start_time, end_time = bound
        start_idx = max(0, int(start_time * fps))
        end_idx = min(int(end_time * fps), total_frames - 1)
    else:
        start_idx = 0
        end_idx = total_frames - 1

    if start_idx >= end_idx:
        return np.full(num_segments, start_idx, dtype=int)

    seg_size = (end_idx - start_idx) / num_segments

    # Sample from the center of each segment
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + idx * seg_size)
        for idx in range(num_segments)
    ])

    return np.clip(frame_indices, 0, total_frames - 1).astype(int)


def load_video(
    video_path: Union[str, Path],
    num_segments: int = 16,
    input_size: int = 448,
    max_num: int = 1,
    bound: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, List[int], float]:
    """
    Load and preprocess a video for feature extraction.

    Args:
        video_path: Path to the video file.
        num_segments: Number of frames to sample.
        input_size: Size of the image tiles.
        max_num: Maximum number of tiles per frame.
        bound: Optional tuple of (start_time, end_time) in seconds.

    Returns:
        Tuple of (pixel_values, num_patches_list, fps).
        - pixel_values: Tensor of shape (total_patches, 3, input_size, input_size)
        - num_patches_list: List of patch counts per frame
        - fps: Video frame rate
    """
    try:
        from decord import VideoReader, cpu
    except ImportError:
        raise ImportError("decord is required for video loading. Install with: pip install decord")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Load video
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    except Exception as e:
        logger.error(f"Error opening video {video_path}: {e}")
        # Return dummy data
        dummy_pixel_values = torch.zeros(
            (num_segments * max_num, 3, input_size, input_size),
            dtype=torch.bfloat16,
        )
        return dummy_pixel_values, [max_num] * num_segments, 30.0

    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps()) or 30.0

    if max_frame < 0:
        logger.warning(f"Video {video_path} has no frames")
        dummy_pixel_values = torch.zeros(
            (num_segments * max_num, 3, input_size, input_size),
            dtype=torch.bfloat16,
        )
        return dummy_pixel_values, [max_num] * num_segments, fps

    # Get frame indices
    frame_indices = get_frame_indices(max_frame + 1, num_segments, fps, bound)

    # Build transform
    transform = build_transform(input_size=input_size)

    # Process frames
    pixel_values_list = []
    num_patches_list = []

    for frame_idx in frame_indices:
        try:
            img_array = vr[frame_idx].asnumpy()
            img = Image.fromarray(img_array).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to read frame {frame_idx}: {e}")
            img = Image.new("RGB", (input_size, input_size), (0, 0, 0))

        # Dynamic preprocessing
        tiles = dynamic_preprocess(
            img,
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_num,
        )

        pixel_values_frame = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(pixel_values_frame.shape[0])
        pixel_values_list.append(pixel_values_frame)

    # Concatenate all frames
    pixel_values = torch.cat(pixel_values_list)

    return pixel_values, num_patches_list, fps


class VideoLoader:
    """
    Video loader with configurable preprocessing.

    This class provides a convenient interface for loading and preprocessing
    videos for feature extraction from MLLMs.
    """

    def __init__(
        self,
        num_segments: int = 16,
        input_size: int = 448,
        max_num: int = 1,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the video loader.

        Args:
            num_segments: Number of frames to sample per video.
            input_size: Size of the image tiles.
            max_num: Maximum number of tiles per frame.
            dtype: Data type for the output tensor.
        """
        self.num_segments = num_segments
        self.input_size = input_size
        self.max_num = max_num
        self.dtype = dtype

    def load(
        self,
        video_path: Union[str, Path],
        bound: Optional[Tuple[float, float]] = None,
    ) -> Tuple[torch.Tensor, List[int], float]:
        """
        Load and preprocess a video.

        Args:
            video_path: Path to the video file.
            bound: Optional tuple of (start_time, end_time) in seconds.

        Returns:
            Tuple of (pixel_values, num_patches_list, fps).
        """
        pixel_values, num_patches_list, fps = load_video(
            video_path,
            num_segments=self.num_segments,
            input_size=self.input_size,
            max_num=self.max_num,
            bound=bound,
        )

        # Convert to specified dtype
        pixel_values = pixel_values.to(self.dtype)

        return pixel_values, num_patches_list, fps

    def load_to_device(
        self,
        video_path: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
        bound: Optional[Tuple[float, float]] = None,
    ) -> Tuple[torch.Tensor, List[int], float]:
        """
        Load a video and move to device.

        Args:
            video_path: Path to the video file.
            device: Target device.
            bound: Optional tuple of (start_time, end_time) in seconds.

        Returns:
            Tuple of (pixel_values, num_patches_list, fps).
        """
        pixel_values, num_patches_list, fps = self.load(video_path, bound)
        pixel_values = pixel_values.to(device)
        return pixel_values, num_patches_list, fps
