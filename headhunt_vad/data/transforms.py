"""Image transformation utilities for HeadHunt-VAD."""

from typing import List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(
    input_size: int = 448,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD,
) -> T.Compose:
    """
    Build the image transformation pipeline.

    Args:
        input_size: Target image size (both width and height).
        mean: Normalization mean values for each channel.
        std: Normalization standard deviation values for each channel.

    Returns:
        Composed transformation pipeline.
    """
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return transform


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    """
    Find the closest aspect ratio from a list of target ratios.

    Args:
        aspect_ratio: Current image aspect ratio.
        target_ratios: List of target (width, height) ratios.
        width: Original image width.
        height: Original image height.
        image_size: Base tile size.

    Returns:
        Best matching (width_tiles, height_tiles) ratio.
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # Prefer larger area if aspect ratios are equal
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    """
    Dynamically preprocess an image into tiles based on aspect ratio.

    This function tiles the image to preserve aspect ratio while keeping
    the total number of tiles within bounds.

    Args:
        image: Input PIL Image.
        min_num: Minimum number of tiles.
        max_num: Maximum number of tiles.
        image_size: Size of each tile (square).
        use_thumbnail: Whether to add a thumbnail of the whole image.

    Returns:
        List of preprocessed image tiles.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate all valid tile configurations
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find best matching aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize image
    resized_img = image.resize((target_width, target_height))

    # Split into tiles
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    # Optionally add thumbnail
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def preprocess_frame(
    frame: Image.Image,
    input_size: int = 448,
    max_num: int = 1,
    use_thumbnail: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Preprocess a single video frame.

    Args:
        frame: Input PIL Image.
        input_size: Target image size.
        max_num: Maximum number of tiles.
        use_thumbnail: Whether to add a thumbnail.

    Returns:
        Tuple of (pixel_values tensor, number of patches).
    """
    transform = build_transform(input_size=input_size)

    # Get image tiles
    tiles = dynamic_preprocess(
        frame,
        image_size=input_size,
        use_thumbnail=use_thumbnail,
        max_num=max_num,
    )

    # Apply transforms
    pixel_values = torch.stack([transform(tile) for tile in tiles])

    return pixel_values, pixel_values.shape[0]
