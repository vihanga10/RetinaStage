"""Reusable retinal-image preprocessing for RetinaStage.

The functions preserve aspect ratio, remove unnecessary dark borders and
provide optional contrast and edge enhancement. The original source images
are never modified.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration shared by training, evaluation and inference."""

    output_size: int = 224
    crop_retinal_field: bool = True
    crop_margin_fraction: float = 0.02
    use_clahe: bool = False
    use_unsharp_mask: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    unsharp_strength: float = 0.5


def estimate_retinal_mask(image: np.ndarray) -> np.ndarray:
    """Estimate the largest visible retinal field against a dark background."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray > 10, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return np.full(gray.shape, 255, dtype=np.uint8)

    largest_contour = max(contours, key=cv2.contourArea)
    retinal_mask = np.zeros_like(gray)
    cv2.drawContours(
        retinal_mask,
        [largest_contour],
        contourIdx=-1,
        color=255,
        thickness=-1,
    )

    return retinal_mask


def crop_to_retinal_field(
    image: np.ndarray,
    margin_fraction: float = 0.02,
) -> np.ndarray:
    """Crop around the estimated retinal field with a small safety margin."""
    mask = estimate_retinal_mask(image)
    points = cv2.findNonZero(mask)

    if points is None:
        return image.copy()

    x, y, width, height = cv2.boundingRect(points)
    margin = round(max(width, height) * margin_fraction)

    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(image.shape[1], x + width + margin)
    y_end = min(image.shape[0], y + height + margin)

    return image[y_start:y_end, x_start:x_end].copy()


def pad_to_square(image: np.ndarray) -> np.ndarray:
    """Place an image centrally on a black square without distortion."""
    height, width = image.shape[:2]
    side = max(height, width)

    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    x_start = (side - width) // 2
    y_start = (side - height) // 2

    canvas[
        y_start:y_start + height,
        x_start:x_start + width,
    ] = image

    return canvas


def resize_image(image: np.ndarray, output_size: int) -> np.ndarray:
    """Resize a square image to the configured model-input dimensions."""
    interpolation = (
        cv2.INTER_AREA
        if image.shape[0] > output_size
        else cv2.INTER_CUBIC
    )

    return cv2.resize(
        image,
        (output_size, output_size),
        interpolation=interpolation,
    )


def apply_clahe(
    image: np.ndarray,
    clip_limit: float,
    grid_size: int,
) -> np.ndarray:
    """Enhance local contrast on the LAB lightness channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size),
    )
    enhanced_lightness = clahe.apply(lightness)

    enhanced_lab = cv2.merge(
        (enhanced_lightness, channel_a, channel_b)
    )

    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def apply_unsharp_mask(
    image: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Apply mild edge enhancement using a Gaussian unsharp mask."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)

    return cv2.addWeighted(
        image,
        1.0 + strength,
        blurred,
        -strength,
        0,
    )


def preprocess_image(
    image: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Apply the configured deterministic preprocessing pipeline."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")

    processed = image.copy()

    if config.crop_retinal_field:
        processed = crop_to_retinal_field(
            processed,
            margin_fraction=config.crop_margin_fraction,
        )

    processed = pad_to_square(processed)
    processed = resize_image(processed, config.output_size)

    if config.use_clahe:
        processed = apply_clahe(
            processed,
            clip_limit=config.clahe_clip_limit,
            grid_size=config.clahe_grid_size,
        )

    if config.use_unsharp_mask:
        processed = apply_unsharp_mask(
            processed,
            strength=config.unsharp_strength,
        )

    return processed


def preprocess_with_stages(
    image: np.ndarray,
    config: PreprocessingConfig,
) -> dict[str, np.ndarray]:
    """Return intermediate stages for report figures and debugging."""
    cropped = (
        crop_to_retinal_field(
            image,
            margin_fraction=config.crop_margin_fraction,
        )
        if config.crop_retinal_field
        else image.copy()
    )

    padded = pad_to_square(cropped)
    resized = resize_image(padded, config.output_size)
    enhanced = resized.copy()

    if config.use_clahe:
        enhanced = apply_clahe(
            enhanced,
            clip_limit=config.clahe_clip_limit,
            grid_size=config.clahe_grid_size,
        )

    if config.use_unsharp_mask:
        enhanced = apply_unsharp_mask(
            enhanced,
            strength=config.unsharp_strength,
        )

    return {
        "original": image.copy(),
        "cropped": cropped,
        "padded_and_resized": resized,
        "enhanced": enhanced,
    }
