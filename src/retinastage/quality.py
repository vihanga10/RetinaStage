"""Reusable technical image-quality checks for RetinaStage inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np


ANALYSIS_SIZE = 512
REQUIRED_LIMITS = {
    "brightness_low",
    "brightness_high",
    "contrast_low",
    "sharpness_low",
    "coverage_low",
}


@dataclass(frozen=True)
class QualityAssessment:
    """Technical measurements and dataset-derived review flags."""

    metrics: dict[str, float]
    flags: tuple[str, ...]

    @property
    def requires_review(self) -> bool:
        return bool(self.flags)

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics,
            "flags": list(self.flags),
            "requires_review": self.requires_review,
            "interpretation": (
                "Flags identify unusual technical characteristics relative "
                "to the project dataset; they are not clinical gradability "
                "decisions."
            ),
        }


def resize_with_padding(
    image: np.ndarray,
    size: int = ANALYSIS_SIZE,
) -> np.ndarray:
    """Resize without distortion and centre the image on a black square."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty")
    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x_start = (size - resized_width) // 2
    y_start = (size - resized_height) // 2
    canvas[
        y_start:y_start + resized_height,
        x_start:x_start + resized_width,
    ] = resized
    return canvas


def create_retinal_mask(gray: np.ndarray) -> np.ndarray:
    """Estimate the visible retinal field against the dark background."""

    initial_mask = np.where(gray > 10, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned_mask = cv2.morphologyEx(
        initial_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )
    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return np.zeros_like(gray)
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


def measure_quality(image: np.ndarray) -> dict[str, float]:
    """Calculate the same technical measurements used by the data audit."""

    standardised = resize_with_padding(image)
    gray = cv2.cvtColor(standardised, cv2.COLOR_BGR2GRAY)
    retinal_mask = create_retinal_mask(gray)
    retinal_pixels = gray[retinal_mask > 0]
    if retinal_pixels.size == 0:
        raise ValueError("No retinal field could be estimated")

    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11),
    )
    inner_mask = cv2.erode(retinal_mask, erosion_kernel, iterations=1)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness_pixels = laplacian[inner_mask > 0]
    if sharpness_pixels.size == 0:
        sharpness_pixels = laplacian[retinal_mask > 0]

    return {
        "brightness_mean": float(np.mean(retinal_pixels)),
        "contrast_std": float(np.std(retinal_pixels)),
        "sharpness_laplacian_variance": float(
            np.var(sharpness_pixels)
        ),
        "dark_pixel_fraction": float(np.mean(retinal_pixels < 30)),
        "bright_pixel_fraction": float(np.mean(retinal_pixels > 225)),
        "retinal_field_coverage": float(np.mean(retinal_mask > 0)),
    }


def load_review_limits(summary_path: Path) -> dict[str, float]:
    """Load and validate data-derived review limits from the audit summary."""

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    limits = summary.get("review_limits")
    if not isinstance(limits, dict):
        raise ValueError("Quality summary does not contain review_limits")
    missing = REQUIRED_LIMITS - set(limits)
    if missing:
        raise ValueError(
            f"Quality review limits are missing: {sorted(missing)}"
        )
    return {name: float(limits[name]) for name in REQUIRED_LIMITS}


def assess_quality(
    image: np.ndarray,
    review_limits: Mapping[str, float],
) -> QualityAssessment:
    """Measure an image and assign non-clinical technical review flags."""

    missing = REQUIRED_LIMITS - set(review_limits)
    if missing:
        raise ValueError(f"Review limits are missing: {sorted(missing)}")
    metrics = measure_quality(image)
    flags: list[str] = []
    if metrics["brightness_mean"] <= review_limits["brightness_low"]:
        flags.append("LOW_BRIGHTNESS")
    if metrics["brightness_mean"] >= review_limits["brightness_high"]:
        flags.append("HIGH_BRIGHTNESS")
    if metrics["contrast_std"] <= review_limits["contrast_low"]:
        flags.append("LOW_CONTRAST")
    if (
        metrics["sharpness_laplacian_variance"]
        <= review_limits["sharpness_low"]
    ):
        flags.append("LOW_SHARPNESS")
    if metrics["retinal_field_coverage"] <= review_limits["coverage_low"]:
        flags.append("LOW_RETINAL_COVERAGE")
    return QualityAssessment(metrics=metrics, flags=tuple(flags))
