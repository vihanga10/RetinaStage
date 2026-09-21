"""Measure image quality for eligible APTOS retinal photographs.

The audit calculates brightness, contrast, sharpness, exposure and retinal
field coverage using a standardised 512-pixel representation. Data-derived
percentiles identify review candidates; they do not establish clinical
gradability and do not automatically exclude images.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "preparation_manifest.csv"
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "train_images"
OUTPUT_DIR = PROJECT_ROOT / "results" / "quality"
METRICS_CSV = OUTPUT_DIR / "image_quality_metrics.csv"
SUMMARY_JSON = OUTPUT_DIR / "image_quality_summary.json"
PLOT_PATH = OUTPUT_DIR / "quality_distributions.png"

ANALYSIS_SIZE = 512
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def load_eligible_records() -> list[dict[str, str]]:
    """Return records marked INCLUDE in the preparation manifest."""
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as file:
        return [
            row
            for row in csv.DictReader(file)
            if row["decision"] == "INCLUDE"
        ]


def build_image_index() -> dict[str, Path]:
    """Map image identifiers to their original image paths."""
    return {
        path.stem: path
        for path in IMAGE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }


def resize_with_padding(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an image without distortion and pad it to a square canvas."""
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
    """Estimate the visible retinal field and suppress the dark background."""
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

    # Retain only the largest connected bright region as the retinal field.
    largest_contour = max(contours, key=cv2.contourArea)
    retinal_mask = np.zeros_like(gray)
    cv2.drawContours(retinal_mask, [largest_contour], -1, 255, thickness=-1)

    return retinal_mask


def measure_quality(image: np.ndarray) -> dict[str, float]:
    """Calculate standardised technical image-quality measurements."""
    standardised = resize_with_padding(image, ANALYSIS_SIZE)
    gray = cv2.cvtColor(standardised, cv2.COLOR_BGR2GRAY)
    retinal_mask = create_retinal_mask(gray)

    retinal_pixels = gray[retinal_mask > 0]

    if retinal_pixels.size == 0:
        raise ValueError("No retinal field could be estimated")

    # Erode the mask before sharpness calculation to prevent the strong
    # retinal-border edge from artificially increasing the blur score.
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
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
        "dark_pixel_fraction": float(
            np.mean(retinal_pixels < 30)
        ),
        "bright_pixel_fraction": float(
            np.mean(retinal_pixels > 225)
        ),
        "retinal_field_coverage": float(
            np.mean(retinal_mask > 0)
        ),
    }


def percentile_summary(values: list[float]) -> dict[str, float]:
    """Return useful percentiles for one quality measurement."""
    percentiles = [0, 2.5, 5, 25, 50, 75, 95, 97.5, 100]
    calculated = np.percentile(values, percentiles)

    return {
        str(percentile): round(float(value), 6)
        for percentile, value in zip(percentiles, calculated)
    }


def save_distribution_plot(rows: list[dict[str, object]]) -> None:
    """Save four quality distributions as report-ready graphical evidence."""
    plots = [
        ("brightness_mean", "Brightness"),
        ("contrast_std", "Contrast"),
        ("sharpness_laplacian_variance", "Sharpness"),
        ("retinal_field_coverage", "Retinal field coverage"),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    for axis, (field, title) in zip(axes.flat, plots):
        values = [float(row[field]) for row in rows]
        axis.hist(values, bins=40, color="#376D8C", edgecolor="white")
        axis.set_title(title)
        axis.set_xlabel(field.replace("_", " "))
        axis.set_ylabel("Number of images")
        axis.grid(alpha=0.2)

    figure.suptitle("Eligible APTOS image-quality distributions")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Measure every eligible image and save metrics, flags and summaries."""
    records = load_eligible_records()
    image_index = build_image_index()
    measured_rows = []
    failures = []

    for position, record in enumerate(records, start=1):
        image_id = record["image_id"]
        path = image_index.get(image_id)

        if path is None:
            failures.append(
                {"image_id": image_id, "error": "Image file not found"}
            )
            continue

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            failures.append(
                {"image_id": image_id, "error": "OpenCV could not decode image"}
            )
            continue

        try:
            metrics = measure_quality(image)
        except Exception as error:
            failures.append(
                {"image_id": image_id, "error": str(error)}
            )
            continue

        measured_rows.append(
            {
                "image_id": image_id,
                "image_filename": record["image_filename"],
                "diagnosis": int(record["diagnosis"]),
                **metrics,
            }
        )

        if position % 250 == 0 or position == len(records):
            print(f"Measured {position}/{len(records)}", flush=True)

    if failures:
        raise RuntimeError(
            f"Quality measurement failed for {len(failures)} images: "
            f"{failures[:5]}"
        )

    metric_names = [
        "brightness_mean",
        "contrast_std",
        "sharpness_laplacian_variance",
        "dark_pixel_fraction",
        "bright_pixel_fraction",
        "retinal_field_coverage",
    ]

    percentiles = {
        metric: percentile_summary(
            [float(row[metric]) for row in measured_rows]
        )
        for metric in metric_names
    }

    # These flags select unusual images for human review. The limits come
    # from this dataset's distributions and are not clinical thresholds.
    limits = {
        "brightness_low": percentiles["brightness_mean"]["2.5"],
        "brightness_high": percentiles["brightness_mean"]["97.5"],
        "contrast_low": percentiles["contrast_std"]["2.5"],
        "sharpness_low": percentiles[
            "sharpness_laplacian_variance"
        ]["5"],
        "coverage_low": percentiles["retinal_field_coverage"]["2.5"],
    }

    flag_counts = Counter()

    for row in measured_rows:
        flags = []

        if row["brightness_mean"] <= limits["brightness_low"]:
            flags.append("LOW_BRIGHTNESS")
        if row["brightness_mean"] >= limits["brightness_high"]:
            flags.append("HIGH_BRIGHTNESS")
        if row["contrast_std"] <= limits["contrast_low"]:
            flags.append("LOW_CONTRAST")
        if (
            row["sharpness_laplacian_variance"]
            <= limits["sharpness_low"]
        ):
            flags.append("LOW_SHARPNESS")
        if row["retinal_field_coverage"] <= limits["coverage_low"]:
            flags.append("LOW_RETINAL_COVERAGE")

        for flag in flags:
            flag_counts[flag] += 1

        row["review_flags"] = ";".join(flags)
        row["requires_manual_review"] = "YES" if flags else "NO"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "image_filename",
        "diagnosis",
        *metric_names,
        "review_flags",
        "requires_manual_review",
    ]

    with METRICS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(measured_rows)

    summary = {
        "eligible_images": len(records),
        "successfully_measured": len(measured_rows),
        "measurement_failures": failures,
        "analysis_size": ANALYSIS_SIZE,
        "percentiles": percentiles,
        "review_limits": limits,
        "flag_counts": dict(sorted(flag_counts.items())),
        "images_requiring_manual_review": sum(
            row["requires_manual_review"] == "YES"
            for row in measured_rows
        ),
        "interpretation": (
            "Flags identify distribution-based review candidates and do not "
            "establish clinical image gradability."
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    save_distribution_plot(measured_rows)

    print("\nIMAGE QUALITY AUDIT")
    print("Eligible images:", summary["eligible_images"])
    print("Successfully measured:", summary["successfully_measured"])
    print(
        "Images requiring manual review:",
        summary["images_requiring_manual_review"],
    )
    print("Flag counts:", summary["flag_counts"])
    print("Review limits:", summary["review_limits"])
    print("Metrics:", METRICS_CSV)
    print("Summary:", SUMMARY_JSON)
    print("Plot:", PLOT_PATH)


if __name__ == "__main__":
    main()
