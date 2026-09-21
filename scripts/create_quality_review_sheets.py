"""Create visual review sheets for unusual retinal-image quality measurements.

The sheets support human inspection of distribution-based quality flags.
They do not assign clinical gradability or automatically exclude images.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "results" / "quality" / "image_quality_metrics.csv"
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "train_images"
OUTPUT_DIR = PROJECT_ROOT / "results" / "quality" / "review_sheets"
IMAGES_PER_SHEET = 20


def load_rows() -> list[dict[str, str]]:
    """Load the saved image-quality measurements."""
    with METRICS_PATH.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_image_index() -> dict[str, Path]:
    """Map each image identifier to its source path."""
    supported_extensions = {".png", ".jpg", ".jpeg"}

    return {
        path.stem: path
        for path in IMAGE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_extensions
    }


def create_sheet(
    rows: list[dict[str, str]],
    image_index: dict[str, Path],
    title: str,
    metric: str,
    output_name: str,
) -> None:
    """Create one labelled contact sheet from selected retinal images."""
    figure, axes = plt.subplots(4, 5, figsize=(15, 12))

    for axis, row in zip(axes.flat, rows):
        image_id = row["image_id"]
        image_path = image_index[image_id]

        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))

        axis.set_title(
            f"{image_id}\n"
            f"Grade {row['diagnosis']} | {float(row[metric]):.3f}",
            fontsize=8,
        )
        axis.axis("off")

    for axis in axes.flat[len(rows):]:
        axis.axis("off")

    figure.suptitle(title, fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(
        OUTPUT_DIR / output_name,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    """Generate review sheets for the extreme values of each metric."""
    rows = load_rows()
    image_index = build_image_index()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    specifications = [
        (
            "Lowest brightness review candidates",
            "brightness_mean",
            False,
            "lowest_brightness.png",
        ),
        (
            "Highest brightness review candidates",
            "brightness_mean",
            True,
            "highest_brightness.png",
        ),
        (
            "Lowest contrast review candidates",
            "contrast_std",
            False,
            "lowest_contrast.png",
        ),
        (
            "Lowest sharpness review candidates",
            "sharpness_laplacian_variance",
            False,
            "lowest_sharpness.png",
        ),
        (
            "Lowest retinal-field coverage review candidates",
            "retinal_field_coverage",
            False,
            "lowest_retinal_coverage.png",
        ),
    ]

    for title, metric, descending, output_name in specifications:
        selected = sorted(
            rows,
            key=lambda row: float(row[metric]),
            reverse=descending,
        )[:IMAGES_PER_SHEET]

        create_sheet(
            selected,
            image_index,
            title,
            metric,
            output_name,
        )

        print("Created:", OUTPUT_DIR / output_name)

    print("Review sheets created:", len(specifications))


if __name__ == "__main__":
    main()
