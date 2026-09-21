"""Create preprocessing comparison figures for every DR grade.

Examples come only from the validation partition. Each figure compares the
original image, retinal-field cropping, CLAHE and mild unsharp masking.
"""

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from retinastage.preprocessing import (
    PreprocessingConfig,
    preprocess_image,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "split_manifest.csv"
QUALITY_PATH = (
    PROJECT_ROOT / "results" / "quality" / "image_quality_metrics.csv"
)
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "train_images"
OUTPUT_DIR = PROJECT_ROOT / "results" / "preprocessing" / "comparisons"

GRADE_NAMES = {
    "0": "No DR",
    "1": "Mild",
    "2": "Moderate",
    "3": "Severe",
    "4": "Proliferative DR",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def select_examples(
    records: list[dict[str, str]],
) -> list[tuple[str, dict[str, str]]]:
    """Select typical, dark and soft examples from one disease grade."""
    brightness = np.array(
        [float(row["brightness_mean"]) for row in records]
    )
    contrast = np.array(
        [float(row["contrast_std"]) for row in records]
    )
    sharpness = np.array(
        [float(row["sharpness_laplacian_variance"]) for row in records]
    )

    def robust_distance(values: np.ndarray) -> np.ndarray:
        """Return distance from the median scaled by interquartile range."""
        median = np.median(values)
        lower, upper = np.percentile(values, [25, 75])
        scale = max(upper - lower, 1e-6)
        return np.abs(values - median) / scale

    typical_scores = (
        robust_distance(brightness)
        + robust_distance(contrast)
        + robust_distance(sharpness)
    )

    typical = records[int(np.argmin(typical_scores))]
    darkest = min(records, key=lambda row: float(row["brightness_mean"]))
    softest = min(
        records,
        key=lambda row: float(row["sharpness_laplacian_variance"]),
    )

    selected = []
    used_ids = set()

    for label, row in [
        ("Typical quality", typical),
        ("Lowest brightness", darkest),
        ("Lowest sharpness", softest),
    ]:
        if row["image_id"] not in used_ids:
            selected.append((label, row))
            used_ids.add(row["image_id"])

    # Add another deterministic record if two categories selected the same image.
    for row in sorted(records, key=lambda item: item["image_id"]):
        if len(selected) == 3:
            break
        if row["image_id"] not in used_ids:
            selected.append(("Additional example", row))
            used_ids.add(row["image_id"])

    return selected


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image for Matplotlib display."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> None:
    """Generate one three-row comparison figure for each disease grade."""
    split_rows = read_csv(SPLIT_PATH)
    quality_rows = read_csv(QUALITY_PATH)

    validation_ids = {
        row["image_id"]
        for row in split_rows
        if row["split"] == "validation"
    }

    quality_by_grade: defaultdict[str, list[dict[str, str]]] = defaultdict(list)

    for row in quality_rows:
        if row["image_id"] in validation_ids:
            quality_by_grade[row["diagnosis"]].append(row)

    crop_config = PreprocessingConfig(
        output_size=224,
        crop_retinal_field=True,
        use_clahe=False,
        use_unsharp_mask=False,
    )
    clahe_config = PreprocessingConfig(
        output_size=224,
        crop_retinal_field=True,
        use_clahe=True,
        use_unsharp_mask=False,
    )
    sharpened_config = PreprocessingConfig(
        output_size=224,
        crop_retinal_field=True,
        use_clahe=True,
        use_unsharp_mask=True,
        unsharp_strength=0.5,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for diagnosis in sorted(GRADE_NAMES):
        selected = select_examples(quality_by_grade[diagnosis])
        figure, axes = plt.subplots(3, 4, figsize=(14, 11))

        for row_index, (selection_name, record) in enumerate(selected):
            image_path = IMAGE_DIR / record["image_filename"]
            original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

            if original is None:
                raise RuntimeError(f"Could not read {image_path}")

            cropped = preprocess_image(original, crop_config)
            clahe = preprocess_image(original, clahe_config)
            sharpened = preprocess_image(original, sharpened_config)

            displayed_images = [
                original,
                cropped,
                clahe,
                sharpened,
            ]
            column_titles = [
                "Original",
                "Crop, pad and resize",
                "CLAHE",
                "CLAHE and unsharp",
            ]

            for column_index, (displayed, title) in enumerate(
                zip(displayed_images, column_titles)
            ):
                axis = axes[row_index, column_index]
                axis.imshow(to_rgb(displayed))
                axis.axis("off")

                if row_index == 0:
                    axis.set_title(title, fontsize=11)

            axes[row_index, 0].set_ylabel(
                (
                    f"{selection_name}\n"
                    f"{record['image_id']}\n"
                    f"Brightness {float(record['brightness_mean']):.1f}\n"
                    f"Sharpness "
                    f"{float(record['sharpness_laplacian_variance']):.1f}"
                ),
                fontsize=9,
            )

        figure.suptitle(
            f"Grade {diagnosis}: {GRADE_NAMES[diagnosis]}",
            fontsize=16,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))

        output_path = OUTPUT_DIR / f"grade_{diagnosis}_comparison.png"
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

        print("Created:", output_path)

    print("Preprocessing comparison figures: 5")


if __name__ == "__main__":
    main()
