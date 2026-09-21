"""Create side-by-side review sheets for perceptual-hash candidates.

The figures allow human inspection of possible resized, compressed, colour-
adjusted or otherwise altered copies. No candidate is automatically excluded.
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = (
    PROJECT_ROOT
    / "results"
    / "similarity"
    / "near_duplicate_candidates.csv"
)
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "train_images"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "similarity"
    / "review_sheets"
)

PAIRS_PER_PAGE = 4
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def load_candidates() -> list[dict[str, str]]:
    """Load perceptual-hash candidate pairs."""
    with CANDIDATES_PATH.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_image_index() -> dict[str, Path]:
    """Map image identifiers to their original image paths."""
    return {
        path.stem: path
        for path in IMAGE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }


def show_image(
    axis,
    path: Path,
    image_id: str,
    diagnosis: str,
    side: str,
) -> None:
    """Display one labelled image in a candidate-pair figure."""
    with Image.open(path) as image:
        axis.imshow(image.convert("RGB"))

    axis.set_title(
        f"{side}: {image_id}\nGrade {diagnosis}",
        fontsize=10,
    )
    axis.axis("off")


def main() -> None:
    """Generate paginated side-by-side similarity review sheets."""
    candidates = load_candidates()
    image_index = build_image_index()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    page_count = math.ceil(len(candidates) / PAIRS_PER_PAGE)

    for page_number in range(page_count):
        start = page_number * PAIRS_PER_PAGE
        page_pairs = candidates[start:start + PAIRS_PER_PAGE]

        figure, axes = plt.subplots(
            PAIRS_PER_PAGE,
            2,
            figsize=(12, 4 * PAIRS_PER_PAGE),
        )

        for row_index, candidate in enumerate(page_pairs):
            left_id = candidate["left_image_id"]
            right_id = candidate["right_image_id"]

            show_image(
                axes[row_index, 0],
                image_index[left_id],
                left_id,
                candidate["left_diagnosis"],
                "Left",
            )
            show_image(
                axes[row_index, 1],
                image_index[right_id],
                right_id,
                candidate["right_diagnosis"],
                "Right",
            )

            axes[row_index, 0].text(
                0.5,
                -0.08,
                (
                    f"Pair {start + row_index + 1} | "
                    f"pHash distance "
                    f"{candidate['hamming_distance']} | "
                    f"Same label: {candidate['same_label']}"
                ),
                transform=axes[row_index, 0].transAxes,
                ha="center",
                fontsize=9,
            )

        for unused_index in range(len(page_pairs), PAIRS_PER_PAGE):
            axes[unused_index, 0].axis("off")
            axes[unused_index, 1].axis("off")

        figure.suptitle(
            f"Near-duplicate candidate review — page {page_number + 1}",
            fontsize=16,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.97))

        output_path = OUTPUT_DIR / f"near_duplicate_page_{page_number + 1}.png"
        figure.savefig(output_path, dpi=170, bbox_inches="tight")
        plt.close(figure)

        print("Created:", output_path)

    print("Candidate pairs:", len(candidates))
    print("Review pages:", page_count)


if __name__ == "__main__":
    main()
