"""Audit APTOS retinal images for readability, dimensions and duplicates."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CSV_PATH = RAW_DIR / "train.csv"
IMAGE_DIR = RAW_DIR / "train_images"
RESULT_PATH = PROJECT_ROOT / "results" / "audit" / "image_audit.json"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file without loading it all at once."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    """Run the image audit and save its results as JSON."""
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as file:
        labels = {
            row["id_code"]: int(row["diagnosis"])
            for row in csv.DictReader(file)
        }

    image_paths = sorted(
        path
        for path in IMAGE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    exact_file_groups = defaultdict(list)
    rgb_pixel_groups = defaultdict(list)
    dimension_counts = Counter()
    colour_modes = Counter()
    unreadable_images = []

    for position, path in enumerate(image_paths, start=1):
        try:
            exact_file_groups[sha256_file(path)].append(path.stem)

            with Image.open(path) as image:
                image.load()
                dimension_counts[f"{image.width}x{image.height}"] += 1
                colour_modes[image.mode] += 1

                rgb_image = image.convert("RGB")
                rgb_digest = hashlib.sha256()
                rgb_digest.update(
                    f"{rgb_image.width}x{rgb_image.height}:RGB:".encode()
                )
                rgb_digest.update(rgb_image.tobytes())
                rgb_pixel_groups[rgb_digest.hexdigest()].append(path.stem)

        except Exception as error:
            unreadable_images.append(
                {"filename": path.name, "error": str(error)}
            )

        if position % 250 == 0 or position == len(image_paths):
            print(f"Checked {position}/{len(image_paths)}", flush=True)

    exact_duplicates = [
        group for group in exact_file_groups.values() if len(group) > 1
    ]
    pixel_duplicates = [
        group for group in rgb_pixel_groups.values() if len(group) > 1
    ]
    conflicting_labels = [
        {
            "image_ids": group,
            "labels": [labels.get(image_id) for image_id in group],
        }
        for group in pixel_duplicates
        if len({labels.get(image_id) for image_id in group}) > 1
    ]

    result = {
        "csv_records": len(labels),
        "images_examined": len(image_paths),
        "successfully_decoded": len(image_paths) - len(unreadable_images),
        "unreadable_count": len(unreadable_images),
        "unreadable_images": unreadable_images,
        "colour_modes": dict(sorted(colour_modes.items())),
        "distinct_dimensions": len(dimension_counts),
        "dimension_counts": dict(
            sorted(dimension_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "identical_file_group_count": len(exact_duplicates),
        "identical_file_groups": exact_duplicates,
        "identical_rgb_group_count": len(pixel_duplicates),
        "identical_rgb_groups": pixel_duplicates,
        "conflicting_label_group_count": len(conflicting_labels),
        "conflicting_label_groups": conflicting_labels,
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nAUDIT RESULTS")
    print("Images examined:", result["images_examined"])
    print("Successfully decoded:", result["successfully_decoded"])
    print("Unreadable images:", result["unreadable_count"])
    print("Colour modes:", result["colour_modes"])
    print("Distinct dimensions:", result["distinct_dimensions"])
    print("Identical-file groups:", result["identical_file_group_count"])
    print("Identical-RGB-pixel groups:", result["identical_rgb_group_count"])
    print(
        "Duplicate groups with conflicting labels:",
        result["conflicting_label_group_count"],
    )
    print("Detailed results:", RESULT_PATH)


if __name__ == "__main__":
    main()
