"""Find perceptually similar eligible retinal images using pHash.

Exact pixel duplicates were handled earlier. This script crops the estimated
retinal field, calculates a 256-bit perceptual hash and reports close pairs for
human review. Candidate pairs are not automatically excluded.
"""

import csv
import json
from pathlib import Path

import cv2
import imagehash
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "preparation_manifest.csv"
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "train_images"
OUTPUT_DIR = PROJECT_ROOT / "results" / "similarity"
CANDIDATES_CSV = OUTPUT_DIR / "near_duplicate_candidates.csv"
SUMMARY_JSON = OUTPUT_DIR / "near_duplicate_summary.json"

HASH_SIZE = 16
MAX_HAMMING_DISTANCE = 12
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def load_eligible_records() -> list[dict[str, str]]:
    """Load records retained for modelling."""
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as file:
        return [
            row
            for row in csv.DictReader(file)
            if row["decision"] == "INCLUDE"
        ]


def build_image_index() -> dict[str, Path]:
    """Map image identifiers to original image paths."""
    return {
        path.stem: path
        for path in IMAGE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }


def crop_retinal_field(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Crop around the largest visible retinal-field region."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype("uint8") * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(largest)

    return image[y:y + height, x:x + width]


def calculate_phash(path: Path) -> str:
    """Calculate a 256-bit perceptual hash after retinal-field cropping."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"OpenCV could not decode {path.name}")

    cropped = crop_retinal_field(image)
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    return str(imagehash.phash(pil_image, hash_size=HASH_SIZE))


def main() -> None:
    """Calculate hashes and report candidate near-duplicate image pairs."""
    records = load_eligible_records()
    image_index = build_image_index()
    hashed_records = []

    for position, record in enumerate(records, start=1):
        image_id = record["image_id"]
        path = image_index.get(image_id)

        if path is None:
            raise FileNotFoundError(f"No image found for {image_id}")

        hash_text = calculate_phash(path)

        hashed_records.append(
            {
                "image_id": image_id,
                "diagnosis": int(record["diagnosis"]),
                "hash_text": hash_text,
                "hash_integer": int(hash_text, 16),
            }
        )

        if position % 250 == 0 or position == len(records):
            print(f"Hashed {position}/{len(records)}", flush=True)

    candidates = []
    closest_distance = HASH_SIZE * HASH_SIZE

    for left_index, left in enumerate(hashed_records):
        for right in hashed_records[left_index + 1:]:
            distance = (
                left["hash_integer"] ^ right["hash_integer"]
            ).bit_count()

            closest_distance = min(closest_distance, distance)

            if distance <= MAX_HAMMING_DISTANCE:
                candidates.append(
                    {
                        "left_image_id": left["image_id"],
                        "left_diagnosis": left["diagnosis"],
                        "right_image_id": right["image_id"],
                        "right_diagnosis": right["diagnosis"],
                        "hamming_distance": distance,
                        "same_label": (
                            "YES"
                            if left["diagnosis"] == right["diagnosis"]
                            else "NO"
                        ),
                    }
                )

        if (left_index + 1) % 500 == 0:
            print(
                f"Compared {left_index + 1}/{len(hashed_records)} hashes",
                flush=True,
            )

    candidates.sort(
        key=lambda row: (
            row["hamming_distance"],
            row["left_image_id"],
            row["right_image_id"],
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "left_image_id",
        "left_diagnosis",
        "right_image_id",
        "right_diagnosis",
        "hamming_distance",
        "same_label",
    ]

    with CANDIDATES_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    summary = {
        "eligible_images": len(hashed_records),
        "hash_algorithm": "pHash",
        "hash_bits": HASH_SIZE * HASH_SIZE,
        "candidate_distance_threshold": MAX_HAMMING_DISTANCE,
        "closest_observed_distance": closest_distance,
        "candidate_pair_count": len(candidates),
        "same_label_candidate_pairs": sum(
            row["same_label"] == "YES" for row in candidates
        ),
        "different_label_candidate_pairs": sum(
            row["same_label"] == "NO" for row in candidates
        ),
        "interpretation": (
            "Perceptual-hash matches are review candidates and are not proof "
            "that two images show the same retinal photograph."
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nNEAR-DUPLICATE SCREEN")
    print("Eligible images:", summary["eligible_images"])
    print("Hash bits:", summary["hash_bits"])
    print(
        "Candidate threshold:",
        summary["candidate_distance_threshold"],
    )
    print(
        "Closest observed distance:",
        summary["closest_observed_distance"],
    )
    print("Candidate pairs:", summary["candidate_pair_count"])
    print(
        "Same-label candidates:",
        summary["same_label_candidate_pairs"],
    )
    print(
        "Different-label candidates:",
        summary["different_label_candidate_pairs"],
    )
    print("Candidates:", CANDIDATES_CSV)
    print("Summary:", SUMMARY_JSON)


if __name__ == "__main__":
    main()
