"""Create reproducible stratified dataset splits for RetinaStage.

Only records marked INCLUDE in the final manifest are split. Each disease
grade is shuffled independently with a fixed random seed. Validation and test
receive equal per-class counts, calibration receives approximately 5%, and
training receives the remainder.
"""

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "final_manifest.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
SPLIT_MANIFEST = OUTPUT_DIR / "split_manifest.csv"
SPLIT_SUMMARY = OUTPUT_DIR / "split_summary.json"

RANDOM_SEED = 20260921


def load_eligible_records() -> list[dict[str, str]]:
    """Load only records approved for model development."""
    with FINAL_MANIFEST.open(newline="", encoding="utf-8") as file:
        return [
            row
            for row in csv.DictReader(file)
            if row["decision"] == "INCLUDE"
        ]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 fingerprint of a completed split manifest."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def allocate_class(records: list[dict[str, str]], seed: int) -> None:
    """Assign one disease class across the four project partitions."""
    generator = random.Random(seed)
    generator.shuffle(records)

    total = len(records)

    # Matching validation and test counts simplify fair comparison.
    validation_count = round(total * 0.15)
    test_count = validation_count
    calibration_count = round(total * 0.05)
    training_count = (
        total
        - validation_count
        - calibration_count
        - test_count
    )

    boundaries = {
        "train": training_count,
        "validation": training_count + validation_count,
        "calibration": (
            training_count
            + validation_count
            + calibration_count
        ),
    }

    for index, record in enumerate(records):
        if index < boundaries["train"]:
            record["split"] = "train"
        elif index < boundaries["validation"]:
            record["split"] = "validation"
        elif index < boundaries["calibration"]:
            record["split"] = "calibration"
        else:
            record["split"] = "test"


def main() -> None:
    """Generate, validate, save and fingerprint the frozen split manifest."""
    records = load_eligible_records()
    records_by_class: defaultdict[str, list[dict[str, str]]] = defaultdict(list)

    for record in records:
        records_by_class[record["diagnosis"]].append(
            {
                "image_id": record["image_id"],
                "image_filename": record["image_filename"],
                "diagnosis": record["diagnosis"],
            }
        )

    for diagnosis in sorted(records_by_class):
        # A class-specific seed keeps each class reproducible independently.
        allocate_class(
            records_by_class[diagnosis],
            RANDOM_SEED + int(diagnosis),
        )

    split_rows = sorted(
        (
            record
            for class_records in records_by_class.values()
            for record in class_records
        ),
        key=lambda row: row["image_id"],
    )

    if len(split_rows) != 3487:
        raise RuntimeError(
            f"Expected 3,487 eligible records, found {len(split_rows)}"
        )

    if len({row["image_id"] for row in split_rows}) != len(split_rows):
        raise RuntimeError("An image identifier appears more than once")

    overall_counts = Counter(row["split"] for row in split_rows)
    class_split_counts = {
        diagnosis: dict(
            sorted(Counter(row["split"] for row in class_rows).items())
        )
        for diagnosis, class_rows in sorted(records_by_class.items())
    }

    # Every grade must be represented in every split.
    required_splits = {"train", "validation", "calibration", "test"}

    for diagnosis, counts in class_split_counts.items():
        if set(counts) != required_splits:
            raise RuntimeError(
                f"Grade {diagnosis} is missing from one or more splits"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "image_filename",
        "diagnosis",
        "split",
    ]

    with SPLIT_MANIFEST.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_rows)

    fingerprint = sha256_file(SPLIT_MANIFEST)

    summary = {
        "random_seed": RANDOM_SEED,
        "eligible_records": len(split_rows),
        "target_proportions": {
            "train": 0.65,
            "validation": 0.15,
            "calibration": 0.05,
            "test": 0.15,
        },
        "overall_counts": dict(sorted(overall_counts.items())),
        "class_split_counts": class_split_counts,
        "split_manifest_sha256": fingerprint,
        "limitations": [
            (
                "The source labels do not provide explicit patient "
                "identifiers, so patient-level separation cannot be verified."
            )
        ],
    }

    SPLIT_SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("DATASET SPLITS")
    print("Random seed:", RANDOM_SEED)
    print("Eligible records:", summary["eligible_records"])
    print("Overall counts:", summary["overall_counts"])
    print("Per-class counts:")

    for diagnosis, counts in class_split_counts.items():
        print(f"  Grade {diagnosis}: {counts}")

    print("Manifest SHA-256:", fingerprint)
    print("Manifest:", SPLIT_MANIFEST)
    print("Summary:", SPLIT_SUMMARY)


if __name__ == "__main__":
    main()
