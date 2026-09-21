"""Create an auditable modelling manifest from the original APTOS dataset.

The script preserves every original record and assigns one of three decisions:

- INCLUDE
- EXCLUDE_REDUNDANT_DUPLICATE
- EXCLUDE_CONFLICTING_LABEL

No source image or original label is modified.
"""

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_CSV = RAW_DIR / "train.csv"
IMAGE_DIR = RAW_DIR / "train_images"
AUDIT_JSON = PROJECT_ROOT / "results" / "audit" / "image_audit.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "manifests"
MANIFEST_CSV = OUTPUT_DIR / "preparation_manifest.csv"
SUMMARY_JSON = OUTPUT_DIR / "preparation_summary.json"

INCLUDE = "INCLUDE"
EXCLUDE_REDUNDANT = "EXCLUDE_REDUNDANT_DUPLICATE"
EXCLUDE_CONFLICT = "EXCLUDE_CONFLICTING_LABEL"


def read_labels() -> dict[str, int]:
    """Return the original diagnosis label for every image identifier."""
    with TRAIN_CSV.open(newline="", encoding="utf-8-sig") as file:
        rows = csv.DictReader(file)
        return {
            row["id_code"]: int(row["diagnosis"])
            for row in rows
        }


def find_image_filenames() -> dict[str, str]:
    """Map each image identifier to its source filename."""
    supported_extensions = {".png", ".jpg", ".jpeg"}

    return {
        path.stem: path.name
        for path in IMAGE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_extensions
    }


def main() -> None:
    """Apply the duplicate policy and save the complete decision manifest."""
    labels = read_labels()
    filenames = find_image_filenames()
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))

    duplicate_groups = audit["identical_rgb_groups"]

    # Sort groups and their members so repeated runs produce identical output.
    duplicate_groups = sorted(
        [sorted(group) for group in duplicate_groups],
        key=lambda group: group[0],
    )

    decisions: dict[str, dict[str, str]] = {}

    for group_number, group in enumerate(duplicate_groups, start=1):
        group_id = f"DUP{group_number:04d}"
        group_labels = {labels[image_id] for image_id in group}

        if len(group_labels) > 1:
            # The same pixels have different labels, so none can provide an
            # unambiguous supervised-learning target.
            for image_id in group:
                decisions[image_id] = {
                    "duplicate_group_id": group_id,
                    "decision": EXCLUDE_CONFLICT,
                    "reason": "Identical RGB pixels occur with conflicting labels",
                    "representative_id": "",
                }
        else:
            # Retain the alphabetically first identifier. This deterministic
            # rule prevents duplicate pixels from receiving extra influence.
            representative_id = group[0]

            decisions[representative_id] = {
                "duplicate_group_id": group_id,
                "decision": INCLUDE,
                "reason": "Representative retained from same-label duplicate group",
                "representative_id": representative_id,
            }

            for image_id in group[1:]:
                decisions[image_id] = {
                    "duplicate_group_id": group_id,
                    "decision": EXCLUDE_REDUNDANT,
                    "reason": "Redundant copy of same-label representative",
                    "representative_id": representative_id,
                }

    manifest_rows = []

    for image_id in sorted(labels):
        decision = decisions.get(
            image_id,
            {
                "duplicate_group_id": "",
                "decision": INCLUDE,
                "reason": "Unique image",
                "representative_id": image_id,
            },
        )

        manifest_rows.append(
            {
                "image_id": image_id,
                "image_filename": filenames.get(image_id, ""),
                "diagnosis": labels[image_id],
                "duplicate_group_id": decision["duplicate_group_id"],
                "decision": decision["decision"],
                "reason": decision["reason"],
                "representative_id": decision["representative_id"],
            }
        )

    decision_counts = Counter(row["decision"] for row in manifest_rows)
    included_class_counts = Counter(
        str(row["diagnosis"])
        for row in manifest_rows
        if row["decision"] == INCLUDE
    )

    missing_filenames = [
        row["image_id"]
        for row in manifest_rows
        if not row["image_filename"]
    ]

    # Stop instead of producing an incomplete manifest.
    if missing_filenames:
        raise RuntimeError(
            f"{len(missing_filenames)} records have no matching image file"
        )

    if len(manifest_rows) != len(labels):
        raise RuntimeError("Manifest row count does not match train.csv")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "image_filename",
        "diagnosis",
        "duplicate_group_id",
        "decision",
        "reason",
        "representative_id",
    ]

    with MANIFEST_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "original_records": len(manifest_rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "eligible_records": decision_counts[INCLUDE],
        "eligible_class_counts": dict(sorted(included_class_counts.items())),
        "policy": {
            "conflicting_label_duplicates": "Exclude every member",
            "same_label_duplicates": (
                "Retain the alphabetically first image identifier"
            ),
            "unique_images": "Include",
        },
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("PREPARATION MANIFEST")
    print("Original records:", summary["original_records"])
    print("Decision counts:", summary["decision_counts"])
    print("Eligible records:", summary["eligible_records"])
    print("Eligible class counts:", summary["eligible_class_counts"])
    print("Manifest:", MANIFEST_CSV)
    print("Summary:", SUMMARY_JSON)


if __name__ == "__main__":
    main()
