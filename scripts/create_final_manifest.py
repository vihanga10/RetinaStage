"""Apply confirmed near-duplicate review decisions to the preparation manifest.

Connected near-duplicate pairs are treated as groups. All members of a group
are excluded when their labels conflict. For a same-label group, the
alphabetically first image identifier is retained as the representative.
Original images and labels remain unchanged.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARATION_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "preparation_manifest.csv"
)
REVIEW_PATH = (
    PROJECT_ROOT / "data" / "manifests" / "near_duplicate_review.csv"
)
FINAL_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "final_manifest.csv"
FINAL_SUMMARY = (
    PROJECT_ROOT / "data" / "manifests" / "final_manifest_summary.json"
)

INCLUDE = "INCLUDE"
EXCLUDE_CONFLICT = "EXCLUDE_NEAR_DUPLICATE_CONFLICT"
EXCLUDE_REDUNDANT = "EXCLUDE_NEAR_DUPLICATE_REDUNDANT"


class UnionFind:
    """Build connected groups from overlapping candidate pairs."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        """Return the representative root for one image identifier."""
        self.parent.setdefault(item, item)

        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])

        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        """Connect two identifiers into the same similarity group."""
        left_root = self.find(left)
        right_root = self.find(right)

        if left_root != right_root:
            self.parent[right_root] = left_root


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dictionaries."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> None:
    """Create the final modelling manifest and its reconciliation summary."""
    manifest_rows = read_csv(PREPARATION_MANIFEST)
    review_rows = read_csv(REVIEW_PATH)
    manifest_by_id = {
        row["image_id"]: row
        for row in manifest_rows
    }

    confirmed_pairs = [
        row
        for row in review_rows
        if row["review_decision"] == "CONFIRMED_NEAR_DUPLICATE"
    ]

    groups = UnionFind()

    for pair in confirmed_pairs:
        left_id = pair["left_image_id"]
        right_id = pair["right_image_id"]

        if left_id not in manifest_by_id or right_id not in manifest_by_id:
            raise RuntimeError("Reviewed image identifier is missing")

        if (
            manifest_by_id[left_id]["decision"] != INCLUDE
            or manifest_by_id[right_id]["decision"] != INCLUDE
        ):
            raise RuntimeError(
                "Near-duplicate review must reference eligible images"
            )

        groups.union(left_id, right_id)

    connected_components: defaultdict[str, list[str]] = defaultdict(list)

    for image_id in groups.parent:
        connected_components[groups.find(image_id)].append(image_id)

    component_number = 0

    for members in sorted(
        (sorted(group) for group in connected_components.values()),
        key=lambda group: group[0],
    ):
        component_number += 1
        group_id = f"NEAR{component_number:04d}"
        labels = {
            manifest_by_id[image_id]["diagnosis"]
            for image_id in members
        }

        if len(labels) > 1:
            # Different labels make the shared visual evidence ambiguous.
            for image_id in members:
                row = manifest_by_id[image_id]
                row["decision"] = EXCLUDE_CONFLICT
                row["reason"] = (
                    "Confirmed near-duplicate group has conflicting labels"
                )
                row["representative_id"] = ""
                row["near_duplicate_group_id"] = group_id
        else:
            representative_id = members[0]

            for image_id in members:
                row = manifest_by_id[image_id]
                row["near_duplicate_group_id"] = group_id

                if image_id == representative_id:
                    row["reason"] = (
                        "Representative retained from confirmed "
                        "same-label near-duplicate group"
                    )
                    row["representative_id"] = representative_id
                else:
                    row["decision"] = EXCLUDE_REDUNDANT
                    row["reason"] = (
                        "Redundant member of confirmed same-label "
                        "near-duplicate group"
                    )
                    row["representative_id"] = representative_id

    for row in manifest_rows:
        row.setdefault("near_duplicate_group_id", "")

    decision_counts = Counter(row["decision"] for row in manifest_rows)
    eligible_class_counts = Counter(
        row["diagnosis"]
        for row in manifest_rows
        if row["decision"] == INCLUDE
    )

    fieldnames = list(manifest_rows[0].keys())

    with FINAL_MANIFEST.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "original_records": len(manifest_rows),
        "confirmed_candidate_pairs": len(confirmed_pairs),
        "confirmed_connected_groups": len(connected_components),
        "decision_counts": dict(sorted(decision_counts.items())),
        "eligible_records": decision_counts[INCLUDE],
        "eligible_class_counts": dict(sorted(eligible_class_counts.items())),
    }

    FINAL_SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if len(manifest_rows) != 3662:
        raise RuntimeError("Final manifest does not contain 3,662 records")

    if decision_counts[INCLUDE] != 3487:
        raise RuntimeError(
            f"Expected 3,487 eligible records, found "
            f"{decision_counts[INCLUDE]}"
        )

    print("FINAL MANIFEST")
    print("Original records:", summary["original_records"])
    print(
        "Confirmed connected groups:",
        summary["confirmed_connected_groups"],
    )
    print("Decision counts:", summary["decision_counts"])
    print("Eligible records:", summary["eligible_records"])
    print("Eligible class counts:", summary["eligible_class_counts"])
    print("Manifest:", FINAL_MANIFEST)
    print("Summary:", FINAL_SUMMARY)


if __name__ == "__main__":
    main()
