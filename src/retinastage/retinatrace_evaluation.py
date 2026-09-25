"""Reproducible integrity and privacy evaluation for RetinaTrace receipts."""

from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from retinastage.retinatrace import (
    build_receipt,
    serialise_receipt,
    verify_receipt,
)


FIXED_ISSUED_AT = "2026-09-25T05:30:00Z"
CSV_FIELDS = ("check_id", "category", "passed", "evidence")


def evaluation_prediction() -> dict[str, object]:
    """Return one controlled, API-shaped prediction without model inference."""

    return {
        "original_filename": "controlled-private-name.png",
        "input_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "predicted_grade": 2,
        "predicted_label": "Moderate",
        "confidence": 0.42,
        "uncertain": True,
        "requires_human_review": True,
        "review_reasons": ["LOW_MODEL_CONFIDENCE"],
        "probabilities": [
            {"grade": 0, "label": "No DR", "probability": 0.10},
            {"grade": 1, "label": "Mild", "probability": 0.18},
            {"grade": 2, "label": "Moderate", "probability": 0.42},
            {"grade": 3, "label": "Severe", "probability": 0.20},
            {
                "grade": 4,
                "label": "Proliferative DR",
                "probability": 0.10,
            },
        ],
        "quality": {
            "metrics": {
                "brightness_mean": 86.0,
                "contrast_std": 17.0,
                "sharpness_laplacian_variance": 25.0,
                "dark_pixel_fraction": 0.11,
                "bright_pixel_fraction": 0.08,
                "retinal_field_coverage": 0.60,
            },
            "flags": [],
            "requires_review": False,
            "interpretation": "Dataset-relative technical checks only.",
        },
        "policy": {
            "temperature": 1.168,
            "confidence_threshold": 0.551,
        },
        "educational_notice": "Educational research prototype only.",
    }


def evaluation_explanation() -> dict[str, object]:
    """Return controlled Grad-CAM metadata plus data that must be excluded."""

    return {
        "input_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "target_grade": 2,
        "target_label": "Moderate",
        "backbone_layer": "efficientnetb0",
        "input_space": "processed_224_pixel_model_input",
        "heatmap_data_url": "data:image/png;base64,excluded-heatmap",
        "overlay_data_url": "data:image/png;base64,excluded-overlay",
        "interpretation": "Attention evidence is not lesion localization.",
    }


def _check(
    check_id: str,
    category: str,
    condition: bool,
    evidence: str,
) -> dict[str, object]:
    return {
        "check_id": check_id,
        "category": category,
        "passed": bool(condition),
        "evidence": evidence,
    }


def evaluate_retinatrace() -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run fixed receipt checks and return row-level plus summary evidence."""

    receipt = build_receipt(
        evaluation_prediction(),
        evaluation_explanation(),
        issued_at_utc=FIXED_ISSUED_AT,
    )
    serialised = serialise_receipt(receipt)
    unchanged_valid = bool(verify_receipt(receipt)["valid"])

    tamper_operations: Sequence[
        tuple[str, Callable[[dict[str, object]], None]]
    ] = (
        (
            "prediction_tamper",
            lambda value: value["prediction"].__setitem__(
                "predicted_grade",
                4,
            ),
        ),
        (
            "probability_tamper",
            lambda value: value["prediction"]["probabilities"][2].__setitem__(
                "probability",
                0.41,
            ),
        ),
        (
            "review_tamper",
            lambda value: value["prediction"].__setitem__(
                "review_reasons",
                [],
            ),
        ),
    )
    tamper_rows = []
    for check_id, operation in tamper_operations:
        changed = deepcopy(receipt)
        operation(changed)
        detected = not bool(verify_receipt(changed)["valid"])
        tamper_rows.append(
            _check(
                check_id,
                "integrity",
                detected,
                "Modified receipt failed checksum verification.",
            )
        )

    probabilities = receipt["prediction"]["probabilities"]
    rows = [
        _check(
            "unchanged_receipt",
            "integrity",
            unchanged_valid,
            "Unmodified receipt passed checksum verification.",
        ),
        *tamper_rows,
        _check(
            "five_probabilities",
            "completeness",
            len(probabilities) == 5,
            "Receipt contains all five ordered grade probabilities.",
        ),
        _check(
            "traceability_hashes",
            "traceability",
            receipt["traceability"]["input_sha256"] == "1" * 64
            and receipt["traceability"]["model_sha256"] == "2" * 64,
            "Input and model SHA-256 values were preserved.",
        ),
        _check(
            "image_bytes_excluded",
            "privacy",
            "data:image" not in serialised
            and "heatmap_data_url" not in serialised
            and "overlay_data_url" not in serialised,
            "Retinal and Grad-CAM image bytes are absent.",
        ),
        _check(
            "filename_excluded",
            "privacy",
            receipt["privacy"]["original_filename_included"] is False
            and "controlled-private-name.png" not in serialised,
            "Supplied original filename is absent and explicitly excluded.",
        ),
        _check(
            "stable_serialisation",
            "repeatability",
            serialised == serialise_receipt(json.loads(serialised)),
            "Repeated JSON serialization is byte-stable.",
        ),
    ]
    passed = sum(bool(row["passed"]) for row in rows)
    tamper_passed = sum(bool(row["passed"]) for row in tamper_rows)
    summary = {
        "analysis": "retinatrace_receipt_evaluation",
        "evaluation_version": 1,
        "total_checks": len(rows),
        "passed_checks": passed,
        "failed_checks": len(rows) - passed,
        "all_checks_passed": passed == len(rows),
        "pass_rate": round(passed / len(rows), 6),
        "tamper_cases": len(tamper_rows),
        "tamper_detection_rate": round(
            tamper_passed / len(tamper_rows),
            6,
        ),
        "privacy_checks_passed": all(
            bool(row["passed"])
            for row in rows
            if row["category"] == "privacy"
        ),
        "retinal_images_used": False,
        "model_inference_used": False,
        "final_test_split_used": False,
        "cryptographic_signature_used": False,
        "failed_check_ids": [
            str(row["check_id"]) for row in rows if not row["passed"]
        ],
        "limitations": (
            "SHA-256 detects receipt modification but does not authenticate "
            "the issuer. These controlled checks do not evaluate clinical "
            "correctness or model performance."
        ),
    }
    return rows, summary


def write_evaluation_artifacts(
    rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    output_directory: Path,
) -> tuple[Path, Path]:
    """Write stable row-level CSV and JSON summary evidence."""

    output_directory.mkdir(parents=True, exist_ok=True)
    results_path = output_directory / "retinatrace_evaluation_results.csv"
    summary_path = output_directory / "retinatrace_evaluation_summary.json"
    with results_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results_path, summary_path
