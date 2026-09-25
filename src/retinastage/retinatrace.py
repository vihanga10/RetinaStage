"""Canonical, privacy-preserving evidence receipts for RetinaStage results."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import re
from typing import Mapping, Sequence


RECEIPT_TYPE = "retinastage_prediction_evidence"
RECEIPT_SCHEMA_VERSION = 1
CHECKSUM_ALGORITHM = "SHA-256"
CANONICALIZATION = "UTF-8 JSON with sorted keys and compact separators"
INTEGRITY_NOTICE = (
    "The checksum detects changes to this receipt. It is not a digital "
    "signature and does not prove who created the receipt."
)
GRADE_LABELS = (
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
)
QUALITY_METRIC_NAMES = (
    "brightness_mean",
    "contrast_std",
    "sharpness_laplacian_variance",
    "dark_pixel_fraction",
    "bright_pixel_fraction",
    "retinal_field_coverage",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _as_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _as_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _as_integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _as_number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return number


def _as_sha256(value: object, name: str) -> str:
    digest = _as_text(value, name).lower()
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    return digest


def _as_string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an array")
    result = []
    for index, item in enumerate(value):
        result.append(_as_text(item, f"{name}[{index}]"))
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _normalise_timestamp(value: str | None) -> str:
    if value is None:
        moment = datetime.now(timezone.utc)
    else:
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("issued_at_utc must be an ISO-8601 timestamp") from error
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("issued_at_utc must include a timezone")
    return (
        moment.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _normalise_probabilities(
    result: Mapping[str, object],
    predicted_grade: int,
    confidence: float,
) -> list[dict[str, object]]:
    supplied = result.get("probabilities")
    if not isinstance(supplied, Sequence) or isinstance(supplied, (str, bytes)):
        raise TypeError("probabilities must be an array")
    if len(supplied) != len(GRADE_LABELS):
        raise ValueError("probabilities must contain exactly five grades")

    probabilities: list[dict[str, object]] = []
    for expected_grade, item in enumerate(supplied):
        row = _as_mapping(item, f"probabilities[{expected_grade}]")
        grade = _as_integer(
            row.get("grade"),
            f"probabilities[{expected_grade}].grade",
            0,
            4,
        )
        if grade != expected_grade:
            raise ValueError("probabilities must be ordered from grade 0 to 4")
        label = _as_text(
            row.get("label"),
            f"probabilities[{expected_grade}].label",
        )
        if label != GRADE_LABELS[grade]:
            raise ValueError(f"probabilities[{grade}].label is inconsistent")
        probability = _as_number(
            row.get("probability"),
            f"probabilities[{grade}].probability",
            minimum=0.0,
            maximum=1.0,
        )
        probabilities.append(
            {"grade": grade, "label": label, "probability": probability}
        )

    total = sum(float(row["probability"]) for row in probabilities)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("probabilities must sum to 1")
    selected_probability = float(probabilities[predicted_grade]["probability"])
    if not math.isclose(
        confidence,
        selected_probability,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("confidence must match the selected probability")
    return probabilities


def _normalise_quality(result: Mapping[str, object]) -> dict[str, object]:
    quality = _as_mapping(result.get("quality"), "quality")
    metrics = _as_mapping(quality.get("metrics"), "quality.metrics")
    normalised_metrics: dict[str, float] = {}
    for name in QUALITY_METRIC_NAMES:
        maximum = 1.0 if name.endswith("_fraction") or name.endswith("_coverage") else None
        normalised_metrics[name] = _as_number(
            metrics.get(name),
            f"quality.metrics.{name}",
            minimum=0.0,
            maximum=maximum,
        )
    flags = _as_string_list(quality.get("flags"), "quality.flags")
    requires_review = _as_bool(
        quality.get("requires_review"),
        "quality.requires_review",
    )
    if requires_review != bool(flags):
        raise ValueError("quality.requires_review must match quality.flags")
    return {
        "metrics": normalised_metrics,
        "flags": flags,
        "requires_review": requires_review,
        "interpretation": _as_text(
            quality.get("interpretation"),
            "quality.interpretation",
        ),
    }


def _normalise_explanation(
    explanation: Mapping[str, object] | None,
    *,
    input_sha256: str,
    model_sha256: str,
    predicted_grade: int,
) -> dict[str, object]:
    if explanation is None:
        return {"included": False}
    payload = _as_mapping(explanation, "explanation")
    if _as_sha256(payload.get("input_sha256"), "explanation.input_sha256") != input_sha256:
        raise ValueError("explanation input hash does not match prediction")
    if _as_sha256(payload.get("model_sha256"), "explanation.model_sha256") != model_sha256:
        raise ValueError("explanation model hash does not match prediction")
    target_grade = _as_integer(
        payload.get("target_grade"),
        "explanation.target_grade",
        0,
        4,
    )
    if target_grade != predicted_grade:
        raise ValueError("explanation target must match the predicted grade")
    target_label = _as_text(
        payload.get("target_label"),
        "explanation.target_label",
    )
    if target_label != GRADE_LABELS[target_grade]:
        raise ValueError("explanation target label is inconsistent")
    return {
        "included": True,
        "target_grade": target_grade,
        "target_label": target_label,
        "backbone_layer": _as_text(
            payload.get("backbone_layer"),
            "explanation.backbone_layer",
        ),
        "input_space": _as_text(
            payload.get("input_space"),
            "explanation.input_space",
        ),
        "interpretation": _as_text(
            payload.get("interpretation"),
            "explanation.interpretation",
        ),
    }


def _canonical_bytes(receipt: Mapping[str, object]) -> bytes:
    unsigned = json.loads(json.dumps(receipt))
    integrity = _as_mapping(unsigned.get("integrity"), "integrity")
    mutable_integrity = dict(integrity)
    mutable_integrity.pop("receipt_sha256", None)
    unsigned["integrity"] = mutable_integrity
    return json.dumps(
        unsigned,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def receipt_sha256(receipt: Mapping[str, object]) -> str:
    """Return the canonical digest, excluding the checksum field itself."""

    return hashlib.sha256(_canonical_bytes(receipt)).hexdigest()


def build_receipt(
    result: Mapping[str, object],
    explanation: Mapping[str, object] | None = None,
    *,
    issued_at_utc: str | None = None,
) -> dict[str, object]:
    """Validate supplied evidence and create one canonical receipt."""

    prediction = _as_mapping(result, "result")
    input_sha256 = _as_sha256(prediction.get("input_sha256"), "input_sha256")
    model_sha256 = _as_sha256(prediction.get("model_sha256"), "model_sha256")
    predicted_grade = _as_integer(
        prediction.get("predicted_grade"),
        "predicted_grade",
        0,
        4,
    )
    predicted_label = _as_text(
        prediction.get("predicted_label"),
        "predicted_label",
    )
    if predicted_label != GRADE_LABELS[predicted_grade]:
        raise ValueError("predicted_label is inconsistent with predicted_grade")
    confidence = _as_number(
        prediction.get("confidence"),
        "confidence",
        minimum=0.0,
        maximum=1.0,
    )
    probabilities = _normalise_probabilities(
        prediction,
        predicted_grade,
        confidence,
    )
    uncertain = _as_bool(prediction.get("uncertain"), "uncertain")
    review_reasons = _as_string_list(
        prediction.get("review_reasons"),
        "review_reasons",
    )
    requires_review = _as_bool(
        prediction.get("requires_human_review"),
        "requires_human_review",
    )
    if requires_review != bool(review_reasons):
        raise ValueError("requires_human_review must match review_reasons")
    if uncertain != ("LOW_MODEL_CONFIDENCE" in review_reasons):
        raise ValueError("uncertain must match LOW_MODEL_CONFIDENCE")

    policy = _as_mapping(prediction.get("policy"), "policy")
    temperature = _as_number(
        policy.get("temperature"),
        "policy.temperature",
        minimum=0.0,
    )
    if temperature == 0:
        raise ValueError("policy.temperature must be positive")
    confidence_threshold = _as_number(
        policy.get("confidence_threshold"),
        "policy.confidence_threshold",
        minimum=0.0,
        maximum=1.0,
    )

    receipt: dict[str, object] = {
        "receipt_type": RECEIPT_TYPE,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "issued_at_utc": _normalise_timestamp(issued_at_utc),
        "traceability": {
            "input_sha256": input_sha256,
            "model_sha256": model_sha256,
        },
        "prediction": {
            "predicted_grade": predicted_grade,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "uncertain": uncertain,
            "requires_human_review": requires_review,
            "review_reasons": review_reasons,
            "probabilities": probabilities,
        },
        "quality": _normalise_quality(prediction),
        "policy": {
            "temperature": temperature,
            "confidence_threshold": confidence_threshold,
        },
        "explanation": _normalise_explanation(
            explanation,
            input_sha256=input_sha256,
            model_sha256=model_sha256,
            predicted_grade=predicted_grade,
        ),
        "privacy": {
            "retinal_image_included": False,
            "original_filename_included": False,
            "gradcam_image_data_included": False,
        },
        "educational_notice": _as_text(
            prediction.get("educational_notice"),
            "educational_notice",
        ),
        "integrity_notice": INTEGRITY_NOTICE,
        "integrity": {
            "algorithm": CHECKSUM_ALGORITHM,
            "canonicalization": CANONICALIZATION,
            "scope": "all receipt fields except integrity.receipt_sha256",
        },
    }
    integrity = dict(_as_mapping(receipt["integrity"], "integrity"))
    integrity["receipt_sha256"] = receipt_sha256(receipt)
    receipt["integrity"] = integrity
    return receipt


def verify_receipt(receipt: Mapping[str, object]) -> dict[str, object]:
    """Verify checksum integrity without claiming origin authenticity."""

    payload = _as_mapping(receipt, "receipt")
    if payload.get("receipt_type") != RECEIPT_TYPE:
        raise ValueError("Unsupported receipt_type")
    if payload.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("Unsupported schema_version")
    integrity = _as_mapping(payload.get("integrity"), "integrity")
    if integrity.get("algorithm") != CHECKSUM_ALGORITHM:
        raise ValueError("Unsupported integrity algorithm")
    supplied = _as_sha256(
        integrity.get("receipt_sha256"),
        "integrity.receipt_sha256",
    )
    expected = receipt_sha256(payload)
    valid = hmac.compare_digest(supplied, expected)
    return {
        "valid": valid,
        "algorithm": CHECKSUM_ALGORITHM,
        "supplied_sha256": supplied,
        "computed_sha256": expected,
        "message": (
            "Receipt contents match the stored checksum."
            if valid
            else "Receipt contents do not match the stored checksum."
        ),
        "authenticity_warning": (
            "Checksum verification does not prove who created the receipt."
        ),
    }


def serialise_receipt(receipt: Mapping[str, object]) -> str:
    """Return stable, human-readable JSON for download or storage."""

    return json.dumps(receipt, indent=2, sort_keys=True) + "\n"
