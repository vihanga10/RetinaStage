"""Tests for privacy-preserving RetinaTrace evidence receipts."""

from copy import deepcopy
import json
import unittest

from retinastage.retinatrace import (
    build_receipt,
    serialise_receipt,
    verify_receipt,
)


def prediction_fixture() -> dict[str, object]:
    return {
        "original_filename": "private-patient-image.png",
        "input_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "predicted_grade": 2,
        "predicted_label": "Moderate",
        "confidence": 0.6,
        "uncertain": False,
        "requires_human_review": True,
        "review_reasons": ["HIGH_BRIGHTNESS"],
        "probabilities": [
            {"grade": 0, "label": "No DR", "probability": 0.1},
            {"grade": 1, "label": "Mild", "probability": 0.2},
            {"grade": 2, "label": "Moderate", "probability": 0.6},
            {"grade": 3, "label": "Severe", "probability": 0.05},
            {
                "grade": 4,
                "label": "Proliferative DR",
                "probability": 0.05,
            },
        ],
        "quality": {
            "metrics": {
                "brightness_mean": 152.4,
                "contrast_std": 88.6,
                "sharpness_laplacian_variance": 2357.5,
                "dark_pixel_fraction": 0.08,
                "bright_pixel_fraction": 0.36,
                "retinal_field_coverage": 0.58,
            },
            "flags": ["HIGH_BRIGHTNESS"],
            "requires_review": True,
            "interpretation": "Dataset-relative technical checks only.",
        },
        "policy": {
            "temperature": 1.168,
            "confidence_threshold": 0.551,
        },
        "educational_notice": "Educational research prototype only.",
    }


def explanation_fixture() -> dict[str, object]:
    return {
        "input_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "target_grade": 2,
        "target_label": "Moderate",
        "backbone_layer": "efficientnetb0",
        "input_space": "processed_224_pixel_model_input",
        "heatmap_data_url": "data:image/png;base64,private-heatmap",
        "overlay_data_url": "data:image/png;base64,private-overlay",
        "interpretation": "Attention evidence is not lesion localization.",
    }


class RetinaTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.receipt = build_receipt(
            prediction_fixture(),
            explanation_fixture(),
            issued_at_utc="2026-09-25T05:30:00Z",
        )

    def test_unchanged_receipt_verifies(self) -> None:
        verification = verify_receipt(self.receipt)
        self.assertTrue(verification["valid"])
        self.assertEqual(
            verification["supplied_sha256"],
            verification["computed_sha256"],
        )

    def test_changed_probability_fails_verification(self) -> None:
        changed = deepcopy(self.receipt)
        changed["prediction"]["probabilities"][2]["probability"] = 0.59
        self.assertFalse(verify_receipt(changed)["valid"])

    def test_changed_review_reason_fails_verification(self) -> None:
        changed = deepcopy(self.receipt)
        changed["prediction"]["review_reasons"] = []
        self.assertFalse(verify_receipt(changed)["valid"])

    def test_receipt_omits_images_filenames_and_gradcam_bytes(self) -> None:
        serialised = serialise_receipt(self.receipt)
        self.assertNotIn("heatmap_data_url", serialised)
        self.assertNotIn("overlay_data_url", serialised)
        self.assertNotIn("private-heatmap", serialised)
        self.assertNotIn("data:image", serialised)
        self.assertNotIn("private-patient-image.png", serialised)
        self.assertNotIn("filename", self.receipt["traceability"])
        self.assertFalse(self.receipt["privacy"]["retinal_image_included"])

    def test_serialisation_is_stable_and_ends_with_newline(self) -> None:
        first = serialise_receipt(self.receipt)
        second = serialise_receipt(json.loads(first))
        self.assertEqual(first, second)
        self.assertTrue(first.endswith("\n"))

    def test_prediction_and_explanation_hashes_must_match(self) -> None:
        explanation = explanation_fixture()
        explanation["input_sha256"] = "3" * 64
        with self.assertRaisesRegex(ValueError, "input hash"):
            build_receipt(prediction_fixture(), explanation)

    def test_invalid_prediction_hash_is_rejected(self) -> None:
        prediction = prediction_fixture()
        prediction["model_sha256"] = "not-a-sha256"
        with self.assertRaisesRegex(ValueError, "64-character"):
            build_receipt(prediction)


if __name__ == "__main__":
    unittest.main()
