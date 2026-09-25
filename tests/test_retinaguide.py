"""Tests for deterministic, prediction-grounded RetinaGuide responses."""

import unittest

from retinastage.retinaguide import (
    classify_intent,
    context_from_prediction,
    explain_prediction,
)


def prediction_result(
    *,
    confidence: float = 0.60,
    uncertain: bool = False,
    review_reasons: list[str] | None = None,
) -> dict[str, object]:
    reasons = review_reasons or []
    return {
        "predicted_grade": 2,
        "predicted_label": "Moderate",
        "confidence": confidence,
        "uncertain": uncertain,
        "requires_human_review": bool(reasons),
        "review_reasons": reasons,
        "probabilities": [
            {"grade": 0, "label": "No DR", "probability": 0.10},
            {"grade": 1, "label": "Mild", "probability": 0.20},
            {"grade": 2, "label": "Moderate", "probability": confidence},
            {"grade": 3, "label": "Severe", "probability": 0.05},
            {
                "grade": 4,
                "label": "Proliferative DR",
                "probability": max(0.0, 0.65 - confidence),
            },
        ],
        "quality": {
            "metrics": {
                "brightness_mean": 76.3,
                "contrast_std": 17.2,
                "sharpness_laplacian_variance": 25.0,
                "retinal_field_coverage": 0.60,
            },
            "flags": [
                reason
                for reason in reasons
                if reason != "LOW_MODEL_CONFIDENCE"
            ],
        },
        "policy": {"confidence_threshold": 0.55},
    }


class RetinaGuideTests(unittest.TestCase):
    def test_confidence_answer_uses_frozen_threshold(self) -> None:
        result = prediction_result(
            confidence=0.45,
            uncertain=True,
            review_reasons=["LOW_MODEL_CONFIDENCE"],
        )
        response = explain_prediction("Why is this uncertain?", result)
        self.assertEqual(response["intent"], "confidence")
        self.assertIn("45.0%", response["answer"])
        self.assertIn("55.0%", response["answer"])
        self.assertIn("human review", response["answer"])

    def test_quality_answer_separates_flags_from_diagnosis(self) -> None:
        result = prediction_result(review_reasons=["HIGH_BRIGHTNESS"])
        response = explain_prediction("Explain the image quality", result)
        self.assertEqual(response["intent"], "quality")
        self.assertIn("High brightness", response["answer"])
        self.assertIn("not clinical gradability", response["answer"])

    def test_medical_advice_request_is_declined(self) -> None:
        response = explain_prediction(
            "What treatment should I take?",
            prediction_result(),
        )
        self.assertEqual(response["intent"], "medical_advice")
        self.assertIn("cannot diagnose", response["answer"])
        self.assertIn("cannot", response["safety_notice"])

    def test_probability_answer_reports_all_five_grades(self) -> None:
        response = explain_prediction(
            "Show all probabilities",
            prediction_result(),
        )
        self.assertEqual(response["intent"], "probabilities")
        for grade in range(5):
            self.assertIn(f"Grade {grade}", response["answer"])

    def test_context_rejects_mismatched_confidence(self) -> None:
        result = prediction_result()
        result["confidence"] = 0.61
        with self.assertRaisesRegex(ValueError, "confidence must match"):
            context_from_prediction(result)

    def test_intent_classifier_has_grounded_fallback(self) -> None:
        self.assertEqual(classify_intent("Please explain this"), "summary")


if __name__ == "__main__":
    unittest.main()
