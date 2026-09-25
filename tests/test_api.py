"""Lightweight HTTP contract tests that do not load TensorFlow artifacts."""

import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient

    from app.api.main import app
except ImportError:  # pragma: no cover - optional local test environment
    TestClient = None
    app = None


def receipt_prediction() -> dict[str, object]:
    """Return one complete API-shaped prediction for receipt tests."""

    return {
        "input_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "predicted_grade": 0,
        "predicted_label": "No DR",
        "confidence": 0.9,
        "uncertain": False,
        "requires_human_review": False,
        "review_reasons": [],
        "probabilities": [
            {"grade": 0, "label": "No DR", "probability": 0.9},
            {"grade": 1, "label": "Mild", "probability": 0.04},
            {"grade": 2, "label": "Moderate", "probability": 0.03},
            {"grade": 3, "label": "Severe", "probability": 0.02},
            {
                "grade": 4,
                "label": "Proliferative DR",
                "probability": 0.01,
            },
        ],
        "quality": {
            "metrics": {
                "brightness_mean": 88.0,
                "contrast_std": 20.0,
                "sharpness_laplacian_variance": 30.0,
                "dark_pixel_fraction": 0.1,
                "bright_pixel_fraction": 0.1,
                "retinal_field_coverage": 0.6,
            },
            "flags": [],
            "requires_review": False,
            "interpretation": "Dataset-relative technical checks only.",
        },
        "policy": {"temperature": 1.2, "confidence_threshold": 0.55},
        "educational_notice": "Educational research prototype only.",
    }


@unittest.skipIf(TestClient is None, "Application dependencies are missing")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_reports_all_artifact_keys(self) -> None:
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["status"], {"ready", "not_ready"})
        self.assertEqual(
            set(payload["artifacts"]),
            {"model", "calibration", "quality_summary"},
        )

    def test_non_image_upload_is_rejected_before_model_loading(self) -> None:
        response = self.client.post(
            "/api/v1/predict",
            files={"image": ("notes.txt", b"not an image", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)

    def test_non_image_explanation_is_rejected_before_model_loading(self) -> None:
        response = self.client.post(
            "/api/v1/explain?target_grade=0",
            files={"image": ("notes.txt", b"not an image", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)

    def test_explanation_endpoint_forwards_validated_target(self) -> None:
        class FakePredictor:
            def explain_bytes(self, content: bytes, target_grade: int) -> dict:
                return {
                    "byte_count": len(content),
                    "target_grade": target_grade,
                }

        with patch("app.api.main.get_predictor", return_value=FakePredictor()):
            response = self.client.post(
                "/api/v1/explain?target_grade=3",
                files={"image": ("retina.png", b"png-data", "image/png")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"byte_count": 8, "target_grade": 3})

    def test_retinaguide_explains_context_without_loading_model(self) -> None:
        result = {
            "predicted_grade": 2,
            "predicted_label": "Moderate",
            "confidence": 0.6,
            "uncertain": False,
            "requires_human_review": False,
            "review_reasons": [],
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
                    "brightness_mean": 80.0,
                    "contrast_std": 20.0,
                    "sharpness_laplacian_variance": 30.0,
                    "retinal_field_coverage": 0.6,
                },
                "flags": [],
            },
            "policy": {"confidence_threshold": 0.55},
        }
        with patch(
            "app.api.main.get_predictor",
            side_effect=AssertionError("model must not load"),
        ):
            response = self.client.post(
                "/api/v1/retinaguide",
                json={"message": "What does the confidence mean?", "result": result},
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["intent"], "confidence")
        self.assertIn("60.0%", payload["answer"])

    def test_retinaguide_rejects_invalid_prediction_context(self) -> None:
        response = self.client.post(
            "/api/v1/retinaguide",
            json={"message": "Explain this result", "result": {}},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("Invalid prediction context", response.json()["detail"])

    def test_retinatrace_creates_and_verifies_without_model_loading(self) -> None:
        with patch(
            "app.api.main.get_predictor",
            side_effect=AssertionError("model must not load"),
        ):
            response = self.client.post(
                "/api/v1/retinatrace/receipt",
                json={"result": receipt_prediction(), "explanation": None},
            )
        self.assertEqual(response.status_code, 200)
        receipt = response.json()
        self.assertEqual(
            receipt["receipt_type"],
            "retinastage_prediction_evidence",
        )
        self.assertFalse(receipt["privacy"]["retinal_image_included"])

        verification = self.client.post(
            "/api/v1/retinatrace/verify",
            json={"receipt": receipt},
        )
        self.assertEqual(verification.status_code, 200)
        self.assertTrue(verification.json()["valid"])

    def test_retinatrace_detects_modified_receipt(self) -> None:
        response = self.client.post(
            "/api/v1/retinatrace/receipt",
            json={"result": receipt_prediction()},
        )
        receipt = response.json()
        receipt["prediction"]["confidence"] = 0.8
        verification = self.client.post(
            "/api/v1/retinatrace/verify",
            json={"receipt": receipt},
        )
        self.assertEqual(verification.status_code, 200)
        self.assertFalse(verification.json()["valid"])

    def test_retinatrace_rejects_incomplete_prediction(self) -> None:
        response = self.client.post(
            "/api/v1/retinatrace/receipt",
            json={"result": {}},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("Invalid receipt evidence", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
