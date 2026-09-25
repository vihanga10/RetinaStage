"""Lightweight HTTP contract tests that do not load TensorFlow artifacts."""

import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient

    from app.api.main import app
except ImportError:  # pragma: no cover - optional local test environment
    TestClient = None
    app = None


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


if __name__ == "__main__":
    unittest.main()
