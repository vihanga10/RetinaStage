"""Unit tests for calibrated frozen-model inference orchestration."""

import json
from pathlib import Path
import tempfile
import unittest

try:
    import cv2
    import numpy as np

    from retinastage.inference import (
        InferencePolicy,
        RetinaStagePredictor,
        decode_image_bytes,
        load_inference_policy,
    )
except ImportError:  # pragma: no cover - optional local test environment
    cv2 = None
    np = None
    InferencePolicy = None
    RetinaStagePredictor = None
    decode_image_bytes = None
    load_inference_policy = None


class FakeModel:
    def predict(self, inputs: object, verbose: int = 0) -> object:
        del inputs, verbose
        return np.asarray([[0.05, 0.10, 0.65, 0.15, 0.05]])


@unittest.skipIf(cv2 is None, "OpenCV and NumPy are not installed")
class InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.circle(image, (200, 150), 130, (70, 100, 150), -1)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)
        self.image_bytes = encoded.tobytes()
        self.limits = {
            "brightness_low": 0.0,
            "brightness_high": 255.0,
            "contrast_low": 0.0,
            "sharpness_low": 0.0,
            "coverage_low": 0.0,
        }

    def test_invalid_image_bytes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "decodable"):
            decode_image_bytes(b"not an image")

    def test_nested_calibration_policy_is_supported(self) -> None:
        payload = {
            "temperature": 1.25,
            "uncertainty_policy": {"confidence_threshold": 0.55},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration_summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            policy = load_inference_policy(path)
        self.assertEqual(policy.temperature, 1.25)
        self.assertEqual(policy.confidence_threshold, 0.55)

    def test_prediction_contains_calibrated_review_fields(self) -> None:
        predictor = RetinaStagePredictor(
            model=FakeModel(),
            policy=InferencePolicy(
                temperature=1.0,
                confidence_threshold=0.70,
            ),
            quality_limits=self.limits,
            model_sha256="fixed-model-hash",
        )
        result = predictor.predict_bytes(self.image_bytes)
        self.assertEqual(result["predicted_grade"], 2)
        self.assertEqual(result["predicted_label"], "Moderate")
        self.assertTrue(result["uncertain"])
        self.assertTrue(result["requires_human_review"])
        self.assertIn("LOW_MODEL_CONFIDENCE", result["review_reasons"])
        self.assertEqual(len(result["probabilities"]), 5)
        self.assertAlmostEqual(
            sum(item["probability"] for item in result["probabilities"]),
            1.0,
        )

    def test_invalid_explanation_target_is_rejected_early(self) -> None:
        predictor = RetinaStagePredictor(
            model=FakeModel(),
            policy=InferencePolicy(temperature=1.0, confidence_threshold=0.5),
            quality_limits=self.limits,
        )
        with self.assertRaisesRegex(ValueError, "0 to 4"):
            predictor.explain_bytes(self.image_bytes, 5)


if __name__ == "__main__":
    unittest.main()
