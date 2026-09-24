"""Unit tests for inference-time technical image-quality checks."""

import unittest

try:
    import cv2
    import numpy as np

    from retinastage.quality import assess_quality, measure_quality
except ImportError:  # pragma: no cover - optional local test environment
    cv2 = None
    np = None
    assess_quality = None
    measure_quality = None


@unittest.skipIf(cv2 is None, "OpenCV and NumPy are not installed")
class QualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.circle(self.image, (200, 150), 130, (70, 100, 150), -1)

    def test_measurements_are_finite_and_bounded(self) -> None:
        metrics = measure_quality(self.image)
        self.assertGreater(metrics["brightness_mean"], 0)
        self.assertGreater(metrics["retinal_field_coverage"], 0)
        self.assertLessEqual(metrics["retinal_field_coverage"], 1)
        self.assertTrue(all(np.isfinite(list(metrics.values()))))

    def test_review_flags_follow_supplied_limits(self) -> None:
        limits = {
            "brightness_low": 255.0,
            "brightness_high": 256.0,
            "contrast_low": -1.0,
            "sharpness_low": -1.0,
            "coverage_low": -1.0,
        }
        result = assess_quality(self.image, limits)
        self.assertIn("LOW_BRIGHTNESS", result.flags)
        self.assertTrue(result.requires_review)


if __name__ == "__main__":
    unittest.main()
