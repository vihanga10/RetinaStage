"""Tests for calibration calculations that do not require image data."""

import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinastage.calibration import (  # noqa: E402
    expected_calibration_error,
    multiclass_brier_score,
    probabilities_with_temperature,
)


class CalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.probabilities = np.asarray(
            [
                [0.80, 0.10, 0.05, 0.03, 0.02],
                [0.05, 0.10, 0.70, 0.10, 0.05],
            ]
        )
        self.labels = np.asarray([0, 2])

    def test_temperature_scaling_preserves_rows_and_argmax(self) -> None:
        scaled = probabilities_with_temperature(self.probabilities, 1.5)
        np.testing.assert_allclose(scaled.sum(axis=1), 1.0)
        np.testing.assert_array_equal(
            scaled.argmax(axis=1), self.probabilities.argmax(axis=1)
        )

    def test_invalid_temperature_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            probabilities_with_temperature(self.probabilities, 0.0)

    def test_calibration_metrics_are_bounded(self) -> None:
        ece, bins = expected_calibration_error(
            self.labels, self.probabilities
        )
        brier = multiclass_brier_score(
            self.labels, self.probabilities
        )
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)
        self.assertGreaterEqual(brier, 0.0)
        self.assertEqual(int(bins["count"].sum()), len(self.labels))


if __name__ == "__main__":
    unittest.main()
