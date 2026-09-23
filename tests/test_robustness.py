"""TensorFlow-dependent checks for deterministic robustness perturbations."""

import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

if tf is not None:
    from retinastage.robustness import (
        ROBUSTNESS_CONDITIONS,
        apply_perturbation,
        get_perturbation,
    )


@unittest.skipUnless(tf is not None, "TensorFlow is not installed")
class RobustnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = tf.ones((32, 32, 3), dtype=tf.float32) * 128.0

    def test_all_conditions_preserve_shape_and_pixel_range(self) -> None:
        for condition in ROBUSTNESS_CONDITIONS:
            transformed = apply_perturbation(
                self.image,
                tf.constant(7),
                condition.name,
            ).numpy()
            self.assertEqual(transformed.shape, (32, 32, 3))
            self.assertGreaterEqual(float(transformed.min()), 0.0)
            self.assertLessEqual(float(transformed.max()), 255.0)

    def test_noise_is_deterministic_for_same_record(self) -> None:
        first = apply_perturbation(
            self.image, tf.constant(11), "gaussian_noise"
        ).numpy()
        second = apply_perturbation(
            self.image, tf.constant(11), "gaussian_noise"
        ).numpy()
        np.testing.assert_array_equal(first, second)

    def test_unknown_condition_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_perturbation("unknown")


if __name__ == "__main__":
    unittest.main()
