"""Tests for the serialisable hybrid ordinal loss."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

if tf is not None:
    from retinastage.ordinal import HybridOrdinalLoss


@unittest.skipUnless(tf is not None, "TensorFlow is not installed")
class HybridOrdinalLossTests(unittest.TestCase):
    def test_distant_error_has_larger_loss_than_adjacent_error(self) -> None:
        loss = HybridOrdinalLoss(ordinal_weight=1.0)
        true_grade = tf.constant([0])
        adjacent = tf.constant([[0.05, 0.90, 0.03, 0.01, 0.01]])
        distant = tf.constant([[0.05, 0.01, 0.01, 0.03, 0.90]])
        self.assertGreater(
            float(loss(true_grade, distant)),
            float(loss(true_grade, adjacent)),
        )

    def test_config_round_trip(self) -> None:
        original = HybridOrdinalLoss(num_classes=5, ordinal_weight=0.5)
        restored = HybridOrdinalLoss.from_config(original.get_config())
        self.assertEqual(restored.num_classes, 5)
        self.assertEqual(restored.ordinal_weight, 0.5)


if __name__ == "__main__":
    unittest.main()
