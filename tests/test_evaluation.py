"""Regression tests for final-test metric calculations."""

import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinastage.evaluation import calculate_metrics  # noqa: E402


class EvaluationTests(unittest.TestCase):
    def test_reported_final_confusion_matrix_metrics(self) -> None:
        matrix = np.asarray(
            [
                [257, 7, 4, 1, 0],
                [5, 37, 5, 1, 3],
                [0, 27, 83, 18, 10],
                [0, 1, 9, 11, 4],
                [0, 4, 14, 9, 13],
            ]
        )
        true_labels = np.repeat(np.arange(5), matrix.sum(axis=1))
        predictions = np.concatenate(
            [np.repeat(np.arange(5), row) for row in matrix]
        )
        metrics = calculate_metrics(true_labels, predictions)
        self.assertEqual(len(true_labels), 523)
        self.assertAlmostEqual(metrics["accuracy"], 0.7667, places=4)
        self.assertAlmostEqual(metrics["macro_f1"], 0.5833, places=4)
        self.assertAlmostEqual(
            metrics["quadratic_weighted_kappa"], 0.8348, places=4
        )
        self.assertAlmostEqual(metrics["grade_mae"], 0.3212, places=4)
        self.assertAlmostEqual(
            metrics["within_one_grade_accuracy"], 0.9273, places=4
        )


if __name__ == "__main__":
    unittest.main()
