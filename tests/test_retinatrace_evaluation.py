"""Tests for the reproducible RetinaTrace evaluation evidence."""

from pathlib import Path
import tempfile
import unittest

from retinastage.retinatrace_evaluation import (
    evaluate_retinatrace,
    write_evaluation_artifacts,
)


class RetinaTraceEvaluationTests(unittest.TestCase):
    def test_fixed_checks_all_pass(self) -> None:
        rows, summary = evaluate_retinatrace()
        self.assertEqual(len(rows), 9)
        self.assertTrue(summary["all_checks_passed"])
        self.assertEqual(summary["pass_rate"], 1.0)
        self.assertEqual(summary["tamper_detection_rate"], 1.0)
        self.assertTrue(summary["privacy_checks_passed"])

    def test_evaluation_does_not_claim_signature_or_use_model(self) -> None:
        _, summary = evaluate_retinatrace()
        self.assertFalse(summary["cryptographic_signature_used"])
        self.assertFalse(summary["retinal_images_used"])
        self.assertFalse(summary["model_inference_used"])
        self.assertFalse(summary["final_test_split_used"])

    def test_written_evidence_is_byte_stable(self) -> None:
        rows, summary = evaluate_retinatrace()
        with tempfile.TemporaryDirectory() as first_directory:
            first_paths = write_evaluation_artifacts(
                rows,
                summary,
                Path(first_directory),
            )
            first_bytes = [path.read_bytes() for path in first_paths]
        with tempfile.TemporaryDirectory() as second_directory:
            second_paths = write_evaluation_artifacts(
                rows,
                summary,
                Path(second_directory),
            )
            second_bytes = [path.read_bytes() for path in second_paths]
        self.assertEqual(first_bytes, second_bytes)


if __name__ == "__main__":
    unittest.main()
