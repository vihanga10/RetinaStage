"""Tests for reproducible RetinaGuide response-layer evaluation."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from retinastage.retinaguide_evaluation import (
    EVALUATION_CASES,
    evaluate_retinaguide_cases,
    write_evaluation_artifacts,
)


class RetinaGuideEvaluationTests(unittest.TestCase):
    def test_fixed_matrix_passes_all_metrics(self) -> None:
        rows, summary = evaluate_retinaguide_cases()
        self.assertEqual(len(rows), 15)
        self.assertTrue(summary["all_cases_passed"])
        self.assertEqual(summary["intent_accuracy"], 1.0)
        self.assertEqual(summary["grounding_accuracy"], 1.0)
        self.assertEqual(summary["content_accuracy"], 1.0)
        self.assertEqual(summary["determinism_rate"], 1.0)

    def test_safety_cases_all_refuse_medical_advice(self) -> None:
        rows, summary = evaluate_retinaguide_cases()
        safety_rows = [row for row in rows if row["safety_applicable"]]
        self.assertEqual(len(safety_rows), 3)
        self.assertTrue(all(row["safety_compliant"] for row in safety_rows))
        self.assertEqual(summary["safety_compliance_rate"], 1.0)

    def test_confident_answer_does_not_equate_confidence_with_truth(self) -> None:
        rows, _ = evaluate_retinaguide_cases()
        row = next(
            item
            for item in rows
            if item["case_id"] == "confidence_without_trigger"
        )
        self.assertIn(
            "does not prove the prediction is correct",
            row["answer"],
        )

    def test_wrong_expectation_is_reported_as_failure(self) -> None:
        wrong_case = replace(
            EVALUATION_CASES[0],
            expected_intent="quality",
        )
        rows, summary = evaluate_retinaguide_cases((wrong_case,))
        self.assertFalse(rows[0]["passed"])
        self.assertEqual(rows[0]["failure_reasons"], "intent")
        self.assertEqual(summary["failed_case_ids"], ["stage_label"])

    def test_written_artifacts_are_byte_stable(self) -> None:
        rows, summary = evaluate_retinaguide_cases()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_paths = write_evaluation_artifacts(
                rows,
                summary,
                root / "first",
            )
            second_paths = write_evaluation_artifacts(
                rows,
                summary,
                root / "second",
            )
            for first, second in zip(first_paths, second_paths, strict=True):
                self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
