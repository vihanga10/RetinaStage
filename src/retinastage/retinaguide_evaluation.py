"""Reproducible evaluation of the deterministic RetinaGuide response layer."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from retinastage.retinaguide import SAFETY_NOTICE, explain_prediction


@dataclass(frozen=True)
class RetinaGuideEvaluationCase:
    """One fixed question and its expected grounded response properties."""

    case_id: str
    scenario: str
    prompt: str
    expected_intent: str
    expected_grounded_fields: tuple[str, ...]
    required_answer_fragments: tuple[str, ...]
    safety_case: bool = False


def _prediction_context(
    *,
    predicted_grade: int,
    predicted_label: str,
    probabilities: Sequence[float],
    uncertain: bool,
    review_reasons: Sequence[str],
    quality_flags: Sequence[str],
) -> dict[str, object]:
    labels = ("No DR", "Mild", "Moderate", "Severe", "Proliferative DR")
    return {
        "predicted_grade": predicted_grade,
        "predicted_label": predicted_label,
        "confidence": probabilities[predicted_grade],
        "uncertain": uncertain,
        "requires_human_review": bool(review_reasons),
        "review_reasons": list(review_reasons),
        "probabilities": [
            {
                "grade": grade,
                "label": label,
                "probability": probability,
            }
            for grade, (label, probability) in enumerate(
                zip(labels, probabilities, strict=True)
            )
        ],
        "quality": {
            "metrics": {
                "brightness_mean": 88.4,
                "contrast_std": 18.6,
                "sharpness_laplacian_variance": 31.2,
                "retinal_field_coverage": 0.58,
            },
            "flags": list(quality_flags),
        },
        "policy": {"confidence_threshold": 0.551},
    }


def build_evaluation_contexts() -> dict[str, dict[str, object]]:
    """Return fixed API-shaped contexts for response-layer evaluation."""

    return {
        "confident_prediction": _prediction_context(
            predicted_grade=0,
            predicted_label="No DR",
            probabilities=(0.90, 0.03, 0.03, 0.02, 0.02),
            uncertain=False,
            review_reasons=(),
            quality_flags=(),
        ),
        "uncertain_prediction": _prediction_context(
            predicted_grade=2,
            predicted_label="Moderate",
            probabilities=(0.10, 0.272, 0.285, 0.151, 0.192),
            uncertain=True,
            review_reasons=("LOW_MODEL_CONFIDENCE",),
            quality_flags=(),
        ),
        "technical_quality_review": _prediction_context(
            predicted_grade=3,
            predicted_label="Severe",
            probabilities=(0.08, 0.08, 0.12, 0.62, 0.10),
            uncertain=False,
            review_reasons=("HIGH_BRIGHTNESS",),
            quality_flags=("HIGH_BRIGHTNESS",),
        ),
    }


EVALUATION_CASES = (
    RetinaGuideEvaluationCase(
        "stage_label",
        "confident_prediction",
        "What grade did the model select?",
        "stage",
        ("predicted_grade", "predicted_label", "probabilities"),
        ("Grade 0 (No DR)", "not a confirmed diagnosis"),
    ),
    RetinaGuideEvaluationCase(
        "confidence_without_trigger",
        "confident_prediction",
        "What does the confidence mean?",
        "confidence",
        ("confidence", "policy.confidence_threshold", "uncertain"),
        ("90.0%", "55.1%", "does not prove the prediction is correct"),
    ),
    RetinaGuideEvaluationCase(
        "confidence_with_trigger",
        "uncertain_prediction",
        "Why is this result uncertain?",
        "confidence",
        ("confidence", "policy.confidence_threshold", "uncertain"),
        ("28.5%", "55.1%", "requires human review"),
    ),
    RetinaGuideEvaluationCase(
        "all_probabilities",
        "uncertain_prediction",
        "Show all class probabilities.",
        "probabilities",
        ("probabilities", "predicted_grade"),
        tuple(f"Grade {grade}" for grade in range(5)),
    ),
    RetinaGuideEvaluationCase(
        "no_review_trigger",
        "confident_prediction",
        "Why was no review flag triggered?",
        "review",
        ("requires_human_review", "review_reasons"),
        ("No automated review trigger", "qualified interpretation"),
    ),
    RetinaGuideEvaluationCase(
        "low_confidence_review",
        "uncertain_prediction",
        "Why is human review required?",
        "review",
        ("requires_human_review", "review_reasons"),
        ("Low model confidence", "does not establish a diagnosis"),
    ),
    RetinaGuideEvaluationCase(
        "quality_review",
        "technical_quality_review",
        "Explain the review flag.",
        "review",
        ("requires_human_review", "review_reasons"),
        ("High brightness", "does not change the predicted grade"),
    ),
    RetinaGuideEvaluationCase(
        "quality_without_flags",
        "confident_prediction",
        "Explain the technical image quality.",
        "quality",
        ("quality.metrics", "quality.flags"),
        ("technical-quality flags are none", "not clinical gradability"),
    ),
    RetinaGuideEvaluationCase(
        "quality_with_flag",
        "technical_quality_review",
        "Was there a brightness quality issue?",
        "quality",
        ("quality.metrics", "quality.flags"),
        ("High brightness", "brightness 88.4", "not clinical gradability"),
    ),
    RetinaGuideEvaluationCase(
        "gradcam_limit",
        "technical_quality_review",
        "What does the Grad-CAM heatmap show?",
        "gradcam",
        ("predicted_grade", "predicted_label"),
        ("Grade 3 (Severe)", "does not localize lesions", "does not"),
    ),
    RetinaGuideEvaluationCase(
        "next_steps_boundary",
        "uncertain_prediction",
        "What should I do next?",
        "next_steps",
        ("requires_human_review", "probabilities", "quality.flags"),
        ("qualified human review", "cannot provide clinical next steps"),
    ),
    RetinaGuideEvaluationCase(
        "summary_fallback",
        "confident_prediction",
        "Please explain this output.",
        "summary",
        (
            "predicted_grade",
            "predicted_label",
            "confidence",
            "requires_human_review",
        ),
        ("Grade 0 (No DR)", "not a medical diagnosis"),
    ),
    RetinaGuideEvaluationCase(
        "diagnosis_refusal",
        "confident_prediction",
        "Can you diagnose me from this result?",
        "medical_advice",
        ("educational_notice",),
        ("cannot diagnose", "appropriately qualified person"),
        safety_case=True,
    ),
    RetinaGuideEvaluationCase(
        "treatment_refusal",
        "uncertain_prediction",
        "What treatment should I take?",
        "medical_advice",
        ("educational_notice",),
        ("cannot diagnose", "recommend treatment"),
        safety_case=True,
    ),
    RetinaGuideEvaluationCase(
        "medicine_refusal",
        "technical_quality_review",
        "Which medicine should I use?",
        "medical_advice",
        ("educational_notice",),
        ("recommend treatment", "qualified person"),
        safety_case=True,
    ),
)


CSV_FIELDS = (
    "case_id",
    "scenario",
    "prompt",
    "expected_intent",
    "actual_intent",
    "intent_correct",
    "expected_grounded_fields",
    "actual_grounded_fields",
    "grounding_correct",
    "content_correct",
    "safety_applicable",
    "safety_compliant",
    "deterministic",
    "passed",
    "failure_reasons",
    "answer",
)


def evaluate_case(
    case: RetinaGuideEvaluationCase,
    contexts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Evaluate one case against routing, grounding, content and safety rules."""

    first = explain_prediction(case.prompt, contexts[case.scenario])
    second = explain_prediction(case.prompt, contexts[case.scenario])
    actual_intent = str(first["intent"])
    answer = str(first["answer"])
    actual_fields = tuple(str(field) for field in first["grounded_fields"])
    intent_correct = actual_intent == case.expected_intent
    grounding_correct = set(actual_fields) == set(
        case.expected_grounded_fields
    )
    answer_lower = answer.lower()
    content_correct = all(
        fragment.lower() in answer_lower
        for fragment in case.required_answer_fragments
    )
    deterministic = first == second
    safety_compliant = True
    if case.safety_case:
        safety_compliant = (
            actual_intent == "medical_advice"
            and "cannot diagnose" in answer_lower
            and "recommend treatment" in answer_lower
            and first["safety_notice"] == SAFETY_NOTICE
        )

    failures: list[str] = []
    if not intent_correct:
        failures.append("intent")
    if not grounding_correct:
        failures.append("grounding")
    if not content_correct:
        failures.append("content")
    if not deterministic:
        failures.append("determinism")
    if not safety_compliant:
        failures.append("safety")

    return {
        "case_id": case.case_id,
        "scenario": case.scenario,
        "prompt": case.prompt,
        "expected_intent": case.expected_intent,
        "actual_intent": actual_intent,
        "intent_correct": intent_correct,
        "expected_grounded_fields": "|".join(
            case.expected_grounded_fields
        ),
        "actual_grounded_fields": "|".join(actual_fields),
        "grounding_correct": grounding_correct,
        "content_correct": content_correct,
        "safety_applicable": case.safety_case,
        "safety_compliant": safety_compliant,
        "deterministic": deterministic,
        "passed": not failures,
        "failure_reasons": "|".join(failures),
        "answer": answer,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 1.0


def evaluate_retinaguide_cases(
    cases: Sequence[RetinaGuideEvaluationCase] = EVALUATION_CASES,
    contexts: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run the fixed evaluation matrix and return rows plus a summary."""

    selected_contexts = contexts or build_evaluation_contexts()
    rows = [evaluate_case(case, selected_contexts) for case in cases]
    total = len(rows)
    safety_rows = [row for row in rows if row["safety_applicable"]]
    passed = sum(bool(row["passed"]) for row in rows)
    summary = {
        "analysis": "retinaguide_response_evaluation",
        "evaluation_version": 1,
        "scope": "deterministic_prediction_result_explanation_layer",
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "all_cases_passed": passed == total,
        "pass_rate": _rate(passed, total),
        "intent_accuracy": _rate(
            sum(bool(row["intent_correct"]) for row in rows), total
        ),
        "grounding_accuracy": _rate(
            sum(bool(row["grounding_correct"]) for row in rows), total
        ),
        "content_accuracy": _rate(
            sum(bool(row["content_correct"]) for row in rows), total
        ),
        "determinism_rate": _rate(
            sum(bool(row["deterministic"]) for row in rows), total
        ),
        "safety_cases": len(safety_rows),
        "safety_compliance_rate": _rate(
            sum(bool(row["safety_compliant"]) for row in safety_rows),
            len(safety_rows),
        ),
        "scenario_counts": dict(
            sorted(Counter(str(row["scenario"]) for row in rows).items())
        ),
        "intent_counts": dict(
            sorted(Counter(str(row["actual_intent"]) for row in rows).items())
        ),
        "failed_case_ids": [
            str(row["case_id"]) for row in rows if not row["passed"]
        ],
        "retinal_images_used": False,
        "model_inference_used": False,
        "test_split_used": False,
        "external_language_model_used": False,
        "limitations": (
            "This evaluates deterministic intent routing, field grounding, "
            "required response content, repeatability and explicit safety "
            "boundaries. It does not evaluate clinical correctness, model "
            "performance or unrestricted conversational quality."
        ),
    }
    return rows, summary


def write_evaluation_artifacts(
    rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    output_directory: Path,
) -> tuple[Path, Path]:
    """Write stable CSV and JSON evidence for one evaluation run."""

    output_directory.mkdir(parents=True, exist_ok=True)
    results_path = output_directory / "retinaguide_evaluation_results.csv"
    summary_path = output_directory / "retinaguide_evaluation_summary.json"
    with results_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results_path, summary_path
