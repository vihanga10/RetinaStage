"""Grounded, deterministic explanations for RetinaStage prediction results."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, Sequence


SAFETY_NOTICE = (
    "RetinaGuide explains this educational model output only. It cannot "
    "diagnose disease, recommend treatment, or replace review by an "
    "appropriately qualified person."
)

DEFAULT_SUGGESTIONS = (
    "What does the confidence mean?",
    "Why is human review required?",
    "What does the Grad-CAM panel show?",
)

INTENT_KEYWORDS = {
    "gradcam": ("grad-cam", "gradcam", "heatmap", "overlay", "attention"),
    "quality": (
        "quality",
        "brightness",
        "contrast",
        "sharpness",
        "coverage",
        "technical",
    ),
    "probabilities": (
        "probability",
        "probabilities",
        "other grade",
        "other class",
        "classes",
        "distribution",
    ),
    "confidence": (
        "confidence",
        "certain",
        "uncertain",
        "threshold",
        "reliable",
        "sure",
    ),
    "review": ("review", "flag", "hold", "qualified"),
    "next_steps": ("next step", "what next", "what should i do", "what now"),
    "stage": ("stage", "grade", "result", "prediction", "label", "mean"),
}

RESTRICTED_MEDICAL_TERMS = (
    "diagnose",
    "diagnosis",
    "treatment",
    "treat this",
    "medicine",
    "medication",
    "prescription",
    "cure",
    "surgery",
)


@dataclass(frozen=True)
class GuideProbability:
    """One calibrated class probability supplied by the prediction API."""

    grade: int
    label: str
    probability: float


@dataclass(frozen=True)
class GuideContext:
    """Validated subset of prediction fields RetinaGuide is allowed to use."""

    predicted_grade: int
    predicted_label: str
    confidence: float
    uncertain: bool
    requires_human_review: bool
    review_reasons: tuple[str, ...]
    probabilities: tuple[GuideProbability, ...]
    quality_flags: tuple[str, ...]
    quality_metrics: Mapping[str, float]
    confidence_threshold: float


def _bounded_probability(value: object, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{field_name} must be a finite value from 0 to 1")
    return number


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a list")
    return value


def context_from_prediction(result: Mapping[str, object]) -> GuideContext:
    """Validate and reduce an API prediction to explainable fields."""

    predicted_grade = int(result["predicted_grade"])
    if predicted_grade not in range(5):
        raise ValueError("predicted_grade must be an integer from 0 to 4")

    predicted_label = str(result["predicted_label"]).strip()
    if not predicted_label:
        raise ValueError("predicted_label must not be empty")

    raw_probabilities = _sequence(result["probabilities"], "probabilities")
    probabilities: list[GuideProbability] = []
    seen_grades: set[int] = set()
    for item in raw_probabilities:
        row = _mapping(item, "probability item")
        grade = int(row["grade"])
        if grade not in range(5) or grade in seen_grades:
            raise ValueError("probability grades must be unique integers from 0 to 4")
        seen_grades.add(grade)
        probabilities.append(
            GuideProbability(
                grade=grade,
                label=str(row["label"]).strip(),
                probability=_bounded_probability(
                    row["probability"],
                    f"probability for grade {grade}",
                ),
            )
        )
    if seen_grades != set(range(5)):
        raise ValueError("probabilities must contain each grade from 0 to 4")

    selected_probability = next(
        item.probability
        for item in probabilities
        if item.grade == predicted_grade
    )
    confidence = _bounded_probability(result["confidence"], "confidence")
    if not math.isclose(confidence, selected_probability, abs_tol=1e-6):
        raise ValueError("confidence must match the predicted grade probability")

    policy = _mapping(result["policy"], "policy")
    quality = _mapping(result["quality"], "quality")
    raw_metrics = _mapping(quality["metrics"], "quality.metrics")
    metrics: dict[str, float] = {}
    for key, value in raw_metrics.items():
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"quality metric {key} must be finite")
        metrics[str(key)] = number

    review_reasons = tuple(
        str(reason)
        for reason in _sequence(result["review_reasons"], "review_reasons")
    )
    quality_flags = tuple(
        str(flag)
        for flag in _sequence(quality["flags"], "quality.flags")
    )
    return GuideContext(
        predicted_grade=predicted_grade,
        predicted_label=predicted_label,
        confidence=confidence,
        uncertain=bool(result["uncertain"]),
        requires_human_review=bool(result["requires_human_review"]),
        review_reasons=review_reasons,
        probabilities=tuple(sorted(probabilities, key=lambda item: item.grade)),
        quality_flags=quality_flags,
        quality_metrics=metrics,
        confidence_threshold=_bounded_probability(
            policy["confidence_threshold"],
            "confidence_threshold",
        ),
    )


def _normalise_question(message: str) -> str:
    return re.sub(r"\s+", " ", message.strip().lower())


def classify_intent(message: str) -> str:
    """Classify a short question into a supported deterministic intent."""

    normalised = _normalise_question(message)
    if not normalised:
        raise ValueError("message must not be empty")
    if any(term in normalised for term in RESTRICTED_MEDICAL_TERMS):
        return "medical_advice"
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in normalised for keyword in keywords):
            return intent
    return "summary"


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _humanise_flag(flag: str) -> str:
    return flag.replace("_", " ").strip().lower().capitalize()


def _review_reason_text(context: GuideContext) -> str:
    if not context.review_reasons:
        return "no automated confidence or technical-quality trigger was recorded"
    return ", ".join(_humanise_flag(reason) for reason in context.review_reasons)


def _quality_metrics_text(context: GuideContext) -> str:
    labels = (
        ("brightness_mean", "brightness"),
        ("contrast_std", "contrast"),
        ("sharpness_laplacian_variance", "sharpness"),
    )
    parts = [
        f"{label} {context.quality_metrics[key]:.1f}"
        for key, label in labels
        if key in context.quality_metrics
    ]
    if "retinal_field_coverage" in context.quality_metrics:
        parts.append(
            "retinal-field coverage "
            f"{_percent(context.quality_metrics['retinal_field_coverage'])}"
        )
    return ", ".join(parts)


def _answer(intent: str, context: GuideContext) -> tuple[str, list[str]]:
    grade = f"Grade {context.predicted_grade} ({context.predicted_label})"
    if intent == "medical_advice":
        return (
            "I cannot diagnose a condition or recommend treatment. I can "
            "explain the displayed model result, confidence, probabilities, "
            "technical-quality flags, and Grad-CAM limitations. An "
            "appropriately qualified person must interpret the image and "
            "decide any clinical next steps.",
            ["educational_notice"],
        )
    if intent == "confidence":
        relation = "below" if context.uncertain else "at or above"
        consequence = (
            "The result is therefore marked uncertain and requires human review."
            if context.uncertain
            else "The uncertainty rule was not triggered, but this still does not prove the prediction is correct."
        )
        return (
            f"The calibrated confidence is {_percent(context.confidence)}, "
            f"which is {relation} the fixed {_percent(context.confidence_threshold)} "
            f"threshold. {consequence}",
            ["confidence", "policy.confidence_threshold", "uncertain"],
        )
    if intent == "probabilities":
        distribution = "; ".join(
            f"Grade {item.grade} {item.label}: {_percent(item.probability)}"
            for item in context.probabilities
        )
        return (
            f"The calibrated distribution is {distribution}. The model selected "
            f"{grade} because it has the largest value. These probabilities are "
            "model estimates, not clinical certainty.",
            ["probabilities", "predicted_grade"],
        )
    if intent == "review":
        if context.requires_human_review:
            answer = (
                "Human review is required because the recorded trigger set is: "
                f"{_review_reason_text(context)}. The trigger does not change the "
                "predicted grade and does not establish a diagnosis."
            )
        else:
            answer = (
                "No automated review trigger was recorded for confidence or "
                "technical image quality. The output is still an educational "
                "model result and requires appropriately qualified interpretation."
            )
        return answer, ["requires_human_review", "review_reasons"]
    if intent == "quality":
        flags = (
            ", ".join(_humanise_flag(flag) for flag in context.quality_flags)
            if context.quality_flags
            else "none"
        )
        metrics = _quality_metrics_text(context)
        answer = (
            f"The technical-quality flags are {flags}."
            + (f" The measured values are {metrics}." if metrics else "")
            + " These are dataset-derived technical checks, not clinical gradability decisions."
        )
        return answer, ["quality.metrics", "quality.flags"]
    if intent == "gradcam":
        return (
            f"The Grad-CAM panel is targeted to the selected {grade} output. "
            "It highlights processed-image regions associated with that fixed "
            "output. It does not localize lesions, establish causality, or prove "
            "that the selected grade is correct.",
            ["predicted_grade", "predicted_label"],
        )
    if intent == "next_steps":
        if context.requires_human_review:
            action = (
                "The automated policy requests qualified human review before the "
                "result is used for any decision."
            )
        else:
            action = (
                "No automated review trigger fired, but the result should still "
                "be interpreted by an appropriately qualified person."
            )
        return (
            f"{action} Review the selected grade, all five probabilities, "
            "technical-quality information, and Grad-CAM limitation together. "
            "RetinaGuide cannot provide clinical next steps.",
            ["requires_human_review", "probabilities", "quality.flags"],
        )
    if intent == "stage":
        return (
            f"In this project's five-class scheme, the selected output is {grade}. "
            "It is the class with the highest calibrated probability for this "
            "image. It is a model label, not a confirmed diagnosis.",
            ["predicted_grade", "predicted_label", "probabilities"],
        )

    review_sentence = (
        f"Human review is requested because {_review_reason_text(context)}."
        if context.requires_human_review
        else "No automated confidence or technical-quality review trigger was recorded."
    )
    return (
        f"The model selected {grade} with calibrated confidence "
        f"{_percent(context.confidence)}. {review_sentence} Consider the five "
        "probabilities, technical-quality information, and explanation limits "
        "together; this output is not a medical diagnosis.",
        [
            "predicted_grade",
            "predicted_label",
            "confidence",
            "requires_human_review",
        ],
    )


def explain_prediction(
    message: str,
    result: Mapping[str, object],
) -> dict[str, object]:
    """Answer one question using only validated prediction-result fields."""

    if len(message) > 500:
        raise ValueError("message must contain at most 500 characters")
    context = context_from_prediction(result)
    intent = classify_intent(message)
    answer, grounded_fields = _answer(intent, context)
    suggestions = list(DEFAULT_SUGGESTIONS)
    if not context.requires_human_review:
        suggestions[1] = "Why was no review flag triggered?"
    return {
        "answer": answer,
        "intent": intent,
        "grounded_fields": grounded_fields,
        "suggested_questions": suggestions,
        "safety_notice": SAFETY_NOTICE,
    }
