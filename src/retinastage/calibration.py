"""Post-hoc probability calibration for a fixed RetinaStage model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss


def probabilities_with_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply temperature scaling to already normalised probabilities."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probabilities = np.asarray(probabilities, dtype=float)
    clipped = np.clip(probabilities, 1e-7, 1.0)
    logits = np.log(clipped) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def fit_temperature(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Fit one scalar temperature by minimising calibration-set NLL."""

    labels = np.asarray(true_labels, dtype=int)

    def objective(log_temperature: float) -> float:
        scaled = probabilities_with_temperature(
            probabilities, np.exp(log_temperature)
        )
        return float(log_loss(labels, scaled, labels=np.arange(5)))

    result = minimize_scalar(
        objective,
        bounds=(-3.0, 3.0),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimisation failed: {result.message}")
    return float(np.exp(result.x))


def expected_calibration_error(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    """Calculate top-label ECE and return the reliability-bin table."""

    labels = np.asarray(true_labels, dtype=int)
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float | int]] = []
    ece = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == 0:
            selected = (confidence >= lower) & (confidence <= upper)
        else:
            selected = (confidence > lower) & (confidence <= upper)
        count = int(selected.sum())
        mean_confidence = float(confidence[selected].mean()) if count else 0.0
        accuracy = (
            float((predictions[selected] == labels[selected]).mean())
            if count
            else 0.0
        )
        gap = abs(accuracy - mean_confidence)
        ece += (count / len(labels)) * gap
        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )
    return float(ece), pd.DataFrame(rows)


def multiclass_brier_score(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    labels = np.asarray(true_labels, dtype=int)
    targets = np.eye(probabilities.shape[1])[labels]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


@dataclass(frozen=True)
class CalibrationResult:
    temperature: float
    confidence_threshold: float
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float


def calibrate_probabilities(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    uncertainty_quantile: float = 0.20,
) -> tuple[np.ndarray, CalibrationResult, pd.DataFrame, pd.DataFrame]:
    """Fit temperature scaling and a calibration-only abstention threshold."""

    temperature = fit_temperature(true_labels, probabilities)
    scaled = probabilities_with_temperature(probabilities, temperature)
    ece_before, raw_bins = expected_calibration_error(
        true_labels, probabilities
    )
    ece_after, scaled_bins = expected_calibration_error(
        true_labels, scaled
    )
    result = CalibrationResult(
        temperature=temperature,
        confidence_threshold=float(
            np.quantile(scaled.max(axis=1), uncertainty_quantile)
        ),
        nll_before=float(
            log_loss(true_labels, probabilities, labels=np.arange(5))
        ),
        nll_after=float(log_loss(true_labels, scaled, labels=np.arange(5))),
        ece_before=ece_before,
        ece_after=ece_after,
        brier_before=multiclass_brier_score(true_labels, probabilities),
        brier_after=multiclass_brier_score(true_labels, scaled),
    )
    return scaled, result, raw_bins, scaled_bins
