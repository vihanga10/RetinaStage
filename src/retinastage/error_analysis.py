"""Post-hoc descriptive analysis of locked final-test predictions."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from retinastage.evaluation import CLASS_NAMES


def calculate_binary_group_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    threshold: int,
    group_name: str,
) -> dict[str, float | int | str]:
    """Describe performance after grouping grades at a fixed threshold."""

    binary_true = (np.asarray(true_labels) >= threshold).astype(int)
    binary_predicted = (np.asarray(predictions) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        binary_true, binary_predicted, labels=[0, 1]
    ).ravel()

    def divide(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = divide(tp, tp + fp)
    sensitivity = divide(tp, tp + fn)
    return {
        "grouping": group_name,
        "positive_grade_threshold": threshold,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "sensitivity": sensitivity,
        "specificity": divide(tn, tn + fp),
        "precision": precision,
        "f1": divide(2 * precision * sensitivity, precision + sensitivity),
    }


def analyse_prediction_errors(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add error fields and return summary tables as serialisable records."""

    required = {"true_grade", "predicted_grade"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    analysed = records.copy()
    true_labels = analysed["true_grade"].astype(int).to_numpy()
    predictions = analysed["predicted_grade"].astype(int).to_numpy()
    analysed["correct"] = true_labels == predictions
    analysed["grade_distance"] = np.abs(true_labels - predictions)
    analysed["error_direction"] = np.select(
        [predictions < true_labels, predictions > true_labels],
        ["lower-grade prediction", "higher-grade prediction"],
        default="correct",
    )
    errors = analysed.loc[~analysed["correct"]].copy()
    errors["true_grade_name"] = errors["true_grade"].map(
        dict(enumerate(CLASS_NAMES))
    )
    errors["predicted_grade_name"] = errors["predicted_grade"].map(
        dict(enumerate(CLASS_NAMES))
    )
    pairs = (
        errors.groupby(["true_grade_name", "predicted_grade_name"], sort=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    binary = [
        calculate_binary_group_metrics(
            true_labels, predictions, 2, "Moderate-or-higher"
        ),
        calculate_binary_group_metrics(
            true_labels, predictions, 3, "Severe-or-higher"
        ),
    ]
    summary: dict[str, object] = {
        "records": int(len(analysed)),
        "correct_predictions": int(analysed["correct"].sum()),
        "incorrect_predictions": int((~analysed["correct"]).sum()),
        "adjacent_grade_errors": int((analysed["grade_distance"] == 1).sum()),
        "distant_grade_errors": int((analysed["grade_distance"] > 1).sum()),
        "lower_grade_predictions": int(
            (analysed["error_direction"] == "lower-grade prediction").sum()
        ),
        "higher_grade_predictions": int(
            (analysed["error_direction"] == "higher-grade prediction").sum()
        ),
        "top_confusion_pairs": pairs.head(10).to_dict(orient="records"),
        "binary_grade_groups": binary,
        "post_hoc_descriptive_only": True,
    }
    return analysed, summary
