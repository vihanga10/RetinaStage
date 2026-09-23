"""Classification and ordinal evaluation helpers."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)


CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]


def calculate_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calculate class-imbalance and grade-distance aware metrics."""

    y_true = np.asarray(true_labels, dtype=int)
    y_pred = np.asarray(predictions, dtype=int)
    recalls = recall_score(
        y_true,
        y_pred,
        labels=np.arange(5),
        average=None,
        zero_division=0,
    )
    distances = np.abs(y_true - y_pred)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "quadratic_weighted_kappa": cohen_kappa_score(
            y_true, y_pred, weights="quadratic"
        ),
        "grade_mae": mean_absolute_error(y_true, y_pred),
        "within_one_grade_accuracy": np.mean(distances <= 1),
        "distant_error_rate": np.mean(distances > 1),
    }
    for class_id, class_name in enumerate(
        ["no_dr", "mild", "moderate", "severe", "proliferative_dr"]
    ):
        metrics[f"{class_name}_recall"] = recalls[class_id]
    return {key: float(value) for key, value in metrics.items()}


def classification_report_text(
    true_labels: np.ndarray,
    predictions: np.ndarray,
) -> str:
    return classification_report(
        true_labels,
        predictions,
        labels=np.arange(5),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )


def plot_confusion_matrices(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
    *,
    title_prefix: str,
) -> None:
    """Save count and row-normalised confusion matrices together."""

    counts = confusion_matrix(
        true_labels, predictions, labels=np.arange(5)
    )
    row_totals = counts.sum(axis=1, keepdims=True)
    normalised = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals != 0,
    )
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(
        counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0],
    )
    sns.heatmap(
        normalised,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="Greens",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1],
    )
    for axis, suffix in zip(axes, ["Counts", "Row Normalized"]):
        axis.set_title(f"{title_prefix} — {suffix}")
        axis.set_xlabel("Predicted grade")
        axis.set_ylabel("True grade")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_evaluation(
    output_directory: Path,
    records: pd.DataFrame,
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    title_prefix: str,
) -> dict[str, float]:
    """Persist predictions, metrics, report, and confusion matrices."""

    output_directory.mkdir(parents=True, exist_ok=True)
    predictions = np.argmax(probabilities, axis=1)
    metrics = calculate_metrics(true_labels, predictions)
    prediction_records = records.reset_index(drop=True).copy()
    prediction_records["true_grade"] = np.asarray(true_labels, dtype=int)
    prediction_records["predicted_grade"] = predictions
    prediction_records["confidence"] = probabilities.max(axis=1)
    for class_id, class_name in enumerate(
        ["no_dr", "mild", "moderate", "severe", "pdr"]
    ):
        prediction_records[f"probability_{class_name}"] = probabilities[
            :, class_id
        ]
    prediction_records.to_csv(
        output_directory / "predictions.csv", index=False
    )
    (output_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_directory / "classification_report.txt").write_text(
        classification_report_text(true_labels, predictions),
        encoding="utf-8",
    )
    plot_confusion_matrices(
        true_labels,
        predictions,
        output_directory / "confusion_matrices.png",
        title_prefix=title_prefix,
    )
    return metrics
