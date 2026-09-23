"""Evaluate fixed-model robustness on validation data only."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retinastage.calibration import probabilities_with_temperature  # noqa: E402
from retinastage.data_pipeline import DatasetConfig, build_dataset  # noqa: E402
from retinastage.robustness import (  # noqa: E402
    ROBUSTNESS_CONDITIONS,
    build_robustness_dataset,
    calculate_robustness_metrics,
)
from retinastage.training import calculate_sha256  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/training/ordinal_stage2/best_macro_f1_model.keras",
    )
    parser.add_argument(
        "--calibration-summary",
        type=Path,
        default=None,
        help=(
            "Path to calibration_summary.json. When omitted, known project "
            "artifact locations are checked automatically."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts/evaluation/robustness_validation",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def create_figure(results: pd.DataFrame, output_path: Path) -> None:
    """Save a compact comparison of performance and stability."""

    labels = results["condition_label"].tolist()
    positions = np.arange(len(results))
    colours = ["#2e7d32"] + ["#547aa5"] * (len(results) - 1)
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].bar(positions - 0.25, results["accuracy"], 0.25, label="Accuracy")
    axes[0, 0].bar(positions, results["macro_f1"], 0.25, label="Macro F1")
    axes[0, 0].bar(
        positions + 0.25,
        results["quadratic_weighted_kappa"],
        0.25,
        label="QWK",
    )
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("Classification and ordinal performance")
    axes[0, 0].legend()

    axes[0, 1].bar(positions, results["grade_mae"], color=colours)
    axes[0, 1].set_title("Grade mean absolute error (lower is better)")

    axes[1, 0].bar(
        positions, results["prediction_consistency"], color=colours
    )
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Prediction consistency with clean images")

    axes[1, 1].bar(
        positions, results["uncertainty_rate"], color=colours
    )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Fixed-policy uncertainty rate")

    for axis in axes.flat:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("RetinaStage Validation Robustness Analysis", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    calibration_summary_path = arguments.calibration_summary
    if calibration_summary_path is None:
        candidates = [
            PROJECT_ROOT
            / "artifacts/calibration/selected_ordinal_model/"
            "calibration_summary.json",
            PROJECT_ROOT
            / "artifacts/evaluation/calibration/calibration_summary.json",
        ]
        calibration_summary_path = next(
            (path for path in candidates if path.is_file()),
            candidates[0],
        )
    if not arguments.checkpoint.is_file():
        raise FileNotFoundError(
            f"Selected checkpoint not found: {arguments.checkpoint}"
        )
    if not calibration_summary_path.is_file():
        raise FileNotFoundError(
            "Calibration summary not found: "
            f"{calibration_summary_path}"
        )

    fixed_policy = json.loads(
        calibration_summary_path.read_text(encoding="utf-8")
    )
    temperature = float(fixed_policy["temperature"])
    if "confidence_threshold" in fixed_policy:
        confidence_threshold = float(fixed_policy["confidence_threshold"])
    else:
        confidence_threshold = float(
            fixed_policy["uncertainty_policy"]["confidence_threshold"]
        )
    dataset_config = DatasetConfig(
        manifest_path=PROJECT_ROOT / "data/splits/split_manifest.csv",
        image_directory=PROJECT_ROOT / "data/raw/train_images",
        batch_size=arguments.batch_size,
        random_seed=20260921,
    )
    validation_dataset, true_labels = build_dataset(
        dataset_config,
        "validation",
        shuffle=False,
    )
    # Cache preprocessed validation images once; perturbations are then applied
    # to an identical ordered image set without repeating disk preprocessing.
    cached_dataset = validation_dataset.cache()
    validation_records = pd.read_csv(dataset_config.manifest_path)
    validation_records = validation_records.loc[
        validation_records["split"] == "validation"
    ].reset_index(drop=True)
    if len(validation_records) != len(true_labels):
        raise AssertionError("Validation manifest and label counts differ")

    model = keras.models.load_model(arguments.checkpoint, compile=False)
    print("VALIDATION ROBUSTNESS ANALYSIS")
    print("Checkpoint:", arguments.checkpoint)
    print("Validation records:", len(true_labels))
    print("Temperature (fixed):", temperature)
    print("Confidence threshold (fixed):", confidence_threshold)
    print("Test set used: False")

    print("\nEvaluating: Clean validation images")
    clean_raw = model.predict(cached_dataset, verbose=1)
    clean_scaled = probabilities_with_temperature(clean_raw, temperature)
    clean_predictions = np.argmax(clean_scaled, axis=1)
    prediction_table = validation_records[
        ["image_id", "image_filename", "diagnosis", "split"]
    ].copy()
    prediction_table["true_grade"] = true_labels

    result_rows: list[dict[str, object]] = []
    for condition in ROBUSTNESS_CONDITIONS:
        if condition.name == "clean":
            raw_probabilities = clean_raw
        else:
            print(f"\nEvaluating: {condition.label}")
            condition_dataset = build_robustness_dataset(
                cached_dataset,
                condition.name,
                batch_size=arguments.batch_size,
                random_seed=20260921,
            )
            raw_probabilities = model.predict(
                condition_dataset,
                verbose=1,
            )

        metrics, predictions, confidence = calculate_robustness_metrics(
            true_labels,
            raw_probabilities,
            clean_predictions,
            clean_scaled,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
        )
        result_rows.append(
            {
                "condition": condition.name,
                "condition_label": condition.label,
                "parameter": condition.parameter,
                "value": condition.value,
                **metrics,
            }
        )
        prediction_table[f"{condition.name}_prediction"] = predictions
        prediction_table[f"{condition.name}_confidence"] = confidence
        prediction_table[f"{condition.name}_uncertain"] = (
            confidence < confidence_threshold
        )

    results = pd.DataFrame(result_rows)
    clean = results.iloc[0]
    results["accuracy_change_from_clean"] = results["accuracy"] - clean["accuracy"]
    results["macro_f1_change_from_clean"] = results["macro_f1"] - clean["macro_f1"]
    results["qwk_change_from_clean"] = (
        results["quadratic_weighted_kappa"]
        - clean["quadratic_weighted_kappa"]
    )
    results["grade_mae_change_from_clean"] = (
        results["grade_mae"] - clean["grade_mae"]
    )

    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "robustness_metrics.csv", index=False)
    prediction_table.to_csv(
        output / "robustness_predictions.csv", index=False
    )
    create_figure(results, output / "robustness_comparison.png")
    summary = {
        "analysis": "validation_robustness",
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": calculate_sha256(arguments.checkpoint),
        "validation_records": int(len(true_labels)),
        "temperature": temperature,
        "confidence_threshold": confidence_threshold,
        "calibration_summary": str(calibration_summary_path),
        "random_seed": 20260921,
        "conditions": [
            condition.to_dict() for condition in ROBUSTNESS_CONDITIONS
        ],
        "results": results.to_dict(orient="records"),
        "model_parameters_changed": False,
        "threshold_changed": False,
        "test_set_used": False,
        "interpretation_warning": (
            "These fixed synthetic perturbations are sensitivity probes, not "
            "clinical image-quality standards or external validation."
        ),
    }
    (output / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "ROBUSTNESS_ANALYSIS_COMPLETE.txt").write_text(
        "Validation-only robustness analysis completed without retraining, "
        "test-set use, or policy changes.\n",
        encoding="utf-8",
    )

    display_columns = [
        "condition_label",
        "accuracy",
        "macro_f1",
        "quadratic_weighted_kappa",
        "grade_mae",
        "prediction_consistency",
        "uncertainty_rate",
    ]
    print("\nROBUSTNESS RESULTS")
    print(results[display_columns].to_string(index=False))
    print("\nValidation robustness analysis: PASSED")
    print("Saved to:", output)


if __name__ == "__main__":
    main()
