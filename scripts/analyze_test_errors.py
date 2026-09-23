"""Analyse saved final-test predictions without re-running the model."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retinastage.error_analysis import analyse_prediction_errors  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/evaluation/final_test/test_predictions.csv",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts/evaluation/final_test_error_analysis",
    )
    arguments = parser.parse_args()
    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(arguments.predictions)
    analysed, summary = analyse_prediction_errors(records)
    analysed.to_csv(output / "analysed_predictions.csv", index=False)
    (output / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    pairs = pd.DataFrame(summary["top_confusion_pairs"])
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = pairs["true_grade_name"] + " → " + pairs["predicted_grade_name"]
    axes[0].barh(labels[::-1], pairs["count"][::-1])
    axes[0].set_title("Most Frequent Misclassification Pairs")
    axes[0].set_xlabel("Number of test images")
    distance_counts = analysed["grade_distance"].value_counts().sort_index()
    axes[1].bar(distance_counts.index.astype(str), distance_counts.values)
    axes[1].set_title("Prediction Error Distance")
    axes[1].set_xlabel("Absolute grade difference")
    axes[1].set_ylabel("Number of test images")
    figure.tight_layout()
    figure.savefig(output / "error_analysis.png", dpi=200)
    plt.close(figure)
    print("Quantitative error analysis: PASSED")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
