"""Fit temperature scaling using the calibration split only."""

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retinastage.calibration import calibrate_probabilities  # noqa: E402
from retinastage.data_pipeline import DatasetConfig, build_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/training/ordinal_stage2/best_macro_f1_model.keras",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts/evaluation/calibration",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    arguments = parser.parse_args()
    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)

    config = DatasetConfig(
        manifest_path=PROJECT_ROOT / "data/splits/split_manifest.csv",
        image_directory=PROJECT_ROOT / "data/raw/train_images",
        batch_size=arguments.batch_size,
    )
    dataset, labels = build_dataset(config, "calibration", shuffle=False)
    records = pd.read_csv(config.manifest_path)
    records = records.loc[
        records["split"] == "calibration"
    ].reset_index(drop=True)
    model = keras.models.load_model(arguments.checkpoint, compile=False)
    raw = model.predict(dataset, verbose=1)
    scaled, result, raw_bins, scaled_bins = calibrate_probabilities(
        labels, raw
    )

    predictions = records.copy()
    predictions["true_grade"] = labels
    predictions["raw_confidence"] = raw.max(axis=1)
    predictions["calibrated_confidence"] = scaled.max(axis=1)
    predictions.to_csv(output / "calibration_predictions.csv", index=False)
    raw_bins.to_csv(output / "reliability_bins_raw.csv", index=False)
    scaled_bins.to_csv(output / "reliability_bins_scaled.csv", index=False)
    summary = {
        **asdict(result),
        "checkpoint": str(arguments.checkpoint),
        "calibration_records": int(len(labels)),
        "test_set_used": False,
    }
    (output / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for table, label in [(raw_bins, "Raw"), (scaled_bins, "Scaled")]:
        nonempty = table[table["count"] > 0]
        axis.plot(
            nonempty["mean_confidence"],
            nonempty["accuracy"],
            marker="o",
            label=label,
        )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "reliability_diagram.png", dpi=200)
    plt.close(figure)
    print("Calibration execution: PASSED")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
