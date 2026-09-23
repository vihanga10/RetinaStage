"""Run the guarded, one-time final-test evaluation with fixed parameters."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retinastage.calibration import (  # noqa: E402
    expected_calibration_error,
    multiclass_brier_score,
    probabilities_with_temperature,
)
from retinastage.data_pipeline import DatasetConfig, build_dataset  # noqa: E402
from retinastage.evaluation import save_evaluation  # noqa: E402


def main() -> None:
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
        default=PROJECT_ROOT
        / "artifacts/evaluation/calibration/calibration_summary.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts/evaluation/final_test",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    arguments = parser.parse_args()
    output = arguments.output_directory
    marker = output / "TEST_EVALUATION_COMPLETE.txt"
    if marker.exists():
        raise RuntimeError(
            "Final-test evaluation is locked because the completion marker "
            f"already exists: {marker}"
        )
    if not arguments.calibration_summary.is_file():
        raise FileNotFoundError(arguments.calibration_summary)
    fixed = json.loads(
        arguments.calibration_summary.read_text(encoding="utf-8")
    )
    temperature = float(fixed["temperature"])
    threshold = float(fixed["confidence_threshold"])

    config = DatasetConfig(
        manifest_path=PROJECT_ROOT / "data/splits/split_manifest.csv",
        image_directory=PROJECT_ROOT / "data/raw/train_images",
        batch_size=arguments.batch_size,
    )
    dataset, labels = build_dataset(config, "test", shuffle=False)
    records = pd.read_csv(config.manifest_path)
    records = records.loc[records["split"] == "test"].reset_index(drop=True)
    model = keras.models.load_model(arguments.checkpoint, compile=False)
    raw = model.predict(dataset, verbose=1)
    scaled = probabilities_with_temperature(raw, temperature)
    if not np.array_equal(raw.argmax(axis=1), scaled.argmax(axis=1)):
        raise AssertionError("Temperature scaling changed predicted classes")

    output.mkdir(parents=True, exist_ok=True)
    metrics = save_evaluation(
        output,
        records,
        labels,
        scaled,
        title_prefix="Final Test",
    )
    predictions_path = output / "predictions.csv"
    predictions = pd.read_csv(predictions_path)
    predictions["raw_confidence"] = raw.max(axis=1)
    predictions["calibrated_confidence"] = scaled.max(axis=1)
    predictions["uncertain"] = (
        predictions["calibrated_confidence"] < threshold
    )
    predictions.to_csv(output / "test_predictions.csv", index=False)
    predictions_path.unlink()

    raw_ece, raw_bins = expected_calibration_error(labels, raw)
    scaled_ece, scaled_bins = expected_calibration_error(labels, scaled)
    raw_bins.to_csv(output / "reliability_bins_raw.csv", index=False)
    scaled_bins.to_csv(output / "reliability_bins_scaled.csv", index=False)
    accepted = ~predictions["uncertain"].to_numpy()
    predictions_array = scaled.argmax(axis=1)
    summary = {
        **metrics,
        "test_records": int(len(labels)),
        "model_checkpoint": str(arguments.checkpoint),
        "temperature": temperature,
        "confidence_threshold": threshold,
        "raw_nll": float(log_loss(labels, raw, labels=np.arange(5))),
        "scaled_nll": float(log_loss(labels, scaled, labels=np.arange(5))),
        "raw_ece": raw_ece,
        "scaled_ece": scaled_ece,
        "raw_brier": multiclass_brier_score(labels, raw),
        "scaled_brier": multiclass_brier_score(labels, scaled),
        "accepted_records": int(accepted.sum()),
        "uncertain_records": int((~accepted).sum()),
        "coverage": float(accepted.mean()),
        "abstention_rate": float((~accepted).mean()),
        "selective_accuracy": float(
            (predictions_array[accepted] == labels[accepted]).mean()
        ),
        "uncertain_subset_accuracy": float(
            (predictions_array[~accepted] == labels[~accepted]).mean()
        ),
        "parameters_changed_using_test_results": False,
    }
    (output / "final_test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    marker.write_text(
        "Final one-time test evaluation completed. Do not tune model "
        "parameters using these results.\n",
        encoding="utf-8",
    )
    print("Final one-time test evaluation: PASSED")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
