"""Generate deterministic Grad-CAM examples for the locked final model."""

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

from retinastage.evaluation import CLASS_NAMES  # noqa: E402
from retinastage.explainability import (  # noqa: E402
    create_gradcam_heatmap,
    create_overlay,
    load_model_image,
)
from retinastage.training import calculate_sha256  # noqa: E402


def select_samples(records: pd.DataFrame) -> pd.DataFrame:
    """Choose one representative correct and one major error per grade."""

    selected: list[pd.Series] = []
    working = records.copy()
    working["grade_distance"] = abs(
        working["true_grade"] - working["predicted_grade"]
    )
    working["correct"] = working["true_grade"] == working["predicted_grade"]
    for grade in range(5):
        grade_rows = working.loc[working["true_grade"] == grade]
        correct = grade_rows.loc[grade_rows["correct"]].copy()
        if not correct.empty:
            median = correct["confidence"].median()
            correct["median_distance"] = abs(correct["confidence"] - median)
            choice = correct.sort_values(
                ["median_distance", "image_id"]
            ).iloc[0].copy()
            choice["selection_reason"] = "correct_nearest_class_median_confidence"
            selected.append(choice)
        errors = grade_rows.loc[~grade_rows["correct"]].copy()
        if not errors.empty:
            choice = errors.sort_values(
                ["grade_distance", "confidence", "image_id"],
                ascending=[False, False, True],
            ).iloc[0].copy()
            choice["selection_reason"] = "largest_grade_distance_error"
            selected.append(choice)
    return pd.DataFrame(selected).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/training/ordinal_stage2/best_macro_f1_model.keras",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/evaluation/final_test/test_predictions.csv",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts/explainability/gradcam_final_model",
    )
    arguments = parser.parse_args()
    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(arguments.predictions)
    if "confidence" not in records and "calibrated_confidence" in records:
        records["confidence"] = records["calibrated_confidence"]
    samples = select_samples(records)
    model = keras.models.load_model(arguments.checkpoint, compile=False)
    gallery: list[tuple[np.ndarray, str]] = []
    saved: list[dict[str, object]] = []
    for number, row in samples.iterrows():
        image_path = PROJECT_ROOT / "data/raw/train_images" / row["image_filename"]
        image, batch = load_model_image(image_path)
        heatmap = create_gradcam_heatmap(
            model, batch, int(row["predicted_grade"])
        )
        resized, overlay = create_overlay(image, heatmap)
        status = "Correct" if bool(row["correct"]) else "Error"
        title = (
            f"{row['image_id']} | {status}\n"
            f"True: {CLASS_NAMES[int(row['true_grade'])]} | "
            f"Predicted: {CLASS_NAMES[int(row['predicted_grade'])]}\n"
            f"Confidence: {float(row['confidence']):.3f}"
        )
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(np.clip(image / 255.0, 0, 1))
        axes[0].set_title("Model input")
        axes[1].imshow(resized, cmap="turbo", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM heatmap")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(title, fontsize=10)
        figure.tight_layout()
        filename = f"{number + 1:02d}_{row['image_id']}_gradcam.png"
        figure.savefig(output / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)
        gallery.append((overlay, title))
        saved.append(
            {
                "image_id": row["image_id"],
                "true_grade": int(row["true_grade"]),
                "predicted_grade": int(row["predicted_grade"]),
                "confidence": float(row["confidence"]),
                "selection_reason": row["selection_reason"],
                "file": filename,
            }
        )

    figure, axes = plt.subplots(5, 2, figsize=(12, 25))
    for axis, (overlay, title) in zip(axes.flat, gallery):
        axis.imshow(overlay)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    for axis in axes.flat[len(gallery):]:
        axis.axis("off")
    figure.suptitle("RetinaStage Grad-CAM — Fixed Final Model", fontsize=16)
    figure.tight_layout()
    figure.savefig(output / "gradcam_gallery.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    methodology = {
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": calculate_sha256(arguments.checkpoint),
        "predictions": str(arguments.predictions),
        "generated_explanations": len(saved),
        "samples": saved,
        "model_parameters_changed": False,
        "interpretation_warning": (
            "Grad-CAM is qualitative evidence of model attention, not a "
            "clinical lesion-localisation validation."
        ),
    }
    (output / "gradcam_methodology.json").write_text(
        json.dumps(methodology, indent=2), encoding="utf-8"
    )
    print("Grad-CAM execution: PASSED")
    print("Generated explanations:", len(saved))


if __name__ == "__main__":
    main()
