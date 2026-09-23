"""Evaluate one fixed model checkpoint on validation data only."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retinastage.data_pipeline import DatasetConfig, build_dataset  # noqa: E402
from retinastage.evaluation import save_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )
    parser.add_argument("--title", default="Validation")
    parser.add_argument("--batch-size", type=int, default=16)
    arguments = parser.parse_args()

    config = DatasetConfig(
        manifest_path=PROJECT_ROOT / "data/splits/split_manifest.csv",
        image_directory=PROJECT_ROOT / "data/raw/train_images",
        batch_size=arguments.batch_size,
    )
    dataset, labels = build_dataset(
        config, "validation", shuffle=False
    )
    records = pd.read_csv(config.manifest_path)
    records = records.loc[
        records["split"] == "validation"
    ].reset_index(drop=True)
    model = keras.models.load_model(arguments.checkpoint, compile=False)
    probabilities = model.predict(dataset, verbose=1)
    metrics = save_evaluation(
        arguments.output_directory,
        records,
        labels,
        probabilities,
        title_prefix=arguments.title,
    )
    print("Validation checkpoint evaluation: PASSED")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
