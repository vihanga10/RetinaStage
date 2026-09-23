"""Fine-tune the final EfficientNetB0 block with cross-entropy."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retinastage.data_pipeline import DatasetConfig  # noqa: E402
from retinastage.experiments import (  # noqa: E402
    ExperimentTrainingConfig,
    train_fine_tuning_experiment,
)
from retinastage.ordinal import FineTuningConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/baseline_stage1/best_model.keras",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    arguments = parser.parse_args()
    train_fine_tuning_experiment(
        arguments.checkpoint,
        DatasetConfig(
            manifest_path=PROJECT_ROOT / "data/splits/split_manifest.csv",
            image_directory=PROJECT_ROOT / "data/raw/train_images",
            batch_size=arguments.batch_size,
        ),
        FineTuningConfig(),
        ExperimentTrainingConfig(
            output_directory=PROJECT_ROOT
            / "artifacts/training/baseline_stage2_finetuned",
            epochs=arguments.epochs,
        ),
        ordinal=False,
    )
    print("Fine-tuning execution: PASSED")


if __name__ == "__main__":
    main()
