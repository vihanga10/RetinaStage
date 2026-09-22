"""Train or smoke-test the RetinaStage EfficientNetB0 baseline."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src"

if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from retinastage.data_pipeline import DatasetConfig  # noqa: E402
from retinastage.model import ModelConfig  # noqa: E402
from retinastage.training import (  # noqa: E402
    TrainingConfig,
    train_frozen_baseline,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description="Train the RetinaStage baseline model."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one epoch on very few batches without ImageNet.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/training/baseline_stage1",
    )
    return parser.parse_args()


def main() -> None:
    """Configure and execute baseline training."""

    arguments = parse_arguments()

    if arguments.smoke_test:
        epochs = 1
        output_directory = (
            PROJECT_ROOT / "artifacts/training/smoke_test"
        )
        maximum_train_batches = 2
        maximum_validation_batches = 1
        imagenet_weights = False
    else:
        epochs = arguments.epochs
        output_directory = arguments.output_directory
        maximum_train_batches = None
        maximum_validation_batches = None
        imagenet_weights = True

    dataset_config = DatasetConfig(
        manifest_path=PROJECT_ROOT
        / "data/splits/split_manifest.csv",
        image_directory=PROJECT_ROOT
        / "data/raw/train_images",
        image_size=224,
        batch_size=arguments.batch_size,
        random_seed=20260921,
    )

    model_config = ModelConfig(
        image_size=224,
        learning_rate=1e-3,
    )

    training_config = TrainingConfig(
        output_directory=output_directory,
        epochs=epochs,
        random_seed=20260921,
    )

    print("TRAINING CONFIGURATION")
    print("Mode:", "smoke test" if arguments.smoke_test else "full")
    print("Epochs:", epochs)
    print("Batch size:", arguments.batch_size)
    print("ImageNet weights:", imagenet_weights)
    print("Output:", output_directory)

    train_frozen_baseline(
        dataset_config,
        model_config,
        training_config,
        imagenet_weights=imagenet_weights,
        maximum_train_batches=maximum_train_batches,
        maximum_validation_batches=maximum_validation_batches,
    )

    print("Training execution: PASSED")


if __name__ == "__main__":
    main()
