"""Training utilities for the RetinaStage EfficientNet baseline."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from retinastage.data_pipeline import (
    DatasetConfig,
    build_dataset,
    calculate_class_weights,
)
from retinastage.model import (
    ModelConfig,
    build_baseline_model,
    compile_baseline_model,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and output settings for frozen-backbone training."""

    output_directory: Path
    epochs: int = 12
    early_stopping_patience: int = 4
    learning_rate_patience: int = 2
    learning_rate_reduction_factor: float = 0.3
    minimum_learning_rate: float = 1e-6
    random_seed: int = 20260921


def set_reproducibility(random_seed: int) -> None:
    """Set Python, NumPy and TensorFlow random seeds."""

    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        # Some TensorFlow/device combinations do not expose this option.
        pass


def calculate_sha256(path: Path) -> str:
    """Calculate a file hash for experiment provenance."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def create_callbacks(
    config: TrainingConfig,
) -> list[keras.callbacks.Callback]:
    """Create checkpointing and overfitting-prevention callbacks."""

    config.output_directory.mkdir(parents=True, exist_ok=True)

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(
                config.output_directory / "best_model.keras"
            ),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=config.learning_rate_reduction_factor,
            patience=config.learning_rate_patience,
            min_lr=config.minimum_learning_rate,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            filename=str(
                config.output_directory / "training_history.csv"
            ),
            append=False,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]


def save_model_summary(
    model: keras.Model,
    output_path: Path,
) -> None:
    """Save the architecture summary as reproducible evidence."""

    lines: list[str] = []
    model.summary(print_fn=lines.append)
    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def train_frozen_baseline(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    imagenet_weights: bool = True,
    maximum_train_batches: int | None = None,
    maximum_validation_batches: int | None = None,
) -> tuple[keras.Model, keras.callbacks.History]:
    """Train the frozen EfficientNetB0 baseline.

    Batch limits are intended only for local smoke testing. Full experiments
    must leave both limits as None.
    """

    set_reproducibility(training_config.random_seed)

    train_dataset, train_labels = build_dataset(
        dataset_config,
        "train",
        shuffle=True,
    )
    validation_dataset, validation_labels = build_dataset(
        dataset_config,
        "validation",
        shuffle=False,
    )

    if maximum_train_batches is not None:
        train_dataset = train_dataset.take(maximum_train_batches)

    if maximum_validation_batches is not None:
        validation_dataset = validation_dataset.take(
            maximum_validation_batches
        )

    class_weights = calculate_class_weights(train_labels)

    model, backbone = build_baseline_model(
        model_config,
        imagenet_weights=imagenet_weights,
    )
    compile_baseline_model(model, model_config)

    output_directory = training_config.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    save_model_summary(
        model,
        output_directory / "model_summary.txt",
    )

    experiment_metadata = {
        "dataset_config": {
            **asdict(dataset_config),
            "manifest_path": str(dataset_config.manifest_path),
            "image_directory": str(
                dataset_config.image_directory
            ),
        },
        "model_config": asdict(model_config),
        "training_config": {
            **asdict(training_config),
            "output_directory": str(output_directory),
        },
        "manifest_sha256": calculate_sha256(
            dataset_config.manifest_path
        ),
        "training_records": int(len(train_labels)),
        "validation_records": int(len(validation_labels)),
        "class_weights": {
            str(key): value
            for key, value in class_weights.items()
        },
        "backbone": backbone.name,
        "backbone_initially_frozen": not backbone.trainable,
        "imagenet_weights": imagenet_weights,
        "smoke_test_batch_limit": maximum_train_batches,
    }

    (
        output_directory / "experiment_config.json"
    ).write_text(
        json.dumps(experiment_metadata, indent=2),
        encoding="utf-8",
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=training_config.epochs,
        class_weight=class_weights,
        callbacks=create_callbacks(training_config),
        shuffle=False,
        verbose=2,
    )

    serializable_history = {
        metric: [float(value) for value in values]
        for metric, values in history.history.items()
    }

    (output_directory / "history.json").write_text(
        json.dumps(serializable_history, indent=2),
        encoding="utf-8",
    )

    model.save(output_directory / "final_model.keras")

    return model, history
