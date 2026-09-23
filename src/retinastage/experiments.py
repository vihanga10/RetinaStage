"""Reusable training workflows for RetinaStage experiment variants."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from retinastage.data_pipeline import (
    DatasetConfig,
    build_dataset,
    calculate_class_weights,
)
from retinastage.ordinal import (
    FineTuningConfig,
    OrdinalValidationMetrics,
    compile_fine_tuning_model,
    configure_backbone_for_fine_tuning,
)
from retinastage.training import save_model_summary, set_reproducibility


@dataclass(frozen=True)
class ExperimentTrainingConfig:
    output_directory: Path
    epochs: int = 8
    early_stopping_patience: int = 3
    learning_rate_patience: int = 2
    learning_rate_reduction_factor: float = 0.3
    minimum_learning_rate: float = 1e-7
    random_seed: int = 20260921


def train_fine_tuning_experiment(
    baseline_checkpoint: Path,
    dataset_config: DatasetConfig,
    fine_tuning_config: FineTuningConfig,
    training_config: ExperimentTrainingConfig,
    *,
    ordinal: bool,
) -> tuple[keras.Model, keras.callbacks.History]:
    """Train standard or hybrid-ordinal fine-tuning from one checkpoint."""

    if not baseline_checkpoint.is_file():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {baseline_checkpoint}"
        )
    set_reproducibility(training_config.random_seed)
    output = training_config.output_directory
    output.mkdir(parents=True, exist_ok=True)

    train_dataset, training_labels = build_dataset(
        dataset_config, "train", shuffle=True
    )
    validation_dataset, validation_labels = build_dataset(
        dataset_config, "validation", shuffle=False
    )
    class_weights = calculate_class_weights(training_labels)

    model = keras.models.load_model(baseline_checkpoint, compile=False)
    backbone = configure_backbone_for_fine_tuning(
        model, fine_tuning_config
    )
    compile_fine_tuning_model(
        model,
        fine_tuning_config,
        ordinal=ordinal,
    )

    callbacks: list[keras.callbacks.Callback] = []
    if ordinal:
        callbacks.append(
            OrdinalValidationMetrics(
                validation_dataset,
                validation_labels,
            )
        )
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(output / "best_macro_f1_model.keras"),
                monitor="val_macro_f1",
                mode="max",
                save_best_only=True,
                verbose=1,
            )
        )

    val_loss_name = (
        "best_val_loss_model.keras" if ordinal else "best_model.keras"
    )
    # In ordinal training, the custom metric callback must precede checkpoints.
    callbacks.insert(
        1 if ordinal else 0,
        keras.callbacks.ModelCheckpoint(
            filepath=str(output / val_loss_name),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    )
    callbacks.extend(
        [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=training_config.learning_rate_reduction_factor,
                patience=training_config.learning_rate_patience,
                min_lr=training_config.minimum_learning_rate,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=training_config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.CSVLogger(
                str(output / "training_history.csv")
            ),
            keras.callbacks.TerminateOnNaN(),
        ]
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=training_config.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(output / "final_restored_model.keras")
    save_model_summary(model, output / "model_summary.txt")
    serializable_history = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    (output / "history.json").write_text(
        json.dumps(serializable_history, indent=2), encoding="utf-8"
    )
    metadata = {
        "experiment": "ordinal_stage2" if ordinal else "baseline_stage2_finetuned",
        "starting_checkpoint": str(baseline_checkpoint),
        "dataset": {
            **asdict(dataset_config),
            "manifest_path": str(dataset_config.manifest_path),
            "image_directory": str(dataset_config.image_directory),
        },
        "fine_tuning": asdict(fine_tuning_config),
        "training": {
            **asdict(training_config),
            "output_directory": str(output),
        },
        "training_records": int(len(training_labels)),
        "validation_records": int(len(validation_labels)),
        "class_weights": {
            str(key): float(value) for key, value in class_weights.items()
        },
        "trainable_backbone_layers": int(
            np.sum([layer.trainable for layer in backbone.layers])
        ),
        "validation_selection_only": True,
        "test_set_used": False,
    }
    (output / "experiment_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return model, history
