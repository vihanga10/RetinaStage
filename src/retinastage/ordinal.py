"""Fine-tuning and ordinal-learning utilities for RetinaStage."""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    recall_score,
)
from tensorflow import keras


@keras.utils.register_keras_serializable(package="RetinaStage")
class HybridOrdinalLoss(keras.losses.Loss):
    """Cross-entropy plus a penalty for distance between ordered grades."""

    def __init__(
        self,
        num_classes: int = 5,
        ordinal_weight: float = 0.5,
        name: str = "hybrid_ordinal_loss",
        **kwargs: object,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.ordinal_weight = ordinal_weight
        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(
            reduction="none"
        )

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        classification_loss = self.cross_entropy(y_true, y_pred)
        true_distribution = tf.one_hot(
            y_true,
            depth=self.num_classes,
            dtype=y_pred.dtype,
        )
        true_cdf = tf.cumsum(true_distribution, axis=-1)
        predicted_cdf = tf.cumsum(y_pred, axis=-1)
        ordinal_loss = tf.reduce_mean(
            tf.square(true_cdf - predicted_cdf),
            axis=-1,
        )
        return classification_loss + self.ordinal_weight * ordinal_loss

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "ordinal_weight": self.ordinal_weight,
            }
        )
        return config


@dataclass(frozen=True)
class FineTuningConfig:
    """Configuration shared by standard and ordinal fine-tuning."""

    backbone_name: str = "efficientnetb0"
    candidate_backbone_layers: int = 40
    learning_rate: float = 1e-5
    ordinal_weight: float = 0.5
    num_classes: int = 5


def configure_backbone_for_fine_tuning(
    model: keras.Model,
    config: FineTuningConfig,
) -> keras.Model:
    """Unfreeze the final backbone block while keeping batch norm frozen."""

    backbone = model.get_layer(config.backbone_name)
    backbone.trainable = True
    candidate_count = min(
        config.candidate_backbone_layers,
        len(backbone.layers),
    )

    for layer in backbone.layers[:-candidate_count]:
        layer.trainable = False

    for layer in backbone.layers[-candidate_count:]:
        layer.trainable = not isinstance(
            layer,
            keras.layers.BatchNormalization,
        )

    return backbone


def compile_fine_tuning_model(
    model: keras.Model,
    config: FineTuningConfig,
    *,
    ordinal: bool,
) -> None:
    """Compile a selectively unfrozen model for one experiment variant."""

    if ordinal:
        loss: keras.losses.Loss = HybridOrdinalLoss(
            num_classes=config.num_classes,
            ordinal_weight=config.ordinal_weight,
        )
    else:
        loss = keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.learning_rate
        ),
        loss=loss,
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(
                k=2,
                name="top_2_accuracy",
            ),
        ],
    )


class OrdinalValidationMetrics(keras.callbacks.Callback):
    """Add imbalance-aware and ordinal validation metrics each epoch."""

    def __init__(
        self,
        dataset: tf.data.Dataset,
        true_labels: np.ndarray,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.true_labels = np.asarray(true_labels, dtype=int)

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict[str, float] | None = None,
    ) -> None:
        del epoch
        logs = logs if logs is not None else {}
        probabilities = self.model.predict(self.dataset, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        recalls = recall_score(
            self.true_labels,
            predictions,
            labels=np.arange(5),
            average=None,
            zero_division=0,
        )
        logs["val_macro_f1"] = float(
            f1_score(
                self.true_labels,
                predictions,
                labels=np.arange(5),
                average="macro",
                zero_division=0,
            )
        )
        logs["val_severe_recall"] = float(recalls[3])
        logs["val_pdr_recall"] = float(recalls[4])
        logs["val_quadratic_kappa"] = float(
            cohen_kappa_score(
                self.true_labels,
                predictions,
                weights="quadratic",
            )
        )
        logs["val_grade_mae"] = float(
            mean_absolute_error(self.true_labels, predictions)
        )
        print(
            "\nOrdinal validation metrics — "
            f"macro F1: {logs['val_macro_f1']:.4f}, "
            f"Severe recall: {logs['val_severe_recall']:.4f}, "
            f"PDR recall: {logs['val_pdr_recall']:.4f}, "
            f"QWK: {logs['val_quadratic_kappa']:.4f}, "
            f"grade MAE: {logs['val_grade_mae']:.4f}"
        )
