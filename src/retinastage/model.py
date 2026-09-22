"""Transfer-learning models used by RetinaStage."""

from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


NUM_CLASSES = 5


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the baseline EfficientNetB0 classifier."""

    image_size: int = 224
    num_classes: int = NUM_CLASSES
    dropout_rate: float = 0.35
    dense_units: int = 128
    learning_rate: float = 1e-3
    l2_strength: float = 1e-4


def create_training_augmentation() -> keras.Sequential:
    """Create medically reasonable training-only augmentation.

    Horizontal flipping, mild rotation, zoom and contrast variation simulate
    acquisition variability without severely altering retinal anatomy.
    """

    return keras.Sequential(
        [
            layers.RandomFlip(
                mode="horizontal",
                seed=20260921,
            ),
            layers.RandomRotation(
                factor=0.05,
                fill_mode="reflect",
                seed=20260922,
            ),
            layers.RandomZoom(
                height_factor=(-0.10, 0.10),
                width_factor=(-0.10, 0.10),
                fill_mode="reflect",
                seed=20260923,
            ),
            layers.RandomContrast(
                factor=0.10,
                seed=20260924,
            ),
        ],
        name="training_augmentation",
    )


def build_baseline_model(
    config: ModelConfig,
    *,
    imagenet_weights: bool = True,
) -> tuple[keras.Model, keras.Model]:
    """Build a frozen-backbone EfficientNetB0 transfer-learning model.

    Returns both the complete classifier and backbone so the backbone can be
    selectively unfrozen during the later fine-tuning phase.
    """

    inputs = keras.Input(
        shape=(config.image_size, config.image_size, 3),
        name="retinal_image",
    )

    # Keras augmentation layers operate only when training=True.
    augmented = create_training_augmentation()(inputs)

    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet" if imagenet_weights else None,
        input_shape=(
            config.image_size,
            config.image_size,
            3,
        ),
    )
    backbone.trainable = False

    # training=False keeps frozen BatchNormalization statistics unchanged.
    features = backbone(augmented, training=False)
    features = layers.GlobalAveragePooling2D(
        name="global_average_pooling",
    )(features)
    features = layers.BatchNormalization(
        name="classifier_batch_normalization",
    )(features)
    features = layers.Dropout(
        config.dropout_rate,
        name="classifier_dropout_1",
    )(features)
    features = layers.Dense(
        config.dense_units,
        activation="swish",
        kernel_regularizer=regularizers.l2(
            config.l2_strength
        ),
        name="classifier_dense",
    )(features)
    features = layers.Dropout(
        0.20,
        name="classifier_dropout_2",
    )(features)

    outputs = layers.Dense(
        config.num_classes,
        activation="softmax",
        name="diagnosis_probabilities",
    )(features)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="retinastage_efficientnetb0",
    )

    return model, backbone


def compile_baseline_model(
    model: keras.Model,
    config: ModelConfig,
) -> None:
    """Compile the baseline using weighted sparse cross-entropy training."""

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.learning_rate
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            ),
            keras.metrics.SparseTopKCategoricalAccuracy(
                k=2,
                name="top_2_accuracy",
            ),
        ],
    )
