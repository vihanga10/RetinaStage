"""Smoke-test the RetinaStage transfer-learning architecture."""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src"

if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from retinastage.data_pipeline import (  # noqa: E402
    DatasetConfig,
    build_dataset,
)
from retinastage.model import (  # noqa: E402
    ModelConfig,
    build_baseline_model,
    compile_baseline_model,
)


def main() -> None:
    """Check model construction and one inference pass locally."""

    np.random.seed(20260921)
    tf.random.set_seed(20260921)

    dataset_config = DatasetConfig(
        manifest_path=PROJECT_ROOT
        / "data/splits/split_manifest.csv",
        image_directory=PROJECT_ROOT
        / "data/raw/train_images",
        image_size=224,
        batch_size=2,
    )

    model_config = ModelConfig(image_size=224)

    train_dataset, _ = build_dataset(
        dataset_config,
        "train",
        shuffle=False,
    )
    images, labels = next(iter(train_dataset))

    # Random initialization avoids downloading ImageNet weights merely for
    # this architecture test. Actual training will use ImageNet weights.
    model, backbone = build_baseline_model(
        model_config,
        imagenet_weights=False,
    )
    compile_baseline_model(model, model_config)

    predictions = model(images, training=False)
    probability_sums = tf.reduce_sum(
        predictions,
        axis=1,
    ).numpy()

    assert predictions.shape == (2, 5)
    assert np.allclose(probability_sums, 1.0, atol=1e-5)
    assert backbone.trainable is False
    assert all(
        not layer.trainable for layer in backbone.layers
    )

    print("MODEL SMOKE TEST")
    print("Input shape:", images.shape)
    print("Labels:", labels.numpy().tolist())
    print("Output shape:", predictions.shape)
    print(
        "Probability sums:",
        np.round(probability_sums, 5).tolist(),
    )
    print("Backbone:", backbone.name)
    print("Backbone frozen:", not backbone.trainable)
    print("Total parameters:", f"{model.count_params():,}")
    print(
        "Trainable parameters:",
        f"{sum(np.prod(v.shape) for v in model.trainable_weights):,}",
    )
    print("Model smoke test: PASSED")


if __name__ == "__main__":
    main()
