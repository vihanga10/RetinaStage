"""Run a small local test of the RetinaStage data pipeline."""

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
    calculate_class_weights,
)


def main() -> None:
    """Validate loading, preprocessing, labels and class weights."""

    # Fixed seeds make this test and later experiments reproducible.
    np.random.seed(20260921)
    tf.random.set_seed(20260921)

    config = DatasetConfig(
        manifest_path=PROJECT_ROOT
        / "data/splits/split_manifest.csv",
        image_directory=PROJECT_ROOT
        / "data/raw/train_images",
        image_size=224,
        batch_size=4,
        random_seed=20260921,
    )

    train_dataset, train_labels = build_dataset(
        config,
        "train",
        shuffle=False,
    )

    images, labels = next(iter(train_dataset))
    class_weights = calculate_class_weights(train_labels)

    assert images.shape == (4, 224, 224, 3)
    assert images.dtype == tf.float32
    assert labels.shape == (4,)
    assert set(labels.numpy()).issubset(set(range(5)))
    assert float(tf.reduce_min(images)) >= 0.0
    assert float(tf.reduce_max(images)) <= 255.0
    assert len(class_weights) == 5

    print("DATA PIPELINE SMOKE TEST")
    print("Training records:", len(train_labels))
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("Image dtype:", images.dtype.name)
    print(
        "Pixel range:",
        round(float(tf.reduce_min(images)), 2),
        "to",
        round(float(tf.reduce_max(images)), 2),
    )
    print("Batch labels:", labels.numpy().tolist())
    print(
        "Class weights:",
        {
            key: round(value, 4)
            for key, value in class_weights.items()
        },
    )
    print("Data pipeline smoke test: PASSED")


if __name__ == "__main__":
    main()
