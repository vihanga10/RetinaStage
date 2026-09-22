"""TensorFlow dataset pipeline for RetinaStage.

The pipeline reads the frozen split manifest, applies deterministic retinal
preprocessing and produces reproducible TensorFlow datasets. Augmentation is
deliberately excluded here because it must be applied only during training.
"""

from dataclasses import dataclass
import csv
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from retinastage.preprocessing import (
    PreprocessingConfig,
    preprocess_image,
)


NUM_CLASSES = 5
VALID_SPLITS = {"train", "validation", "calibration", "test"}


@dataclass(frozen=True)
class DatasetConfig:
    """Paths and reproducibility settings for dataset construction."""

    manifest_path: Path
    image_directory: Path
    image_size: int = 224
    batch_size: int = 16
    random_seed: int = 20260921
    use_clahe: bool = False
    use_unsharp_mask: bool = False


def read_split_records(
    config: DatasetConfig,
    split_name: str,
) -> tuple[list[str], np.ndarray]:
    """Read filenames and labels for one frozen dataset partition."""

    if split_name not in VALID_SPLITS:
        raise ValueError(
            f"Unknown split {split_name!r}; expected one of "
            f"{sorted(VALID_SPLITS)}"
        )

    with config.manifest_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    required_columns = {
        "image_id",
        "image_filename",
        "diagnosis",
        "split",
    }
    if not rows:
        raise ValueError("Split manifest is empty")

    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise ValueError(
            f"Split manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    selected_rows = [
        row for row in rows if row["split"] == split_name
    ]
    if not selected_rows:
        raise ValueError(f"No records found for split {split_name!r}")

    image_paths = [
        str(config.image_directory / row["image_filename"])
        for row in selected_rows
    ]
    labels = np.asarray(
        [int(row["diagnosis"]) for row in selected_rows],
        dtype=np.int32,
    )

    invalid_labels = sorted(
        set(labels.tolist()) - set(range(NUM_CLASSES))
    )
    if invalid_labels:
        raise ValueError(
            f"Invalid diagnosis labels found: {invalid_labels}"
        )

    missing_images = [
        image_path
        for image_path in image_paths
        if not Path(image_path).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} split images are missing. "
            f"First missing image: {missing_images[0]}"
        )

    return image_paths, labels


def _decode_path(path_value: object) -> str:
    """Convert a value supplied by tf.numpy_function into a path."""

    if isinstance(path_value, np.ndarray):
        path_value = path_value.item()

    if isinstance(path_value, bytes):
        return path_value.decode("utf-8")

    return str(path_value)


def _load_preprocessed_image(
    path_value: object,
    preprocessing_config: PreprocessingConfig,
) -> np.ndarray:
    """Load one image with OpenCV and reuse project preprocessing."""

    image_path = _decode_path(path_value)
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError(f"OpenCV could not decode {image_path}")

    processed_bgr = preprocess_image(
        image_bgr,
        preprocessing_config,
    )

    # TensorFlow and ImageNet models conventionally use RGB ordering.
    processed_rgb = cv2.cvtColor(
        processed_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # EfficientNet's Keras implementation accepts values in [0, 255].
    return processed_rgb.astype(np.float32)


def build_dataset(
    config: DatasetConfig,
    split_name: str,
    *,
    shuffle: bool | None = None,
) -> tuple[tf.data.Dataset, np.ndarray]:
    """Build a deterministic batched TensorFlow dataset."""

    image_paths, labels = read_split_records(config, split_name)

    if shuffle is None:
        shuffle = split_name == "train"

    preprocessing_config = PreprocessingConfig(
        output_size=config.image_size,
        crop_retinal_field=True,
        use_clahe=config.use_clahe,
        use_unsharp_mask=config.use_unsharp_mask,
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=config.random_seed,
            reshuffle_each_iteration=True,
        )

    def load_example(
        image_path: tf.Tensor,
        label: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.numpy_function(
            func=lambda value: _load_preprocessed_image(
                value,
                preprocessing_config,
            ),
            inp=[image_path],
            Tout=tf.float32,
        )
        image.set_shape(
            [config.image_size, config.image_size, 3]
        )
        label = tf.cast(label, tf.int32)
        return image, label

    dataset = dataset.map(
        load_example,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, labels


def calculate_class_weights(
    labels: np.ndarray,
) -> dict[int, float]:
    """Calculate balanced weights for the five diagnosis classes."""

    counts = np.bincount(labels, minlength=NUM_CLASSES)

    if np.any(counts == 0):
        raise ValueError(
            f"Cannot calculate weights with empty classes: {counts}"
        )

    total = int(counts.sum())

    return {
        class_id: total / (NUM_CLASSES * int(class_count))
        for class_id, class_count in enumerate(counts)
    }
