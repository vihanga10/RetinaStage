"""Deterministic validation perturbations for RetinaStage robustness tests."""

from dataclasses import asdict, dataclass

import numpy as np
import tensorflow as tf

from retinastage.calibration import probabilities_with_temperature
from retinastage.evaluation import calculate_metrics


@dataclass(frozen=True)
class PerturbationConfig:
    """One pre-declared image perturbation and its fixed severity."""

    name: str
    label: str
    parameter: str
    value: float | int

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


ROBUSTNESS_CONDITIONS = (
    PerturbationConfig("clean", "Clean validation images", "none", 0),
    PerturbationConfig("brightness_low", "Reduced brightness", "factor", 0.70),
    PerturbationConfig("brightness_high", "Increased brightness", "factor", 1.25),
    PerturbationConfig("contrast_low", "Reduced contrast", "factor", 0.60),
    PerturbationConfig("gaussian_blur", "Gaussian blur", "sigma", 1.20),
    PerturbationConfig("gaussian_noise", "Gaussian noise", "stddev", 12.0),
    PerturbationConfig("jpeg_compression", "JPEG compression", "quality", 40),
)


def get_perturbation(name: str) -> PerturbationConfig:
    """Resolve one supported robustness condition by its stable name."""

    for condition in ROBUSTNESS_CONDITIONS:
        if condition.name == name:
            return condition
    supported = [condition.name for condition in ROBUSTNESS_CONDITIONS]
    raise ValueError(f"Unknown perturbation {name!r}; expected one of {supported}")


def _gaussian_blur(
    image: tf.Tensor,
    sigma: float,
    *,
    kernel_size: int = 5,
) -> tf.Tensor:
    coordinates = tf.range(kernel_size, dtype=tf.float32)
    coordinates -= (kernel_size - 1) / 2.0
    kernel_1d = tf.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel_1d /= tf.reduce_sum(kernel_1d)
    kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)
    kernel = kernel_2d[:, :, tf.newaxis, tf.newaxis]
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    blurred = tf.nn.depthwise_conv2d(
        image[tf.newaxis, ...],
        kernel,
        strides=[1, 1, 1, 1],
        padding="SAME",
    )
    return blurred[0]


def apply_perturbation(
    image: tf.Tensor,
    record_index: tf.Tensor,
    condition_name: str,
    *,
    random_seed: int = 20260921,
) -> tf.Tensor:
    """Apply one deterministic condition to a preprocessed RGB image."""

    condition = get_perturbation(condition_name)
    image = tf.cast(image, tf.float32)

    if condition.name == "clean":
        transformed = image
    elif condition.name in {"brightness_low", "brightness_high"}:
        transformed = image * float(condition.value)
    elif condition.name == "contrast_low":
        transformed = tf.image.adjust_contrast(
            image, float(condition.value)
        )
    elif condition.name == "gaussian_blur":
        transformed = _gaussian_blur(image, float(condition.value))
    elif condition.name == "gaussian_noise":
        noise = tf.random.stateless_normal(
            tf.shape(image),
            seed=tf.stack(
                [
                    tf.cast(random_seed, tf.int32),
                    tf.cast(record_index, tf.int32),
                ]
            ),
            mean=0.0,
            stddev=float(condition.value),
            dtype=tf.float32,
        )
        transformed = image + noise
    elif condition.name == "jpeg_compression":
        encoded = tf.io.encode_jpeg(
            tf.cast(tf.round(image), tf.uint8),
            quality=int(condition.value),
            chroma_downsampling=True,
        )
        transformed = tf.cast(
            tf.io.decode_jpeg(encoded, channels=3),
            tf.float32,
        )
        transformed.set_shape(image.shape)
    else:  # pragma: no cover - protected by get_perturbation
        raise AssertionError(condition.name)

    return tf.clip_by_value(transformed, 0.0, 255.0)


def build_robustness_dataset(
    base_dataset: tf.data.Dataset,
    condition_name: str,
    *,
    batch_size: int,
    random_seed: int = 20260921,
) -> tf.data.Dataset:
    """Transform a non-shuffled dataset while preserving record order."""

    get_perturbation(condition_name)
    if condition_name == "clean":
        return base_dataset

    unbatched = base_dataset.unbatch().enumerate()

    def transform(
        index: tf.Tensor,
        example: tuple[tf.Tensor, tf.Tensor],
    ) -> tuple[tf.Tensor, tf.Tensor]:
        image, label = example
        transformed = apply_perturbation(
            image,
            index,
            condition_name,
            random_seed=random_seed,
        )
        transformed.set_shape(image.shape)
        return transformed, label

    return (
        unbatched.map(
            transform,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def calculate_robustness_metrics(
    true_labels: np.ndarray,
    raw_probabilities: np.ndarray,
    clean_predictions: np.ndarray,
    clean_scaled_probabilities: np.ndarray,
    *,
    temperature: float,
    confidence_threshold: float,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    """Calculate performance, stability, and fixed-policy uncertainty metrics."""

    scaled = probabilities_with_temperature(raw_probabilities, temperature)
    predictions = np.argmax(scaled, axis=1)
    confidence = np.max(scaled, axis=1)
    accepted = confidence >= confidence_threshold
    metrics: dict[str, float | int] = {
        **calculate_metrics(true_labels, predictions),
        "prediction_consistency": float(
            np.mean(predictions == clean_predictions)
        ),
        "mean_probability_l1_shift": float(
            np.mean(
                np.sum(
                    np.abs(scaled - clean_scaled_probabilities),
                    axis=1,
                )
            )
        ),
        "mean_calibrated_confidence": float(np.mean(confidence)),
        "accepted_records": int(np.sum(accepted)),
        "uncertain_records": int(np.sum(~accepted)),
        "uncertainty_rate": float(np.mean(~accepted)),
        "selective_accuracy": (
            float(np.mean(predictions[accepted] == true_labels[accepted]))
            if np.any(accepted)
            else 0.0
        ),
    }
    return metrics, predictions, confidence
