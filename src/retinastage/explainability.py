"""Grad-CAM helpers for the fixed final RetinaStage model."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from retinastage.preprocessing import PreprocessingConfig, preprocess_image


def load_model_image(
    image_path: Path,
    *,
    image_size: int = 224,
) -> tuple[np.ndarray, tf.Tensor]:
    """Load an image with the same deterministic preprocessing as training."""

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not decode {image_path}")
    processed_bgr = preprocess_image(
        image_bgr,
        PreprocessingConfig(output_size=image_size),
    )
    image_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB).astype(
        np.float32
    )
    return image_rgb, tf.convert_to_tensor(image_rgb[np.newaxis, ...])


def create_gradcam_heatmap(
    model: keras.Model,
    image_batch: tf.Tensor,
    target_class: int,
    *,
    backbone_name: str = "efficientnetb0",
) -> np.ndarray:
    """Calculate Grad-CAM at the final spatial backbone output."""

    with tf.GradientTape() as tape:
        activations = image_batch
        feature_maps = None
        for layer in model.layers:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            try:
                activations = layer(activations, training=False)
            except TypeError:
                activations = layer(activations)
            if layer.name == backbone_name:
                feature_maps = activations
                tape.watch(feature_maps)
        if feature_maps is None:
            raise ValueError(f"Backbone {backbone_name!r} was not found")
        if len(feature_maps.shape) != 4:
            raise ValueError("Backbone output is not a spatial feature map")
        target_score = activations[:, int(target_class)]

    gradients = tape.gradient(target_score, feature_maps)
    if gradients is None:
        raise RuntimeError("Gradients could not be calculated")
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(
        feature_maps[0] * pooled_gradients,
        axis=-1,
    )
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    return heatmap.numpy()


def create_overlay(
    image_array: np.ndarray,
    heatmap: np.ndarray,
    *,
    image_weight: float = 0.60,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize a heatmap and blend it with its model input image."""

    height, width = image_array.shape[:2]
    resized = tf.image.resize(
        heatmap[..., np.newaxis], (height, width)
    ).numpy().squeeze()
    coloured = plt.get_cmap("turbo")(resized)[..., :3]
    normalised = np.clip(image_array / 255.0, 0, 1)
    overlay = np.clip(
        image_weight * normalised + (1.0 - image_weight) * coloured,
        0,
        1,
    )
    return resized, overlay
