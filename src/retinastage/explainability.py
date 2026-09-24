"""Grad-CAM helpers for the fixed final RetinaStage model."""

from __future__ import annotations

import base64
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
    heatmap = tf.reduce_sum(feature_maps[0] * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    return heatmap.numpy()


def colourise_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Convert a normalized heatmap to an RGB turbo-colour image."""

    clipped = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    return plt.get_cmap("turbo")(clipped)[..., :3]


def create_overlay(
    image_array: np.ndarray,
    heatmap: np.ndarray,
    *,
    image_weight: float = 0.60,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize a heatmap and blend it with its model input image."""

    if not 0.0 <= image_weight <= 1.0:
        raise ValueError("image_weight must be between 0 and 1")
    height, width = image_array.shape[:2]
    resized = tf.image.resize(
        heatmap[..., np.newaxis], (height, width)
    ).numpy().squeeze()
    coloured = colourise_heatmap(resized)
    normalised = np.clip(image_array / 255.0, 0, 1)
    overlay = np.clip(
        image_weight * normalised + (1.0 - image_weight) * coloured,
        0,
        1,
    )
    return resized, overlay


def encode_rgb_png_data_url(image_rgb: np.ndarray) -> str:
    """Encode a float or uint8 RGB image as an in-memory PNG data URL."""

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG data URL input must have shape (H, W, 3)")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise RuntimeError("Grad-CAM image could not be encoded")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"
