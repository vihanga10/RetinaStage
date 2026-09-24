"""Frozen-model inference orchestration for the RetinaStage application."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Protocol

import cv2
import numpy as np

from retinastage.calibration import probabilities_with_temperature
from retinastage.preprocessing import PreprocessingConfig, preprocess_image
from retinastage.quality import assess_quality, load_review_limits


GRADE_LABELS = (
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
)
EDUCATIONAL_NOTICE = (
    "Educational research prototype only. This output is not a medical "
    "diagnosis and requires review by an appropriately qualified person."
)


class PredictionModel(Protocol):
    """Small protocol shared by Keras models and deterministic test doubles."""

    def predict(self, inputs: np.ndarray, verbose: int = 0) -> Any:
        ...


@dataclass(frozen=True)
class InferencePolicy:
    """Frozen post-hoc policy selected before final-test evaluation."""

    temperature: float
    confidence_threshold: float


def load_inference_policy(path: Path) -> InferencePolicy:
    """Read notebook-style or script-style calibration summaries."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    temperature = float(payload["temperature"])
    if "confidence_threshold" in payload:
        confidence_threshold = float(payload["confidence_threshold"])
    else:
        confidence_threshold = float(
            payload["uncertainty_policy"]["confidence_threshold"]
        )
    if temperature <= 0:
        raise ValueError("Calibration temperature must be positive")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1")
    return InferencePolicy(
        temperature=temperature,
        confidence_threshold=confidence_threshold,
    )


def decode_image_bytes(
    content: bytes,
    *,
    max_pixels: int = 40_000_000,
) -> np.ndarray:
    """Decode one uploaded image and reject invalid or extreme inputs."""

    if not content:
        raise ValueError("Uploaded image is empty")
    encoded = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a decodable image")
    height, width = image.shape[:2]
    if height * width > max_pixels:
        raise ValueError("Uploaded image dimensions are too large")
    return image


class RetinaStagePredictor:
    """Apply frozen preprocessing, calibration and review policies."""

    def __init__(
        self,
        model: PredictionModel,
        policy: InferencePolicy,
        quality_limits: Mapping[str, float],
        *,
        model_sha256: str | None = None,
        image_size: int = 224,
    ) -> None:
        self.model = model
        self.policy = policy
        self.quality_limits = dict(quality_limits)
        self.model_sha256 = model_sha256
        self.preprocessing_config = PreprocessingConfig(
            output_size=image_size,
            crop_retinal_field=True,
            use_clahe=False,
            use_unsharp_mask=False,
        )
        self._prediction_lock = Lock()

    @classmethod
    def from_artifacts(
        cls,
        model_path: Path,
        calibration_path: Path,
        quality_summary_path: Path,
    ) -> "RetinaStagePredictor":
        """Load the selected Keras checkpoint and its frozen policies."""

        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not calibration_path.is_file():
            raise FileNotFoundError(
                f"Calibration summary not found: {calibration_path}"
            )
        if not quality_summary_path.is_file():
            raise FileNotFoundError(
                f"Quality summary not found: {quality_summary_path}"
            )
        from tensorflow import keras

        model = keras.models.load_model(model_path, compile=False)
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        return cls(
            model=model,
            policy=load_inference_policy(calibration_path),
            quality_limits=load_review_limits(quality_summary_path),
            model_sha256=model_hash,
        )

    def predict_bytes(self, content: bytes) -> dict[str, object]:
        """Return one auditable, calibrated five-grade prediction."""

        image_bgr = decode_image_bytes(content)
        quality = assess_quality(image_bgr, self.quality_limits)
        processed_bgr = preprocess_image(
            image_bgr,
            self.preprocessing_config,
        )
        processed_rgb = cv2.cvtColor(
            processed_bgr,
            cv2.COLOR_BGR2RGB,
        ).astype(np.float32)
        batch = processed_rgb[np.newaxis, ...]
        with self._prediction_lock:
            raw_output = self.model.predict(batch, verbose=0)
        raw_probabilities = np.asarray(raw_output, dtype=float)
        if raw_probabilities.shape != (1, len(GRADE_LABELS)):
            raise RuntimeError(
                "Model output must have shape (1, 5); received "
                f"{raw_probabilities.shape}"
            )
        if not np.isfinite(raw_probabilities).all():
            raise RuntimeError("Model output contains non-finite values")
        if np.any(raw_probabilities < 0):
            raise RuntimeError("Model output contains negative probabilities")
        row_sum = raw_probabilities.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise RuntimeError("Model output probabilities sum to zero")
        raw_probabilities = raw_probabilities / row_sum
        calibrated = probabilities_with_temperature(
            raw_probabilities,
            self.policy.temperature,
        )[0]
        predicted_grade = int(np.argmax(calibrated))
        confidence = float(calibrated[predicted_grade])
        uncertain = confidence < self.policy.confidence_threshold
        review_reasons = list(quality.flags)
        if uncertain:
            review_reasons.append("LOW_MODEL_CONFIDENCE")

        return {
            "input_sha256": hashlib.sha256(content).hexdigest(),
            "model_sha256": self.model_sha256,
            "predicted_grade": predicted_grade,
            "predicted_label": GRADE_LABELS[predicted_grade],
            "confidence": confidence,
            "uncertain": uncertain,
            "requires_human_review": bool(review_reasons),
            "review_reasons": review_reasons,
            "probabilities": [
                {
                    "grade": grade,
                    "label": label,
                    "probability": float(calibrated[grade]),
                }
                for grade, label in enumerate(GRADE_LABELS)
            ],
            "quality": quality.to_dict(),
            "policy": {
                "temperature": self.policy.temperature,
                "confidence_threshold": self.policy.confidence_threshold,
            },
            "educational_notice": EDUCATIONAL_NOTICE,
        }
