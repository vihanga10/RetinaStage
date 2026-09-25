"""FastAPI entry point for the RetinaStage educational prototype."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from retinastage.inference import RetinaStagePredictor
from retinastage.retinaguide import explain_prediction


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(
    os.getenv(
        "RETINASTAGE_MODEL_PATH",
        PROJECT_ROOT
        / "artifacts/training/ordinal_stage2/best_macro_f1_model.keras",
    )
)
CALIBRATION_PATH = Path(
    os.getenv(
        "RETINASTAGE_CALIBRATION_PATH",
        PROJECT_ROOT
        / "artifacts/calibration/selected_ordinal_model/"
        "calibration_summary.json",
    )
)
QUALITY_SUMMARY_PATH = Path(
    os.getenv(
        "RETINASTAGE_QUALITY_SUMMARY_PATH",
        PROJECT_ROOT / "results/quality/image_quality_summary.json",
    )
)
MAX_UPLOAD_BYTES = int(
    os.getenv("RETINASTAGE_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))
)
ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
}


app = FastAPI(
    title="RetinaStage API",
    version="0.1.0",
    description=(
        "Educational five-grade diabetic-retinopathy research prototype. "
        "Not validated for clinical diagnosis."
    ),
)
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "RETINASTAGE_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_predictor: RetinaStagePredictor | None = None
_load_lock = Lock()


class RetinaGuideRequest(BaseModel):
    """One question grounded in the current prediction response."""

    message: str = Field(min_length=1, max_length=500)
    result: dict[str, object]


def get_predictor() -> RetinaStagePredictor:
    """Load the fixed model once, on the first prediction request."""

    global _predictor
    if _predictor is None:
        with _load_lock:
            if _predictor is None:
                _predictor = RetinaStagePredictor.from_artifacts(
                    MODEL_PATH,
                    CALIBRATION_PATH,
                    QUALITY_SUMMARY_PATH,
                )
    return _predictor


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "RetinaStage API",
        "status": "running",
        "documentation": "/docs",
        "intended_use": "educational research prototype",
    }


@app.get("/api/v1/health")
def health() -> dict[str, object]:
    """Report artifact readiness without loading the TensorFlow model."""

    artifacts = {
        "model": MODEL_PATH.is_file(),
        "calibration": CALIBRATION_PATH.is_file(),
        "quality_summary": QUALITY_SUMMARY_PATH.is_file(),
    }
    return {
        "status": "ready" if all(artifacts.values()) else "not_ready",
        "artifacts": artifacts,
        "model_loaded": _predictor is not None,
    }


@app.post("/api/v1/predict")
async def predict(
    image: UploadFile = File(...),
) -> dict[str, object]:
    """Predict one uploaded PNG or JPEG retinal photograph."""

    content = await read_uploaded_image(image)
    try:
        predictor = get_predictor()
        return predictor.predict_bytes(content)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


async def read_uploaded_image(image: UploadFile) -> bytes:
    """Validate and read one supported upload without retaining it."""

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only PNG and JPEG images are supported.",
        )
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    await image.close()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Uploaded image exceeds the configured size limit.",
        )
    return content


@app.post("/api/v1/explain")
async def explain(
    image: UploadFile = File(...),
    target_grade: int = Query(..., ge=0, le=4),
) -> dict[str, object]:
    """Generate Grad-CAM evidence for a previously predicted grade."""

    content = await read_uploaded_image(image)
    try:
        predictor = get_predictor()
        return predictor.explain_bytes(content, target_grade)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/v1/retinaguide")
def retinaguide(request: RetinaGuideRequest) -> dict[str, object]:
    """Explain supplied prediction fields without loading or changing the model."""

    try:
        return explain_prediction(request.message, request.result)
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid prediction context: {error}",
        ) from error
