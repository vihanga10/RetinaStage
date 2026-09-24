# RetinaStage application architecture

The application layer wraps the fixed, selected experiment checkpoint. It
does not retrain the model, fit calibration, change the uncertainty threshold,
or use the final-test split.

## Request flow

1. The browser uploads one PNG or JPEG retinal photograph.
2. The API validates the file type, byte size and decoded dimensions.
3. Technical quality measurements are compared with the dataset-derived audit
   limits. Flags request review; they do not declare clinical gradability.
4. The image follows the same retinal-field crop, square padding and 224-pixel
   resizing used during training and evaluation.
5. The selected EfficientNetB0 checkpoint produces five class probabilities.
6. The frozen temperature scales the probabilities.
7. The frozen confidence threshold marks uncertain predictions.
8. The response includes the grade, calibrated confidence, all five
   probabilities, technical-quality flags and a human-review decision.
9. The interface requests Grad-CAM for the returned grade. The API generates
   a heatmap and overlay from the processed model input, entirely in memory.
10. The React interface renders the result, explanation, policy and audit fields without
   converting them into clinical advice.

## Safety and evidence boundaries

- This is an educational research prototype, not a clinical diagnostic tool.
- Quality flags are dataset-relative technical warnings.
- An uncertain result or any quality flag requires human review.
- Input and model SHA-256 values support traceability.
- Uploaded images are processed in memory and are not stored by the API.
- Grad-CAM is qualitative attention evidence, not lesion localization or
  clinical evidence. Its failure does not remove an otherwise valid prediction.
- RetinaGuide will be connected in a later application stage.

## Local API configuration

The default artifact locations are:

```text
artifacts/training/ordinal_stage2/best_macro_f1_model.keras
artifacts/calibration/selected_ordinal_model/calibration_summary.json
results/quality/image_quality_summary.json
```

They can be overridden with `RETINASTAGE_MODEL_PATH`,
`RETINASTAGE_CALIBRATION_PATH` and `RETINASTAGE_QUALITY_SUMMARY_PATH`.

Install and run from the repository root:

```bash
python -m pip install -r requirements-app.txt
PYTHONPATH=src uvicorn app.api.main:app --reload
```

Then inspect `http://127.0.0.1:8000/docs` or call:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

## Local frontend configuration

The browser application is located in `frontend/`. Its only runtime
configuration value is the API base URL:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Copy `.env.example` to `.env.local` and change the port when required. The
FastAPI CORS configuration already permits the Vite development origins on
port 5173.

```bash
cd frontend
npm install
npm test
npm run build
npm run dev
```

The interface contains no model parameters. It sends a multipart image first
to `/api/v1/predict` and then to `/api/v1/explain` using the returned grade as
the explanation target. Keeping the endpoints separate ensures explanation
failure cannot silently alter or erase the fixed inference result. RetinaGuide
remains a separate future endpoint.
