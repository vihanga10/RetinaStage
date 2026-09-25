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
10. RetinaGuide receives a question and the current prediction response. Its
    deterministic intent rules validate and explain only those returned fields;
    the retinal image is not sent to the chatbot endpoint.
11. RetinaTrace validates the current response and optional explanation
    metadata, then creates a canonical JSON evidence receipt. The image,
    filename, heatmap and overlay bytes are excluded.
12. A saved receipt can be uploaded to the verification endpoint. The API
    recomputes its canonical SHA-256 checksum and reports whether the content
    is unchanged.
13. The React interface renders the result, explanation, policy, audit fields
    and grounded chat response without converting them into clinical advice.

## Safety and evidence boundaries

- This is an educational research prototype, not a clinical diagnostic tool.
- Quality flags are dataset-relative technical warnings.
- An uncertain result or any quality flag requires human review.
- Input and model SHA-256 values support traceability.
- Uploaded images are processed in memory and are not stored by the API.
- Grad-CAM is qualitative attention evidence, not lesion localization or
  clinical evidence. Its failure does not remove an otherwise valid prediction.
- RetinaGuide is a deterministic result-explanation layer, not a general medical
  chatbot. It cannot diagnose, recommend treatment or infer facts absent from
  the current prediction response.
- RetinaGuide does not use an external language-model service and does not
  receive the uploaded image.
- RetinaTrace receipts omit the retinal image, original filename, heatmap and
  overlay data URLs. The input hash supports matching to a known input without
  embedding that input.
- The RetinaTrace checksum detects modification of receipt content. It is not
  a digital signature, does not authenticate the issuer and does not establish
  that a prediction is medically correct.

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
failure cannot silently alter or erase the fixed inference result. The browser
sends only a question and current prediction JSON to `/api/v1/retinaguide`.
The chatbot response includes its detected intent, grounded field names,
suggested follow-up questions and an explicit safety notice.

The browser sends the current prediction and sanitized explanation metadata to
`/api/v1/retinatrace/receipt`. Only the explanation hashes, target, layer,
input-space label and limitation text are retained; Grad-CAM image data URLs
never enter the receipt. Saved receipts are checked by
`/api/v1/retinatrace/verify`.

## RetinaGuide response-layer evaluation

The deterministic chatbot can be evaluated without loading TensorFlow or
opening any retinal image:

```bash
python scripts/evaluate_retinaguide.py
```

The runner applies 15 fixed prompts to three controlled prediction contexts
covering a confident result, a low-confidence result and a technical-quality
review. It verifies intent selection, the exact grounded fields, required
answer content, identical repeated responses and explicit refusal of diagnosis,
treatment and medication requests. It writes row-level CSV evidence and a JSON
summary to `results/retinaguide_evaluation/`.

This evaluation deliberately does not use the retinal dataset, the trained
model, the calibration split or the final-test split. It establishes the
behaviour of the result-explanation layer only; it does not add evidence about
clinical validity or classifier performance. The complete protocol and results
are documented in `docs/RETINAGUIDE_EVALUATION.md`.

## RetinaTrace receipt evaluation

The receipt layer can be evaluated without TensorFlow, the retinal dataset or
the final-test split:

```bash
python scripts/evaluate_retinatrace.py
```

The fixed matrix checks unchanged verification, three controlled content
modifications, five-class completeness, traceability hashes, privacy omissions
and byte-stable serialization. It writes row-level CSV evidence and a JSON
summary to `results/retinatrace_evaluation/`. This is software-integrity and
privacy evidence, not classifier or clinical evaluation. See
`docs/RETINATRACE_RECEIPT.md` for the complete protocol and limitations.
