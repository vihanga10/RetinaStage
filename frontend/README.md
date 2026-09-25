# RetinaStage web interface

React and TypeScript interface for the frozen RetinaStage inference API.

## Local setup

Use Node.js 22 or newer. Start the FastAPI service from the repository root,
then use a second terminal:

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open `http://127.0.0.1:5173`. If FastAPI is running on port 8001, update
`.env.local` to:

```text
VITE_API_BASE_URL=http://127.0.0.1:8001
```

Restart Vite after changing the environment file.

## Verification

```bash
npm test
npm run build
```

The production bundle is written to `frontend/dist/` and is excluded from Git.

## Evidence boundary

The browser displays the API response; it does not train the model, recalibrate
probabilities, change the uncertainty threshold or save uploaded images. A
quality warning or uncertain result triggers human review. The interface is an
educational research prototype and is not a medical diagnostic system.

After a prediction succeeds, the browser requests Grad-CAM evidence for the
returned grade from `/api/v1/explain`. The heatmap and overlay represent the
processed 224-pixel model input. They indicate associations in the fixed model
and must not be interpreted as lesion localization, causality, or diagnosis.

RetinaGuide sends the user's question and the current prediction JSON to
`/api/v1/retinaguide`; it does not send the retinal image. Responses are
deterministic and limited to the displayed grade, probabilities, confidence,
review decision, technical-quality fields and Grad-CAM limitations. Requests
for diagnosis or treatment are declined with the research safety boundary.
The conversation log scrolls internally to the newest message so longer chats
do not move or obscure the result sections around it.

The response engine's fixed evaluation is run from the repository root with
`python scripts/evaluate_retinaguide.py`. It tests controlled prediction JSON,
not retinal images or model predictions; see `docs/RETINAGUIDE_EVALUATION.md`.

RetinaTrace creates a verification JSON receipt from the current result and
optional Grad-CAM metadata. The interface separately offers **Print / Save
PDF**, which opens the browser print dialog with a human-readable A4 summary.
The JSON remains the machine-verifiable record, while the PDF is intended for
reading and presentation. Both omit the retinal image, original filename,
heatmap and overlay bytes. A saved JSON receipt can be uploaded to verify that
its canonical content still matches its SHA-256 checksum. This detects
modification; it is not a digital signature and does not authenticate the
receipt issuer. Run the fixed receipt evaluation from the repository root with
`python scripts/evaluate_retinatrace.py`; see
`docs/RETINATRACE_RECEIPT.md` for scope and limitations.
