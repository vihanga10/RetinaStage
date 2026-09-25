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
