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
