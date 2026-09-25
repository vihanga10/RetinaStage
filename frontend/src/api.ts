import type {
  ExplanationResponse,
  HealthResponse,
  PredictionResponse,
  RetinaGuideResponse,
  RetinaTraceReceipt,
  RetinaTraceVerification,
} from "./types";

const FALLBACK_API_URL = "http://127.0.0.1:8000";

export function normaliseApiBaseUrl(value?: string): string {
  return (value?.trim() || FALLBACK_API_URL).replace(/\/+$/, "");
}

export const API_BASE_URL = normaliseApiBaseUrl(
  import.meta.env.VITE_API_BASE_URL,
);

async function extractError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
  } catch {
    // The service may return a plain response if it fails before FastAPI.
  }
  return `Request failed with status ${response.status}.`;
}

export async function fetchHealth(
  signal?: AbortSignal,
): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/health`, { signal });
  if (!response.ok) {
    throw new Error(await extractError(response));
  }
  return (await response.json()) as HealthResponse;
}

export async function predictImage(
  file: File,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("image", file);
  const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
    method: "POST",
    body: formData,
    signal,
  });
  if (!response.ok) {
    throw new Error(await extractError(response));
  }
  return (await response.json()) as PredictionResponse;
}

export async function explainImage(
  file: File,
  targetGrade: number,
  signal?: AbortSignal,
): Promise<ExplanationResponse> {
  const formData = new FormData();
  formData.append("image", file);
  const response = await fetch(
    `${API_BASE_URL}/api/v1/explain?target_grade=${targetGrade}`,
    {
      method: "POST",
      body: formData,
      signal,
    },
  );
  if (!response.ok) {
    throw new Error(await extractError(response));
  }
  return (await response.json()) as ExplanationResponse;
}

export async function askRetinaGuide(
  message: string,
  result: PredictionResponse,
  signal?: AbortSignal,
): Promise<RetinaGuideResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/retinaguide`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, result }),
    signal,
  });
  if (!response.ok) {
    throw new Error(await extractError(response));
  }
  return (await response.json()) as RetinaGuideResponse;
}

export async function createRetinaTraceReceipt(
  result: PredictionResponse,
  explanation: ExplanationResponse | null,
  signal?: AbortSignal,
): Promise<RetinaTraceReceipt> {
  const explanationEvidence = explanation
    ? {
        input_sha256: explanation.input_sha256,
        model_sha256: explanation.model_sha256,
        target_grade: explanation.target_grade,
        target_label: explanation.target_label,
        backbone_layer: explanation.backbone_layer,
        input_space: explanation.input_space,
        interpretation: explanation.interpretation,
      }
    : null;
  const response = await fetch(`${API_BASE_URL}/api/v1/retinatrace/receipt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ result, explanation: explanationEvidence }),
    signal,
  });
  if (!response.ok) {
    throw new Error(await extractError(response));
  }
  return (await response.json()) as RetinaTraceReceipt;
}

export async function verifyRetinaTraceReceipt(
  receipt: RetinaTraceReceipt,
  signal?: AbortSignal,
): Promise<RetinaTraceVerification> {
  const response = await fetch(`${API_BASE_URL}/api/v1/retinatrace/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ receipt }),
    signal,
  });
  if (!response.ok) {
    throw new Error(await extractError(response));
  }
  return (await response.json()) as RetinaTraceVerification;
}
