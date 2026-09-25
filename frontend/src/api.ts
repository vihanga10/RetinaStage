import type {
  ExplanationResponse,
  HealthResponse,
  PredictionResponse,
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
