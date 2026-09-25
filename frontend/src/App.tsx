import {
  type CSSProperties,
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  API_BASE_URL,
  askRetinaGuide,
  createRetinaTraceReceipt,
  explainImage,
  fetchHealth,
  predictImage,
  verifyRetinaTraceReceipt,
} from "./api";
import {
  createGuideWelcome,
  initialGuideQuestions,
  RETINAGUIDE_MESSAGE_LIMIT,
  validGuideMessage,
} from "./retinaguide";
import {
  MAX_RECEIPT_BYTES,
  parseRetinaTraceReceipt,
  receiptDownloadName,
  serialiseRetinaTraceReceipt,
  shortReceiptHash,
} from "./retinatrace";
import type {
  ExplanationResponse,
  HealthResponse,
  PredictionResponse,
  RetinaGuideMessage,
  RetinaTraceReceipt,
  RetinaTraceVerification,
} from "./types";
import { formatFlag, formatPercent, reviewState } from "./view-model";

const MAX_FILE_BYTES = 20 * 1024 * 1024;
const ACCEPTED_TYPES = new Set(["image/png", "image/jpeg"]);

type AnalysisState = "idle" | "loading" | "complete" | "error";
type ExplanationState = "idle" | "loading" | "complete" | "error";

function Icon({ name }: { name: "eye" | "upload" | "shield" | "spark" }) {
  const paths = {
    eye: (
      <>
        <path d="M2 12s3.7-6 10-6 10 6 10 6-3.7 6-10 6S2 12 2 12Z" />
        <circle cx="12" cy="12" r="3" />
      </>
    ),
    upload: (
      <>
        <path d="M12 16V4" />
        <path d="m7 9 5-5 5 5" />
        <path d="M5 20h14" />
      </>
    ),
    shield: <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />,
    spark: (
      <>
        <path d="m12 3 1.25 4.25L17.5 8.5l-4.25 1.25L12 14l-1.25-4.25L6.5 8.5l4.25-1.25L12 3Z" />
        <path d="m18.5 14 .65 2.35L21.5 17l-2.35.65L18.5 20l-.65-2.35L15.5 17l2.35-.65.65-2.35Z" />
      </>
    ),
  };
  return (
    <svg
      aria-hidden="true"
      className="icon"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[name]}
    </svg>
  );
}

function ServiceStatus({ health }: { health: HealthResponse | null }) {
  const ready = health?.status === "ready";
  return (
    <div className={`service-status ${ready ? "ready" : "waiting"}`}>
      <span className="status-dot" aria-hidden="true" />
      <span>{ready ? "Analysis service ready" : "Checking analysis service"}</span>
    </div>
  );
}

function ProbabilityChart({ result }: { result: PredictionResponse }) {
  return (
    <div className="probability-list" aria-label="Five-grade probabilities">
      {result.probabilities.map((item) => {
        const selected = item.grade === result.predicted_grade;
        return (
          <div className={`probability-row ${selected ? "selected" : ""}`} key={item.grade}>
            <div className="probability-label">
              <span className="grade-number">{item.grade}</span>
              <span>{item.label}</span>
              <strong>{formatPercent(item.probability)}</strong>
            </div>
            <div className="probability-track" aria-hidden="true">
              <span style={{ width: `${Math.max(item.probability * 100, 0.35)}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ExplanationPanel({
  explanation,
  state,
  error,
}: {
  explanation: ExplanationResponse | null;
  state: ExplanationState;
  error: string | null;
}) {
  return (
    <div className="result-section explanation-section">
      <div className="subheading">
        <h3>Grad-CAM attention evidence</h3>
        <span>
          {state === "complete"
            ? `Target: Grade ${explanation?.target_grade}`
            : "Processed model input"}
        </span>
      </div>

      {state === "loading" && (
        <div className="explanation-loading" role="status">
          <span className="spinner dark" /> Generating attention maps…
        </div>
      )}

      {state === "error" && (
        <div className="explanation-error" role="status">
          <strong>Prediction available; explanation unavailable.</strong>
          <span>{error}</span>
        </div>
      )}

      {explanation && state === "complete" && (
        <>
          <div className="explanation-grid">
            <figure>
              <img
                src={explanation.heatmap_data_url}
                alt={`Grad-CAM heatmap for ${explanation.target_label}`}
              />
              <figcaption>Attention heatmap</figcaption>
            </figure>
            <figure>
              <img
                src={explanation.overlay_data_url}
                alt={`Grad-CAM overlay for ${explanation.target_label}`}
              />
              <figcaption>Heatmap over processed input</figcaption>
            </figure>
          </div>
          <p className="explanation-note">{explanation.interpretation}</p>
        </>
      )}
    </div>
  );
}

function RetinaGuidePanel({ result }: { result: PredictionResponse }) {
  const [messages, setMessages] = useState<RetinaGuideMessage[]>(() => [
    createGuideWelcome(result),
  ]);
  const [suggestions, setSuggestions] = useState<string[]>(() =>
    initialGuideQuestions(result),
  );
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [groundedFields, setGroundedFields] = useState<string[]>([]);
  const [safetyNotice, setSafetyNotice] = useState(
    "RetinaGuide explains only the current model result and does not provide medical advice.",
  );
  const requestRef = useRef<AbortController | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const messageCounter = useRef(0);

  useEffect(() => {
    return () => {
      requestRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    const messageLog = messagesRef.current;
    if (messageLog) {
      messageLog.scrollTop = messageLog.scrollHeight;
    }
  }, [messages, loading]);

  function messageId(role: RetinaGuideMessage["role"]): string {
    messageCounter.current += 1;
    return `guide-${role}-${messageCounter.current}`;
  }

  async function sendQuestion(question: string) {
    const content = question.trim();
    if (!validGuideMessage(content) || loading) return;

    setMessages((current) => [
      ...current,
      { id: messageId("user"), role: "user", content },
    ]);
    setDraft("");
    setError(null);
    setLoading(true);

    const controller = new AbortController();
    requestRef.current = controller;
    try {
      const response = await askRetinaGuide(
        content,
        result,
        controller.signal,
      );
      if (controller.signal.aborted) return;
      setMessages((current) => [
        ...current,
        {
          id: messageId("assistant"),
          role: "assistant",
          content: response.answer,
        },
      ]);
      setSuggestions(response.suggested_questions);
      setGroundedFields(response.grounded_fields);
      setSafetyNotice(response.safety_notice);
    } catch (requestError) {
      if (!controller.signal.aborted) {
        setError(
          requestError instanceof Error
            ? requestError.message
            : "RetinaGuide could not answer this question.",
        );
      }
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        if (!controller.signal.aborted) setLoading(false);
      }
    }
  }

  function submitQuestion(event: FormEvent) {
    event.preventDefault();
    void sendQuestion(draft);
  }

  return (
    <div className="result-section retinaguide-section">
      <div className="subheading">
        <h3>RetinaGuide result explanation</h3>
        <span>Grounded in this result only</span>
      </div>

      <div className="guide-card">
        <div className="guide-heading">
          <span className="guide-mark"><Icon name="spark" /></span>
          <div>
            <strong>Ask RetinaGuide</strong>
            <p>No image upload, diagnosis, or treatment advice</p>
          </div>
        </div>

        <div
          ref={messagesRef}
          className="guide-messages"
          role="log"
          aria-live="polite"
          aria-label="RetinaGuide conversation"
        >
          {messages.map((chatMessage) => (
            <div
              className={`guide-message ${chatMessage.role}`}
              key={chatMessage.id}
            >
              <span>{chatMessage.role === "assistant" ? "RetinaGuide" : "You"}</span>
              <p>{chatMessage.content}</p>
            </div>
          ))}
          {loading && (
            <div className="guide-message assistant pending" role="status">
              <span>RetinaGuide</span>
              <p><span className="spinner dark" /> Checking the current result…</p>
            </div>
          )}
        </div>

        <div className="guide-suggestions" aria-label="Suggested questions">
          {suggestions.map((question) => (
            <button
              type="button"
              key={question}
              disabled={loading}
              onClick={() => void sendQuestion(question)}
            >
              {question}
            </button>
          ))}
        </div>

        {error && <p className="guide-error" role="alert">{error}</p>}

        <form className="guide-form" onSubmit={submitQuestion}>
          <input
            aria-label="Question for RetinaGuide"
            value={draft}
            maxLength={RETINAGUIDE_MESSAGE_LIMIT}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Ask about this displayed result…"
          />
          <button type="submit" disabled={!validGuideMessage(draft) || loading}>
            Ask
          </button>
        </form>

        {groundedFields.length > 0 && (
          <p className="guide-grounding">
            Grounded fields: {groundedFields.join(", ")}
          </p>
        )}
        <p className="guide-boundary">{safetyNotice}</p>
      </div>
    </div>
  );
}

function RetinaTracePanel({
  result,
  explanation,
}: {
  result: PredictionResponse;
  explanation: ExplanationResponse | null;
}) {
  const [receipt, setReceipt] = useState<RetinaTraceReceipt | null>(null);
  const [verification, setVerification] =
    useState<RetinaTraceVerification | null>(null);
  const [activity, setActivity] = useState<"idle" | "creating" | "verifying">(
    "idle",
  );
  const [error, setError] = useState<string | null>(null);
  const verifyInputRef = useRef<HTMLInputElement>(null);

  async function createReceipt() {
    setActivity("creating");
    setError(null);
    setVerification(null);
    try {
      setReceipt(await createRetinaTraceReceipt(result, explanation));
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The evidence receipt could not be created.",
      );
    } finally {
      setActivity("idle");
    }
  }

  function downloadReceipt() {
    if (!receipt) return;
    const blob = new Blob([serialiseRetinaTraceReceipt(receipt)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = receiptDownloadName(receipt);
    link.click();
    URL.revokeObjectURL(url);
  }

  async function verifyReceiptFile(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    event.target.value = "";
    if (!selected) return;
    setError(null);
    setVerification(null);
    if (selected.size > MAX_RECEIPT_BYTES) {
      setError("The selected receipt exceeds the 512 KB verification limit.");
      return;
    }
    setActivity("verifying");
    try {
      const parsed = parseRetinaTraceReceipt(await selected.text());
      setVerification(await verifyRetinaTraceReceipt(parsed));
    } catch (verificationError) {
      setError(
        verificationError instanceof Error
          ? verificationError.message
          : "The selected receipt could not be verified.",
      );
    } finally {
      setActivity("idle");
    }
  }

  return (
    <div className="result-section retinatrace-section">
      <div className="subheading">
        <h3>RetinaTrace evidence receipt</h3>
        <span>No retinal image stored</span>
      </div>

      <div className="trace-card">
        <div className="trace-intro">
          <span className="trace-mark"><Icon name="shield" /></span>
          <div>
            <strong>Portable integrity evidence</strong>
            <p>
              Save the result, policy, review decision and traceability hashes
              as JSON. Grad-CAM image data and the original filename are omitted.
            </p>
          </div>
        </div>

        <div className="trace-actions">
          <button
            type="button"
            onClick={() => void createReceipt()}
            disabled={activity !== "idle"}
          >
            {activity === "creating" ? "Creating…" : "Create receipt"}
          </button>
          {receipt && (
            <button type="button" className="secondary" onClick={downloadReceipt}>
              Download JSON
            </button>
          )}
          <label className={`trace-file-button ${activity !== "idle" ? "disabled" : ""}`}>
            Verify saved receipt
            <input
              ref={verifyInputRef}
              type="file"
              accept="application/json,.json"
              disabled={activity !== "idle"}
              onChange={(event) => void verifyReceiptFile(event)}
            />
          </label>
        </div>

        {receipt && (
          <div className="trace-summary" role="status">
            <span>Receipt checksum</span>
            <code>{shortReceiptHash(receipt)}</code>
            <small>
              {receipt.explanation.included
                ? "Grad-CAM metadata included; heatmap pixels excluded."
                : "Receipt created without Grad-CAM metadata."}
            </small>
          </div>
        )}

        {verification && (
          <div
            className={`trace-verification ${verification.valid ? "valid" : "invalid"}`}
            role="status"
          >
            <strong>
              {verification.valid
                ? "Receipt integrity verified"
                : "Receipt modification detected"}
            </strong>
            <span>{verification.message}</span>
          </div>
        )}

        {error && <p className="trace-error" role="alert">{error}</p>}
        <p className="trace-boundary">
          A checksum detects changes; it is not a digital signature and does
          not prove who created the receipt.
        </p>
      </div>
    </div>
  );
}

function ResultPanel({
  result,
  explanation,
  explanationState,
  explanationError,
}: {
  result: PredictionResponse;
  explanation: ExplanationResponse | null;
  explanationState: ExplanationState;
  explanationError: string | null;
}) {
  const state = reviewState(result);
  const quality = result.quality.metrics;
  return (
    <section className="results-card" aria-labelledby="results-title">
      <div className="card-heading result-heading">
        <div>
          <span className="section-label">Calibrated result</span>
          <h2 id="results-title">Five-stage assessment</h2>
        </div>
        <span className={`grade-badge grade-${result.predicted_grade}`}>
          Grade {result.predicted_grade}
        </span>
      </div>

      <div className="primary-result">
        <div>
          <span className="result-caption">Predicted stage</span>
          <h3>{result.predicted_label}</h3>
          <p>
            Calibrated confidence <strong>{formatPercent(result.confidence)}</strong>
          </p>
        </div>
        <div className="confidence-ring" style={{ "--confidence": `${result.confidence * 360}deg` } as CSSProperties}>
          <span>{formatPercent(result.confidence, 0)}</span>
        </div>
      </div>

      <div className={`review-banner ${state.tone}`}>
        <div className="review-icon">
          <Icon name="shield" />
        </div>
        <div>
          <span>{state.eyebrow}</span>
          <strong>{state.title}</strong>
          {result.review_reasons.length > 0 ? (
            <div className="flag-list">
              {result.review_reasons.map((reason) => (
                <span key={reason}>{formatFlag(reason)}</span>
              ))}
            </div>
          ) : (
            <p>No confidence or technical-quality flags were triggered.</p>
          )}
        </div>
      </div>

      <ExplanationPanel
        explanation={explanation}
        state={explanationState}
        error={explanationError}
      />

      <div className="result-section">
        <div className="subheading">
          <h3>Class probabilities</h3>
          <span>Temperature scaled</span>
        </div>
        <ProbabilityChart result={result} />
      </div>

      <div className="result-section">
        <div className="subheading">
          <h3>Technical image quality</h3>
          <span>{result.quality.flags.length ? "Review flagged" : "Within audit range"}</span>
        </div>
        <div className="metrics-grid">
          <Metric label="Brightness" value={quality.brightness_mean.toFixed(1)} />
          <Metric label="Contrast" value={quality.contrast_std.toFixed(1)} />
          <Metric label="Sharpness" value={quality.sharpness_laplacian_variance.toFixed(0)} />
          <Metric label="Retinal coverage" value={formatPercent(quality.retinal_field_coverage)} />
        </div>
        <p className="quality-note">{result.quality.interpretation}</p>
      </div>

      <RetinaGuidePanel key={result.input_sha256} result={result} />

      <RetinaTracePanel
        key={`${result.input_sha256}-receipt`}
        result={result}
        explanation={explanationState === "complete" ? explanation : null}
      />

      <details className="audit-details">
        <summary>Audit and policy details</summary>
        <dl>
          <div>
            <dt>Confidence threshold</dt>
            <dd>{result.policy.confidence_threshold.toFixed(4)}</dd>
          </div>
          <div>
            <dt>Temperature</dt>
            <dd>{result.policy.temperature.toFixed(4)}</dd>
          </div>
          <div className="hash-row">
            <dt>Input SHA-256</dt>
            <dd>{result.input_sha256}</dd>
          </div>
          <div className="hash-row">
            <dt>Model SHA-256</dt>
            <dd>{result.model_sha256}</dd>
          </div>
        </dl>
      </details>

      <div className="educational-notice">
        <Icon name="shield" />
        <p>{result.educational_notice}</p>
      </div>
    </section>
  );
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [explanation, setExplanation] =
    useState<ExplanationResponse | null>(null);
  const [explanationState, setExplanationState] =
    useState<ExplanationState>("idle");
  const [explanationError, setExplanationError] = useState<string | null>(
    null,
  );
  const [analysisState, setAnalysisState] = useState<AnalysisState>("idle");
  const [message, setMessage] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    const checkHealth = () => {
      fetchHealth(controller.signal)
        .then((response) => {
          setHealth(response);
          setHealthError(false);
        })
        .catch(() => {
          if (!controller.signal.aborted) {
            setHealth(null);
            setHealthError(true);
          }
        });
    };
    checkHealth();
    const interval = window.setInterval(checkHealth, 15_000);
    return () => {
      window.clearInterval(interval);
      controller.abort();
    };
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function selectFile(candidate?: File) {
    setMessage(null);
    setResult(null);
    setExplanation(null);
    setExplanationState("idle");
    setExplanationError(null);
    setAnalysisState("idle");
    if (!candidate) return;
    if (!ACCEPTED_TYPES.has(candidate.type)) {
      setMessage("Choose a PNG or JPEG retinal photograph.");
      return;
    }
    if (candidate.size > MAX_FILE_BYTES) {
      setMessage("The selected image is larger than the 20 MB upload limit.");
      return;
    }
    setFile(candidate);
    setPreviewUrl(URL.createObjectURL(candidate));
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    selectFile(event.target.files?.[0]);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
  }

  async function runAnalysis(event: FormEvent) {
    event.preventDefault();
    if (!file) {
      setMessage("Select a retinal photograph before starting analysis.");
      return;
    }
    setAnalysisState("loading");
    setMessage(null);
    try {
      const response = await predictImage(file);
      setResult(response);
      setExplanationState("loading");
      try {
        const explanationResponse = await explainImage(
          file,
          response.predicted_grade,
        );
        if (
          explanationResponse.input_sha256 !== response.input_sha256 ||
          explanationResponse.model_sha256 !== response.model_sha256
        ) {
          throw new Error(
            "Explanation traceability does not match the prediction.",
          );
        }
        setExplanation(explanationResponse);
        setExplanationState("complete");
      } catch (error) {
        setExplanationError(
          error instanceof Error
            ? error.message
            : "Grad-CAM generation failed.",
        );
        setExplanationState("error");
      } finally {
        setAnalysisState("complete");
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Analysis failed.");
      setAnalysisState("error");
    }
  }

  function reset() {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setExplanation(null);
    setExplanationState("idle");
    setExplanationError(null);
    setMessage(null);
    setAnalysisState("idle");
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="RetinaStage home">
          <span className="brand-mark"><Icon name="eye" /></span>
          <span>
            <strong>RetinaStage</strong>
            <small>Evidence-aware retinal review</small>
          </span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#analyse">Analyse</a>
          <a href="#method">How it works</a>
          <a href="#research">Research</a>
        </nav>
        <ServiceStatus health={health} />
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <span className="eyebrow"><Icon name="spark" /> Educational computer-vision prototype</span>
            <h1>Retinal stage analysis with evidence attached.</h1>
            <p>
              A five-grade research workflow that keeps calibrated confidence,
              technical image quality and human-review decisions visible.
            </p>
            <div className="hero-actions">
              <a className="button primary" href="#analyse">Analyse an image</a>
              <a className="button secondary" href="#method">View the workflow</a>
            </div>
            {healthError && (
              <p className="connection-warning" role="status">
                The API at {API_BASE_URL} is not reachable. Start FastAPI before analysing an image.
              </p>
            )}
          </div>
          <div className="hero-visual" aria-hidden="true">
            <div className="retina-orbit orbit-one" />
            <div className="retina-orbit orbit-two" />
            <div className="retina-core"><Icon name="eye" /></div>
            <div className="signal-card signal-top"><span>5</span> ordered stages</div>
            <div className="signal-card signal-bottom"><span>✓</span> calibrated review</div>
          </div>
        </section>

        <section className="trust-strip" aria-label="System characteristics">
          <div><strong>Five-stage</strong><span>Ordinal classification</span></div>
          <div><strong>Calibrated</strong><span>Confidence policy</span></div>
          <div><strong>Quality-aware</strong><span>Technical review flags</span></div>
          <div><strong>Private by design</strong><span>In-memory processing</span></div>
        </section>

        <section className="analysis-section" id="analyse">
          <div className="section-intro">
            <span className="section-label">Analysis workspace</span>
            <h2>Upload one retinal photograph</h2>
            <p>PNG or JPEG, up to 20 MB. Images are processed in memory and are not retained by the API.</p>
          </div>

          <div className={`analysis-grid ${result ? "with-result" : ""}`}>
            <form className="upload-card" onSubmit={runAnalysis}>
              <div className="card-heading">
                <div>
                  <span className="section-label">Step 01</span>
                  <h2>Choose an image</h2>
                </div>
                {file && <button className="text-button" type="button" onClick={reset}>Clear</button>}
              </div>

              <div
                className={`drop-zone ${dragging ? "dragging" : ""} ${previewUrl ? "has-preview" : ""}`}
                onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
              >
                {previewUrl ? (
                  <img src={previewUrl} alt="Selected retinal photograph preview" />
                ) : (
                  <div className="drop-prompt">
                    <span className="upload-icon"><Icon name="upload" /></span>
                    <strong>Drop a retinal image here</strong>
                    <span>or browse from your computer</span>
                  </div>
                )}
                <input
                  ref={inputRef}
                  id="retinal-image"
                  type="file"
                  accept="image/png,image/jpeg"
                  onChange={onFileChange}
                />
                <label className="drop-action" htmlFor="retinal-image">
                  {file ? "Replace image" : "Browse image"}
                </label>
              </div>

              {file && (
                <div className="file-meta">
                  <div><span>Selected file</span><strong>{file.name}</strong></div>
                  <span>{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                </div>
              )}

              {message && <p className="form-error" role="alert">{message}</p>}

              <button
                className="analyse-button"
                type="submit"
                disabled={!file || analysisState === "loading" || health?.status !== "ready"}
              >
                {analysisState === "loading" ? <><span className="spinner" /> Analysing image…</> : <><Icon name="spark" /> Run calibrated analysis</>}
              </button>
              <p className="form-footnote">Research use only · Not a medical diagnosis</p>
            </form>

            {result ? (
              <ResultPanel
                result={result}
                explanation={explanation}
                explanationState={explanationState}
                explanationError={explanationError}
              />
            ) : (
              <aside className="empty-results" aria-live="polite">
                <div className="empty-graphic"><Icon name="eye" /></div>
                <span className="section-label">Step 02</span>
                <h2>Your evidence panel will appear here</h2>
                <p>After analysis, review all five probabilities, image-quality measurements and the human-review decision together.</p>
                <ul>
                  <li><span>01</span> Calibrated stage confidence</li>
                  <li><span>02</span> Technical-quality flags</li>
                  <li><span>03</span> Grad-CAM attention evidence</li>
                  <li><span>04</span> Grounded RetinaGuide explanations</li>
                  <li><span>05</span> Auditable input and model hashes</li>
                </ul>
              </aside>
            )}
          </div>
        </section>

        <section className="method-section" id="method">
          <div className="section-intro centered">
            <span className="section-label">Transparent by design</span>
            <h2>One result, several independent checks</h2>
            <p>The interface keeps model output separate from technical-quality and uncertainty policies.</p>
          </div>
          <div className="method-grid">
            <article><span>01</span><h3>Prepare</h3><p>Validate, crop the retinal field, preserve aspect ratio and resize to the model input.</p></article>
            <article><span>02</span><h3>Classify</h3><p>Apply the fixed EfficientNetB0 ordinal checkpoint across five ordered stages.</p></article>
            <article><span>03</span><h3>Calibrate</h3><p>Scale probabilities with the temperature fixed before final-test evaluation.</p></article>
            <article><span>04</span><h3>Review</h3><p>Combine confidence and technical-quality flags without treating either as diagnosis.</p></article>
          </div>
        </section>

        <section className="research-section" id="research">
          <div>
            <span className="section-label light">Research roadmap</span>
            <h2>Evidence and explanation stay connected.</h2>
            <p>Grad-CAM shows model-associated regions, while RetinaGuide explains only the returned prediction, quality and policy fields.</p>
          </div>
          <div className="roadmap-cards">
            <article><span>Integrated in application</span><h3>Grad-CAM</h3><p>Qualitative model-attention evidence with an explicit interpretation warning.</p></article>
            <article><span>Integrated in application</span><h3>RetinaGuide</h3><p>Deterministic plain-language explanations with explicit medical-safety boundaries.</p></article>
          </div>
        </section>
      </main>

      <footer>
        <div className="brand footer-brand"><span className="brand-mark"><Icon name="eye" /></span><span><strong>RetinaStage</strong><small>Educational research prototype</small></span></div>
        <p>Not validated for clinical diagnosis. Outputs require appropriately qualified review.</p>
      </footer>
    </div>
  );
}
