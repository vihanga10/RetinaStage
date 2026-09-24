import {
  type CSSProperties,
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
  useEffect,
  useRef,
  useState,
} from "react";

import { API_BASE_URL, fetchHealth, predictImage } from "./api";
import type { HealthResponse, PredictionResponse } from "./types";
import { formatFlag, formatPercent, reviewState } from "./view-model";

const MAX_FILE_BYTES = 20 * 1024 * 1024;
const ACCEPTED_TYPES = new Set(["image/png", "image/jpeg"]);

type AnalysisState = "idle" | "loading" | "complete" | "error";

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

function ResultPanel({ result }: { result: PredictionResponse }) {
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
      setAnalysisState("complete");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Analysis failed.");
      setAnalysisState("error");
    }
  }

  function reset() {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
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
              <ResultPanel result={result} />
            ) : (
              <aside className="empty-results" aria-live="polite">
                <div className="empty-graphic"><Icon name="eye" /></div>
                <span className="section-label">Step 02</span>
                <h2>Your evidence panel will appear here</h2>
                <p>After analysis, review all five probabilities, image-quality measurements and the human-review decision together.</p>
                <ul>
                  <li><span>01</span> Calibrated stage confidence</li>
                  <li><span>02</span> Technical-quality flags</li>
                  <li><span>03</span> Auditable input and model hashes</li>
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
            <h2>The evidence layer is ready for explanation.</h2>
            <p>Grad-CAM and RetinaGuide are the next application integrations. They will explain a fixed result without changing the selected model or clinical boundary.</p>
          </div>
          <div className="roadmap-cards">
            <article><span>Available in research workflow</span><h3>Grad-CAM</h3><p>Qualitative model-attention evidence with an explicit interpretation warning.</p></article>
            <article><span>Next application stage</span><h3>RetinaGuide</h3><p>Plain-language explanations grounded only in the returned prediction, quality and policy fields.</p></article>
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
