export interface ArtifactReadiness {
  model: boolean;
  calibration: boolean;
  quality_summary: boolean;
}

export interface HealthResponse {
  status: "ready" | "not_ready";
  artifacts: ArtifactReadiness;
  model_loaded: boolean;
}

export interface GradeProbability {
  grade: number;
  label: string;
  probability: number;
}

export interface QualityMetrics {
  brightness_mean: number;
  contrast_std: number;
  sharpness_laplacian_variance: number;
  dark_pixel_fraction: number;
  bright_pixel_fraction: number;
  retinal_field_coverage: number;
}

export interface QualityAssessment {
  metrics: QualityMetrics;
  flags: string[];
  requires_review: boolean;
  interpretation: string;
}

export interface InferencePolicy {
  temperature: number;
  confidence_threshold: number;
}

export interface PredictionResponse {
  input_sha256: string;
  model_sha256: string;
  predicted_grade: number;
  predicted_label: string;
  confidence: number;
  uncertain: boolean;
  requires_human_review: boolean;
  review_reasons: string[];
  probabilities: GradeProbability[];
  quality: QualityAssessment;
  policy: InferencePolicy;
  educational_notice: string;
}
