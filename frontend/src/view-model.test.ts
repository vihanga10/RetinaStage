import { describe, expect, it } from "vitest";

import type { PredictionResponse } from "./types";
import { formatFlag, formatPercent, reviewState } from "./view-model";

function resultWithReview(required: boolean): PredictionResponse {
  return {
    input_sha256: "input",
    model_sha256: "model",
    predicted_grade: 0,
    predicted_label: "No DR",
    confidence: 0.9,
    uncertain: false,
    requires_human_review: required,
    review_reasons: required ? ["HIGH_BRIGHTNESS"] : [],
    probabilities: [],
    quality: {
      metrics: {
        brightness_mean: 0,
        contrast_std: 0,
        sharpness_laplacian_variance: 0,
        dark_pixel_fraction: 0,
        bright_pixel_fraction: 0,
        retinal_field_coverage: 0,
      },
      flags: [],
      requires_review: required,
      interpretation: "",
    },
    policy: { temperature: 1, confidence_threshold: 0.5 },
    educational_notice: "",
  };
}

describe("result view model", () => {
  it("formats probabilities and review flags for display", () => {
    expect(formatPercent(0.99576)).toBe("99.6%");
    expect(formatFlag("LOW_MODEL_CONFIDENCE")).toBe(
      "Low Model Confidence",
    );
  });

  it("keeps quality review separate from model confidence", () => {
    expect(reviewState(resultWithReview(true)).tone).toBe("review");
    expect(reviewState(resultWithReview(false)).tone).toBe("clear");
  });
});
