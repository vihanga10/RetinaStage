import { describe, expect, it } from "vitest";

import {
  createGuideWelcome,
  initialGuideQuestions,
  RETINAGUIDE_MESSAGE_LIMIT,
  validGuideMessage,
} from "./retinaguide";
import type { PredictionResponse } from "./types";

function resultWithReview(required: boolean): PredictionResponse {
  return {
    input_sha256: "input",
    model_sha256: "model",
    predicted_grade: 2,
    predicted_label: "Moderate",
    confidence: 0.45,
    uncertain: required,
    requires_human_review: required,
    review_reasons: required ? ["LOW_MODEL_CONFIDENCE"] : [],
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
      requires_review: false,
      interpretation: "",
    },
    policy: { temperature: 1, confidence_threshold: 0.55 },
    educational_notice: "",
  };
}

describe("RetinaGuide view helpers", () => {
  it("grounds the welcome message in the current result", () => {
    const welcome = createGuideWelcome(resultWithReview(true));
    expect(welcome.content).toContain("Grade 2 (Moderate)");
    expect(welcome.content).toContain("requests human review");
  });

  it("adapts the review suggestion to the current policy result", () => {
    expect(initialGuideQuestions(resultWithReview(true))[1]).toContain(
      "required",
    );
    expect(initialGuideQuestions(resultWithReview(false))[1]).toContain(
      "no review flag",
    );
  });

  it("rejects blank and overlong messages", () => {
    expect(validGuideMessage("   ")).toBe(false);
    expect(validGuideMessage("Explain confidence")).toBe(true);
    expect(validGuideMessage("x".repeat(RETINAGUIDE_MESSAGE_LIMIT + 1))).toBe(
      false,
    );
  });
});
