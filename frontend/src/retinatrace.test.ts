import { describe, expect, it } from "vitest";

import {
  formatReceiptTimestamp,
  parseRetinaTraceReceipt,
  receiptDownloadName,
  serialiseRetinaTraceReceipt,
  shortReceiptHash,
} from "./retinatrace";
import type { RetinaTraceReceipt } from "./types";

function receiptFixture(): RetinaTraceReceipt {
  return {
    receipt_type: "retinastage_prediction_evidence",
    schema_version: 1,
    issued_at_utc: "2026-09-25T05:30:00Z",
    traceability: {
      input_sha256: "1".repeat(64),
      model_sha256: "2".repeat(64),
    },
    prediction: {
      predicted_grade: 0,
      predicted_label: "No DR",
      confidence: 0.9,
      uncertain: false,
      requires_human_review: false,
      review_reasons: [],
      probabilities: [],
    },
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
      interpretation: "Technical checks only.",
    },
    policy: { temperature: 1, confidence_threshold: 0.55 },
    explanation: { included: false },
    privacy: {
      retinal_image_included: false,
      original_filename_included: false,
      gradcam_image_data_included: false,
    },
    educational_notice: "Educational research prototype only.",
    integrity_notice: "Checksum only.",
    integrity: {
      algorithm: "SHA-256",
      canonicalization: "canonical JSON",
      scope: "all fields except checksum",
      receipt_sha256: "a".repeat(64),
    },
  };
}

describe("RetinaTrace frontend helpers", () => {
  it("round-trips one receipt as readable JSON", () => {
    const receipt = receiptFixture();
    const serialised = serialiseRetinaTraceReceipt(receipt);
    expect(parseRetinaTraceReceipt(serialised)).toEqual(receipt);
    expect(serialised.endsWith("\n")).toBe(true);
  });

  it("creates an auditable filename without the original image name", () => {
    expect(receiptDownloadName(receiptFixture())).toBe(
      "retinatrace_111111111111_20260925T053000Z.json",
    );
  });

  it("rejects unrelated or malformed JSON files", () => {
    expect(() => parseRetinaTraceReceipt("not-json")).toThrow("valid JSON");
    expect(() => parseRetinaTraceReceipt('{"receipt_type":"other"}')).toThrow(
      "not a RetinaTrace receipt",
    );
  });

  it("shortens a checksum while preserving both ends", () => {
    expect(shortReceiptHash(receiptFixture())).toBe(
      "aaaaaaaaaaaa…aaaaaaaaaaaa",
    );
  });

  it("formats the UTC issue time without depending on browser locale", () => {
    expect(formatReceiptTimestamp(receiptFixture().issued_at_utc)).toBe(
      "2026-09-25 05:30:00 UTC",
    );
    expect(formatReceiptTimestamp("unrecognised-time")).toBe(
      "unrecognised-time",
    );
  });
});
