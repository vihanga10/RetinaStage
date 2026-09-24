import type { PredictionResponse } from "./types";

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatFlag(flag: string): string {
  return flag
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function reviewState(result: PredictionResponse): {
  tone: "review" | "clear";
  eyebrow: string;
  title: string;
} {
  if (result.requires_human_review) {
    return {
      tone: "review",
      eyebrow: "Review required",
      title: "Hold for qualified human review",
    };
  }
  return {
    tone: "clear",
    eyebrow: "No automated flag",
    title: "No technical review trigger detected",
  };
}
