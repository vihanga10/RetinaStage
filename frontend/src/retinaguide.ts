import type { PredictionResponse, RetinaGuideMessage } from "./types";

export const RETINAGUIDE_MESSAGE_LIMIT = 500;

export function createGuideWelcome(
  result: PredictionResponse,
): RetinaGuideMessage {
  const review = result.requires_human_review
    ? "The automated policy requests human review."
    : "No automated review trigger was recorded.";
  return {
    id: "guide-welcome",
    role: "assistant",
    content:
      `I can explain the displayed Grade ${result.predicted_grade} ` +
      `(${result.predicted_label}) result. ${review} Ask about confidence, ` +
      "probabilities, technical quality, review flags, or Grad-CAM.",
  };
}

export function initialGuideQuestions(
  result: PredictionResponse,
): string[] {
  return [
    "What does the confidence mean?",
    result.requires_human_review
      ? "Why is human review required?"
      : "Why was no review flag triggered?",
    "What does the Grad-CAM panel show?",
  ];
}

export function validGuideMessage(value: string): boolean {
  const length = value.trim().length;
  return length > 0 && length <= RETINAGUIDE_MESSAGE_LIMIT;
}
