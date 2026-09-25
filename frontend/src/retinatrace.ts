import type { RetinaTraceReceipt } from "./types";

export const MAX_RECEIPT_BYTES = 512 * 1024;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function parseRetinaTraceReceipt(text: string): RetinaTraceReceipt {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error("The selected file is not valid JSON.");
  }
  if (!isRecord(value)) {
    throw new Error("The selected JSON does not contain a receipt object.");
  }
  if (value.receipt_type !== "retinastage_prediction_evidence") {
    throw new Error("The selected JSON is not a RetinaTrace receipt.");
  }
  if (value.schema_version !== 1) {
    throw new Error("The RetinaTrace receipt version is not supported.");
  }
  if (!isRecord(value.integrity)) {
    throw new Error("The receipt integrity section is missing.");
  }
  const checksum = value.integrity.receipt_sha256;
  if (typeof checksum !== "string" || !/^[0-9a-f]{64}$/.test(checksum)) {
    throw new Error("The receipt checksum is missing or invalid.");
  }
  return value as unknown as RetinaTraceReceipt;
}

export function serialiseRetinaTraceReceipt(
  receipt: RetinaTraceReceipt,
): string {
  return `${JSON.stringify(receipt, null, 2)}\n`;
}

export function receiptDownloadName(receipt: RetinaTraceReceipt): string {
  const inputPrefix = receipt.traceability.input_sha256.slice(0, 12);
  const timestamp = receipt.issued_at_utc.replace(/[^0-9TZ]/g, "");
  return `retinatrace_${inputPrefix}_${timestamp}.json`;
}

export function shortReceiptHash(receipt: RetinaTraceReceipt): string {
  const checksum = receipt.integrity.receipt_sha256;
  return `${checksum.slice(0, 12)}…${checksum.slice(-12)}`;
}
