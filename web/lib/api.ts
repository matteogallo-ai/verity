/**
 * Typed API client for the Verity FastAPI backend.
 *
 * No secrets on the client — every LLM call is server-side. The only
 * client-visible env is `NEXT_PUBLIC_API_BASE` (defaults to
 * `http://localhost:8000`, i.e. the FastAPI dev server started by
 * `uv run verity serve`).
 */

import type {
  AskResponse,
  IngestResponse,
  StatusResponse,
} from "./types";

const DEFAULT_BASE = "http://localhost:8000";

function apiBase(): string {
  const base = process.env.NEXT_PUBLIC_API_BASE;
  return (base && base.length > 0 ? base : DEFAULT_BASE).replace(/\/$/, "");
}

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(`API ${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
  }
}

/**
 * The message to show for an error body. FastAPI puts a string in `detail` for most
 * errors and an object with a `message` for 503 outages; every error body carries
 * the `request_id` that finds the server-side log line.
 */
export function errorDetail(body: unknown, fallback: string): string {
  if (!body || typeof body !== "object") return fallback;
  const { detail, request_id: requestId } = body as {
    detail?: unknown;
    request_id?: unknown;
  };
  let message = fallback;
  if (typeof detail === "string") {
    message = detail;
  } else if (detail && typeof detail === "object" && "message" in detail) {
    message = String((detail as { message: unknown }).message);
  }
  return typeof requestId === "string" && requestId
    ? `${message} (request ${requestId})`
    : message;
}

async function jsonOrThrow(response: Response): Promise<unknown> {
  if (!response.ok) {
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      /* body was not json — fall back to statusText */
    }
    throw new ApiError(response.status, errorDetail(body, response.statusText));
  }
  return response.json();
}

export async function fetchStatus(signal?: AbortSignal): Promise<StatusResponse> {
  const response = await fetch(`${apiBase()}/status`, {
    signal,
    cache: "no-store",
  });
  return (await jsonOrThrow(response)) as StatusResponse;
}

export async function ask(question: string, signal?: AbortSignal): Promise<AskResponse> {
  const response = await fetch(`${apiBase()}/ask`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
    cache: "no-store",
  });
  return (await jsonOrThrow(response)) as AskResponse;
}

export async function ingest(
  file: File,
  signal?: AbortSignal
): Promise<IngestResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${apiBase()}/ingest`, {
    method: "POST",
    body: formData,
    signal,
    cache: "no-store",
  });
  return (await jsonOrThrow(response)) as IngestResponse;
}
