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

async function jsonOrThrow(response: Response): Promise<unknown> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (body && typeof body === "object" && "detail" in body) {
        detail = String((body as { detail: unknown }).detail);
      }
    } catch {
      /* body was not json — fall back to statusText */
    }
    throw new ApiError(response.status, detail);
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
