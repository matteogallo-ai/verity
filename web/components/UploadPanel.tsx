"use client";

import { useRef, useState } from "react";
import { ApiError, ingest } from "@/lib/api";
import type { IngestResponse } from "@/lib/types";

export interface UploadPanelProps {
  onIngested: (response: IngestResponse) => void;
}

export function UploadPanel({ onIngested }: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    setError(null);
    setPending(true);
    try {
      const response = await ingest(file);
      onIngested(response);
      if (inputRef.current) inputRef.current.value = "";
    } catch (exc) {
      const msg = exc instanceof ApiError ? exc.detail : String(exc);
      setError(msg);
    } finally {
      setPending(false);
    }
  };

  return (
    <section className="panel" aria-labelledby="upload-heading">
      <div className="panel-head">
        <h2 id="upload-heading">Upload</h2>
        <p className="hint">
          Add a Markdown, PDF, DOCX, or TXT document. Content-addressed IDs —
          same bytes produce the same document.
        </p>
      </div>
      <label className="dropzone" htmlFor="upload-input">
        <span>{pending ? "Ingesting…" : "Drop a file or click to browse"}</span>
        <input
          ref={inputRef}
          id="upload-input"
          type="file"
          accept=".pdf,.docx,.md,.markdown,.txt"
          disabled={pending}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
      </label>
      {error && (
        <div role="alert" className="error">
          <strong>upload failed:</strong> {error}
        </div>
      )}
      <style jsx>{`
        .panel {
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
        }
        .panel-head h2 {
          margin: 0;
          font-size: 13px;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--fg-dim);
        }
        .hint {
          margin: 4px 0 0;
          color: var(--fg-mute);
          font-size: 12px;
        }
        .dropzone {
          border: 1px dashed var(--border-hi);
          border-radius: var(--radius);
          padding: 24px 12px;
          display: flex;
          justify-content: center;
          align-items: center;
          background: var(--bg-elev);
          color: var(--fg-dim);
          cursor: pointer;
          transition: border-color 120ms ease;
        }
        .dropzone:hover, .dropzone:focus-within {
          border-color: var(--accent);
          color: var(--fg);
        }
        .dropzone input[type="file"] {
          position: absolute;
          width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden;
          clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
        }
        .error {
          padding: var(--space-2) var(--space-3);
          background: color-mix(in oklab, var(--error) 15%, transparent);
          border: 1px solid color-mix(in oklab, var(--error) 40%, transparent);
          color: var(--error);
          border-radius: var(--radius);
          font-size: 13px;
        }
      `}</style>
    </section>
  );
}
