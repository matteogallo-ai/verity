"use client";

import { useCallback, useEffect, useState } from "react";
import { AnswerCard } from "@/components/AnswerCard";
import { AskBox } from "@/components/AskBox";
import { ProvenanceBadge } from "@/components/ProvenanceBadge";
import { SourcePanel } from "@/components/SourcePanel";
import { StubModeBanner } from "@/components/StubModeBanner";
import { UploadPanel } from "@/components/UploadPanel";
import { ApiError, ask, fetchStatus } from "@/lib/api";
import type {
  AskResponse,
  Citation,
  IngestResponse,
  RetrievedChunk,
  StatusResponse,
} from "@/lib/types";

type Status =
  | { kind: "unknown" }
  | { kind: "up"; response: StatusResponse }
  | { kind: "down"; detail: string };

export default function HomePage() {
  const [status, setStatus] = useState<Status>({ kind: "unknown" });
  const [asking, setAsking] = useState(false);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [askError, setAskError] = useState<string | null>(null);
  const [activeCitation, setActiveCitation] = useState<number | null>(null);
  const [activeChunkId, setActiveChunkId] = useState<string | null>(null);

  const refreshStatus = useCallback(async () => {
    try {
      const response = await fetchStatus();
      setStatus({ kind: "up", response });
    } catch (exc) {
      const detail = exc instanceof ApiError ? exc.detail : String(exc);
      setStatus({ kind: "down", detail });
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
  }, [refreshStatus]);

  const handleAsk = useCallback(async (question: string) => {
    setAskError(null);
    setActiveCitation(null);
    setActiveChunkId(null);
    setAsking(true);
    try {
      const response = await ask(question);
      setResult(response);
    } catch (exc) {
      const detail = exc instanceof ApiError ? exc.detail : String(exc);
      setAskError(detail);
    } finally {
      setAsking(false);
    }
  }, []);

  const handleIngested = useCallback(
    async (_response: IngestResponse) => {
      void _response; // consumed via status refresh
      await refreshStatus();
    },
    [refreshStatus]
  );

  const onCitationClick = useCallback(
    (marker: { index: number; citation: Citation }) => {
      setActiveCitation(marker.index);
      setActiveChunkId(marker.citation.chunk_id);
    },
    []
  );

  const onChunkClick = useCallback((chunk: RetrievedChunk) => {
    setActiveChunkId(chunk.chunk_id);
    // If this chunk is cited, sync the citation highlight too.
    const currentCitations = result?.answer.citations ?? [];
    const idx = currentCitations.findIndex((c) => c.chunk_id === chunk.chunk_id);
    setActiveCitation(idx >= 0 ? idx + 1 : null);
  }, [result]);

  const provenance = result?.provenance ?? (status.kind === "up" ? status.response.provenance : null);

  return (
    <>
      <StubModeBanner provenance={provenance} />
      <header className="topbar">
        <div className="brand">
          <span className="mark mono">verity</span>
          <span className="tagline">evaluated, observable RAG</span>
        </div>
        <div className="topbar-right">
          {status.kind === "up" && provenance && (
            <ProvenanceBadge provenance={provenance} />
          )}
          {status.kind === "up" && (
            <span className="mono corpus">
              corpus · {status.response.corpus.n_documents} docs ·{" "}
              {status.response.corpus.n_chunks} chunks
            </span>
          )}
          {status.kind === "down" && (
            <span role="alert" className="mono down">backend down: {status.detail}</span>
          )}
        </div>
      </header>

      <main className="page">
        <section className="left">
          <UploadPanel onIngested={handleIngested} />
          <div className="ask-wrap">
            <AskBox disabled={asking || status.kind !== "up"} onAsk={handleAsk} />
            {asking && <p className="pending mono">asking…</p>}
            {askError && (
              <p role="alert" className="err">
                <b>ask failed:</b> {askError}
              </p>
            )}
          </div>
          {result && (
            <AnswerCard
              answer={result.answer}
              provenance={result.provenance}
              activeCitationIndex={activeCitation}
              activeChunkId={activeChunkId}
              onCitationClick={onCitationClick}
            />
          )}
          {!result && !asking && status.kind === "up" && (
            <div className="empty">
              <h3>Ready.</h3>
              <p>
                Ask a question about the ingested documents. Try one of the demo
                questions above (or upload a doc first).
              </p>
            </div>
          )}
        </section>
        <SourcePanel
          answer={result?.answer ?? null}
          activeChunkId={activeChunkId}
          onChunkClick={onChunkClick}
        />
      </main>

      <style jsx>{`
        .topbar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 14px 24px;
          border-bottom: 1px solid var(--border);
          gap: var(--space-4);
        }
        .brand {
          display: flex;
          align-items: baseline;
          gap: 12px;
        }
        .mark {
          font-size: 16px;
          font-weight: 600;
          color: var(--fg);
          letter-spacing: 0.04em;
        }
        .tagline {
          color: var(--fg-mute);
          font-size: 12px;
        }
        .topbar-right {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .corpus {
          font-size: 11px;
          color: var(--fg-mute);
        }
        .down {
          font-size: 11px;
          color: var(--error);
        }
        .page {
          display: grid;
          grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr);
          gap: var(--space-5);
          padding: var(--space-5);
          max-width: 1400px;
          margin: 0 auto;
        }
        .left {
          display: flex;
          flex-direction: column;
          gap: var(--space-4);
          min-width: 0;
        }
        .ask-wrap {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }
        .pending { font-size: 11px; color: var(--fg-mute); }
        .err {
          padding: var(--space-2) var(--space-3);
          background: color-mix(in oklab, var(--error) 15%, transparent);
          border: 1px solid color-mix(in oklab, var(--error) 40%, transparent);
          color: var(--error);
          border-radius: var(--radius);
          font-size: 13px;
        }
        .empty {
          padding: var(--space-6) var(--space-4);
          border: 1px dashed var(--border-hi);
          border-radius: var(--radius);
          background: var(--bg-panel);
          color: var(--fg-dim);
        }
        .empty h3 { margin: 0 0 6px; color: var(--fg); font-size: 15px; }
        .empty p { margin: 0; font-size: 13px; }
        @media (max-width: 900px) {
          .page {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </>
  );
}
