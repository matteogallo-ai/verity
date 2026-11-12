"use client";

import { useCallback, useEffect, useState } from "react";
import { AnswerCard } from "@/components/AnswerCard";
import { AskBox } from "@/components/AskBox";
import { PresentationAnswer } from "@/components/PresentationAnswer";
import { ProvenanceBadge } from "@/components/ProvenanceBadge";
import { SourcePanel } from "@/components/SourcePanel";
import { StubModeBanner } from "@/components/StubModeBanner";
import { UploadPanel } from "@/components/UploadPanel";
import { ApiError, ask, fetchStatus } from "@/lib/api";
import { labelsFor, labelsLanguage } from "@/lib/labels";
import type {
  AskResponse,
  Citation,
  IngestResponse,
  RetrievedChunk,
  StatusResponse,
} from "@/lib/types";

// A demo started before the API is up must find it without a reload.
const STATUS_RETRY_MS = 3000;

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

  useEffect(() => {
    if (status.kind !== "down") return;
    const timer = setTimeout(() => void refreshStatus(), STATUS_RETRY_MS);
    return () => clearTimeout(timer);
  }, [status, refreshStatus]);

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
  const language = labelsLanguage(status.kind === "up" ? status.response.language : undefined);
  const labels = labelsFor(language);
  const questions = status.kind === "up" ? (status.response.suggested_questions ?? []) : [];
  const presenting = status.kind === "up" && status.response.ui_mode === "presentation";
  // Until /status answers, the mode is unknown: show neither view's chrome, so a
  // client never sees the developer read-outs flash before the presentation.
  const developer = status.kind === "down" || (status.kind === "up" && !presenting);
  const corpusName = status.kind === "up" ? status.response.corpus_name : null;
  const title = corpusName ? `${labels.presentationTitle} \u2013 ${corpusName}` : labels.presentationTitle;

  return (
    <>
      <StubModeBanner provenance={provenance} notice={presenting ? labels.stubNotice : undefined} />
      <header className="topbar">
        {presenting && <h1 className="title">{title}</h1>}
        {developer && (
          <div className="brand">
            <span className="mark mono">verity</span>
            <span className="tagline">evaluated, observable RAG</span>
          </div>
        )}
        <div className="topbar-right">
          {status.kind === "up" && !presenting && provenance && (
            <ProvenanceBadge provenance={provenance} />
          )}
          {status.kind === "up" && !presenting && (
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

      <main className={developer ? "page" : "page presenting"}>
        <section className="left">
          {developer && <UploadPanel onIngested={handleIngested} />}
          <div className="ask-wrap">
            <AskBox
              disabled={asking || status.kind !== "up"}
              questions={questions}
              language={language}
              onAsk={handleAsk}
            />
            {asking && <p className="pending mono">{labels.asking}</p>}
            {askError && (
              <p role="alert" className="err">
                {presenting ? (
                  labels.presentationAskFailed
                ) : (
                  <>
                    <b>{labels.askFailed}</b> {askError}
                  </>
                )}
              </p>
            )}
          </div>
          {result && presenting && (
            <PresentationAnswer answer={result.answer} language={language} />
          )}
          {result && !presenting && (
            <AnswerCard
              answer={result.answer}
              provenance={result.provenance}
              activeCitationIndex={activeCitation}
              activeChunkId={activeChunkId}
              onCitationClick={onCitationClick}
              language={language}
            />
          )}
          {!result && !asking && status.kind === "up" && (
            <div className="empty">
              <h2>{labels.ready}</h2>
              <p>
                {presenting
                  ? questions.length > 0
                    ? labels.presentationReadyHintWithQuestions
                    : labels.presentationReadyHint
                  : questions.length > 0
                    ? labels.readyHintWithQuestions
                    : labels.readyHint}
              </p>
            </div>
          )}
        </section>
        {developer && (
          <SourcePanel
            answer={result?.answer ?? null}
            activeChunkId={activeChunkId}
            onChunkClick={onChunkClick}
            language={language}
          />
        )}
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
        .title {
          margin: 0;
          font-size: 18px;
          font-weight: 600;
        }
        .page.presenting {
          grid-template-columns: minmax(0, 1fr);
          max-width: 860px;
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
        .empty h2 { margin: 0 0 6px; color: var(--fg); font-size: 15px; }
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
