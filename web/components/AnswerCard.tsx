"use client";

import { useMemo } from "react";
import { segmentAnswer } from "@/lib/highlight";
import type { AnswerView, Citation, Provenance } from "@/lib/types";
import { ConfidenceReadout } from "./ConfidenceReadout";
import { ProvenanceBadge } from "./ProvenanceBadge";
import { RefusalState } from "./RefusalState";

export interface AnswerCardProps {
  answer: AnswerView;
  provenance: Provenance;
  activeCitationIndex: number | null;
  activeChunkId: string | null;
  onCitationClick: (marker: { index: number; citation: Citation }) => void;
}

export function AnswerCard(props: AnswerCardProps) {
  const { answer, provenance, activeCitationIndex, activeChunkId, onCitationClick } = props;

  // Compute segments unconditionally so React hooks stay in a stable order
  // across refused / non-refused renders. When the answer is refused we never
  // read `segments`, but the hook still runs — cheap and rule-of-hooks safe.
  const segments = useMemo(
    () => segmentAnswer(answer.text, answer.citations),
    [answer.text, answer.citations]
  );

  if (answer.confidence.refused) {
    return <RefusalState answer={answer} provenance={provenance} />;
  }

  return (
    <article className="answer" role="region" aria-labelledby="answer-heading">
      <header>
        <h2 id="answer-heading" className="mono">answer</h2>
        <ProvenanceBadge provenance={provenance} />
      </header>
      <p className="text">
        {segments.map((segment, i) => {
          if (segment.kind === "text") {
            return <span key={i}>{segment.content}</span>;
          }
          const { marker } = segment;
          const isActive =
            activeCitationIndex === marker.index ||
            activeChunkId === marker.citation.chunk_id;
          return (
            <button
              key={i}
              type="button"
              className={`cite ${isActive ? "cite-active" : ""}`}
              aria-label={`Citation ${marker.index}: ${marker.citation.quote}`}
              aria-pressed={isActive}
              onClick={() => onCitationClick(marker)}
              data-testid={`citation-${marker.index}`}
            >
              [{marker.index}]
            </button>
          );
        })}
      </p>
      <ConfidenceReadout answer={answer} provenance={provenance} />
      <ul className="citations">
        {answer.citations.map((c, i) => {
          const idx = i + 1;
          const isActive =
            activeCitationIndex === idx || activeChunkId === c.chunk_id;
          return (
            <li key={c.chunk_id} className={isActive ? "active" : ""}>
              <button
                type="button"
                className="cite-row"
                aria-pressed={isActive}
                onClick={() => onCitationClick({ index: idx, citation: c })}
              >
                <span className="marker">[{idx}]</span>
                <span className="quote">“{c.quote}”</span>
                <span className="ids mono">chunk {c.chunk_id.slice(0, 8)}</span>
              </button>
            </li>
          );
        })}
      </ul>
      <footer className="usage mono" aria-label="Usage">
        <span>latency <b>{answer.usage.latency_ms.toFixed(0)} ms</b></span>
        <span>llm calls <b>{answer.usage.llm_calls}</b></span>
        <span>in tokens <b>{answer.usage.input_tokens}</b></span>
        <span>out tokens <b>{answer.usage.output_tokens}</b></span>
        <span>cost <b>${answer.usage.cost_usd.toFixed(4)}</b></span>
        <span title={answer.trace_id}>trace <b>{answer.trace_id.slice(0, 8)}</b></span>
      </footer>
      <style jsx>{`
        .answer {
          background: var(--bg-panel);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: var(--space-5);
          display: flex;
          flex-direction: column;
          gap: var(--space-4);
        }
        header {
          display: flex;
          align-items: center;
          justify-content: space-between;
        }
        h2 { margin: 0; font-size: 13px; color: var(--fg-dim); text-transform: uppercase; letter-spacing: 0.06em; }
        .text {
          margin: 0;
          font-size: 15px;
          line-height: 1.65;
        }
        .cite {
          display: inline;
          padding: 1px 6px;
          margin: 0 2px;
          font-family: var(--mono);
          font-size: 11px;
          border: 1px solid color-mix(in oklab, var(--accent) 40%, var(--border));
          background: var(--accent-soft);
          color: var(--accent);
          border-radius: 4px;
          vertical-align: baseline;
        }
        .cite:hover, .cite-active {
          background: color-mix(in oklab, var(--accent) 30%, transparent);
          color: var(--fg);
        }
        .citations {
          list-style: none;
          padding: 0;
          margin: 0;
          display: flex;
          flex-direction: column;
          gap: 4px;
          border-top: 1px solid var(--border);
          padding-top: var(--space-3);
        }
        .cite-row {
          display: flex;
          align-items: baseline;
          gap: 8px;
          width: 100%;
          text-align: left;
          padding: 6px 8px;
          background: transparent;
          border: 1px solid transparent;
          color: var(--fg);
        }
        li.active .cite-row {
          border-color: color-mix(in oklab, var(--accent) 40%, transparent);
          background: var(--accent-soft);
        }
        .marker { color: var(--accent); font-family: var(--mono); font-size: 11px; }
        .quote { color: var(--fg); font-size: 13px; }
        .ids { color: var(--fg-mute); margin-left: auto; }
        .usage {
          display: flex;
          gap: 14px;
          flex-wrap: wrap;
          padding-top: var(--space-3);
          border-top: 1px solid var(--border);
          font-size: 11px;
          color: var(--fg-mute);
        }
        .usage b { color: var(--fg); font-weight: 600; }
      `}</style>
    </article>
  );
}
