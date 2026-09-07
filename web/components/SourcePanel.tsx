"use client";

import type { AnswerView, RetrievedChunk } from "@/lib/types";

export interface SourcePanelProps {
  answer: AnswerView | null;
  activeChunkId: string | null;
  onChunkClick: (chunk: RetrievedChunk) => void;
}

/**
 * Right-hand panel — the honesty panel. Renders EVERY retrieved chunk the
 * agent reasoned over, marked cited vs not-cited, with the reranker score
 * as it came back from the API. Never a synthetic composite; never rounded
 * to hide a value.
 */
export function SourcePanel({ answer, activeChunkId, onChunkClick }: SourcePanelProps) {
  if (!answer) {
    return (
      <aside className="panel">
        <div className="empty">
          <p>The evidence used by the agent will appear here after you ask a question.</p>
        </div>
        <style jsx>{scoped}</style>
      </aside>
    );
  }
  return (
    <aside className="panel" aria-labelledby="sources-heading">
      <header>
        <h2 id="sources-heading" className="mono">sources · {answer.hits_used.length}</h2>
        <span className="mono note">
          rerank score · cited flag from API set-membership
        </span>
      </header>
      <ol className="chunks">
        {answer.hits_used.map((hit) => {
          const isActive = activeChunkId === hit.chunk_id;
          return (
            <li key={hit.chunk_id} className={isActive ? "active" : ""}>
              <button
                type="button"
                aria-pressed={isActive}
                onClick={() => onChunkClick(hit)}
                data-testid={`chunk-${hit.chunk_id}`}
                className="chunk-row"
              >
                <div className="head">
                  <span className={`badge ${hit.cited ? "cited" : "uncited"} mono`}>
                    {hit.cited ? "cited" : "not-cited"}
                  </span>
                  <span className="section" title={hit.section ?? ""}>
                    {hit.section ?? "(no section)"}
                  </span>
                  <span className="score mono">
                    {hit.kind}: {hit.score.toFixed(3)}
                  </span>
                </div>
                <p className="body">{hit.text}</p>
                <div className="meta mono">
                  <span title={hit.chunk_id}>chunk {hit.chunk_id.slice(0, 8)}</span>
                  <span title={hit.document_id}>doc {hit.document_id.slice(0, 8)}</span>
                  <span>chars {hit.char_start}–{hit.char_end}</span>
                </div>
              </button>
            </li>
          );
        })}
      </ol>
      <style jsx>{scoped}</style>
    </aside>
  );
}

const scoped = `
  .panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    max-height: 78vh;
    overflow: auto;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
  }
  h2 {
    margin: 0;
    font-size: 13px;
    color: var(--fg-dim);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .note { font-size: 10px; color: var(--fg-mute); }
  .empty {
    color: var(--fg-mute);
    font-size: 13px;
    padding: var(--space-5) 0;
  }
  .chunks {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .chunk-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
    width: 100%;
    text-align: left;
    padding: 10px 12px;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--fg);
  }
  li.active .chunk-row {
    border-color: color-mix(in oklab, var(--accent) 45%, var(--border-hi));
    background: var(--accent-soft);
  }
  .head {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  .badge {
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 10px;
    text-transform: lowercase;
    border: 1px solid var(--border-hi);
  }
  .badge.cited {
    color: var(--accent);
    border-color: color-mix(in oklab, var(--accent) 40%, transparent);
    background: var(--accent-soft);
  }
  .badge.uncited {
    color: var(--fg-mute);
    background: transparent;
  }
  .section {
    font-size: 12px;
    color: var(--fg-dim);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .score {
    font-size: 11px;
    color: var(--fg-dim);
  }
  .body {
    margin: 4px 0 0;
    font-size: 13px;
    line-height: 1.55;
    color: var(--fg);
    max-height: 6.5em;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 5;
    -webkit-box-orient: vertical;
  }
  .meta {
    display: flex;
    gap: 12px;
    font-size: 10px;
    color: var(--fg-mute);
  }
`;
