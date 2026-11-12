"use client";

import { useMemo } from "react";
import { segmentAnswer } from "@/lib/highlight";
import { labelsFor, type Language } from "@/lib/labels";
import type { AnswerView } from "@/lib/types";

/**
 * A member's view of an answer (VERITY_UI_MODE=presentation): the text, then each
 * citation's article and quote; or the refusal message alone. The developer
 * read-outs of AnswerCard and RefusalState (model, confidence, rationale, usage,
 * ids) are left out on purpose: in a client meeting they read as noise.
 */
export function PresentationAnswer({
  answer,
  language,
}: {
  answer: AnswerView;
  language?: Language;
}) {
  const t = labelsFor(language);
  const segments = useMemo(
    () => segmentAnswer(answer.text, answer.citations),
    [answer.text, answer.citations],
  );
  const sectionOf = useMemo(
    () => new Map(answer.hits_used.map((hit) => [hit.chunk_id, hit.section])),
    [answer.hits_used],
  );
  const refused = answer.confidence.refused;

  return (
    <article
      className={refused ? "card refused" : "card"}
      role="region"
      aria-labelledby="presentation-heading"
    >
      <h2 id="presentation-heading">{refused ? t.refused : t.answer}</h2>
      <p className="text">
        {refused
          ? answer.text
          : segments.map((segment, i) =>
              segment.kind === "text" ? (
                <span key={i}>{segment.content}</span>
              ) : (
                <sup key={i} className="marker">
                  [{segment.marker.index}]
                </sup>
              ),
            )}
      </p>
      {!refused && answer.citations.length > 0 && (
        <ol className="citations" aria-label={t.sources}>
          {answer.citations.map((c, i) => {
            const where = sectionOf.get(c.chunk_id);
            return (
              <li key={`${c.chunk_id}-${i}`}>
                <span className="marker">[{i + 1}]</span>
                {where ? <span className="where">{where}</span> : null}
                <blockquote>{c.quote}</blockquote>
              </li>
            );
          })}
        </ol>
      )}
      <style jsx>{`
        .card {
          background: var(--bg-panel);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: var(--space-5);
          display: flex;
          flex-direction: column;
          gap: var(--space-4);
        }
        .refused {
          background: var(--refuse-soft);
          border-color: color-mix(in oklab, var(--refuse) 45%, var(--border));
        }
        h2 {
          margin: 0;
          font-size: 13px;
          color: var(--fg-dim);
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }
        .text {
          margin: 0;
          font-size: 17px;
          line-height: 1.65;
        }
        .marker {
          color: var(--accent);
          font-size: 11px;
          margin: 0 2px;
        }
        .citations {
          list-style: none;
          margin: 0;
          padding: var(--space-3) 0 0;
          border-top: 1px solid var(--border);
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
        }
        .where {
          font-weight: 600;
          font-size: 14px;
          margin-left: 4px;
        }
        blockquote {
          margin: 4px 0 0;
          padding-left: 12px;
          border-left: 3px solid var(--accent);
          color: var(--fg-dim);
          font-size: 14px;
        }
      `}</style>
    </article>
  );
}
