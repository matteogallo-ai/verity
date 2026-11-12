import { labelsFor, type Language } from "@/lib/labels";
import type { AnswerView, Provenance } from "@/lib/types";

/**
 * Rendering of ``answer.confidence.score``.
 *
 * The score is NOT a composite computed by the UI. It is a passthrough of the
 * value the agent's confidence scorer returned for this exact request — a
 * canned constant in stub mode (0.88 for on-theme, 0.15 for out-of-scope),
 * the scorer model's self-reported value in live mode (not calibrated).
 *
 * Rule of iron: never present a bare percentage authoritative next to an
 * answer. This component:
 *
 * 1. Labels the number "agent-reported (<agent_model>)" so the reader knows
 *    what produced it.
 * 2. Shows the three raw signals a careful analyst would look at BEFORE
 *    trusting a confidence number: refused flag, count of citations backing
 *    the answer, and the top rerank score of the evidence set.
 * 3. Never applies a colour that reads as "high" or "low quality" — the
 *    number is neutrally styled so the surrounding signals do the framing.
 */
export function ConfidenceReadout({
  answer,
  provenance,
  language,
}: {
  answer: AnswerView;
  provenance: Provenance;
  language?: Language;
}) {
  const t = labelsFor(language);
  const topScore = answer.hits_used.length > 0 ? answer.hits_used[0].score : null;
  const scoreDisplay = answer.confidence.score.toFixed(2);
  const label = provenance.is_stub
    ? `${t.agentReported} (stub: ${provenance.agent_model})`
    : `${t.agentReported} (${provenance.agent_model})`;

  return (
    <section
      className="readout"
      role="group"
      aria-label={t.readoutGroup}
    >
      <div className="head">
        <span className="lbl">{t.confidence}</span>
        <span className="mono val">{scoreDisplay}</span>
        <span className="prov mono">· {label}</span>
      </div>
      <ul className="signals" aria-label={t.signalsGroup}>
        <li>
          <span className="k">{t.refusedSignal}</span>
          <span className="v mono">{answer.confidence.refused ? t.yes : t.no}</span>
        </li>
        <li>
          <span className="k">{t.citations}</span>
          <span className="v mono">{answer.citations.length}</span>
        </li>
        <li>
          <span className="k">{t.topRerankScore}</span>
          <span className="v mono">
            {topScore !== null ? topScore.toFixed(3) : t.notAvailable}
          </span>
        </li>
        <li>
          <span className="k">{t.evidenceHits}</span>
          <span className="v mono">{answer.hits_used.length}</span>
        </li>
      </ul>
      {answer.confidence.rationale && (
        <p className="rationale">
          <span className="lbl">{t.rationale} · </span>
          {answer.confidence.rationale}
        </p>
      )}
      <p className="note">{t.scoreNote}</p>
      <style jsx>{`
        .readout {
          display: flex;
          flex-direction: column;
          gap: 6px;
          padding: 10px 12px;
          border: 1px solid var(--border);
          border-radius: var(--radius);
          background: var(--bg-elev);
        }
        .head { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
        .lbl {
          font-family: var(--mono);
          font-size: 10px;
          text-transform: lowercase;
          letter-spacing: 0.04em;
          color: var(--fg-mute);
        }
        .val { font-size: 16px; color: var(--fg); }
        .prov { font-size: 11px; color: var(--fg-dim); }
        .signals {
          list-style: none;
          padding: 0;
          margin: 0;
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
          gap: 6px 12px;
        }
        .signals li {
          display: flex;
          justify-content: space-between;
          gap: 6px;
          font-size: 11px;
        }
        .signals .k { color: var(--fg-mute); }
        .signals .v { color: var(--fg); }
        .rationale {
          margin: 0;
          font-size: 12px;
          color: var(--fg-dim);
        }
        .note {
          margin: 0;
          font-size: 10px;
          color: var(--fg-mute);
          font-style: italic;
        }
      `}</style>
    </section>
  );
}
