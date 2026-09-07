import type { AnswerView, Provenance } from "@/lib/types";
import { ConfidenceReadout } from "./ConfidenceReadout";
import { ProvenanceBadge } from "./ProvenanceBadge";

/**
 * First-class refusal panel. Not an error — a deliberate output the agent
 * chose because the retrieved evidence did not warrant an answer.
 *
 * Confidence is surfaced through ``ConfidenceReadout`` (labelled agent-reported,
 * underlying signals visible). Never a bare authoritative percentage.
 */
export function RefusalState({
  answer,
  provenance,
}: {
  answer: AnswerView;
  provenance: Provenance;
}) {
  return (
    <article
      className="refusal"
      role="region"
      aria-labelledby="refusal-heading"
    >
      <header>
        <h2 id="refusal-heading">
          <span className="mono">refused</span>
        </h2>
        <ProvenanceBadge provenance={provenance} />
      </header>
      <p className="text">{answer.text}</p>
      <ConfidenceReadout answer={answer} provenance={provenance} />
      <div className="ops mono" aria-label="Operational">
        <span>latency <b>{answer.usage.latency_ms.toFixed(0)} ms</b></span>
        <span>llm calls <b>{answer.usage.llm_calls}</b></span>
      </div>
      <p className="note">
        Calibrated refusal is a headline feature — the agent honoured its
        threshold rather than fabricating an answer. The hits used to make
        this decision are still shown on the right panel for auditability.
      </p>
      <style jsx>{`
        .refusal {
          background: var(--refuse-soft);
          border: 1px solid color-mix(in oklab, var(--refuse) 45%, var(--border));
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
        h2 { margin: 0; color: var(--refuse); font-size: 14px; }
        .text { margin: 0; font-size: 15px; }
        .ops {
          display: flex;
          gap: 14px;
          font-size: 11px;
          color: var(--fg-mute);
        }
        .ops b { color: var(--fg); font-weight: 600; }
        .note {
          margin: 0;
          font-size: 12px;
          color: var(--fg-mute);
        }
      `}</style>
    </article>
  );
}
