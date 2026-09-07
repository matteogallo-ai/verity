import type { Provenance } from "@/lib/types";

/**
 * The single most important pixel on the page: the badge that tells the user
 * WHICH agent produced the answer they're looking at. Sourced verbatim from
 * `provenance.agent_model` and `provenance.is_stub`.
 */
export function ProvenanceBadge({ provenance }: { provenance: Provenance }) {
  const label = provenance.is_stub ? "stub-agent" : provenance.agent_model;
  const tone = provenance.is_stub ? "stub" : "live";
  return (
    <span
      className={`prov prov-${tone}`}
      title={
        provenance.is_stub
          ? "Deterministic scripted stub — no real LLM. Answers are canned per theme."
          : `Real LLM agent (${provenance.agent_model}).`
      }
      aria-label={`Agent provenance: ${label}${provenance.is_stub ? " (stub)" : " (live)"}`}
    >
      <span aria-hidden="true" className="prov-dot" />
      <span className="prov-label">{label}</span>
      <style jsx>{`
        .prov {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 3px 8px;
          border-radius: 999px;
          font-family: var(--mono);
          font-size: 11px;
          letter-spacing: 0.02em;
          text-transform: lowercase;
          border: 1px solid var(--border-hi);
          background: var(--bg-elev);
          color: var(--fg);
        }
        .prov-stub {
          border-color: color-mix(in oklab, var(--warn) 40%, var(--border-hi));
          background: var(--warn-soft);
          color: var(--warn);
        }
        .prov-live {
          border-color: color-mix(in oklab, var(--ok) 40%, var(--border-hi));
          background: color-mix(in oklab, var(--ok) 8%, transparent);
          color: var(--ok);
        }
        .prov-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: currentColor;
        }
      `}</style>
    </span>
  );
}
