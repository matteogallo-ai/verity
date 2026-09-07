import type { Provenance } from "@/lib/types";

/**
 * Persistent banner across the top of the page whenever the API is serving
 * the stub agent. Rendered iff `provenance.is_stub === true` — the same
 * condition the tests assert against.
 */
export function StubModeBanner({ provenance }: { provenance: Provenance | null }) {
  if (!provenance || !provenance.is_stub) return null;
  return (
    <div role="status" className="banner">
      <strong>Stub mode</strong>
      <span> — no LLM key configured. Answers are deterministic and only cover the four demo themes shipped with the corpus. Set{" "}
        <code>VERITY_ANTHROPIC_API_KEY</code> and restart <code>verity serve</code> for live answers (S7).</span>
      <style jsx>{`
        .banner {
          display: flex;
          gap: var(--space-3);
          padding: 10px 14px;
          background: var(--warn-soft);
          border-bottom: 1px solid color-mix(in oklab, var(--warn) 40%, transparent);
          color: var(--warn);
          font-size: 13px;
        }
        .banner strong {
          font-family: var(--mono);
          font-weight: 600;
          text-transform: lowercase;
        }
        .banner code {
          font-family: var(--mono);
          background: rgba(0,0,0,0.2);
          padding: 1px 5px;
          border-radius: 4px;
        }
        .banner span {
          color: var(--fg);
        }
      `}</style>
    </div>
  );
}
