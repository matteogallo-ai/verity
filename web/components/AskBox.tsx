"use client";

import { useState } from "react";

export interface AskBoxProps {
  disabled?: boolean;
  onAsk: (question: string) => void;
}

const DEMO_QUESTIONS = [
  "What is the notice period for terminating the master services agreement?",
  "Which entity is liable for data-processing breaches, and up to what cap?",
  "Where is personal data stored, and what mechanism covers EU-to-US transfers?",
  "What is the response time for a Priority 1 support request?",
  // Out-of-scope — triggers the refusal panel.
  "What is the CEO's home address?",
];

export function AskBox({ disabled, onAsk }: AskBoxProps) {
  const [text, setText] = useState("");
  const canSubmit = text.trim().length > 0 && !disabled;
  return (
    <form
      className="ask"
      onSubmit={(e) => {
        e.preventDefault();
        if (canSubmit) onAsk(text.trim());
      }}
    >
      <textarea
        aria-label="Question"
        placeholder="Ask a question about the ingested documents…"
        value={text}
        rows={2}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && canSubmit) {
            e.preventDefault();
            onAsk(text.trim());
          }
        }}
      />
      <div className="row">
        <div className="demo">
          {DEMO_QUESTIONS.map((q) => (
            <button
              key={q}
              type="button"
              className="chip"
              disabled={disabled}
              onClick={() => {
                setText(q);
                onAsk(q);
              }}
            >
              {q.length > 40 ? q.slice(0, 40) + "…" : q}
            </button>
          ))}
        </div>
        <button
          type="submit"
          className="primary"
          disabled={!canSubmit}
          aria-label="Ask"
        >
          Ask ↵
        </button>
      </div>
      <style jsx>{`
        .ask {
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
        }
        textarea {
          width: 100%;
          resize: vertical;
          min-height: 60px;
        }
        .row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: var(--space-3);
          flex-wrap: wrap;
        }
        .demo {
          display: flex;
          gap: 6px;
          flex-wrap: wrap;
        }
        .chip {
          padding: 4px 10px;
          border: 1px solid var(--border);
          background: var(--bg-elev);
          color: var(--fg-dim);
          border-radius: 999px;
          font-size: 11px;
          font-family: var(--mono);
        }
        .chip:hover:not(:disabled) {
          border-color: var(--accent);
          color: var(--fg);
        }
        .primary {
          padding: 8px 16px;
          border: 1px solid var(--accent);
          background: var(--accent-soft);
          color: var(--accent);
          border-radius: var(--radius);
          font-family: var(--mono);
          font-size: 12px;
        }
        .primary:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }
      `}</style>
    </form>
  );
}
