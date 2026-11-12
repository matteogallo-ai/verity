/**
 * VERITY_UI_MODE=presentation (/status.ui_mode): what a member would see in a client
 * meeting. The question, the answer, each citation's article and quote, the refusal
 * message and the suggested questions; none of the developer read-outs. The API
 * module is mocked: no network.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AskResponse } from "@/lib/types";
import { CITED_ANSWER, LIVE_PROVENANCE, REFUSED_ANSWER } from "./fixtures";

const state = vi.hoisted(() => ({
  uiMode: undefined as string | undefined,
  answer: null as AskResponse | null,
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    fetchStatus: vi.fn(async () => ({
      version: "test",
      provenance: LIVE_PROVENANCE,
      corpus: { n_documents: 2, n_chunks: 117, sources: [] },
      otel_endpoint: null,
      language: "fr",
      suggested_questions: ["Combien de jours de vacances ?"],
      ...(state.uiMode === undefined
        ? {}
        : { ui_mode: state.uiMode, corpus_name: "CCT Métal-Vaud" }),
    })),
    ask: vi.fn(async () => state.answer),
  };
});

import HomePage from "@/app/page";

const ANSWERED: AskResponse = { ...CITED_ANSWER, provenance: LIVE_PROVENANCE };
const REFUSED: AskResponse = {
  ...REFUSED_ANSWER,
  provenance: LIVE_PROVENANCE,
  answer: {
    ...REFUSED_ANSWER.answer,
    text: "Les documents fournis ne permettent pas de répondre. Contactez le secrétariat.",
  },
};

async function askWith(answer: AskResponse) {
  state.answer = answer;
  render(<HomePage />);
  fireEvent.click(await screen.findByRole("button", { name: "Combien de jours de vacances ?" }));
  return screen.findByRole("region", { name: /^(réponse|refus)$/i });
}

const HIDDEN = [
  /claude-sonnet-4-6/, // model badge
  /evaluated, observable RAG/, // English subtitle
  /confiance/i,
  /justification/i,
  /latence/i,
  /jetons/i,
  /coût/i,
  /ef55252e/, // chunk and document ids
  /aa6a5898/, // trace id
  /car\./, // character positions
  /5\.234|3\.912/, // rerank scores
  /Termination for Cause/, // an uncited source
  /117 passages|117 chunks/,
];

describe("presentation mode", () => {
  beforeEach(() => {
    state.uiMode = undefined;
    state.answer = null;
  });

  it("shows a sober French title and no upload", async () => {
    state.uiMode = "presentation";
    render(<HomePage />);
    expect(
      await screen.findByRole("heading", { name: "Assistant CCT – CCT Métal-Vaud" }),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Drop a file/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Combien de jours de vacances ?" })).toBeEnabled();
  });

  it("shows the answer and each citation's article and quote, and nothing else", async () => {
    state.uiMode = "presentation";
    const region = await askWith(ANSWERED);
    expect(region).toHaveTextContent("Either party may terminate for convenience");
    expect(region).toHaveTextContent("1.2 Termination for Convenience");
    expect(region).toHaveTextContent("thirty (30) days");
    for (const hidden of HIDDEN) {
      expect(document.body.textContent).not.toMatch(hidden);
    }
  });

  it("shows the refusal message alone", async () => {
    state.uiMode = "presentation";
    const region = await askWith(REFUSED);
    expect(region).toHaveTextContent("Contactez le secrétariat");
    expect(document.body.textContent).not.toMatch(/refusing rather than guessing/);
    expect(document.body.textContent).not.toMatch(/L'agent a refusé/);
    for (const hidden of HIDDEN) {
      expect(document.body.textContent).not.toMatch(hidden);
    }
  });

  it("keeps the developer view when /status names no mode", async () => {
    const region = await askWith(ANSWERED);
    expect(region).toHaveTextContent("claude-sonnet-4-6");
    expect(screen.getByText(/Drop a file/)).toBeInTheDocument();
    expect(document.body.textContent).toMatch(/Termination for Cause/);
  });
});
