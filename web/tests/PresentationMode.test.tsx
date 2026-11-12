/**
 * VERITY_UI_MODE=presentation (/status.ui_mode): what a member would see in a client
 * meeting. The question, the answer, each citation's article and quote, the refusal
 * message and the suggested questions; none of the developer read-outs. The API
 * module is mocked: no network.
 */
import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/api";
import type { AskResponse } from "@/lib/types";
import { CITED_ANSWER, LIVE_PROVENANCE, REFUSED_ANSWER, STUB_PROVENANCE } from "./fixtures";

const state = vi.hoisted(() => ({
  uiMode: undefined as string | undefined,
  answer: null as AskResponse | null,
  stub: false,
  statusFailures: 0,
  statusPending: false,
  askError: null as Error | null,
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    fetchStatus: vi.fn(async () => {
      if (state.statusPending) return new Promise(() => {});
      if (state.statusFailures > 0) {
        state.statusFailures -= 1;
        throw new actual.ApiError(0, "connection refused");
      }
      return {
      version: "test",
      provenance: state.stub ? STUB_PROVENANCE : LIVE_PROVENANCE,
      corpus: { n_documents: 2, n_chunks: 117, sources: [] },
      otel_endpoint: null,
      language: "fr",
      suggested_questions: ["Combien de jours de vacances ?"],
      ...(state.uiMode === undefined
        ? {}
        : { ui_mode: state.uiMode, corpus_name: "CCT Métal-Vaud" }),
      };
    }),
    ask: vi.fn(async () => {
      if (state.askError) throw state.askError;
      return state.answer;
    }),
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
    state.stub = false;
    state.statusFailures = 0;
    state.statusPending = false;
    state.askError = null;
  });

  afterEach(() => {
    vi.useRealTimers();
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

  it("says in French, in one line, that a stub answers", async () => {
    state.uiMode = "presentation";
    state.stub = true;
    render(<HomePage />);
    expect(await screen.findByText(/Démonstration hors ligne/)).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/Stub mode|VERITY_ANTHROPIC_API_KEY/);
  });

  it("does not invite the member to upload a document", async () => {
    state.uiMode = "presentation";
    render(<HomePage />);
    expect(await screen.findByText(/questions proposées/)).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/ajoutez d'abord un document/);
  });

  it("shows no developer view while /status has not answered", async () => {
    state.statusPending = true;
    render(<HomePage />);
    await act(async () => {});
    expect(document.body.textContent).not.toMatch(/evaluated, observable RAG|Drop a file/);
  });

  it("retries /status until the API is up", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    state.uiMode = "presentation";
    state.statusFailures = 1;
    render(<HomePage />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/connection refused/);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });
    expect(
      await screen.findByRole("heading", { name: "Assistant CCT – CCT Métal-Vaud" }),
    ).toBeInTheDocument();
  });

  it("hides the technical detail of a failed question", async () => {
    state.uiMode = "presentation";
    state.askError = new ApiError(502, "upstream error request_id=req_0123");
    render(<HomePage />);
    fireEvent.click(await screen.findByRole("button", { name: "Combien de jours de vacances ?" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/Réessayez/);
    expect(document.body.textContent).not.toMatch(/req_0123|502/);
  });

  it("shows the developer read-outs the presentation hides", async () => {
    await askWith(ANSWERED);
    for (const shown of [/claude-sonnet-4-6/, /evaluated, observable RAG/, /confiance/i, /117 chunks/]) {
      expect(document.body.textContent).toMatch(shown);
    }
  });
});
