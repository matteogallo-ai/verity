/**
 * The page passes the language /status reports to the answer and evidence panels.
 * The API module is mocked: no network.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { REFUSED_ANSWER } from "./fixtures";

const status = vi.hoisted(() => ({ language: undefined as string | undefined }));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    fetchStatus: vi.fn(async () => ({
      version: "test",
      provenance: REFUSED_ANSWER.provenance,
      corpus: { n_documents: 1, n_chunks: 1, sources: [] },
      otel_endpoint: null,
      ...(status.language === undefined ? {} : { language: status.language }),
    })),
    ask: vi.fn(async () => REFUSED_ANSWER),
  };
});

import HomePage from "@/app/page";

async function refuse() {
  render(<HomePage />);
  const box = await screen.findByRole("textbox", { name: "Question" });
  fireEvent.change(box, { target: { value: "Quelle indemnité de télétravail ?" } });
  fireEvent.click(screen.getByRole("button", { name: /^Ask/ }));
  return screen.findByRole("region", { name: /refus/i });
}

describe("page language", () => {
  beforeEach(() => {
    status.language = undefined;
  });

  it("labels the refusal and the evidence in French when /status says fr", async () => {
    status.language = "fr";
    const refusal = await refuse();
    expect(refusal).toHaveTextContent("L'agent a refusé plutôt que de livrer le brouillon");
    expect(screen.getAllByText(/non citée|non citée/).length).toBeGreaterThan(0);
  });

  it("stays in English when /status reports no language", async () => {
    const refusal = await refuse();
    expect(refusal).toHaveTextContent("The agent declined");
    expect(screen.getAllByText("not-cited").length).toBeGreaterThan(0);
  });
});
