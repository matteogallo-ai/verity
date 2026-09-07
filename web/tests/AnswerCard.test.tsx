import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AnswerCard } from "@/components/AnswerCard";
import { CITED_ANSWER, REFUSED_ANSWER } from "./fixtures";

describe("AnswerCard — cited answer", () => {
  it("renders the answer text with an inline citation marker", () => {
    render(
      <AnswerCard
        answer={CITED_ANSWER.answer}
        provenance={CITED_ANSWER.provenance}
        activeCitationIndex={null}
        activeChunkId={null}
        onCitationClick={() => {}}
      />
    );
    // The full sentence should be rendered somewhere in the article.
    expect(screen.getByRole("region", { name: /answer/i })).toBeInTheDocument();
    // There is at least one inline citation marker.
    expect(screen.getByTestId("citation-1")).toHaveTextContent("[1]");
    // The verbatim quote appears at least once (in the citations list; the
    // aria-label on the marker button also carries it).
    expect(screen.getAllByText(/thirty \(30\) days/i).length).toBeGreaterThan(0);
  });

  it("surfaces the stub provenance badge on every answer", () => {
    render(
      <AnswerCard
        answer={CITED_ANSWER.answer}
        provenance={CITED_ANSWER.provenance}
        activeCitationIndex={null}
        activeChunkId={null}
        onCitationClick={() => {}}
      />
    );
    expect(
      screen.getByLabelText(/agent provenance: stub-agent \(stub\)/i)
    ).toBeInTheDocument();
  });

  it("fires onCitationClick with the marker index + citation payload", () => {
    const onCitationClick = vi.fn();
    render(
      <AnswerCard
        answer={CITED_ANSWER.answer}
        provenance={CITED_ANSWER.provenance}
        activeCitationIndex={null}
        activeChunkId={null}
        onCitationClick={onCitationClick}
      />
    );
    fireEvent.click(screen.getByTestId("citation-1"));
    expect(onCitationClick).toHaveBeenCalledTimes(1);
    const arg = onCitationClick.mock.calls[0][0];
    expect(arg.index).toBe(1);
    expect(arg.citation.chunk_id).toBe(CITED_ANSWER.answer.citations[0].chunk_id);
  });

  it("marks the active citation and applies aria-pressed", () => {
    render(
      <AnswerCard
        answer={CITED_ANSWER.answer}
        provenance={CITED_ANSWER.provenance}
        activeCitationIndex={1}
        activeChunkId={CITED_ANSWER.answer.citations[0].chunk_id}
        onCitationClick={() => {}}
      />
    );
    const marker = screen.getByTestId("citation-1");
    expect(marker).toHaveAttribute("aria-pressed", "true");
    expect(marker.className).toMatch(/cite-active/);
  });

  it("shows real latency + trace_id from the API response — never fabricated", () => {
    render(
      <AnswerCard
        answer={CITED_ANSWER.answer}
        provenance={CITED_ANSWER.provenance}
        activeCitationIndex={null}
        activeChunkId={null}
        onCitationClick={() => {}}
      />
    );
    // The latency value comes verbatim from usage.latency_ms rounded to int ms.
    expect(screen.getByText(/90 ms/)).toBeInTheDocument();
    // The trace_id prefix is shown, full value in title.
    expect(screen.getByText(/^aa6a5898$/)).toBeInTheDocument();
  });
});

describe("AnswerCard — refusal", () => {
  it("renders the refusal state instead of an answer card", () => {
    render(
      <AnswerCard
        answer={REFUSED_ANSWER.answer}
        provenance={REFUSED_ANSWER.provenance}
        activeCitationIndex={null}
        activeChunkId={null}
        onCitationClick={() => {}}
      />
    );
    // Region label reads "refused", not an error.
    expect(
      screen.getByRole("region", { name: /refused/i })
    ).toBeInTheDocument();
    expect(screen.getByText(/i don't know/i)).toBeInTheDocument();
    // The rationale is surfaced verbatim.
    expect(
      screen.getByText(REFUSED_ANSWER.answer.confidence.rationale)
    ).toBeInTheDocument();
    // Provenance badge visible on refusal too.
    expect(
      screen.getByLabelText(/agent provenance: stub-agent \(stub\)/i)
    ).toBeInTheDocument();
    // Zero citations rendered on refusal.
    expect(screen.queryByTestId("citation-1")).toBeNull();
  });
});
