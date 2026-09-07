/**
 * Integration test for the citation ↔ chunk highlight interaction.
 * Uses a small controlled parent that mimics the page's state wiring — no
 * network, no MSW — so the flow is entirely deterministic.
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { AnswerCard } from "@/components/AnswerCard";
import { SourcePanel } from "@/components/SourcePanel";
import { CITED_ANSWER } from "./fixtures";

function Harness() {
  const [activeCitation, setActiveCitation] = useState<number | null>(null);
  const [activeChunkId, setActiveChunkId] = useState<string | null>(null);
  const citations = CITED_ANSWER.answer.citations;
  return (
    <>
      <AnswerCard
        answer={CITED_ANSWER.answer}
        provenance={CITED_ANSWER.provenance}
        activeCitationIndex={activeCitation}
        activeChunkId={activeChunkId}
        onCitationClick={(m) => {
          setActiveCitation(m.index);
          setActiveChunkId(m.citation.chunk_id);
        }}
      />
      <SourcePanel
        answer={CITED_ANSWER.answer}
        activeChunkId={activeChunkId}
        onChunkClick={(hit) => {
          setActiveChunkId(hit.chunk_id);
          const idx = citations.findIndex((c) => c.chunk_id === hit.chunk_id);
          setActiveCitation(idx >= 0 ? idx + 1 : null);
        }}
      />
    </>
  );
}

describe("citation ↔ chunk highlight interaction", () => {
  it("clicking a citation marker highlights the matching chunk row", () => {
    render(<Harness />);
    const marker = screen.getByTestId("citation-1");
    const chunkId = CITED_ANSWER.answer.citations[0].chunk_id;
    const chunkRow = screen.getByTestId(`chunk-${chunkId}`);

    expect(chunkRow).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(marker);
    expect(chunkRow).toHaveAttribute("aria-pressed", "true");
    expect(marker).toHaveAttribute("aria-pressed", "true");
  });

  it("clicking a cited chunk highlights its citation marker (and vice versa)", () => {
    render(<Harness />);
    const chunkId = CITED_ANSWER.answer.hits_used[0].chunk_id;
    const chunkRow = screen.getByTestId(`chunk-${chunkId}`);
    const marker = screen.getByTestId("citation-1");

    fireEvent.click(chunkRow);
    expect(marker).toHaveAttribute("aria-pressed", "true");
    expect(chunkRow).toHaveAttribute("aria-pressed", "true");
  });

  it("clicking a not-cited chunk highlights it but no citation marker", () => {
    render(<Harness />);
    const uncitedChunkId = CITED_ANSWER.answer.hits_used[1].chunk_id;
    const row = screen.getByTestId(`chunk-${uncitedChunkId}`);
    const marker = screen.getByTestId("citation-1");
    fireEvent.click(row);
    expect(row).toHaveAttribute("aria-pressed", "true");
    expect(marker).toHaveAttribute("aria-pressed", "false");
  });
});
