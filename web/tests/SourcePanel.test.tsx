import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SourcePanel } from "@/components/SourcePanel";
import { CITED_ANSWER } from "./fixtures";

describe("SourcePanel", () => {
  it("renders every chunk with its rerank score and cited flag", () => {
    render(
      <SourcePanel
        answer={CITED_ANSWER.answer}
        activeChunkId={null}
        onChunkClick={() => {}}
      />
    );
    // Two chunks in the fixture: one cited, one not.
    const cited = screen.getByText("cited");
    const uncited = screen.getByText("not-cited");
    expect(cited).toBeInTheDocument();
    expect(uncited).toBeInTheDocument();
    // The reranker score is rendered verbatim (3-decimal precision).
    expect(screen.getByText(/reranked: 5\.234/)).toBeInTheDocument();
    expect(screen.getByText(/reranked: 3\.912/)).toBeInTheDocument();
  });

  it("fires onChunkClick with the full chunk on activation", () => {
    const onChunkClick = vi.fn();
    render(
      <SourcePanel
        answer={CITED_ANSWER.answer}
        activeChunkId={null}
        onChunkClick={onChunkClick}
      />
    );
    const chunk = CITED_ANSWER.answer.hits_used[0];
    fireEvent.click(screen.getByTestId(`chunk-${chunk.chunk_id}`));
    expect(onChunkClick).toHaveBeenCalledTimes(1);
    expect(onChunkClick.mock.calls[0][0].chunk_id).toBe(chunk.chunk_id);
  });

  it("marks the active chunk row via aria-pressed", () => {
    const chunkId = CITED_ANSWER.answer.hits_used[0].chunk_id;
    render(
      <SourcePanel
        answer={CITED_ANSWER.answer}
        activeChunkId={chunkId}
        onChunkClick={() => {}}
      />
    );
    const row = screen.getByTestId(`chunk-${chunkId}`);
    expect(row).toHaveAttribute("aria-pressed", "true");
  });

  it("renders a friendly empty state before any question is asked", () => {
    render(
      <SourcePanel answer={null} activeChunkId={null} onChunkClick={() => {}} />
    );
    expect(
      screen.getByText(/evidence used by the agent will appear here/i)
    ).toBeInTheDocument();
  });
});
