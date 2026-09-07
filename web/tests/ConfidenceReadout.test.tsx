import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ConfidenceReadout } from "@/components/ConfidenceReadout";
import { CITED_ANSWER, LIVE_PROVENANCE, REFUSED_ANSWER, STUB_PROVENANCE } from "./fixtures";

describe("ConfidenceReadout", () => {
  it("labels stub provenance explicitly next to the number", () => {
    render(
      <ConfidenceReadout answer={CITED_ANSWER.answer} provenance={STUB_PROVENANCE} />
    );
    // Score verbatim from the API — no rescaling, no composite.
    expect(screen.getByText("0.88")).toBeInTheDocument();
    // Provenance is adjacent to the number and marked as stub.
    expect(screen.getByText(/agent-reported \(stub: stub-agent\)/i)).toBeInTheDocument();
  });

  it("labels live provenance with the actual model id", () => {
    render(
      <ConfidenceReadout answer={CITED_ANSWER.answer} provenance={LIVE_PROVENANCE} />
    );
    expect(
      screen.getByText(/agent-reported \(claude-sonnet-4-6\)/i)
    ).toBeInTheDocument();
    // Never labelled "stub" when the badge says live.
    expect(screen.queryByText(/agent-reported \(stub/i)).toBeNull();
  });

  it("shows the underlying signals a careful reader would want", () => {
    render(
      <ConfidenceReadout answer={CITED_ANSWER.answer} provenance={STUB_PROVENANCE} />
    );
    // refused, citations, top rerank score, evidence hits — all from the API.
    expect(screen.getByText("refused")).toBeInTheDocument();
    expect(screen.getByText("false")).toBeInTheDocument();
    expect(screen.getByText("citations")).toBeInTheDocument();
    expect(screen.getByText("top rerank score")).toBeInTheDocument();
    // Top rerank score from the first hit in the fixture (5.234, 3-decimal).
    expect(screen.getByText("5.234")).toBeInTheDocument();
    expect(screen.getByText("evidence hits")).toBeInTheDocument();
  });

  it("renders refused=true and the rationale verbatim on a refusal", () => {
    render(
      <ConfidenceReadout answer={REFUSED_ANSWER.answer} provenance={STUB_PROVENANCE} />
    );
    expect(screen.getByText("true")).toBeInTheDocument();
    // Rationale text verbatim.
    expect(
      screen.getByText(REFUSED_ANSWER.answer.confidence.rationale)
    ).toBeInTheDocument();
    // Zero citations shown on the citations signal.
    const items = screen.getAllByRole("listitem");
    const citationsItem = items.find((el) => el.textContent?.startsWith("citations"));
    expect(citationsItem?.textContent).toContain("0");
  });

  it("makes the passthrough nature explicit — no client-side composite", () => {
    render(
      <ConfidenceReadout answer={CITED_ANSWER.answer} provenance={STUB_PROVENANCE} />
    );
    expect(
      screen.getByText(/passed through verbatim from the confidence scorer/i)
    ).toBeInTheDocument();
  });
});
