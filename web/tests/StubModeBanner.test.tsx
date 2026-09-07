import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StubModeBanner } from "@/components/StubModeBanner";
import { LIVE_PROVENANCE, STUB_PROVENANCE } from "./fixtures";

describe("StubModeBanner", () => {
  it("appears when provenance.is_stub is true", () => {
    render(<StubModeBanner provenance={STUB_PROVENANCE} />);
    expect(screen.getByText(/stub mode/i)).toBeInTheDocument();
    expect(screen.getByText(/no LLM key configured/i)).toBeInTheDocument();
  });

  it("does NOT appear when provenance.is_stub is false", () => {
    render(<StubModeBanner provenance={LIVE_PROVENANCE} />);
    expect(screen.queryByText(/stub mode/i)).toBeNull();
  });

  it("does NOT appear when provenance is null (backend down / unknown)", () => {
    render(<StubModeBanner provenance={null} />);
    expect(screen.queryByText(/stub mode/i)).toBeNull();
  });
});
