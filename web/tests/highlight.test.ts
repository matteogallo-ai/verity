import { describe, expect, it } from "vitest";
import { segmentAnswer } from "@/lib/highlight";

describe("segmentAnswer", () => {
  it("returns a single text segment when there are no citations", () => {
    const segments = segmentAnswer("Just plain answer.", []);
    expect(segments).toEqual([{ kind: "text", content: "Just plain answer." }]);
  });

  it("places a marker right after the first occurrence of the citation quote", () => {
    const segments = segmentAnswer(
      "Either party may terminate with thirty (30) days notice.",
      [
        {
          chunk_id: "c1",
          document_id: "d1",
          quote: "thirty (30) days",
          char_start: 0,
          char_end: 16,
        },
      ]
    );
    expect(segments).toEqual([
      { kind: "text", content: "Either party may terminate with thirty (30) days" },
      {
        kind: "marker",
        marker: {
          index: 1,
          citation: {
            chunk_id: "c1",
            document_id: "d1",
            quote: "thirty (30) days",
            char_start: 0,
            char_end: 16,
          },
        },
      },
      { kind: "text", content: " notice." },
    ]);
  });

  it("appends the marker at the end when the quote is not present in the answer", () => {
    const segments = segmentAnswer("A paraphrased answer.", [
      {
        chunk_id: "c1",
        document_id: "d1",
        quote: "some other verbatim span",
        char_start: 0,
        char_end: 24,
      },
    ]);
    // No text remains after the quote fallback (end-of-text placement).
    expect(segments[0]).toEqual({ kind: "text", content: "A paraphrased answer." });
    expect(segments[1]).toEqual({
      kind: "marker",
      marker: {
        index: 1,
        citation: {
          chunk_id: "c1",
          document_id: "d1",
          quote: "some other verbatim span",
          char_start: 0,
          char_end: 24,
        },
      },
    });
  });

  it("preserves marker order by their position in the answer", () => {
    const segments = segmentAnswer("alpha beta gamma.", [
      { chunk_id: "c2", document_id: "d", quote: "gamma", char_start: 0, char_end: 5 },
      { chunk_id: "c1", document_id: "d", quote: "alpha", char_start: 0, char_end: 5 },
    ]);
    const markers = segments.filter((s) => s.kind === "marker");
    expect(markers.map((m) => (m.kind === "marker" ? m.marker.citation.chunk_id : ""))).toEqual([
      "c1",
      "c2",
    ]);
  });
});
