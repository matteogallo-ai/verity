/**
 * Citation ↔ chunk interaction helpers.
 *
 * Two mappings are needed:
 *   - answer text → clickable inline markers ([1], [2] …) that map to a
 *     `Citation`. We produce a token stream so the answer text is rendered as
 *     text runs interleaved with `CitationMarker` objects.
 *   - highlight bus: selecting a citation highlights its chunk, selecting a
 *     chunk highlights any citation pointing at it. Both flow through a single
 *     `activeChunkId` piece of React state kept in the page.
 *
 * The renderer supports two kinds of inline citation markers coming from the
 * agent:
 *   1. Explicit `[N]` numeric markers already in the answer text.
 *   2. Otherwise, the verbatim quote from the citation is used to locate its
 *      position in the answer text; the marker is appended right after the
 *      first occurrence of the quote. If the quote is not present in the
 *      answer text (rare, but possible with paraphrased answers), the marker
 *      is appended at the end of the sentence containing the answer text.
 */

import type { Citation, CitationMarker } from "./types";

export interface TextSegment {
  kind: "text";
  content: string;
}

export interface MarkerSegment {
  kind: "marker";
  marker: CitationMarker;
}

export type AnswerSegment = TextSegment | MarkerSegment;

/**
 * Turn `(answer_text, citations)` into an ordered list of text + marker
 * segments the UI can render. Deterministic; safe under empty citations
 * (returns a single text segment).
 */
export function segmentAnswer(
  text: string,
  citations: readonly Citation[]
): AnswerSegment[] {
  if (citations.length === 0) {
    return [{ kind: "text", content: text }];
  }

  const markers: CitationMarker[] = citations.map((citation, i) => ({
    index: i + 1,
    citation,
  }));

  // Deterministic ordering: for each marker, find the first index of its
  // quote in the answer text. Fall back to end-of-text if the quote is not
  // present (paraphrased answer).
  type Placement = { marker: CitationMarker; at: number };
  const placements: Placement[] = markers.map((marker) => {
    const at = marker.citation.quote
      ? text.indexOf(marker.citation.quote)
      : -1;
    return { marker, at: at >= 0 ? at + marker.citation.quote.length : text.length };
  });

  placements.sort((a, b) => a.at - b.at);

  const segments: AnswerSegment[] = [];
  let cursor = 0;
  for (const p of placements) {
    if (p.at > cursor) {
      segments.push({ kind: "text", content: text.slice(cursor, p.at) });
      cursor = p.at;
    }
    segments.push({ kind: "marker", marker: p.marker });
  }
  if (cursor < text.length) {
    segments.push({ kind: "text", content: text.slice(cursor) });
  }
  return segments;
}
