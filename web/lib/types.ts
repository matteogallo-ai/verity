/**
 * Wire types — a 1:1 mirror of `verity/api/models.py`. Kept explicit so a
 * schema drift on the Python side shows up as a TypeScript compile error the
 * moment the fixture doesn't parse.
 *
 * Rule of iron: no additional client-side field. Anything shown in the UI
 * must have a direct counterpart here (and therefore in the API response).
 */

export interface Provenance {
  agent_model: string; // "stub-agent" | "claude-sonnet-4-6" | ...
  is_stub: boolean;
  embedding_model: string;
  reranker_model: string;
}

export interface Citation {
  chunk_id: string;
  document_id: string;
  quote: string;
  char_start: number;
  char_end: number;
}

export interface Confidence {
  score: number; // [0, 1]
  refused: boolean;
  rationale: string;
}

export interface UsageStats {
  input_tokens: number;
  output_tokens: number;
  llm_calls: number;
  cost_usd: number;
  latency_ms: number;
}

export type HitKind = "dense" | "sparse" | "fused" | "reranked";

export interface RetrievedChunk {
  chunk_id: string;
  document_id: string;
  text: string;
  section: string | null;
  ordinal: number;
  char_start: number;
  char_end: number;
  score: number;
  rank: number;
  kind: HitKind | string; // permissive to survive additive schema changes
  cited: boolean;
}

export interface AnswerView {
  query: string;
  text: string;
  citations: Citation[];
  confidence: Confidence;
  hits_used: RetrievedChunk[];
  usage: UsageStats;
  trace_id: string;
}

export interface AskResponse {
  answer: AnswerView;
  provenance: Provenance;
}

export interface CorpusSummary {
  n_documents: number;
  n_chunks: number;
  sources: string[];
}

export interface IngestResponse {
  document_id: string;
  uri: string;
  title: string | null;
  n_chunks: number;
  corpus: CorpusSummary;
}

export interface StatusResponse {
  version: string;
  provenance: Provenance;
  corpus: CorpusSummary;
  otel_endpoint: string | null;
}

// -----------------------------------------------------------------------------
// UI-only view models — cheap structural transforms of the wire types. NEVER
// invent numerics. Every field here is derived from real fields above.
// -----------------------------------------------------------------------------

export interface CitationMarker {
  index: number; // 1-based, order of appearance in the answer
  citation: Citation;
}
