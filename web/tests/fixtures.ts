/**
 * Deterministic fixtures. Every shape here mirrors verity/api/models.py — if
 * the API schema drifts, TypeScript compilation of the tests will fail.
 */

import type { AskResponse, Provenance } from "@/lib/types";

export const STUB_PROVENANCE: Provenance = {
  agent_model: "stub-agent",
  is_stub: true,
  embedding_model: "BAAI/bge-small-en-v1.5",
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2",
};

export const LIVE_PROVENANCE: Provenance = {
  agent_model: "claude-sonnet-4-6",
  is_stub: false,
  embedding_model: "BAAI/bge-small-en-v1.5",
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2",
};

export const CITED_ANSWER: AskResponse = {
  provenance: STUB_PROVENANCE,
  answer: {
    query: "What is the notice period for terminating the master services agreement?",
    text:
      "Either party may terminate for convenience with thirty (30) days' prior written notice.",
    citations: [
      {
        chunk_id: "ef55252e-8732-54d8-8131-f07d4a438bfd",
        document_id: "34d00639-e532-5c59-8563-7535bad66a11",
        quote: "thirty (30) days",
        char_start: 628,
        char_end: 644,
      },
    ],
    confidence: {
      score: 0.88,
      refused: false,
      rationale: "Answer is grounded in a verbatim clause from the retrieved evidence.",
    },
    hits_used: [
      {
        chunk_id: "ef55252e-8732-54d8-8131-f07d4a438bfd",
        document_id: "34d00639-e532-5c59-8563-7535bad66a11",
        text:
          "### 1.2 Termination for Convenience\n\nEither party may terminate this Agreement or any Order Form for its convenience by providing the other party with at least thirty (30) days' prior written notice.",
        section: "Master Services Agreement > 1. Term and Termination > 1.2 Termination for Convenience",
        ordinal: 2,
        char_start: 550,
        char_end: 900,
        score: 5.234,
        rank: 0,
        kind: "reranked",
        cited: true,
      },
      {
        chunk_id: "bac3e0ee-f254-51b3-8df5-b3dc0c5bd891",
        document_id: "34d00639-e532-5c59-8563-7535bad66a11",
        text: "### 1.3 Termination for Cause\n\nEither party may terminate this Agreement immediately upon written notice if the other party materially breaches this Agreement.",
        section: "Master Services Agreement > 1. Term and Termination > 1.3 Termination for Cause",
        ordinal: 3,
        char_start: 900,
        char_end: 1180,
        score: 3.912,
        rank: 1,
        kind: "reranked",
        cited: false,
      },
    ],
    usage: {
      input_tokens: 2393,
      output_tokens: 85,
      llm_calls: 3,
      cost_usd: 0.0,
      latency_ms: 89.7,
    },
    trace_id: "aa6a5898-72c7-4d91-a67c-8957943c95ca",
  },
};

export const REFUSED_ANSWER: AskResponse = {
  provenance: STUB_PROVENANCE,
  answer: {
    query: "What is the CEO's home address?",
    text: "I don't know based on the provided documents.",
    citations: [],
    confidence: {
      score: 0.15,
      refused: true,
      rationale:
        "The retrieved evidence does not address the question — refusing rather than guessing.",
    },
    hits_used: [
      {
        chunk_id: "c82ccae0-1000-0000-0000-000000000001",
        document_id: "34d00639-e532-5c59-8563-7535bad66a11",
        text: "General preamble about Acme Corporation…",
        section: "Master Services Agreement",
        ordinal: 0,
        char_start: 0,
        char_end: 200,
        score: -11.07,
        rank: 0,
        kind: "reranked",
        cited: false,
      },
    ],
    usage: {
      input_tokens: 1213,
      output_tokens: 20,
      llm_calls: 3,
      cost_usd: 0.0,
      latency_ms: 91.3,
    },
    trace_id: "c19862cb-a727-4e09-8a00-c216434589af",
  },
};
