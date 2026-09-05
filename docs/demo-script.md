# 90-second demo script (S7)

Order is deliberate: lead with the differentiator (refusal + measurement), not the upload.

1. **0:00–0:10 — The claim.** One line on screen: "RAG that refuses instead of
   hallucinating — and proves it." Show the README scorecard (real numbers).
2. **0:10–0:30 — Ask an answerable question.** Answer appears with inline citations;
   click a citation → the exact source passage highlights. Confidence panel shows high
   confidence + which chunks were used.
3. **0:30–0:50 — Ask an out-of-scope question.** System refuses: honest "I don't know"
   with a rationale, not a fabricated answer. This is the money shot.
4. **0:50–1:10 — Show the eval.** Run `verity eval run` in a terminal → scorecard prints
   (retrieval P/R, faithfulness, hallucination rate, refusal precision/recall, latency,
   cost). `verity eval compare` → quality tracked across commits.
5. **1:10–1:30 — Show the trace.** One query's end-to-end trace: per-stage latency,
   tokens, cost, retrieval scores. Close on the architecture diagram + repo link.

Record at S7 once the pipeline and UI are live. Keep it silent-captioned; no voiceover
needed if the on-screen actions are legible.
