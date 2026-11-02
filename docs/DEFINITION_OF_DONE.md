# Definition of Done — portfolio repo (Matteo)

Sources: OpenSSF Scorecard · OpenSSF Best Practices Badge · SLSA / GitHub artifact attestations · OWASP Top 10 for LLM Applications 2025 · Model Cards (Mitchell et al., 2019) · Google SRE Production Readiness Review · 2026 LLM evaluation practice.
A repo is "done" only when every line is checked, or marked N/A with a written reason, and section F is passed. Never declare done otherwise.

## A. Repository and supply chain (OpenSSF Scorecard)
- A1 main protected: PR required, required CI checks, no force-push, no deletion.
- A2 Dependabot for the repo ecosystem + github-actions, weekly, grouped.
- A3 Every action pinned by full commit SHA with a version comment; lockfiles committed.
- A4 Workflow tokens read-only by default (`permissions: contents: read`); widened per job, with a reason in a comment.
- A5 CodeQL (SAST) on every PR and weekly.
- A6 OSV-Scanner on lockfiles in CI; no unfixed known vulnerability, or a written exception.
- A7 No dangerous workflow (no `pull_request_target` that checks out PR code; no untrusted input interpolated into `run:`).
- A8 SECURITY.md. A9 LICENSE with SPDX id.
- A10 Released artefacts carry provenance (npm provenance or actions/attest-build-provenance, SLSA Build L2).
- A11 Scorecard Action with publish_results and a live badge; checks that are low by nature on a solo repo (Code-Review, Contributors) are explained, not hidden.
- A12 Secret scanning and push protection enabled.

## B. LLM quality
- B1 Frozen, versioned eval set with a dev/test split.
- B2 LLM judge validated against a human (kappa) and against perturbed false claims (sensitivity).
- B3 Deterministic checks run before any judge or model-based score.
- B4 Every proportion reported with n and a 95% interval; noise band measured (same run repeated) before claiming an improvement.
- B5 CI gate on deterministic metrics; live evals are manual, budget-capped, and committed.
- B6 Live artefacts committed, immutable, checksummed.

## C. LLM security and system documentation
- C1 docs/THREAT_MODEL.md: OWASP LLM Top 10 2025 item by item — applicable or not (why), mitigation, test that proves it.
- C2 Bounded consumption: budget cap, max tokens, timeouts, all tested.
- C3 docs/SYSTEM_CARD.md: intended use, out of scope, data, measured performance (n, CI), known limits, owner.
- C4 Logs never contain secrets, DSNs or personal data; tested.

## D. Production readiness (Google SRE PRR)
- D1 Written SLOs: p95 latency, error rate, cost per request.
- D2 Measured p50/p95 latency and cost per request, from a committed artefact.
- D3 docs/RUNBOOK.md: the five most likely failures — symptom, detection, response.
- D4 Health check; failures distinguishable from normal refusals.
- D5 Fresh clone to running in one documented command, tested in CI.

## E. Delivery and presentation
- E1 README first screen: what it is, why it exists, what was hard. Setup comes after.
- E2 One GIF or screenshot (10–20 s).
- E3 Something a stranger can try: live URL, installable package, or a real committed output.
- E4 ADRs plus a "What I'd do differently" section.
- E5 SemVer tag + GitHub Release with notes for every release.
- E6 Only dynamic badges that tell the truth (CI, Scorecard, release, license).

## G. Above the bar (tests that are themselves tested)
- G1 Property-based tests (Hypothesis / fast-check) on every parser, validator and input boundary; fuzzing on parsers (Scorecard Fuzzing check).
- G2 Mutation testing (mutmut / Stryker) on the core modules: mutation score measured, committed, and surviving mutants explained or killed.
- G3 GitHub Release per tag with attached, signed artefacts (attest-build-provenance) so Scorecard Signed-Releases scores; never publish an unsigned GitHub Release.

## F. Closing
- F1 A context-free agent audits the repo as a senior looking for the flaw; every finding is fixed or refuted in writing.
- F2 OpenSSF Best Practices badge at "passing" (bestpractices.dev, self-certified by Matteo).
- F3 Matteo can explain every checked line without notes.
