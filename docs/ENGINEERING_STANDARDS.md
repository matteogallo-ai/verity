# Engineering Standards

> These standards apply to every repository I publish. Each rule exists because an
> independent audit of my own public repos (2026-11-02) found the corresponding failure.
> They are enforced by CI wherever a machine can check them, and by the PR template
> where a human must.

I build AI-natively: coding agents (Claude Code) write most first drafts. That makes these
rules more important, not less. An agent produces fluent, plausible output whether it is right
or wrong. My job is to specify, verify, and own the result.

---

## The rules

### 1. Every public claim is backed by code and a test
If the README, a doc, a badge or a comparison table says the project does something, there is
code that does it and a test that proves it. Anything not shipped is labelled **Roadmap**, never ✅.

- **Mechanism:** `docs/CLAIMS.md` lists each claim → implementing module → test ID → status
  (`shipped` / `roadmap`).
- **Enforced by:** CI job `claims` runs every listed test explicitly and fails if a `shipped`
  claim has no passing test, or if a `roadmap` item appears as shipped in the README.
- *Failure that created this rule:* "SemVer for prompts ✅", "Native tests ✅",
  "calibrated refusal", "feature-complete" — all claimed, none fully implemented.

### 2. Docs are code: no stale narrative
Shipped code does not talk about the future or about sprints.

- **Forbidden in `src/` and `docs/`:** `lands in S<n>`, `coming in v<x>`,
  `Implementation coming`, sprint IDs (`S4.2`), `TODO`/`FIXME` without an issue link,
  version numbers hard-coded outside the package metadata.
- **Enforced by:** CI job `stale-narrative` (pattern scan). The version is read from a single
  source (`pyproject.toml` / `package.json`).
- *Failure:* `VERSION = "0.1.0-alpha.0"` exported in v1.2.0; "Implementations land in S3"
  docstrings after release; docs describing directories that are empty.

### 3. Mocks prove plumbing; live runs prove behaviour
A stub, mock or fixture can prove that the code is wired correctly. It cannot prove that the
model behaves. **No score computed on mock or stub output is ever published as a quality
metric.**

- Any LLM feature described as working has a committed live-run artefact under `runs/live/`
  (date, model IDs, prompt versions, tokens, cost, outcome). Failed live runs are committed too.
- CI gates that run on stubs are labelled *pipeline smoke tests*, not quality gates.
- *Failure:* a benchmark scoring "5.0/5" on hand-written fixtures; a CI refusal-recall gate that
  is 1.000 by construction.

### 4. One metric, one definition, one source of truth
- The definition in `docs/evaluation-methodology.md` matches the code exactly
  (including micro vs macro aggregation). When aggregation matters, both are reported.
- Every number states its `n` and its denominator.
- Headline numbers are generated from the artefact by a script; CI runs `--check` so the README
  can never drift from the data.
- An LLM-judged metric reports judge–human agreement (Cohen's κ on a hand-labelled sample) or is
  explicitly labelled *unvalidated*. Judge and system under test are different model families
  when possible.
- Thresholds are tuned on a dev split and reported on a held-out test split.
- *Failure:* a methodology defining faithfulness per claim while the code averaged per question
  (0.618 vs 0.722 published); two committed artefacts with two different numbers.

### 5. Untrusted content is data, never instructions
- Everything retrieved, ingested, scraped, or returned by a tool is wrapped in explicit
  delimiters and labelled untrusted in the prompt.
- Every LLM entry point has at least one adversarial prompt-injection test.
- API inputs are bounded: size, type (allow-list), count. Rejections use correct status codes.
- Errors returned to clients are generic; details go to logs. Secrets only via environment.
- `SECURITY.md` is accurate for the current version and states the threat model.
- *Failure:* raw document text injected into the prompt that decides whether to refuse; a 2 MB
  question accepted; a random `.exe` ingested into a shared store; internal stack traces returned.

### 6. External calls fail predictably
- Every network/provider call has an explicit timeout and bounded retries on retryable errors only.
- Errors are mapped by type (`isinstance`), never by string-matching class names.
- There is a test for "provider down" at the public entry point.
- **An infrastructure failure is never reported as a model decision** (e.g. a database outage must
  not be counted as a "refusal").
- *Failure:* SDK clients without timeouts; a Postgres failure silently turned into a refusal.

### 7. A fresh clone works
- The README quickstart runs on a clean machine. CI proves it (clean checkout, documented commands).
- The Docker image is built and smoke-tested in CI (`/health`, one real request path).
- No dependency on sibling directories or local paths. Packages consumed by other projects are
  published and consumable from their registry.
- *Failure:* a Docker image missing `prompts/` and `migrations/`; a quickstart that needs
  `../promptlang` next to it.

### 8. Small, reviewable history
- Conventional Commits, enforced in CI.
- One logical change per commit; target ≤ 400 changed lines excluding lockfiles and generated files.
- All work lands through a pull request using the template checklist, even when working alone.
- `main` is protected; CI must be green to merge.
- *Failure:* single commits of 8 000+ lines; a CONTRIBUTING.md demanding Conventional Commits that
  the history ignored.

### 9. No dead weight
- Dead code, unused dependencies and duplicated helpers are caught by tooling
  (Python: `ruff` + `vulture` + `deptry`; TypeScript: `biome` + `knip`).
- A helper is extracted into a shared module the moment a second copy appears.
- *Failure:* the same helper copy-pasted into 7 files; declared dependencies never imported.

### 10. Ownership before merge
- Nothing merges until the author can explain every changed module: **what it does, why this
  design, how it is verified, how it fails.**
- Every PR includes a plain-language walkthrough and the answers to the review questions.
- Any architectural change gets an ADR, including the alternatives rejected. An ADR contradicted
  by the code is updated or superseded in the same PR.

### 11. Release gate
A tag is created only when: CI is green · `claims` is green · CHANGELOG entry exists ·
docs are in sync · a live artefact exists if behaviour changed · `SECURITY.md` reviewed.

---

## How AI-assisted work flows here

1. **Spec** — I write the objective, constraints and acceptance criteria.
2. **Build** — a coding agent implements on a branch.
3. **Independent review** — a second agent, with no access to the builder's reasoning,
   reviews the diff against these standards.
4. **Human audit** — I verify the claims, run the critical paths, and challenge anything I
   cannot explain.
5. **Gate** — CI (including `claims` and `stale-narrative`) must pass.
6. **Merge** — only after rule 10 is satisfied.

The coding agent is a fast junior collaborator whose output I am accountable for.

---

## Appendix A — How coding agents must work in my repos

Every repo's `CLAUDE.md` reproduces this appendix verbatim, below a short repo-specific
section (what it is, layout, exact commands). Its purpose: make the agent engineer, not
generate.

### Engineer, don't generate
1. **Read before you write.** Find the existing module, contract and test for what you are
   changing. Extend it; never create a parallel version.
2. **Smallest change that meets the acceptance criteria.** No speculative abstractions, no
   "for future use" parameters, no options nobody asked for.
3. **Second copy = extract.** About to paste a helper? Move it to a shared module and test it once.
4. **Failing test first** for every bug: show it red, then fix, then show it green.
5. **Name the failure modes** of what you write (timeout, empty input, malformed model output,
   provider down, oversized input, hostile content) and test the ones that matter.
6. **Comments explain why — never what, never when.** No sprint IDs, no "will be added later",
   no restating the code.
7. **Deterministic code over model calls** whenever the answer is computable. The model
   proposes; code validates.
8. **Never report success without the command output that proves it.**
9. **Challenge with evidence.** If an instruction, finding or ADR looks wrong, say so with file,
   line and output — do not comply silently, do not "fix" something that is not broken.
10. **Explain trade-offs** in the PR: what you chose, what you rejected, why.
11. **Independent review before handing back**: a fresh-context reviewer subagent checks the
    diff against these standards; report what it found, what you fixed, what you rejected and why.

### Forbidden
- Claiming in README/docs a feature that has no passing test listed in `docs/CLAIMS.md`.
- Future tense or sprint references in source code or docs.
- Presenting any score computed on a stub, mock or fixture as a quality measure.
- Concatenating untrusted content (retrieved, ingested, scraped, tool output) into a prompt
  without delimiters and an explicit untrusted-content instruction.
- Catch-all exception handlers without a justifying comment and a typed re-raise or log.
- Secrets held in plain strings where the framework offers a secret type
  (e.g. pydantic `SecretStr`).
- Any paid model call without a cost estimate and explicit approval.
- Commits over ~400 changed lines (lockfiles and generated artefacts excluded).
- Pushing, tagging or publishing: the human does that.

### Report format (end of every work unit, in French for Matteo)
1. What changed, point by point. 2. Evidence: commands + outputs. 3. Audit findings disproved,
with proof. 4. Reviewer findings: fixed / rejected + why. 5. Open items. 6. Walkthrough per
changed module — what it does, why this design, how it is verified, how it breaks — written
for someone who pilots agents well but did not hand-write the code. 7. Five comprehension
questions Matteo must answer before merging (expected answers folded at the end).
