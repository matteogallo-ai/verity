# Supply-chain security records

## What runs

| Control | Where | What it does |
|---|---|---|
| Dependabot | `.github/dependabot.yml` | Weekly version updates for uv (`uv.lock`), npm in `web/` (`web/pnpm-lock.yaml`), GitHub Actions and the image digests of the `Dockerfile` and `docker-compose.yml`: minor and patch bumps in one grouped PR per ecosystem, each major bump in its own PR, read and merged or closed on its own. |
| OSV-Scanner | `osv-scanner` job in `.github/workflows/ci.yml` | Fails on PRs to `main` and pushes to `main` when `uv.lock` or `web/pnpm-lock.yaml` resolves a package with a known vulnerability in the OSV database. The scanner image is pinned by digest. Accepted exceptions, each with a reason and an expiry date, are in `osv-scanner.toml`. |
| CodeQL | `.github/workflows/codeql.yml` | Static analysis of Python and TypeScript on PRs to `main`, pushes to `main` and weekly. |
| Pinned actions | every workflow | Each action is referenced by full commit SHA with a `# vX.Y.Z` comment; Dependabot moves both. |
| Checkout without stored token | every workflow | `actions/checkout` runs with `persist-credentials: false`, so no later step finds the job token in `.git/config` (`tests/unit/test_container.py::test_no_workflow_checkout_keeps_the_token_in_git_config`). |
| Read-only token | every workflow | `permissions: contents: read` at the top; a job that needs more says so, with a comment. |
| Scorecard | `.github/workflows/scorecard.yml` | OpenSSF Scorecard on each push to `main`, on branch-protection changes and weekly; publishes the result behind the README badge and uploads the SARIF to code scanning. Until it has run once on `main`, the badge shows no score. |

A failing OSV-Scanner or CodeQL job blocks a merge only once its check name is in the
required status checks of the `main` ruleset.

`web/package.json` overrides `next>postcss` to 8.5.32: next 15.5 pins postcss 8.4.31
exactly, which carries known advisories. The override is maintained by hand, not by
Dependabot; drop it when a next release requires a postcss without advisories.

## `scorecard-before.json`

[OpenSSF Scorecard](https://github.com/ossf/scorecard) v5.6.0 on `main` at `a0c5e8f`
(2026-11-02), before the supply-chain changes above. Overall score 3.3/10. Produced
with:

```bash
GITHUB_AUTH_TOKEN=$(gh auth token) docker run --rm -e GITHUB_AUTH_TOKEN \
  ghcr.io/ossf/scorecard@sha256:c0a21a1c33a2d8819937d0a8a62e6f4189c800639abd9f0bfbe61c90c71fae90 \
  --repo=github.com/matteogallo-ai/verity --format=json --show-details
```

The image is pinned by digest. The file is the command's output, re-indented with
`json.dumps(indent=2)`; the token is passed through the environment and appears
nowhere in it.

Checks at 0 that the changes above address: Dependency-Update-Tool (Dependabot),
Pinned-Dependencies (actions by SHA), SAST (CodeQL), Token-Permissions (read-only
default) and Vulnerabilities (129 known at the time; the dependency upgrades and two
written exceptions brought the OSV-Scanner job to zero; one exception remains in
`osv-scanner.toml`).
Pinned-Dependencies also counted the `python:3.12-slim` base image in the `Dockerfile`,
referenced by tag at the time; it, the `uv` image and the `pgvector` image are now
pinned by digest (`tests/unit/test_container.py::test_every_external_image_is_pinned_by_digest`).
Its two "downloadThenRun" warnings on
`scripts/docker_smoke.sh` are false positives: those lines `curl` the local Compose
stack and pipe the JSON response into `python3 -c`; nothing is downloaded from the
internet.

Maintained scored 0 because Scorecard gives 0 to any repository created less than 90
days before the run (this one was created on 2026-09-05), whatever its activity.

## Checks that stay low on this repository

| Check | Before | Why it stays low |
|---|---|---|
| Code-Review | 0 | Counts merged changesets approved by a second human. With one maintainer nobody else approves, and GitHub forbids approving one's own PR. |
| Contributors | 3 | Counts contributors from distinct organizations over recent commits. One maintainer is one organization. |
| Branch-Protection | 3 | Its top tiers require one or two approving reviews and code-owner review, which a solo repository cannot satisfy without blocking its only author. The `main` ruleset requires a pull request and green status checks, and forbids force-push and deletion. |
| Signed-Releases | -1 | Scorecard reports "no releases found": the GitHub releases (v0.2.0 to v1.0.1) carry no assets, hence no signature or provenance file for it to inspect. It stays unscored as long as releases are not signed (for example with Sigstore or SLSA provenance attached as assets). |
| Fuzzing | 0 | No fuzzer (OSS-Fuzz, ClusterFuzzLite, Atheris) runs against the code. |
| CII-Best-Practices | 0 | Requires registering the project for an OpenSSF Best Practices badge on bestpractices.dev, a manual self-assessment outside the repository. |
| Packaging | -1 | Looks for a workflow that publishes a package to a registry. Verity publishes none; releases are tags. |
