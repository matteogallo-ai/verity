# Supply-chain security records

## What runs

| Control | Where | What it does |
|---|---|---|
| Dependabot | `.github/dependabot.yml` | One grouped version-update PR per week each for uv (`uv.lock`), npm in `web/` (`web/pnpm-lock.yaml`) and GitHub Actions. |
| OSV-Scanner | `osv-scanner` job in `.github/workflows/ci.yml` | Fails a PR or push whose `uv.lock` or `web/pnpm-lock.yaml` resolves a package with a known vulnerability in the OSV database. Accepted exceptions, each with a reason and an expiry date, are in `osv-scanner.toml`. |
| CodeQL | `.github/workflows/codeql.yml` | Static analysis of Python and TypeScript on PRs to `main`, pushes to `main` and weekly. |
| Pinned actions | every workflow | Each action is referenced by full commit SHA with a `# vX.Y.Z` comment; Dependabot moves both. |
| Read-only token | every workflow | `permissions: contents: read` at the top; a job that needs more says so, with a comment. |
| Scorecard | `.github/workflows/scorecard.yml` | OpenSSF Scorecard on each push to `main`, on branch-protection changes and weekly; publishes the result behind the README badge and uploads the SARIF to code scanning. Until it has run once on `main`, the badge shows no score. |

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
default) and Vulnerabilities (129 known at the time; the dependency bumps on the same
branch and the two written exceptions in `osv-scanner.toml` bring the OSV-Scanner job
to zero). Pinned-Dependencies also counts what this branch does not pin: the
`python:3.12-slim` base image in the `Dockerfile` (referenced by tag, not by digest)
and two download-then-run steps in `scripts/docker_smoke.sh`.

## Checks that stay low on a single-maintainer repository

These scores measure properties a solo project does not have; they are not defects
the code can fix.

| Check | Before | Why it stays low |
|---|---|---|
| Code-Review | 0 | Counts merged changesets approved by a second human. With one maintainer nobody else approves, and GitHub forbids approving one's own PR. |
| Contributors | 3 | Counts contributors from distinct organizations over recent commits. One maintainer is one organization. |
| Maintained | 0 | Scores 0 for any repository created less than 90 days before the run, whatever its activity. |
| Branch-Protection | 3 | Its top tiers require at least one or two approving reviews and code-owner review, which a solo repository cannot satisfy without blocking its only author. The `main` ruleset requires a pull request and green status checks, and forbids force-push and deletion. |
| Signed-Releases | -1 | Scorecard reports "no releases found": the GitHub releases (v0.5.0 to v1.0.1) carry no assets, hence no signature or provenance file for it to inspect. It stays unscored as long as releases are not signed (for example with Sigstore or SLSA provenance attached as assets). |
| Fuzzing | 0 | No fuzzer (OSS-Fuzz, ClusterFuzzLite, Atheris, property-based fuzz targets) runs against the code. |
| CII-Best-Practices | 0 | Requires registering the project for an OpenSSF Best Practices badge on bestpractices.dev, a manual self-assessment outside the repository. |
| Packaging | -1 | Looks for a workflow that publishes a package to a registry. Verity publishes none; releases are tags. |
