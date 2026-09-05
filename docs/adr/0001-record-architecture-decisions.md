# ADR 0001 — Record architecture decisions

**Status:** accepted · 2026-09

## Context
Verity's whole point is to demonstrate production engineering discipline. Decisions with
long-lived consequences (datastore, refusal semantics, dogfooding) should be legible to
a reviewer without archaeology through commit history.

## Decision
Use lightweight ADRs (one markdown file per decision, numbered, immutable once accepted;
superseded rather than edited). Each records context, the decision, and its consequences.

## Consequences
A senior engineer can reconstruct *why* the system looks the way it does in minutes.
Reversals are explicit (a new ADR supersedes an old one), which is itself a signal of
disciplined change management.
