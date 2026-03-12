<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: agents.toml -->

# Agent and Contributor Operating Manual

## Rust First

This repository is a pure Rust codebase first and foremost.

All new automation, analysis, verification, ingestion, artifact generation, and claim-support implementations MUST be written in Rust crates, Rust binaries, or Rust tests.

Python, notebooks, and ad hoc scripting are migration debt, not acceptable end states for new work.

If a workflow currently exists only in a non-Rust form, the required action is to migrate it to Rust rather than add another non-Rust layer beside it.

## Policy Link

`agents.toml` is the canonical machine-readable policy file for this repo.
Read it with this file, keep the two in sync, and follow the stricter Rust-first interpretation if they ever drift.
