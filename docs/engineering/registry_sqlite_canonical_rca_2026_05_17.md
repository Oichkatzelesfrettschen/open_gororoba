---
title: RCA -- registry/book_docs.toml direct edits bypassing the SQLite canonical write path
date: 2026-05-17
status: investigation_closed
related_pr: 48
related_memory: feedback_registry_sqlite_canonical
---

# RCA: book_docs.toml + schema_signatures.toml drift (2026-05-17)

## Symptom

Code review on the registry-touching PRs surfaced two coupled
defects:

1. `registry/book_docs.toml` was being edited directly in feature
   PRs (notably the 2026-05-12..14 Moreno paper-row updates),
   despite the file's first-line header declaring
   `# AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT.`
2. `registry/schema_signatures.toml::SIG-0002` recorded
   `row_count: 485` for `registry/binaries.toml`, but the
   checked-in TOML carried only 452 rows. The 33-row gap is by
   design (filtered compat lane hides retired binaries) but the
   signature row must describe the file on disk, not the canonical
   SQLite count.

PR #48 (`fix(registry): regenerate schema_signatures.toml so binary
row_count matches export`) addressed the immediate signature
mismatch. The underlying RCA -- why the AUTO-GENERATED header was
ignored in the first place -- is documented here.

## Background: SQLite-canonical policy

The policy has been in effect since 2026-03-23 (see
[feedback_registry_sqlite_canonical](../../registry/feedback_registry_sqlite_canonical.md)
if mirrored into the repo, otherwise the per-project memory at
`~/.claude/projects/-home-eirikr-Github-open-gororoba/memory/feedback_registry_sqlite_canonical.md`).
The four-layer flow is documented in
[`registry_canonical_architecture.md`](registry_canonical_architecture.md):

```
SQLite (canonical, control_plane.sqlite3)
   -> registry/*.toml (compatibility export, READ-ONLY)
   -> crates/data_core/src/registry_mirrors/*.rs (Rust mirrors)
   -> docs/*.md (human-readable reflections)
```

Every TOML in `registry/` carries this header on line 1:

```
# AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT.
```

The mutation workflow (also in `AGENTS.md` "Registry: SQLite-canonical"):

1. Edit via `gororoba-db` CLI against `registry/canonical/control_plane.sqlite3`.
2. Re-export via `provenance export-control-plane`.
3. Refresh hashes via `make integrity-resolution`.
4. Commit SQLite + regenerated TOMLs + regenerated markdown atomically.

## What went wrong

The Moreno paper-row update in `registry/book_docs.toml` was
hand-edited as part of a downstream feature PR (the proof-related
work landing the Moreno 1.16 arbitrary-a witness lane).

Direct cause: the file editor (LLM + human review chain) operated
on `registry/book_docs.toml` as if it were source-of-truth, not as
a generated export. The AUTO-GENERATED header is the only signal
that points at the canonical SQLite, and editor agents do not
always re-read line 1 before mutating.

Contributing factors:

- The per-project memory + AGENTS.md describe the policy in prose,
  but the lint surface that ENFORCES it (the
  `governance-gate-readonly` step in the pre-push 6-gate chain)
  catches signature drift only AFTER the bad commit lands locally.
  By the time the gate fires, the working tree already carries the
  hand-edit.
- The signature mismatch in SIG-0002 (485 vs 452) had been carried
  forward for some time; the canonical SQLite had moved past the
  TOML's row count without a corresponding re-export. This made
  the gate complain about content_sha mismatch on EVERY registry
  PR, which trained agents to "fix it by regenerating signatures"
  rather than "fix it by going through the canonical path".

## What was done

- **PR #48**: regenerated `registry/schema_signatures.toml` via
  `cargo run --release -p gororoba_cli_data --bin integrity-resolution`
  so SIG-0002..0011 hash against the actual on-disk TOMLs. Single-
  file commit, no canonical SQLite mutation. This unblocks the
  governance gate on subsequent registry-touching PRs.

- **AGENTS.md import (PR #50)**: the imported steinmarder
  comment-hygiene policy + mesa-26 commit-trailer policy include
  the SQLite-canonical mutation workflow as a top-line operating
  rule. Future agents reading AGENTS.md see the four-step workflow
  before touching `registry/*`.

## What is still open

Hardening to prevent recurrence:

1. **Pre-edit lint**: a pre-commit hook that refuses any change to
   a `registry/*.toml` whose first line contains the AUTO-GENERATED
   marker, unless the same commit ALSO touches
   `registry/canonical/control_plane.sqlite3`. The hook would
   reject the bad pattern at write time rather than at push time.
2. **Linter for `book_docs.toml`-style markdown bodies inside TOML
   strings**: the Moreno row in book_docs.toml is a single long
   markdown-flavored string. Diffing it as TOML is awkward; the
   canonical SQLite stores the same text in `bibliography.notes`.
   Auditing whether the SQLite content matches the TOML row
   content for each book_docs entry would catch silent drift that
   the content_sha doesn't (the sha matches both sides of a
   drifted pair if both were edited together by hand).
3. **Audit trail**: every registry mutation through
   `gororoba-db` should write a row to
   `registry/canonical/audit_log` (table already exists) with the
   tool name (`gororoba-db`), the timestamp, and the affected
   table+row. Edits via any other path (hand-edit, IDE, agent
   tool) leave no audit trail and are thus invisible to the
   integrity check.

Status: items 1-3 above are tracked under DEBT-GENERATED-ARTIFACT
in `plans/repo_debt_roadmap_2026_04_11.toml`. Implementation order
depends on the broader build-out of `gororoba-db`'s mutation API
surface and the pre-commit linter framework.

## Cross-links

- [`registry_canonical_architecture.md`](registry_canonical_architecture.md)
- [`repo_audit_metric_taxonomy.md`](repo_audit_metric_taxonomy.md)
- `AGENTS.md` "Registry: SQLite-canonical" section (canonical
  4-step workflow).
- `~/.claude/projects/-home-eirikr-Github-open-gororoba/memory/feedback_registry_sqlite_canonical.md`
  (per-project memory; auto-loaded into agent sessions).
- PR #48 commit: `7fff56db`
  (`fix(registry): regenerate schema_signatures.toml so binary row_count matches export`).
- PR #50 commit: `b9a3db70`
  (`docs: expand open_gororoba agent guide`) -- carries the
  imported AGENTS.md hygiene policy that surfaces the SQLite
  workflow as a top-line operating rule.
