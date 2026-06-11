---
description: Claude-specific entry point for open_gororoba; canonical policies live in AGENTS.md
last_verified: 2026-05-17
---

# open_gororoba -- Claude operating guide

This file is the Claude-specific entry point for this repository.
The canonical project-wide operating guide lives in `AGENTS.md` at
the same directory level; this file states only the Claude-specific
behaviour that sits on top of it.

## Read first (in this order)

1. `./AGENTS.md` (sibling of this file) -- canonical project rules,
   comment-hygiene policy, commit-trailer policy, GPU helper
   foundation, registry mutation workflow.
2. `~/.claude/CLAUDE.md` -- global user policies (ASCII,
   warnings-as-errors, no shortcuts, TodoWrite discipline,
   AskUserQuestion exhaustively, no destructive operations without
   explicit user authorisation).
3. `~/.claude/projects/-home-eirikr-Github-open-gororoba/memory/MEMORY.md`
   -- per-project memory index (claim counts, proof patterns,
   pitfalls, baseline tags).
4. `~/AGENTS.md` -- home-level cross-project policies.

If a Claude tool or agent skill conflicts with `AGENTS.md`, follow
`AGENTS.md` and surface the conflict to the user.

## Claude tool discipline

- **TaskCreate / TaskUpdate**: granular tracking is MANDATORY for
  multi-step work. Mark each subtask `in_progress` before starting
  and `completed` immediately after. One `in_progress` at a time.
- **AskUserQuestion**: ask EARLY and OFTEN when ambiguity exists.
  Do NOT proceed with implementation until the todo list reflects
  the clarified scope.
- **Agent + Explore**: prefer the Explore agent for broad codebase
  searches over running 5+ greps yourself. Use the Task tool for
  specialised work (code-reviewer, plugin-validator, etc.) when the
  agent description matches the task.
- **Plan mode**: use `ExitPlanMode` for plan approval, not
  `AskUserQuestion`. Use `AskUserQuestion` to clarify requirements
  WITHIN a plan.

## Commit + PR conventions for Claude

Sourced from `AGENTS.md` "Commit messages and PR descriptions". The
short form for Claude-authored commits:

- USE the mesa-canonical `Assisted-by:` trailer (not
  `Co-authored-by:` -- mesa reserves that for human co-authors).
  Canonical form:
  ```text
  Assisted-by: Claude (Opus 4.7 1M context)
  ```
- DO NOT use the legacy
  `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
  trailer in new commits. Historical commits keep it as-is; do not
  force-push to scrub.
- WHY / WHAT / HOW commit body structure remains the project norm.
  Lead with the load-bearing claim; cite primary sources by name;
  list verification commands explicitly.

## Memory hygiene

- `MEMORY.md` is the index; one line per memory file under ~150
  chars. Topic files in the same directory hold the body.
- Save memory only when content is non-obvious from the codebase /
  git history / docs. Code patterns, file paths, and recent commits
  are derivable; don't duplicate them.
- For feedback memories (corrections + confirmations), record the
  _why_ so future-you can judge edge cases. The "rule" alone rots
  faster than "rule + reason + when-to-apply".
- For project memories, convert relative dates to absolute on save
  ("Thursday" -> "2026-05-22").
- Before recommending from memory: verify named files / functions /
  flags still exist (the memory describes a snapshot in time;
  rename and removal happen).

## Project-specific overlays (delta on top of `AGENTS.md`)

These overlays exist to keep `AGENTS.md` more general; Claude reads
both:

### Encoding policy (NON-NEGOTIABLE)

- ASCII only in every authored text file under this repository.
- No emojis, no smart quotes, no en/em dashes, no box-drawing
  characters.
- The `ansi-check` and `terminology-gate` pre-push hooks reject
  violations.

### Build and test gate

- Toolchain: nightly-2026-03-05 pinned via `rust-toolchain.toml`.
  Edition 2024.
- Warnings-as-errors via `[workspace.lints]` in root `Cargo.toml`.
  Do NOT bypass with crate-local `#![allow(warnings)]`. If a lint
  must be suppressed, write `#[allow(clippy::<lint>)]` at the
  narrowest scope with a documented rationale.
- Default target dir: `CARGO_TARGET_DIR=.cache/gate-target` for
  gate runs.
- Pre-push hook at `.githooks/pre-push` (active via
  `core.hooksPath`): six-gate chain documented in `AGENTS.md`.
- Verify hook state: `git config --get core.hooksPath` should print
  `.githooks`.

### Registry: SQLite-canonical

See `AGENTS.md` section "Registry: SQLite-canonical" for the
mutation workflow. The summary: edit via `gororoba-db` against
`registry/canonical/control_plane.sqlite3`; re-export via
`provenance export-control-plane`; refresh hashes via
`make integrity-resolution`; commit atomically.

### Documentation policy

- `*.md` is denied by `.gitignore` at the workspace root with
  explicit allowlist entries (see `.gitignore` lines 197-227+). Do
  not create new top-level `.md` files without first allowlisting
  them.
- `CLAUDE.md`, `AGENTS.md`, `GEMINI.md`, `README.md` are
  case-insensitively allowlisted at root and in every subdirectory.
- The `docs/` tree allows specific files only. Markdown under
  `docs/book/` is mdBook-managed.

### No-Python, no-symlinks, no-shell

- No `.sh` scripts in this Rust repo. Use Makefile targets that
  invoke `$(CARGO_ENV) cargo run --release -p <crate> --bin <name>`.
- No `.py` analysis scripts. Use Rust crates (`zarrs`, `rayon`,
  `wide`, `nalgebra`, `polars`) or wrap C/Python libs via PyO3
  inside a typed Rust binary.
- Never use `ln -s` as a workaround. If multiple branches need to
  share a cache, use a separate `CARGO_TARGET_DIR` per worktree.

### Workflows that matter

- `make cpd-audit CPD_TOP=20`: PMD-driven duplication audit.
- `make rust-clippy`: workspace clippy with deny warnings.
- `make rust-semver-check`: semver-checks for crate API stability.
- `make cargo-deny-check`: license, advisory, source policy gate.
- `make dep-audit`: cargo-audit advisory scan.
- `make docs-freshness`: confirms generated docs match source
  registries.
- `make integrity`: runs the verify lane (mirror + license +
  overflow checks).
- `make integrity-resolution`: regenerates
  `schema_signatures.toml`.

### Scientific stack pinning

- `statrs` 0.18.0 requires `nalgebra` 0.33; do not upgrade
  `nalgebra` without resolving `statrs` first.
- `gauss-quad` 0.2.4, `kodama` 0.3.0, `kiddo` 5.2.4, `petgraph`
  0.7, `wide` 0.7.
- `cudarc` 0.19.1 (NVRTC runtime compilation), `burn` 0.16+.
- See the per-project memory at
  `~/.claude/projects/-home-eirikr-Github-open-gororoba/memory/MEMORY.md`
  for full pinning rationale and Rocq proof patterns.

## What new agents should read first

1. This file (`CLAUDE.md`).
2. `AGENTS.md` (sibling).
3. `~/.claude/CLAUDE.md` (inherited).
4. The per-project memory index at
   `~/.claude/projects/-home-eirikr-Github-open-gororoba/memory/MEMORY.md`.
5. `data/output/debt_baseline_2026_04_30.toml` (current baseline
   numbers).
6. `plans/repo_debt_roadmap_2026_04_11.toml` (architectural debt
   classes).
7. `~/.claude/plans/stage-b-debt-resolution.md` (immediate-action
   plan).
8. `docs/REQUIREMENTS.md` (toolchain prerequisites).
