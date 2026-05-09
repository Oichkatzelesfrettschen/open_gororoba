# open_gororoba: agent operating guide

This file is the canonical guide for any AI coding agent operating in this
repository. It sits beside `CLAUDE.md` (the Claude-specific guide); both share
the same project-specific policies. If they ever disagree, `CLAUDE.md` is
authoritative.

## Read first (in this order)

1. `./CLAUDE.md` (sibling of this file) -- project-specific policies.
2. `~/.claude/CLAUDE.md` -- global user policies (ASCII, warnings-as-errors,
   no shortcuts, TodoWrite discipline, AskUserQuestion exhaustively).
3. `~/.claude/projects/-home-eirikr-Github-open-gororoba/memory/MEMORY.md`
   -- per-project memory index.

## Top-line operating rules

- ASCII only. No emojis. No smart quotes.
- Warnings-as-errors. Do not bypass.
- SQLite-canonical registry. `registry/*.toml` are READ-ONLY exports.
- Pure Rust. No `.sh` scripts; no `.py` analysis scripts. Use PyO3 if a
  Python library must be wrapped, but call it from a typed Rust binary.
- No symlinks. Use separate `CARGO_TARGET_DIR` per worktree if needed.
- Pre-push hook at `.githooks/pre-push` runs 6 gates; do not skip with
  `--no-verify` unless explicitly directed.

## Editing the registry

The 36 TOML files under `registry/` are auto-generated. The canonical write
path is `registry/canonical/control_plane.sqlite3`. To change a claim, an
insight, an experiment, or a binary entry:

1. Edit via `gororoba-db` CLI (in `crates/gororoba_db/`) against the SQLite.
2. Re-export compatibility lanes:
   `cargo run -p gororoba_cli_data --bin provenance -- export-control-plane`.
3. Refresh integrity hashes: `make integrity-resolution`.
4. Commit the SQLite delta and the regenerated TOMLs and markdown atomically.

Never hand-edit a file whose first line is
`# AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT.` -- those are derived
artifacts.

## Build environment

- Toolchain: `rust-toolchain.toml` pins nightly-2026-03-05, edition 2024.
- Default target dir for gates: `CARGO_TARGET_DIR=.cache/gate-target`.
- Per-worktree experimental dirs: `.cache/exp-<name>-target/`.
- Cache budget: gate-target <= 200G; full `.cache` <= 250G; sweep with
  `make cache-sweep` (cargo-sweep --maxsize 100GB).

## Stage gates and audit baselines

- Stage A audit (forensic): `data/output/audit/2026-04-30/`.
- Debt baseline TOML: `data/output/debt_baseline_2026_04_30.toml`.
- Git tag for baseline: `debt-baseline-v0`.
- Stage B execution plan: `~/.claude/plans/stage-b-debt-resolution.md`.
- Architectural debt roadmap: `plans/repo_debt_roadmap_2026_04_11.toml`.

## Scientific stack notes

- Cayley-Dickson tower depth trichotomy: 8-16D = discrimination,
  32-64D = nonlinearity, 128D+ = channel-starved.
- MaNGA real-data null is dimension-independent (C-1366); do not confuse the
  CD-tower synthetic regime with real-data sensitivity.
- Pantheon+ data is retained as evidence; only the orthoplex diffusion MODEL
  (C-932) is FALSIFIED. Pantheon+ is required by C-441, C-787, C-788.
- VF (void fraction): `vf = void_count / total_cells` in LBM grids;
  `(1 - vf)` is the baryon filling fraction. CDG-2 constraint:
  `(1 - vf) < 0.0006`.

## When ambiguity arises

- Use `AskUserQuestion` early and often.
- Use `TaskCreate`/`TaskUpdate` to plan and track work granularly.
- Prefer the EXPLORE agent for broad codebase searches over running multiple
  greps yourself.
- Never invent file paths; verify with `find`/`grep` first.
