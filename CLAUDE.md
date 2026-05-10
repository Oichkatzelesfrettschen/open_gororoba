# open_gororoba: project-scoped agent guide

Purpose: a pure-Rust scientific computing workspace for Cayley-Dickson algebras,
cosmology, fluid dynamics (LBM), formal verification (Rocq 9.1.1), and a
TOML/SQLite-driven research-claim registry.

This file inherits from `~/.claude/CLAUDE.md` (global) and `/home/eirikr/AGENTS.md`
(home). It states only what is project-specific or overrides the inherited
guidance.

## Encoding policy (NON-NEGOTIABLE)

- ASCII only in every authored text file under this repository.
- No emojis, no smart quotes, no en/em dashes, no box-drawing characters.
- The `ansi-check` and `terminology-gate` pre-push hooks reject violations.

## Build and test gate

- Toolchain: nightly-2026-03-05 pinned via `rust-toolchain.toml`. Edition 2024.
- Warnings-as-errors via `[workspace.lints]` in root `Cargo.toml`. Do NOT bypass
  with crate-local `#![allow(warnings)]`. If a lint must be suppressed, write
  `#[allow(clippy::<lint>)]` at the narrowest scope and document the rationale.
- Default target dir: `CARGO_TARGET_DIR=.cache/gate-target` for gate runs.
- Pre-push hook at `.githooks/pre-push` (active via `core.hooksPath`):
  1. git-lfs handoff
  2. cache-check (size budget)
  3. terminology-gate (8 banned patterns)
  4. ansi-check (no emojis)
  5. rust-regression-scoped (warnings-as-errors)
  6. governance-gate-readonly
- Verify hook state: `git config --get core.hooksPath` should print `.githooks`.
  The file at `.git/hooks/pre-push` is an unused git-lfs stub.

## Registry: SQLite-canonical (since 2026-03-23)

- Canonical write target: `registry/canonical/control_plane.sqlite3`.
- `registry/*.toml` are AUTO-GENERATED read-only compatibility exports.
- Every TOML file in `registry/` starts with the header
  `# AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT.` Do not hand-edit them.
- Edit path: SQLite via `gororoba-db` CLI; then run
  `make registry-export-markdown` to regenerate compatibility lanes; then run
  `make integrity-resolution` to refresh `registry/schema_signatures.toml`.
- Source manifest: `registry/source_manifest.toml` declares the 36 TOMLs that
  participate in compatibility round-trip verification.
- Architecture walkthrough:
  `docs/engineering/registry_canonical_architecture.md` (4-layer flow:
  SQLite -> compat TOMLs -> Rust mirrors -> docs; mutation surface;
  splice mechanism; audit trail; common footguns).
- Audit metric taxonomy:
  `docs/engineering/repo_audit_metric_taxonomy.md` (every repo-audit
  metric, anchoring strategy, and what's "real debt" vs "valid
  suppression"; SQLite revisions integration via `--sqlite` flag).

## Documentation policy

- `*.md` is denied by `.gitignore` at the workspace root with explicit allowlist
  entries (see `.gitignore` lines 197-227+). Do not create new top-level `.md`
  files without first allowlisting them.
- `CLAUDE.md`, `AGENTS.md`, `GEMINI.md`, `README.md` are case-insensitively
  allowlisted at root and in every subdirectory.
- The `docs/` tree allows specific files only. Markdown under `docs/book/` is
  mdBook-managed.

## No-Python and no-symlinks policies

- No `.sh` scripts in this repo. Use Makefile targets that invoke
  `$(CARGO_ENV) cargo run --release -p <crate> --bin <name>`.
- No `.py` analysis scripts. Use Rust crates (`zarrs`, `rayon`, `wide`,
  `nalgebra`, `polars`) or wrap C/Python libs via PyO3 inside a typed Rust
  binary.
- Never use `ln -s` as a workaround. If multiple branches need to share a
  cache, use a separate `CARGO_TARGET_DIR` per worktree.

## Stage references and baselines

- Stage A audit pack: `data/output/audit/2026-04-30/` (30+ artifacts).
- Debt baseline TOML: `data/output/debt_baseline_2026_04_30.toml`.
- Baseline git tag: `debt-baseline-v0` on commit `970b4da3`.
- Active in-repo roadmap: `plans/repo_debt_roadmap_2026_04_11.toml`.
- Active user-scope plan: `~/.claude/plans/stage-b-debt-resolution.md`.

## Scientific debt classes (eleven)

Tracked in `plans/repo_debt_roadmap_2026_04_11.toml`:

1. DEBT-NUMERICAL-ALGORITHM (e.g., bounded_nelder_mead duplication)
2. DEBT-STRUCTURAL-ARCHITECTURE (e.g., data_core fetch/parse mixing)
3. DEBT-DUPLICATION (CPD clusters; PMD CPD lane runs in `make cpd-audit`)
4. DEBT-TEST-VERIFICATION (gate-audit narrowness; coverage)
5. DEBT-BUILD-WORKSPACE (Makefile sprawl; xtask migration target)
6. DEBT-DATA-PROVENANCE (default features and network isolation)
7. DEBT-GENERATED-ARTIFACT (registry_mirrors quarantine)
8. DEBT-DOCUMENTATION-REQUIREMENTS (audit lane coverage in REQUIREMENTS.md)
9. DEBT-SCIENTIFIC-EVIDENCE (claims linked to reproducible artifacts)
10. DEBT-SUPPLY-CHAIN (cargo-deny, geiger, machete)
11. DEBT-FORMAL-VERIFICATION (5 admits, 130 axioms, 64 parameters; Rocq 9.1.1)

## Workflows that matter

- `make cpd-audit CPD_TOP=20`: PMD-driven duplication audit.
- `make rust-clippy`: workspace clippy with deny warnings.
- `make rust-semver-check`: semver-checks for crate API stability.
- `make cargo-deny-check`: license, advisory, source policy gate.
- `make dep-audit`: cargo-audit advisory scan.
- `make docs-freshness`: confirms generated docs match source registries.
- `make integrity`: runs the verify lane (mirror + license + overflow checks).
- `make integrity-resolution`: regenerates `schema_signatures.toml`.

## Scientific-stack pinning

- statrs 0.18.0 requires nalgebra 0.33; do not upgrade nalgebra without
  resolving statrs first.
- gauss-quad 0.2.4, kodama 0.3.0, kiddo 5.2.4, petgraph 0.7, wide 0.7.
- cudarc 0.19.1 (NVRTC runtime compilation), burn 0.16+.
- See `~/.claude/projects/.../memory/MEMORY.md` for full pinning rationale and
  Rocq proof patterns (ring_simplify+lra, cbv+ring_simplify, tower rewrites,
  fuel recursion, Boolean reflection).

## What new agents should read first

1. This file.
2. `~/.claude/CLAUDE.md` (inherited).
3. `~/.claude/projects/-home-eirikr-Github-open-gororoba/memory/MEMORY.md`
   (project memory index).
4. `data/output/debt_baseline_2026_04_30.toml` (current baseline numbers).
5. `plans/repo_debt_roadmap_2026_04_11.toml` (architectural debt classes).
6. `~/.claude/plans/stage-b-debt-resolution.md` (immediate-action plan).
7. `docs/REQUIREMENTS.md` (toolchain prerequisites).
