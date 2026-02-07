# Agent and Contributor Guide

This is the single source of truth for agents (Claude, Gemini, Copilot, etc.)
and human contributors working in this repository.  Both `CLAUDE.md` and
`GEMINI.md` redirect here.

---

## Overview

open_gororoba is a research-style codebase mixing:

- **Rust domain crates** -- 13 workspace members under `crates/` covering
  algebra, cosmology, GR, optics, materials, quantum, statistics, spectral,
  LBM, control, data providers, CLI binaries, and PyO3 bindings.
- **Python layer** -- visualization and notebooks under `src/gemini_physics/`.
- **Artifacts** -- reproducible CSVs and plots under `data/` and `curated/`.
- **Narrative documents** -- theoretical analysis, claims tracking, and audit
  reports under `docs/`.
- **Formal proofs** -- Coq files under `curated/01_theory_frameworks/`.

Python-to-Rust migration is COMPLETE (all 15 modules ported, 855 Rust tests,
0 clippy warnings).  See `docs/ROADMAP.md` for architecture and evolution.

Many physics-facing statements are **hypotheses**.  Treat them as unverified
unless backed by first-party sources and a reproducible test or artifact.

---

## Hard Rules

These are non-negotiable across every agent and every PR.

### 1. ASCII-only

All repo-authored code and docs MUST use ASCII characters only.

- No Unicode punctuation (smart quotes, em dashes, arrows, Greek letters).
- Use ASCII spellings: `grad_`, `eps`, `A_infty`, `<=`, `->`.
- **Exception:** transcripts under `convos/` may contain Unicode and are
  treated as immutable inputs.
- Enforced by `make ascii-check` (runs `python3 bin/ascii_check.py --check`).
- To sanitize: `python3 bin/ascii_check.py --fix`, then clean up any
  `<U+....>` placeholders by hand.

### 2. Warnings-as-errors

Treat warnings as failures everywhere.

- Python: run checks and tests with `PYTHONWARNINGS=error`.
- Prefer `-X dev` when debugging locally.
- Rust: `cargo clippy --workspace -- -D warnings` (clippy lint failures block merges).
- Compiler and linter warnings are errors in CI.

### 3. Source-first

Do not treat `convos/` text as authoritative.

- Every claim becomes a testable hypothesis tied to first-party sources.
- Claims are tracked in `docs/CLAIMS_EVIDENCE_MATRIX.md` (435 rows).
- Domain breakdowns live in `docs/claims/by_domain/*.md` (16 files).

### 4. Provenance

- Do not commit large external binaries.
- Record hashes and provenance in `data/external/` or via `make provenance`.
- If a dataset is missing, either fetch it explicitly or label any synthetic
  replacement as synthetic.  Record provenance in `docs/BIBLIOGRAPHY.md`.

### 5. Citations

Never delete existing sources.  Only append or clarify as validation proceeds.

---

## Build and Test

Requires Python >= 3.11.  All commands use the project venv.

```
make install              # Create venv, install editable + dev deps
make test                 # pytest (warnings-as-errors)
make lint                 # ruff check on src/gemini_physics + tests
make smoke                # compileall + lint stats + artifact verifiers
make check                # test + lint + smoke (CI entry point)
make ascii-check          # Verify ASCII-only policy
make doctor               # Environment diagnostics
```

### Artifact generation (deterministic, reproducible)

```
make artifacts            # Regenerate all core artifact sets
make artifacts-motifs     # CD motif census (16D, 32D)
make artifacts-boxkites   # De Marrais boxkite geometry
make artifacts-reggiani   # Reggiani annihilator statistics
make artifacts-m3         # M3 transfer table
make artifacts-dimensional # Dimensional geometry sweeps
```

### External data

```
make fetch-data           # Download external datasets
make provenance           # Hash data/external/* into PROVENANCE.local.json
```

### Cleanup

```
make clean-artifacts      # Remove generated CSV/images/HDF5/LaTeX output
make clean                # Remove venv, caches, bytecode
make clean-all            # clean + clean-artifacts
```

Regenerate everything from scratch: `make clean-all && make install && make artifacts`.

### Rust workspace (parallel builds)

All Rust crates live under `crates/`.  Always build and test with maximum
parallelism -- this machine has 12 cores (AMD Ryzen 5 5600X3D).

**Build optimization**: `.cargo/config.toml` sets:
- `target-cpu=native` (enables AVX2, FMA, BMI2 for Zen 3)
- `opt-level = 1` for test profile (major speedup for compute-heavy algebra tests)
- `jobs = 12` (parallel compilation)

```
cargo build --workspace -j$(nproc)              # Full parallel build
cargo test  --workspace -j$(nproc)              # Full parallel test (all crates)
cargo clippy --workspace -j$(nproc) -- -D warnings  # Lint (warnings = errors)
cargo test  -p algebra_core -j$(nproc)          # Single-crate test (fastest iteration)
cargo test  -p stats_core -j$(nproc)            # Single-crate test
cargo test  --workspace -j$(nproc) -- --nocapture  # Tests with stdout
```

**During iteration**: target only the crate you changed rather than full workspace.
`algebra_core` is the heaviest (~240 tests, ~30-50s with opt-level=1).

Individual analysis binaries (25 total in `crates/gororoba_cli/src/bin/`):

```
# Algebra
cargo run --bin zd-search          -- --help   # Zero-divisor search
cargo run --bin oct-field          -- --help   # Octonion field computations

# Cosmology
cargo run --bin real-cosmo-fit     -- --help   # Pantheon+ / DESI BAO fitting
cargo run --bin bounce-cosmology   -- --help   # Bounce cosmology simulation
cargo run --bin neg-dim-eigen      -- --help   # Negative-dimension eigenvalues
cargo run --bin gravastar-sweep    -- --help   # Gravastar parameter sweep

# GR
cargo run --bin kerr-shadow        -- --help   # Kerr black hole shadow

# Optics / Materials
cargo run --bin grin-trace         -- --help   # GRIN lens ray tracing
cargo run --bin tcmt-sweep         -- --help   # Temporal coupled-mode sweep
cargo run --bin ema-calc           -- --help   # Effective medium approximation

# Quantum
cargo run --bin mera-entropy       -- --help   # MERA entropy analysis
cargo run --bin tensor-network     -- --help   # Tensor network contraction
cargo run --bin frac-schrodinger   -- --help   # Fractional Schrodinger
cargo run --bin frac-laplacian     -- --help   # Fractional Laplacian
cargo run --bin harper-chern       -- --help   # Harper-Chern topological

# Statistics / Ultrametric
cargo run --bin dm-ultrametric     -- --help   # DM-based ultrametricity
cargo run --bin baire-compact      -- --help   # Baire metric compactness
cargo run --bin frb-cascades       -- --help   # FRB temporal cascades
cargo run --bin frb-ultrametric    -- --help   # FRB multi-attribute ultrametric
cargo run --bin gw-merger-tree     -- --help   # GW merger tree analysis
cargo run --bin cosmic-dendrogram  -- --help   # Cosmic dendrogram
cargo run --bin multi-dataset-ultrametric -- --help  # Cross-dataset analysis

# Fluid dynamics
cargo run --bin lbm-poiseuille     -- --help   # Lattice Boltzmann Poiseuille

# Data
cargo run --bin fetch-datasets     -- --help   # Fetch external datasets
cargo run --bin fetch-chime-frb    -- --help   # Fetch CHIME/FRB catalog
```

### Other runtimes

- **Quantum/Qiskit:** use Docker (`docs/requirements/qiskit.md`).
- **Coq:** `make coq` (compiles axiom stubs for theorem inventories).
- **LaTeX:** `make latex` (builds `MASTER_SYNTHESIS.pdf`).

---

## Linting

### Python
- Tool: **ruff** (>= 0.6.0)
- Line length: 100
- Target: Python 3.11
- Rules: E, F, I, W, B
- First-party package: `gemini_physics`
- Excluded dirs: `.mamba`, `venv`, `data`, `curated`, `archive`, `convos`
- Config: `pyproject.toml` `[tool.ruff]`

### Rust
- Tool: **clippy** (ships with rustup)
- Policy: `-D warnings` (all warnings are errors)
- Format: `cargo fmt --all -- --check`
- Command: `cargo clippy --workspace -j$(nproc) -- -D warnings`
- Every PR must pass clippy with zero warnings before merge.

---

## Project Layout

```
crates/                     # Rust workspace (13 members)
  algebra_core/             #   Cayley-Dickson, Clifford, E8, box-kites, Reggiani, M3
  cosmology_core/           #   TOV, bounce, dimensional geometry, observational fitting
  gr_core/                  #   Kerr geodesics, gravastar, shadow, Schwarzschild
  materials_core/           #   Metamaterials, absorbers, optical database
  optics_core/              #   GRIN solver, ray tracing, warp metrics
  quantum_core/             #   Tensor networks, MERA, fractional Schrodinger, Casimir
  stats_core/               #   Frechet, bootstrap, Haar, ultrametric suite
  spectral_core/            #   Fractional Laplacian (periodic/Dirichlet)
  lbm_core/                 #   Lattice Boltzmann D2Q9
  control_core/             #   Control theory utilities
  data_core/                #   18 dataset providers + provenance + parsers
  gororoba_py/              #   PyO3 bindings (thin wrappers -> domain crates)
  gororoba_cli/             #   25 analysis binaries + fetch-datasets
src/
  gemini_physics/           # Python layer (visualization, remaining scripts)
  scripts/                  # Analysis, reporting, visualization, simulation
  verification/             # Artifact and claims verification scripts
  quantum/                  # Qiskit circuits (Docker recommended)
tests/                      # pytest suite
data/
  csv/                      # Generated CSVs (make artifacts)
  artifacts/images/         # Generated plots and dashboards
  external/                 # Downloaded datasets (gitignored, make fetch-data)
  h5/                       # HDF5 simulation output (gitignored, make run)
docs/
  ROADMAP.md                # Consolidated roadmap (architecture, crates, GR port plan)
  ULTRA_ROADMAP.md          # Granular test-driven claim-to-evidence checklists
  TODO.md                   # Current sprint tracker
  NEXT_ACTIONS.md           # Priority queue (top-5 items)
  claims/                   # Claims domain map, index, 16 domain breakdowns
  theory/                   # Physics reconciliation and derivations
  tickets/                  # Batch audit tickets
  engineering/              # Engineering analysis docs
  external_sources/         # Source references
  latex/                    # LaTeX manuscript sources
  convos/                   # Structured conversation extracts
curated/
  01_theory_frameworks/     # Coq proofs and theorem inventories
  04_observational_datasets/  # Curated observational data
convos/                     # Original brainstorming transcripts (immutable)
bin/                        # Utility scripts (ascii_check, doctor, provenance)
reports/                    # Generated analysis reports
```

---

## Claims Workflow

1. Hypotheses originate from brainstorming transcripts in `convos/`.
2. Each is formalized in `docs/CLAIMS_EVIDENCE_MATRIX.md` with a C-nnn ID.
3. Domain mapping: `docs/claims/CLAIMS_DOMAIN_MAP.csv` (436 rows).
4. Per-domain breakdowns: `docs/claims/by_domain/*.md` (16 domains).
5. Audit tickets track verification batches: `docs/tickets/C*_claims_audit.md`.
6. Status levels: Speculative -> Modeled -> Partially verified -> Verified.
7. Every claim must have a WHERE STATED reference to source code or artifact.

---

## Visualization Standards

Grand visualization specs (3160x2820, dark mode, annotated) are documented in
`docs/agents.md`.  All generated visual artifacts should follow those standards.

---

## Common Pitfalls

1. **Stale venv:** if imports fail after branch switches, `make clean && make install`.
2. **Non-ASCII sneaking in:** run `make ascii-check` before committing; Greek
   letters in docs are the usual culprit.
3. **Large files:** never commit files > 1 MB without discussion.  Use
   `make fetch-data` + `.gitignore` for external datasets.
4. **Warnings hidden:** always use `PYTHONWARNINGS=error`; silent warnings
   mask real issues.

---

## Documenting Insights

When implementing features or running analyses, non-obvious discoveries,
design decisions, and mathematical connections should be recorded in
`docs/INSIGHTS.md`. Each entry should include:

- A unique ID (I-nnn)
- Date and context (which module, which analysis)
- Related claims (C-nnn IDs from CLAIMS_EVIDENCE_MATRIX.md)
- The insight itself: what was discovered and why it matters

Examples of insight-worthy observations:
- A mathematical identity that simplifies computation
- A crate or library that prevents reimplementing known algorithms
- A representation change that transforms a null result into a signal
- A numerical method choice that affects accuracy or performance
- A connection between datasets that enables cross-validation

Do NOT document routine implementation details. Focus on knowledge that
would save time or prevent mistakes in future sessions.

---

## Rust Workspace Conventions

All external crates MUST be declared at the workspace level in the root
`Cargo.toml` under `[workspace.dependencies]` and referenced in sub-crate
`Cargo.toml` files with `workspace = true`. Never add version numbers
directly in sub-crate Cargo.toml files.

Before implementing any module in Rust, search for existing crates on
crates.io and lib.rs. Prefer composition (wrapping/extending existing
crates) over reimplementation. Document the decision in module-level docs
if no suitable crate exists.

---

## References

- `docs/ROADMAP.md` -- consolidated roadmap (architecture, crates, evolution, GR port plan)
- `docs/CLAIMS_EVIDENCE_MATRIX.md` -- master claims tracker
- `docs/claims/INDEX.md` -- claims navigation index
- `docs/INSIGHTS.md` -- research insights and design decisions
- `docs/BIBLIOGRAPHY.md` -- external source citations
- `docs/agents.md` -- visualization standards
- `docs/REPO_STRUCTURE.md` -- directory layout details
- `Cargo.toml` -- Rust workspace and dependency declarations
- `pyproject.toml` -- Python dependencies and linter config
- `Makefile` -- all build/test/artifact targets
