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

Python-to-Rust migration is COMPLETE (all 15 modules ported, 1806 Rust tests,
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

Individual analysis binaries (30 total in `crates/gororoba_cli/src/bin/`):

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

# Audit / Verification
cargo run --bin claims-audit       -- --help   # Claims matrix audit
cargo run --bin claims-verify      -- --help   # Claims verification runner
cargo run --bin materials-baseline -- --help   # Materials science baselines
cargo run --bin mass-clumping      -- --help   # GWTC mass dip test
cargo run --bin motif-census       -- --help   # CD motif census
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

## Document Hierarchy

The project uses several tracking documents with distinct roles:

| Document | Role | Update cadence |
|----------|------|----------------|
| `docs/ROADMAP.md` | Architecture, crate map, evolution history | Sprint boundaries |
| `docs/TODO.md` | Current sprint checklist (active execution) | Every work session |
| `docs/NEXT_ACTIONS.md` | Priority queue for next sprint | Sprint boundaries |
| `docs/ULTRA_ROADMAP.md` | Granular claim-to-evidence mapping (historical) | Append-only |
| `docs/CLAIMS_EVIDENCE_MATRIX.md` | Master claims tracker (442 rows) | Per-claim updates |
| `docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md` | Reproducible experiment registry (E-001..E-010) | Per-experiment |
| `docs/MATH_CONVENTIONS.md` | Mathematical and numerical conventions | As conventions change |
| `docs/RISKS_AND_GAPS.md` | Current technical risks and gaps | Sprint boundaries |
| `docs/BIBLIOGRAPHY.md` | External source citations | As citations are added |

Overlapping content between TODO.md and NEXT_ACTIONS.md is intentional:
TODO.md tracks the current sprint's in-progress items; NEXT_ACTIONS.md
is the backlog for the following sprint.

---

## Binary Contracts

All 30 analysis binaries under `crates/gororoba_cli/src/bin/`:

| Binary | Input | Output | Claims | Det. |
|--------|-------|--------|--------|------|
| zd-search | None | stdout | C-050..C-060 | Yes |
| oct-field | None | stdout | -- | Yes |
| motif-census | None | data/csv/motif_census_*.csv | C-100..C-110 | Yes |
| real-cosmo-fit | data/external/Pantheon+SH0ES.dat | stdout | C-200..C-210 | Yes |
| bounce-cosmology | None | stdout/CSV | -- | Yes |
| neg-dim-eigen | None | stdout/CSV | C-420..C-425 | Yes |
| gravastar-sweep | None | CSV | C-400..C-410 | Yes |
| kerr-shadow | None | stdout/CSV | C-301..C-310 | Yes |
| grin-trace | None | stdout/CSV | -- | Yes |
| tcmt-sweep | None | stdout/CSV | -- | Yes |
| ema-calc | None | stdout | -- | Yes |
| mera-entropy | None | stdout | -- | Seed |
| tensor-network | None | stdout/CSV | C-350..C-360 | Seed |
| frac-schrodinger | None | stdout | -- | Yes |
| frac-laplacian | None | stdout | -- | Yes |
| harper-chern | None | stdout/CSV | -- | Yes |
| dm-ultrametric | data/external/chime*.csv | data/csv/ | C-071 | Seed |
| baire-compact | data/external/chime*.csv | data/csv/ | C-071 | Seed |
| frb-cascades | data/external/chime*.csv | data/csv/ | C-438 | Seed |
| frb-ultrametric | data/external/chime*.csv | data/csv/ | C-071 | Seed |
| gw-merger-tree | data/external/gwosc*.csv | data/csv/ | C-439 | Seed |
| cosmic-dendrogram | data/external/ | data/csv/ | C-440 | Seed |
| multi-dataset-ultrametric | data/external/ | data/csv/ | C-071, I-011 | Seed |
| lbm-poiseuille | None | stdout/CSV | -- | Yes |
| fetch-datasets | Network | data/external/ | -- | -- |
| fetch-chime-frb | Network | data/external/ | -- | -- |
| claims-audit | docs/ | stdout | -- | Yes |
| claims-verify | data/ | stdout | -- | Seed |
| materials-baseline | data/external/ | stdout | C-450..C-455 | Seed |
| mass-clumping | data/external/gwosc*.csv | data/csv/ | C-007 | Seed |

Det. column: Yes = fully deterministic, Seed = deterministic given --seed flag.

---

## References

- `docs/ROADMAP.md` -- consolidated roadmap (architecture, crates, evolution, GR port plan)
- `docs/CLAIMS_EVIDENCE_MATRIX.md` -- master claims tracker (442 rows)
- `docs/claims/INDEX.md` -- claims navigation index
- `docs/INSIGHTS.md` -- research insights and design decisions
- `docs/BIBLIOGRAPHY.md` -- external source citations
- `docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md` -- reproducible experiment registry (E-001..E-010)
- `docs/MATH_CONVENTIONS.md` -- mathematical and numerical conventions
- `docs/RISKS_AND_GAPS.md` -- current technical risks and gaps
- `docs/agents.md` -- visualization standards
- `docs/REPO_STRUCTURE.md` -- directory layout details
- `Cargo.toml` -- Rust workspace and dependency declarations
- `pyproject.toml` -- Python dependencies and linter config
- `Makefile` -- all build/test/artifact targets
