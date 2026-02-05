# Agent and Contributor Guide

This is the single source of truth for agents (Claude, Gemini, Copilot, etc.)
and human contributors working in this repository.  Both `CLAUDE.md` and
`GEMINI.md` redirect here.

---

## Overview

open_gororoba is a research-style codebase mixing:

- **Executable experiments** -- Python under `src/` (NumPy, SciPy, Numba, SymPy).
- **Artifacts** -- reproducible CSVs and plots under `data/` and `curated/`.
- **Narrative documents** -- theoretical analysis, claims tracking, and audit
  reports under `docs/`.
- **Formal proofs** -- Coq files under `curated/01_theory_frameworks/`.

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

### Other runtimes

- **Quantum/Qiskit:** use Docker (`docs/requirements/qiskit.md`).
- **Coq:** `make coq` (compiles axiom stubs for theorem inventories).
- **LaTeX:** `make latex` (builds `MASTER_SYNTHESIS.pdf`).

---

## Linting

- Tool: **ruff** (>= 0.6.0)
- Line length: 100
- Target: Python 3.11
- Rules: E, F, I, W, B
- First-party package: `gemini_physics`
- Excluded dirs: `.mamba`, `venv`, `data`, `curated`, `archive`, `convos`
- Config: `pyproject.toml` `[tool.ruff]`

---

## Project Layout

```
src/
  gemini_physics/         # Core library (Cayley-Dickson, optics, groups)
  scripts/                # Analysis, reporting, visualization, simulation
  verification/           # Artifact and claims verification scripts
  quantum/                # Qiskit circuits (Docker recommended)
tests/                    # pytest suite (51 tests)
data/
  csv/                    # Generated CSVs (make artifacts)
  artifacts/images/       # Generated plots and dashboards
  external/               # Downloaded datasets (gitignored, make fetch-data)
  h5/                     # HDF5 simulation output (gitignored, make run)
docs/
  claims/                 # Claims domain map, index, 16 domain breakdowns
  theory/                 # Physics reconciliation and derivations
  tickets/                # Batch audit tickets
  engineering/            # Engineering analysis docs
  external_sources/       # Source references
  latex/                  # LaTeX manuscript sources
  convos/                 # Structured conversation extracts
curated/
  01_theory_frameworks/   # Coq proofs and theorem inventories
  04_observational_datasets/  # Curated observational data
convos/                   # Original brainstorming transcripts (immutable)
bin/                      # Utility scripts (ascii_check, doctor, provenance)
reports/                  # Generated analysis reports
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

## References

- `docs/CLAIMS_EVIDENCE_MATRIX.md` -- master claims tracker
- `docs/claims/INDEX.md` -- claims navigation index
- `docs/BIBLIOGRAPHY.md` -- external source citations
- `docs/agents.md` -- visualization standards
- `docs/REPO_STRUCTURE.md` -- directory layout details
- `pyproject.toml` -- dependencies and linter config
- `Makefile` -- all build/test/artifact targets
