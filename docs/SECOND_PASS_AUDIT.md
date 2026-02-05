# Second-Pass Audit (2026-01-27)

This document records a second audit pass focused on:
- reproducible checks (`make test`, `make verify`, `make smoke`)
- repo-wide lint visibility (without breaking the existing kernel/test lint gate)
- internal doc integrity (broken relative links)
- provenance hardening for `data/external/`
- reducing over-claiming in high-level narrative docs

## What was run

- `PYTHONWARNINGS=error make test lint verify doctor`
- `make lint-all-stats` (visibility only; does not fail the build)
- `PYTHONWARNINGS=error venv/bin/python3 -m compileall -q src`
- `make verify-grand` (enforces "grand" resolution only for files named `*3160x2820*.png`)
- `make smoke`
- A markdown-local-link audit across repo docs (excluding `convos/`)

## Changes made

### Code / build tooling

- Added repo-wide lint visibility targets:
  - `make lint-all` (strict; currently fails)
  - `make lint-all-stats` (reports counts; exit 0)
  - `make lint-all-fix-safe` (whitespace + import-order fixes)
- Reduced repo-wide Ruff issues from ~1000 -> ~160 by:
  - applying safe whitespace/import-order fixes across `src/`
  - fixing several high-visibility scripts so they are Ruff-clean
- Added artifact compliance tooling:
  - `make verify-grand` via `src/verification/verify_grand_images.py`
  - `make smoke` to run compile, lint, artifact verification, and grand-image reporting
- Added local provenance hashing:
  - `make provenance` writes `data/external/PROVENANCE.local.json`

### Documentation integrity

- Fixed broken image links in `docs/FINAL_MANUSCRIPT.md` (docs-relative paths needed `../data/...`).
- Reduced over-claiming in `docs/THE_GEMINI_PROTOCOL.md` and `docs/RESEARCH_STATUS.md` by:
  - labeling physics interpretations as speculative unless validated
  - pointing to the actual code paths under `src/gemini_physics/`
- Updated `docs/BIBLIOGRAPHY.md` with additional first-party sources:
  - GWOSC GWTC-3 event portal + data-release docs
  - Kinyon & Sagle (2006) on sedenion subalgebras and G2 action
- Updated `docs/CLAIMS_EVIDENCE_MATRIX.md` statuses to reflect the presence of primary sources
  (while keeping "implementation/replication" work outstanding).

## Current known gaps (not failures)

- Repo-wide Ruff for all `src/*.py` still reports ~160 issues (mostly line length, semicolons, unused vars/imports).
- Many images under `data/artifacts/images/` are not "grand" resolution; only a subset is currently enforced.
- Several high-level physics claims remain speculative and require:
  - strict provenance for external datasets (GWOSC/GWpy re-fetch)
  - explicit statistical tests (selection effects, null models)
  - clearer mapping between toy operators and physical models

## Next document

See `docs/NEXT_ACTIONS.md` for a prioritized, testable follow-up list.

