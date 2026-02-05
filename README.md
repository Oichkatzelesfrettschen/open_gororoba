# Gemini Experiments: Evidence-First Algebra/Physics Sandbox

This repository is a research-style sandbox focused on reproducible, offline-checkable
math/physics experiments (with a heavy emphasis on Cayley-Dickson algebras and related
computational invariants).

Repo policy: treat narrative text as hypotheses unless it is backed by a primary source and
an in-repo check (test/verifier/script + artifact).

## Start here

- Requirements: `docs/REQUIREMENTS.md`
- Repo structure and conventions: `docs/REPO_STRUCTURE.md`
- Claims and evidence hooks: `docs/CLAIMS_EVIDENCE_MATRIX.md`
- Task tracker: `docs/CLAIMS_TASKS.md`

Quick commands:

- Full gate (tests + lint + verifiers): `PYTHONWARNINGS=error make check`
- Verification-only gate (no artifact generation): `PYTHONWARNINGS=error make smoke`
- Full gate, parallel pytest workers (opt-in): `PYTHONWARNINGS=error make check-parallel`
- Tests only, parallel pytest workers (opt-in): `PYTHONWARNINGS=error make test-parallel`
- Opt-in experiments/simulations smoke: `PYTHONWARNINGS=error make experiments-smoke`
- Opt-in offline experiments suite (slower): `PYTHONWARNINGS=error make experiments`
- Opt-in metadata hygiene reports: `PYTHONWARNINGS=error make metadata-hygiene`
- Opt-in audit bundle (structure/network/pythonpath/provenance): `PYTHONWARNINGS=error make audits`
- Generate selected artifacts (optional): `make artifacts-motifs`, `make artifacts-materials`, `make artifacts-gwtc3`
- Run network-using artifact targets explicitly (opt-in): `GEMINI_ALLOW_NETWORK=1 make artifacts-net`

## What is in this repo

- Algebra engine and invariants under `src/gemini_physics/` (validated by tests under `tests/`)
- Offline-quality gates under `src/verification/`
- Script entrypoints under `src/scripts/` (some require opt-in network access; see Makefile targets)
- Cached external sources under `data/external/` with a local provenance registry

## Notes

- ASCII-only: repo-authored docs/code should remain ASCII-only (immutable transcripts under `convos/` are exempt).
- Warnings-as-errors: run checks/tests with `PYTHONWARNINGS=error` to catch fragile imports and deprecations early.
- Parallel tests + GPU notes: `docs/COMPUTING_SPECS.md`
- Naming: repo title is "Gemini Experiments"; the Python package name is `gemini-experiments` (see `pyproject.toml`).

### Artifact: Annotated Grand Warp Ring
- File: `data/artifacts/images/warp_ring_annotated.png` (~1.5 MB)
- SHA256: `b5e2fe84c603ea0d9b56d90c2472d859925e7b123a0326e46a9816b70245bf58`
- Summary: Publication-ready annotated visualization combining volumetric Sedenion glow with refractive starfield lensing; includes scale bar (10 Planck Lengths), callouts (Metamaterial Ergoregion, Sedenion ZD Projection), and legend (Toroidal Alcubierre metric, S16 algebra).
