# Repository Audit Report (2026-01-27)

## 1) What this repository is (honest scope)

This repo is a research-style sandbox combining:
- Python code experiments (`src/`)
- large AI-generated CSV corpora (`curated/`, `data/csv/legacy/`)
- plots and reports (`data/artifacts/`)
- narrative documents describing hypotheses (`docs/`)

Some mathematical facts used in the repo are standard (e.g., Cayley-Dickson non-associativity beyond 8D).
Many physics-facing statements are **hypotheses** and are not currently supported by first-party sources or
reproducible statistical tests.

## 2) High-confidence vs. speculative content

**High-confidence (verified via tests or standard math):**
- Cayley-Dickson algebras are non-associative from 8D upward.
- Norm composition fails in 16D (sedenions); zero divisors exist.
- Analytic continuation of sphere/ball formulas in dimension using Gamma is mathematically valid (meromorphic).

**Speculative / unverified (requires sourcing + tests):**
- "Negative-dimension vacuum" <-> LIGO mass peak mappings.
- Strong physical claims in `docs/THE_GEMINI_PROTOCOL.md` and related docs.
- Any claim that repo simulations "confirm" cosmological or particle-physics mechanisms without controlled
  models and comparisons.

Tracking file: `docs/CLAIMS_EVIDENCE_MATRIX.md`.

## 3) Changes made in this audit pass

### Reproducibility & quality gates
- Added packaging config for `gemini_physics` (`pyproject.toml`, `src/gemini_physics/__init__.py`).
- Added tests with warnings-as-errors (`pytest.ini`, `tests/`).
- Added linting for the kernel + tests (`ruff`, `make lint`).
- Added environment/compatibility report (`bin/doctor.py`, `make doctor`).
- Added artifact verification (image resolution + CSV schemas) (`make verify`).
- Added GitHub Actions CI for core tests + lint (`.github/workflows/ci.yml`).

### Mathematics: -4D -> 32D
- Implemented analytic continuation of d-ball volume and (d-1)-sphere area (`src/gemini_physics/dimensional_geometry.py`).
- Generated "grand" plots + CSV outputs for d in [-4,16] and [0,32] (`src/vis_dimensional_geometry.py`).
- Documented scope-safe interpretation (`docs/DIMENSIONAL_GEOMETRY.md`).

### Materials science: real property data + 4D->32D experiments
- Implemented a reproducible JARVIS-DFT subset downloader with provenance (`src/fetch_materials_jarvis_subset.py`,
  `src/gemini_physics/materials_jarvis.py`).
- Implemented 118D composition -> {4,8,16,32}D PCA experiments with distance-preservation metrics and "grand" plots
  (`src/materials_embedding_experiments.py`).
- Documented datasets and outputs (`docs/MATERIALS_DATASETS.md`).

### Coq integration (roadmap item)
- Added a minimal Coq model stub and a build workflow (`curated/01_theory_frameworks/ConfineModel.v`,
  `bin/coq_prepare_confine.py`, `make coq`).
- Updated Coq README to avoid overstating proof status (`curated/01_theory_frameworks/README_COQ.md`).

### Documentation corrections + citations
- Updated `README.md` to be scope-safe and consistent with the repo's actual state.
- Updated `docs/BIBLIOGRAPHY.md` with first-party sources for: de Marrais, Reggiani, dimensional regularization,
  Parisi-Sourlas, and JARVIS.
- Updated `docs/SEDENION_ATLAS.md` to cite primary sources and mark "per abstract" claims appropriately.
- Replaced `GEMINI.md` with repo guardrails instead of role-play "memory".
- Added `docs/REPO_OVERVIEW.md` and `docs/ULTRA_ROADMAP.md`.

## 4) Issues found

- Packaging mismatch: `src/gemini_physics/` existed but was not a real importable package.
- No reliable test runner in the committed venv (pytest missing).
- Several Python `SyntaxWarning`s from invalid LaTeX string escapes (fixed).
- Coq files were not compilable as-is (missing model + proofs); now compile as **axiom stubs** for interface checking.
- Several docs contained strong claims without citations or tests; now tracked and partially cited, but still largely
  unverified.

## 5) How to reproduce the new checks

```bash
make test
make lint
make verify
make coq
```

Optional artifacts:
```bash
make artifacts-dimensional
make artifacts-materials
```

## 6) Recommended next steps

1. Expand `docs/CLAIMS_EVIDENCE_MATRIX.md` into "citation + test" tasks, one claim at a time.
2. Replace "toy" mappings (e.g. to LIGO peaks) with falsifiable statistical models and null-model comparisons.
3. Add a second materials dataset backend (OQMD or NOMAD) to cross-check the JARVIS pipeline.
4. Decide whether to invest in a C++ accelerator (Numba may suffice).

