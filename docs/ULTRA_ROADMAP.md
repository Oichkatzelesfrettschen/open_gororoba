# Ultra Roadmap (Granular, Test-Driven)

**Date:** 2026-01-27

This roadmap is organized so that every major claim becomes:
1) a cited statement, and
2) a small reproducible experiment or test.

## A. Reproducibility & Quality Gates

- [x] Package `gemini_physics` properly (`pyproject.toml`, `src/gemini_physics/__init__.py`).
- [x] Add `pytest` with warnings-as-errors (`pytest.ini`, `make test`).
- [x] Add `ruff` lint gate for kernel + tests (`make lint`).
- [x] Add environment report (`make doctor`).
- [x] Add artifact verifier for resolution/schema (`make verify`).
- [ ] Extend linting from `src/gemini_physics/` to all `src/*.py` (phased).
- [x] Add CI workflow (GitHub Actions) mirroring `make test lint`.
- [x] Add repo-wide lint visibility targets (`make lint-all-stats`, `make lint-all`).

## B. Claims -> Evidence (High Priority)

- [ ] For each item in `docs/CLAIMS_EVIDENCE_MATRIX.md`, add:
  - [ ] a primary-source citation (or explicitly mark speculative),
  - [ ] a reproducible script/test validating the narrowest computable part.
- [ ] GWTC-3 provenance hardening:
  - [ ] re-fetch and checksum `data/external/GWTC-3_GWpy_Official.csv`,
  - [ ] record query parameters + date,
  - [ ] tie plots to that provenance.
- [ ] Reggiani (arXiv:2411.18881) integration:
  - [ ] extract the exact definitions of `Z(S)` and `ZD(S)` as used in the paper,
  - [ ] align repo terminology to the paper's.

## C. Dimensional Continuation (-4D -> 32D)

- [x] Implement analytic-continuation formulas (`src/gemini_physics/dimensional_geometry.py`).
- [x] Add tests for integer dimensions and identities (`tests/test_dimensional_geometry.py`).
- [x] Generate "grand" plots and CSV outputs (`src/vis_dimensional_geometry.py`).
- [ ] Add pole-aware plotting (optional: residue plots / annotated singularities).

## D. Materials Science (Real Data)

- [x] Add JARVIS-DFT subset fetcher + provenance (`src/fetch_materials_jarvis_subset.py`).
- [x] Add 118D composition -> {4,8,16,32}D PCA experiments (`src/materials_embedding_experiments.py`).
- [ ] Add a second dataset backend (OQMD or NOMAD) for cross-validation.
- [ ] Add element-property featurization (e.g., periodic-table descriptors) and compare vs. pure composition.
- [ ] Add predictive baselines (e.g., linear regression for formation energy / bandgap) with proper train/test splits.

## E. Algebra Engine (Sedenions and Beyond)

- [x] Add tests for associativity and norm composition behavior (`tests/test_cayley_dickson_properties.py`).
- [ ] Replace "random search for zero divisors" with a literature-derived explicit construction and tests.
- [ ] Add a fast basis-element multiplication table generator (for 16D and 32D) with cache + checksum.
- [ ] (Optional) Add a C++/pybind11 accelerator if Numba becomes a bottleneck.

## F. Coq Formalization

- [x] Add a buildable Coq model stub + compile workflow (`ConfineModel.v`, `make coq`).
- [ ] Decide the actual semantics for `has_right` / `reachable_delegation`.
- [ ] Prove a minimal non-trivial theorem end-to-end (no axioms) to validate the pipeline.

## G. Documentation & Paper

- [x] Update `README.md` to be scope-safe and honest.
- [x] Add `docs/REPO_OVERVIEW.md` and `GEMINI.md` operational guardrails.
- [ ] Add a "paper-ready" LaTeX build pipeline (`make latex`) and minimal draft outline.
- [ ] Convert high-level narratives into a structured "hypotheses + tests + results" format.

## H. Universal-to-Local Dataset Pillars (Cosmology -> Solar-System -> Earth)

This section is the active missing-datasets execution plan. Each item should
end with cached artifacts, provenance hashes, and offline validation hooks.

- [x] Wire missing-provider modules into Rust `fetch-datasets` registry.
- [x] Add `Union3` provider for legacy supernova likelihood chains.
- [x] Add `Hipparcos` provider for legacy local-distance anchoring.
- [x] Add `EHT` M87* and Sgr A* public data-bundle providers.
- [x] Add `DE440` and `DE441` ephemeris kernel providers.
- [x] Add `GRACE-FO` provider for time-variable Earth gravity.
- [x] Add `EGM2008` provider for static geoid.
- [x] Add `Swarm` magnetic-field sample provider.
- [x] Add `SORCE` TSI provider as legacy irradiance complement.
- [x] Add `Landsat` STAC metadata provider for Earth reflectivity pipelines.
- [x] Update `docs/external_sources/DATASET_MANIFEST.md` to include all new providers.
- [ ] Add `fetch-datasets --pillar {candle,map,image,threebody,gravity,magnetism,light}` grouping flags.
- [ ] Implement parser-level schema checks for every newly added dataset.
- [ ] Add deterministic row-count and column-integrity tests for Union3 chains.
- [ ] Add deterministic gzip-table sanity checks for Hipparcos.
- [ ] Add archive member and expected-filename checks for EHT bundles.
- [ ] Add SPK kernel checksum pinning for DE440/DE441 in provenance.
- [ ] Add GFC harmonic-degree sanity checks for GRACE-FO and EGM2008.
- [ ] Add Swarm CSV header and timestamp monotonicity tests.
- [ ] Add SORCE/TSIS overlap comparison script with reproducible CSV output.
- [ ] Add Landsat STAC asset-schema validation and cloud-cover extraction.
- [ ] Add a `dataset_coverage` report that maps claims to concrete dataset files.
- [ ] Link dataset pillars to claim IDs in `docs/CLAIMS_EVIDENCE_MATRIX.md`.
- [ ] Add a verifier ensuring every dataset in manifest has a provider in CLI.
- [ ] Add a verifier ensuring every cached dataset hash is present in provenance JSON.
- [ ] Add a verifier ensuring docs source-index links resolve to cached files.
- [ ] Add benchmark scripts for parser throughput (rows/s) on large catalogs.
- [ ] Add benchmark scripts for ephemeris interpolation accuracy vs Horizons snapshots.
- [ ] Add benchmark scripts for gravity-harmonic truncation error curves.
- [ ] Add benchmark scripts for magnetic-field sample coverage by mission day.
- [ ] Add benchmark scripts for irradiance time-series gap detection.
- [ ] Add benchmark scripts for Landsat scene metadata filtering throughput.
- [ ] Integrate new verifiers into `make smoke` after violation burn-down.
- [ ] Add date-stamped status snapshots in `docs/CLAIMS_TASKS.md` with one canonical count.
- [ ] Run `make check`, `cargo test --workspace -j$(nproc)`, and `cargo clippy --workspace -j$(nproc) -- -D warnings` as release gate.
