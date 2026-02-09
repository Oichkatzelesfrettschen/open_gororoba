<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_root_narratives.toml -->

# Ultra Roadmap (Granular, Test-Driven)

**Date:** 2026-01-27 (updated 2026-02-06)
**See also:** [`docs/ROADMAP.md`](ROADMAP.md) for architecture, crate ecosystem, and GR port plan.

This roadmap is organized so that every major claim becomes:
1) a cited statement, and
2) a small reproducible experiment or test.

**Migration status:** COMPLETE. All 15 Python modules ported to Rust domain crates.
1693 Rust tests (unit + integration + doc) pass, 0 clippy warnings. Section H: 21/21 DONE.
Remaining open items are forward-looking research (Sections B partial, C, D, F, G).

## A. Reproducibility & Quality Gates

- [x] Package `gemini_physics` properly (`pyproject.toml`, `src/gemini_physics/__init__.py`).
- [x] Add `pytest` with warnings-as-errors (`pytest.ini`, `make test`).
- [x] Add `ruff` lint gate for kernel + tests (`make lint`).
- [x] Add environment report (`make doctor`).
- [x] Add artifact verifier for resolution/schema (`make verify`).
- [x] ~~Extend linting from `src/gemini_physics/` to all `src/*.py` (phased).~~ Obsolete: Python migration complete, only `__init__.py` remains.
- [x] Add CI workflow (GitHub Actions) mirroring `make test lint`.
- [x] Add repo-wide lint visibility targets (`make lint-all-stats`, `make lint-all`).

## B. Claims -> Evidence (High Priority)

- [ ] For each item in `docs/CLAIMS_EVIDENCE_MATRIX.md`, add:
  - [ ] a primary-source citation (or explicitly mark speculative),
  - [ ] a reproducible script/test validating the narrowest computable part.
- [x] GWTC-3 provenance hardening:
  - [x] re-fetch and checksum `data/external/GWTC-3_GWpy_Official.csv`,
  - [x] record query parameters + date,
  - [x] tie plots to that provenance.
  - Done: combined GWTC catalog (219 events) in data_core with PROVENANCE.local.json.
- [x] Reggiani (arXiv:2411.18881) integration:
  - [x] extract the exact definitions of `Z(S)` and `ZD(S)` as used in the paper,
  - [x] align repo terminology to the paper's.
  - Done: reggiani.rs (84 standard ZDs), annihilator.rs (SVD nullspace), is_reggiani_zd().

## C. Dimensional Continuation (-4D -> 32D)

- [x] Implement analytic-continuation formulas (Rust: `crates/cosmology_core/src/dimensional_geometry.rs`).
- [x] Add tests for integer dimensions and identities (Rust unit tests in same file).
- [x] Generate "grand" plots and CSV outputs (historical: `src/vis_dimensional_geometry.py`).
- [ ] Add pole-aware plotting (optional: residue plots / annotated singularities).

## D. Materials Science (Real Data)

- [x] Add JARVIS-DFT subset fetcher + provenance (`src/fetch_materials_jarvis_subset.py`).
- [x] Add 118D composition -> {4,8,16,32}D PCA experiments (`src/materials_embedding_experiments.py`).
- [x] Add AFLOW dataset backend (full database via AFLUX REST API) for cross-validation.
- [x] Add Magpie-style element-property featurization (10 properties x 5 stats + 4 global = 54-dim).
- [x] Add OLS linear regression baselines (formation energy + band gap) with 80/20 train/test split.

## E. Algebra Engine (Sedenions and Beyond)

- [x] Add tests for associativity and norm composition behavior (Rust: `crates/algebra_core/src/cayley_dickson.rs`).
- [x] Replace random search with explicit constructions: Reggiani standard ZDs (84 count), annihilator SVD, M3 classification (Rust: `annihilator.rs`, `reggiani.rs`, `m3.rs`).
- [x] Motif census at dim=16 and dim=32 with graph component analysis (Rust: `boxkites.rs`).
- [ ] Add a fast basis-element multiplication table generator (for 16D and 32D) with cache + checksum.
- [x] Rust native performance replaces Numba: 3.5-5.3x speedup over Python (no C++ needed).

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
- [x] Add `fetch-datasets --pillar` grouping flags (7 pillars: candle, gravitational, electromagnetic, survey, cmb, solar, geophysical).
- [x] Implement parser-level schema checks for newly added datasets (Swarm, EHT, GFC, SPK, Landsat STAC, Union3).
- [x] Add deterministic row-count and column-integrity tests for Union3 chains.
- [x] Add deterministic fixed-width format sanity checks for Hipparcos (line width, pipe count, HIP field, marker).
- [x] Add archive member and expected-filename checks for EHT bundles.
- [x] Add SPK kernel magic-number validation for DE440/DE441.
- [x] Add GFC harmonic-degree sanity checks for GRACE-FO and EGM2008.
- [x] Add Swarm CSV header and timestamp monotonicity tests.
- [x] Add SORCE/TSIS overlap comparison function with synthetic tests.
- [x] Add Landsat STAC asset-schema validation and cloud-cover extraction.
- [x] Add a `dataset_coverage` report mapping claims to datasets (`docs/DATASET_COVERAGE.md`, `data_core::provider_pillar`, `data_core::claims_for_provider`).
- [x] Link dataset pillars to claim IDs (7 pillars, 12/30 datasets claim-backed, 3 coverage tests).
- [x] Add a verifier ensuring every dataset in manifest has a provider in CLI.
- [x] Add a verifier ensuring cached dataset hashes match provenance JSON (`data_core::provenance`, 7 tests).
- [x] Add a verifier ensuring docs source-index links resolve to files (`data_core::doc_links`, 84 Rust paths at 100%, 6 tests).
- [x] Add benchmark scripts for parser throughput (rows/s) on large catalogs (`data_core::benchmarks::benchmark_parser_throughput`, 6 parsers, 3 tests).
- [x] Add benchmark scripts for ephemeris interpolation accuracy vs Horizons snapshots (`data_core::benchmarks::benchmark_ephemeris_accuracy`, 4 bodies, 1 test).
- [x] Add benchmark scripts for gravity-harmonic truncation error curves (`data_core::benchmarks::benchmark_gravity_truncation`, Kaula-rule synthetic GFC, 1 convergence test).
- [x] Add benchmark scripts for magnetic-field sample coverage by mission day (`data_core::benchmarks::benchmark_magnetic_coverage`, ISO 8601 gap analysis, 3 tests).
- [x] Add benchmark scripts for irradiance time-series gap detection (`data_core::benchmarks::detect_irradiance_gaps`, TSIS + SORCE, 5 tests).
- [x] Add benchmark scripts for Landsat scene metadata filtering throughput (`data_core::benchmarks::benchmark_landsat_filtering`, STAC schema + cloud filter, 2 tests).
- [x] Integrate Rust verifiers into `make rust-smoke` target (clippy + tests, includes provenance + doc-link verifiers).
- [x] Add date-stamped status snapshot in `docs/CLAIMS_TASKS.md` (2026-02-07, 15 metrics, key completions).
- [x] Release gate passed: `cargo clippy --workspace -j$(nproc) -- -D warnings` (0 warnings), `cargo test --workspace -j$(nproc)` (1464 tests, 0 failures).
