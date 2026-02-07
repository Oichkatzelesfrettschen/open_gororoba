# TODO (updated 2026-02-06): executable, test-driven tracker

Primary trackers:
- `docs/ROADMAP.md` (consolidated roadmap: architecture, crates, migration history, GR port plan)
- `docs/ULTRA_ROADMAP.md` (test-driven, claim->evidence oriented)
- `docs/CLAIMS_TASKS.md` (claim-specific implementation backlog)
- `docs/NEXT_ACTIONS.md` (short priority queue)

Note:
- Python-to-Rust migration is COMPLETE (all 15 modules ported, gororoba_kernels removed).
- 1464 Rust tests (unit + doc) pass, 0 clippy warnings.
- Historical planning docs (`RUST_MIGRATION_PLAN.md`, `RUST_REFACTOR_PLAN.md`, `docs/RESEARCH_ROADMAP.md`) deleted; content absorbed into `docs/ROADMAP.md`.

## Current sprint (blockers first)
- [x] Source-first research passes (no deletions; only append/clarify sources):
  - [x] Wheels (division-by-zero) vs wheel graphs vs wheeled operads (CX-017). Already in wheels.rs + doc.
  - [x] de Marrais primary sources + terminology alignment (CX-002). Audit complete: dim=16 terms fully aligned, higher-dim generic naming is correct.
  - [x] Reggiani alignment (Z(S), ZD(S)) and repo terminology alignment. Audit complete: is_reggiani_zd() correct, 84 standard ZDs, nullity-4 verified, sign variants orthogonal (C-005 empirical correction documented).
  - [x] Fractional Laplacian sources: Riesz vs spectral vs extension (CX-004). Audit complete: periodic=Riesz, Dirichlet=spectral correctly labeled. C-S doc clarified. All 5 sources cited.
  - [x] p-adic operator sources (Vladimirov/Kozyrev) (CX-006). Audit complete: foundations correct (14 tests), Vladimirov/Kozyrev correctly deferred, sources cited in BIBLIOGRAPHY.md.
  - [x] Exceptional/Jordan/Albert references to correct overclaims (CX-007). Audit complete: no overclaims found. nilpotent_orbits.rs is generic (no exceptional claims), magic square verified, H3(O) correctly deferred.
- [x] Execute missing dataset pillars roadmap from `docs/ULTRA_ROADMAP.md` Section H (21/21 done, 1464 tests).
- [ ] Implementation (keep every warning as an error):
  - [x] Wheels axioms checker + unit tests (CX-017). Already in wheels.rs (WheelQ, 8 axioms, 10 tests).
  - [x] XOR-balanced search extension + tests (CX-003). 10 new tests.
  - [x] Motif census: exact to 256D with scaling laws (CX-002). 16 new tests.
  - [ ] Visualization hygiene and artifact saving policy (CX-019).

## Quality gates
- [x] `cargo clippy --workspace -j$(nproc) -- -D warnings`
- [x] `cargo test --workspace -j$(nproc)` (1464 tests, 0 failures)
- [ ] `make ascii-check`

## Completed (keep for provenance)
- [x] Package `gemini_physics` and add tests (`pyproject.toml`, `src/gemini_physics/`, `tests/`).
- [x] Coq stub builds via `make coq` (see `curated/01_theory_frameworks/README_COQ.md`).
- [x] Materials ingestion and embedding benchmarks.
- [x] Python-to-Rust migration: all 15 modules ported.
- [x] Consolidate PyO3 bridges: gororoba_kernels removed, gororoba_py is sole bridge.
- [x] Real cosmology fitting: Pantheon+ + DESI DR1 BAO (Omega_m=0.3284, H0=71.65).
- [x] Ultrametric analysis directions C-436 through C-440 completed.
- [x] GPU ultrametric exploration: 82/472 sig at FDR<0.05, 9 catalogs (I-011).
- [x] Blackhole C++ port: COMPLETE (18 gr_core modules, 394 tests).
- [x] CUDA compute engine: cudarc 0.19.1, 10M triples/test on RTX 4070 Ti.
- [x] Deleted stale planning docs: RUST_MIGRATION_PLAN.md, RUST_REFACTOR_PLAN.md, RESEARCH_ROADMAP.md.
- [x] Motif census extended to dim=64/128/256 exact. 5 scaling laws verified, 16 new tests.
- [x] XOR-balanced search extension (CX-003): mixed 2-blade/4-blade graph, necessity statistics. 10 new tests.
- [x] Source-first research passes: 6/6 complete (CX-017, CX-002, Reggiani, CX-004, CX-006, CX-007). No overclaims found. C-S doc clarified in spectral_core.
- [x] Dataset pillar validation: +33 new tests in data_core (Union3, Swarm, EHT, GFC, SPK, Landsat STAC, SORCE/TSIS overlap, provider verifier). 9/21 Section H items done.
