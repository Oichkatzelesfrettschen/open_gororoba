# TODO (updated 2026-02-06): executable, test-driven tracker

Primary trackers:
- `docs/ROADMAP.md` (consolidated roadmap: architecture, crates, migration history, GR port plan)
- `docs/ULTRA_ROADMAP.md` (test-driven, claim->evidence oriented)
- `docs/CLAIMS_TASKS.md` (claim-specific implementation backlog)
- `docs/NEXT_ACTIONS.md` (short priority queue)

Note:
- Python-to-Rust migration is COMPLETE (all 15 modules ported, gororoba_kernels removed).
- 1370 Rust unit tests + 7 doc-tests pass, 0 clippy warnings.
- Historical planning docs (`RUST_MIGRATION_PLAN.md`, `RUST_REFACTOR_PLAN.md`, `docs/RESEARCH_ROADMAP.md`) deleted; content absorbed into `docs/ROADMAP.md`.

## Current sprint (blockers first)
- [ ] Source-first research passes (no deletions; only append/clarify sources):
  - [ ] Wheels (division-by-zero) vs wheel graphs vs wheeled operads (CX-017).
  - [ ] de Marrais primary sources + terminology alignment (CX-002).
  - [ ] Reggiani alignment (Z(S), ZD(S)) and repo terminology alignment.
  - [ ] Fractional Laplacian sources: Riesz vs spectral vs extension (CX-004).
  - [ ] p-adic operator sources (Vladimirov/Kozyrev) (CX-006).
  - [ ] Exceptional/Jordan/Albert references to correct overclaims (CX-007).
- [ ] Execute missing dataset pillars roadmap from `docs/ULTRA_ROADMAP.md` Section H.
- [ ] Implementation (keep every warning as an error):
  - [ ] Wheels axioms checker + unit tests (CX-017).
  - [ ] XOR-balanced search extension + tests (CX-003).
  - [ ] Motif census: extend to 64D/128D exact and 256D sampled + plot (CX-002).
  - [ ] Visualization hygiene and artifact saving policy (CX-019).

## Quality gates
- [x] `cargo clippy --workspace -j$(nproc) -- -D warnings`
- [x] `cargo test --workspace -j$(nproc)` (1370 unit + 7 doc-tests)
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
