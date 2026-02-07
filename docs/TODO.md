# TODO (Jan 2026): executable, test-driven tracker

Primary trackers:
- `docs/ULTRA_ROADMAP.md` (test-driven, claim->evidence oriented)
- `docs/CLAIMS_TASKS.md` (claim-specific implementation backlog)
- `docs/NEXT_ACTIONS.md` (short priority queue)
- `docs/CONVOS_CONCEPTS_STATUS_INDEX.md` (Convos -> Concepts -> Status -> implement-next)
- `docs/convos/implement_next_from_1_read_nonuser_lines_cont.md` (latest convos audit backlog)

Note:
- `convos/`-derived trackers are non-authoritative input streams.
- Source-first execution authority is `docs/ULTRA_ROADMAP.md` + `docs/CLAIMS_TASKS.md`.

## Current sprint (blockers first)
- [x] Audit `convos/1_read_nonuser_lines_cont.md` in chunks (C1-0001..C1-0041).
  - See: `docs/convos/audit_1_read_nonuser_lines_cont.md` and `docs/convos/keywords_1_read_nonuser_lines_cont.md`.
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
  - [ ] m3 artifact export + stronger invariants tests (CX-001).
  - [ ] XOR-balanced search extension + tests (CX-003).
  - [ ] Motif census: extend to 64D/128D exact and 256D sampled + plot (CX-002).
  - [ ] Visualization hygiene and artifact saving policy (CX-019).
  - [ ] Optional "atlas/plates" reproducible artifact pipeline (CX-020).

## Quality gates
- [ ] `make ascii-check`
- [ ] `PYTHONWARNINGS=error make check`

## Completed (keep for provenance)
- [x] Package `gemini_physics` and add tests (`pyproject.toml`, `src/gemini_physics/`, `tests/`).
- [x] Coq stub builds via `make coq` (see `curated/01_theory_frameworks/README_COQ.md`).
- [x] Materials ingestion and embedding benchmarks:
  - `src/fetch_materials_jarvis_subset.py`
  - `src/materials_embedding_experiments.py`
