# Risks and Gaps (Open Items)

Updated 2026-02-07 (Sprint 8 complete).

## Remaining open items

- **nalgebra version pin**: Workspace pinned to nalgebra 0.33; num-dual 0.13.2
  requires nalgebra 0.34.  Blocks generic autodiff Christoffel computation.
  Workaround: closed-form Christoffels for known metrics (Schwarzschild, Kerr).
  Resolution requires ode_solvers to update its nalgebra dependency.

- **PEPS scaling validation**: `peps_boundary_entropy()` is implemented (Sprint 8)
  but only validated on small grids (rows*cols <= 16).  Boundary MPS contraction
  for large grids needs numerical stability testing on 2D frustrated systems.

- **Primary-source citation gap**: 10 claims were cited in Sprint 8, but ~20
  claims in the evidence matrix still lack WHERE STATED or bibliography refs.

- **GPU test coverage**: 4 CUDA tests require RTX GPU; CI environments without
  NVIDIA hardware skip them silently via `#[ignore]`.  No CI GPU runner configured.

## Documentation health

- **Test count**: 1693 Rust tests (unit + integration + doc), synchronized across
  all 7 tracking documents (ROADMAP, TODO, ULTRA_ROADMAP, NEXT_ACTIONS,
  CLAIMS_TASKS, AGENTS, this file).
- **CLAIMS_TASKS.md snapshot**: updated Sprint 8 with correct per-crate counts.

## Resolved in Sprint 8

- ~~PEPS boundary entropy stub~~: DONE (exact + boundary MPS, 3 new tests).
- ~~Casimir nonadditivity stub~~: DONE (Bimonte 2012 derivative expansion, 3 new tests).
- ~~Hartigan dip test~~: DONE (GCM/LCM algorithm, 5 new tests in stats_core::dip).
- ~~Mass clumping binary~~: DONE (mass-clumping binary, N=176, all 3 columns MULTIMODAL).
- ~~Experiments portfolio~~: DONE (`docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md`, 10 entries).
- ~~Math conventions~~: DONE (`docs/MATH_CONVENTIONS.md`, 9 conventions).
- ~~Test count drift~~: DONE (harmonized from stale 1539 -> accurate 1645).
- ~~CdMultTable generator~~: DONE (O(1) basis lookup with SHA-256 checksum, 6 tests).

## Previously resolved

- ~~Claims-to-evidence conversion~~: DONE (459 claims, 118 backfill verified).
- ~~Second materials dataset~~: DONE (AFLOW provider added alongside MP).
- ~~GWTC-3 reproducible fetch~~: DONE (deterministic parser in data_core).
- ~~Section H dataset pillars~~: DONE (21/21 complete).
- ~~make lint scope~~: N/A (Python is viz-only; Rust clippy covers all code).
- ~~C++ accelerator~~: N/A (cudarc 0.19.1 CUDA engine replaces C++ roadmap).
- ~~Qiskit containerization~~: N/A (quantum_core is pure Rust; no Qiskit dependency).
