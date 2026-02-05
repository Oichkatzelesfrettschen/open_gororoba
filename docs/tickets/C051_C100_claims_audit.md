# Ticket: Claims audit batch C-051..C-100

Owner: agent
Created: 2026-02-02
Status: IN PROGRESS

## Progress

- 2026-02-03: Started hardening C-053 and C-068 (sources + deterministic artifacts + offline tests).
  - C-053: `src/scripts/analysis/unified_spacetime_synthesis.py` currently reports n_eff=2.2361, but the matrix note states ~6.8 (to be reconciled).
  - C-068: matrix notes indicate heavy spectral degeneracy (only 6 distinct eigenvalues); promote to a deterministic reproduction artifact + unit test.
- 2026-02-03: C-053 closed as a mechanically-checkable toy mapping.
  - Added dedicated sources index: `docs/external_sources/C053_PATHION_METAMATERIAL_MAPPING_SOURCES.md` (and cached arXiv PDFs under `data/external/papers/`).
  - Added deterministic artifacts + tests: `src/scripts/analysis/c053_pathion_metamaterial_mapping.py`, `data/csv/c053_pathion_tmm_summary.csv`, `tests/test_c053_pathion_metamaterial_mapping.py`.
  - Updated claim wording/status in `docs/CLAIMS_EVIDENCE_MATRIX.md` to reflect diagonal degeneracy (std=0) and n_eff ~= sqrt(5).
- 2026-02-03: C-068 refuted for the standard 84 diagonal ZD set (degenerate spectrum).
  - Added deterministic spectrum artifact + test: `src/scripts/analysis/c068_zd_interaction_spectrum_degeneracy.py`, `data/csv/c068_zd_interaction_eigen_summary.csv`, `tests/test_c068_zd_interaction_spectrum_degeneracy.py`.
- 2026-02-03: C-070 closed as Not supported under comparison discipline.
  - Added deterministic audit + artifacts + test: `src/scripts/analysis/c070_nanograv_shape_match_audit.py`, `data/csv/c070_nanograv_shape_match_summary.csv`, `tests/test_c070_nanograv_shape_match_audit.py`.
  - Updated decision rule in `docs/external_sources/C070_NANOGRAV_SPECTRUM_MATCH_SOURCES.md`.
- 2026-02-03: C-075 verified as a narrow spectral-diversity metric under the legacy v3 exp3 construction.
  - Added deterministic reproduction + artifacts + test: `src/scripts/analysis/c075_pathion_zd_interaction_spectrum.py`, `data/csv/c075_pathion_interaction_summary.csv`, `tests/test_c075_pathion_zd_interaction_spectrum.py`.
- 2026-02-03: C-074 promoted to Partially verified (fit sanity + sources index already present).
  - Tracker cleanup: updated `docs/CLAIMS_TASKS.md` row and refreshed "Where stated" pointers.
- 2026-02-03: C-077 closed as Not supported (associator mixing matrix is near-democratic, far from PMNS).
  - Added deterministic audit extractor + artifacts + test: `src/scripts/analysis/c077_associator_mixing_pmns_audit.py`, `data/csv/c077_associator_mixing_summary.csv`, `tests/test_c077_associator_mixing_pmns_audit.py`.
- 2026-02-03: C-078 closed as Not supported under the legacy diagonal-form ZD ensemble (64D does not improve over 32D).
  - Added deterministic audit extractor + artifacts + test: `src/scripts/analysis/c078_higher_dim_zd_coverage_audit.py`, `data/csv/c078_higher_dim_zd_coverage_summary.csv`, `tests/test_c078_higher_dim_zd_coverage_audit.py`.
- 2026-02-03: C-087 promoted to Partially verified (derivation + stored Monte Carlo cross-term decay).
  - Added audit extractor + artifacts + test: `src/scripts/analysis/c087_associator_independence_audit.py`, `data/csv/c087_associator_independence_summary.csv`, `tests/test_c087_associator_independence_audit.py`.
- 2026-02-03: C-097 verified as a pure graph invariant statement (84 diagonal ZDs).
  - Added deterministic graph audit + artifacts + test: `src/scripts/analysis/c097_zd_interaction_graph_audit.py`, `data/csv/c097_zd_interaction_graph_summary.csv`, `tests/test_c097_zd_interaction_graph_audit.py`.
- 2026-02-03: C-082 verified as a mechanically-checkable extraction of the v4 dim=1024 saturation extension.
  - Added deterministic extractor + artifacts + test: `src/scripts/analysis/c082_associator_saturation_extended_audit.py`, `data/csv/c082_associator_saturation_summary.csv`, `tests/test_c082_associator_saturation_extended_audit.py`.
- 2026-02-03: C-092 closed as Not supported under the cached v6 orbit-classification diagnostic.
  - Added deterministic extractor + artifacts + test: `src/scripts/analysis/c092_so7_orbit_structure_audit.py`, `data/csv/c092_so7_orbit_structure_summary.csv`, `tests/test_c092_so7_orbit_structure_audit.py`.
- 2026-02-03: C-094 verified as a guardrail: mixing non-2^n truncations breaks the saturation fit (as expected).
  - Added deterministic extractor + artifacts + test: `src/scripts/analysis/c094_non_power_two_dims_fit_audit.py`, `data/csv/c094_non_power_two_dims_summary.csv`, `tests/test_c094_non_power_two_dims_fit_audit.py`.
- 2026-02-03: C-096 verified as a basis-tensor transition at dim=16 (octonion vs sedenion symmetry diagnostics).
  - Added deterministic extractor + artifacts + test: `src/scripts/analysis/c096_associator_tensor_transitions_audit.py`, `data/csv/c096_associator_tensor_summary.csv`, `tests/test_c096_associator_tensor_transitions_audit.py`.
- 2026-02-03: C-099 verified as a mechanically-checkable geometry summary for the cached v8 non-diagonal ZD search.
  - Added deterministic extractor + artifact + test: `src/scripts/analysis/c099_nondiag_zd_geometry_audit.py`, `data/csv/c099_nondiag_zd_geometry_summary.csv`, `tests/test_c099_nondiag_zd_geometry_audit.py`.

## Goal

Make the C-051..C-100 segment of `docs/CLAIMS_EVIDENCE_MATRIX.md` mechanically tractable for
claim-by-claim auditing:
- each open claim has (1) a clear scope boundary, (2) primary sources indexed and cached when possible,
  and (3) an offline check hook (unit test, verifier, or deterministic artifact pipeline) where feasible.

This ticket is primarily about de-risking legacy/"suggestive" algebra-to-physics mappings by:
- splitting observational vs interpretive statements,
- keeping any physics links explicitly speculative unless supported by sources + robust statistics,
- converting legacy experiment notes into reproducible scripts + tests where possible.

## Planning snapshot (auto-generated)

- Backlog report: `reports/claims_batch_backlog_C051_C100.md`

Open claims in-range at time of writing (see report for full details):
Current open claims in-range (2026-02-03; see report for full details):
- C-053, C-074, C-087, C-090.

## Deliverables

- Range backlog report kept current (regenerate as needed):
  - `PYTHONWARNINGS=error venv/bin/python3 src/scripts/analysis/claims_batch_backlog.py --id-from 51 --id-to 100 --out reports/claims_batch_backlog_C051_C100.md`
- For each open claim in-range:
  - tighten claim wording (separate "math/experiment" vs "physics interpretation" subclaims)
  - ensure source index coverage (`docs/external_sources/*.md`, avoid relying on the generic inbox index)
  - cache primary sources under `data/external/` when access is available; otherwise cache traces
  - add at least one offline check hook, or explicitly mark as "Speculative" with symmetric falsification criteria

## Acceptance checks

- `PYTHONWARNINGS=error make smoke`
- `PYTHONWARNINGS=error make check-parallel`
- `PYTHONWARNINGS=error make metadata-hygiene`
