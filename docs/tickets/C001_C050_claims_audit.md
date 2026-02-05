# Ticket: Claims audit batch C-001..C-050

Owner: agent
Created: 2026-02-02
Status: IN PROGRESS

## Goal

Make the C-001..C-050 segment of `docs/CLAIMS_EVIDENCE_MATRIX.md` mechanically tractable for
claim-by-claim auditing:
- each open claim has (1) a clear scope boundary, (2) primary sources indexed and cached when possible,
  and (3) an offline check hook (unit test, verifier, or deterministic artifact pipeline) where feasible.

This ticket does not attempt to "prove the theory". It upgrades traceability and falsifiability.

## Planning snapshot (auto-generated)

- Backlog report: `reports/claims_batch_backlog_C001_C050.md`

Open claims in-range at time of writing (see report for full details):
- C-005, C-007, C-008, C-010, C-011, C-022, C-023, C-026, C-027, C-030, C-031, C-032, C-033, C-034,
  C-037, C-039, C-040, C-041, C-043, C-047, C-048, C-050.

## Progress

- [x] Confirmed strong offline check hooks already exist for:
  - C-007 (GWTC-3 multimodality pipeline + prereg docs)
  - C-008 (Parisi-Sourlas toy operator + tests)
  - C-010 (absorber mapping hypothesis is grounded by baseline absorber models; mapping currently negative)
  - C-011 (gravastar hypothesis explicitly obstructed; tests and artifacts exist)
  - C-022/C-023 (toy models; scripts + tests)
  - C-033 (SU(5) structural tests)
  - C-039 (spectral dimension: tests + deterministic artifacts; qualitative only)
  - C-040 (primordial tilt: tests + deterministic sweep artifacts; refuted under stated mapping)
  - C-043 (compact objects: integration pipeline + offline artifacts target)
- [x] C-026: added an offline observational baseline artifact (GWTC-3 lower-mass-gap occupancy metrics)
  plus a mechanism plan with explicit falsification boundaries (`docs/C026_MASS_GAP_MECHANISM.md`).
- [x] C-027: closed placeholder by adding an explicit decision rule + deterministic artifacts + tests
  (`docs/C027_DEFF_HORIZON_TEST.md`, `data/csv/deff_horizon_mass_scaling*.csv`,
  `tests/test_c027_deff_horizon_mass_scaling.py`).
- [x] C-032: added minimal, offline reproduction artifacts for Tang (2025) preprint claim:
  Table 2 extraction + associator subalgebra stats (`src/scripts/analysis/c032_tang_2025_min_reproduction.py`).
  - Claim-specific sources index: `docs/external_sources/C032_TANG_2025_SEDENIONIC_QED_SOURCES.md`.
- [x] C-034: added minimal, offline structural reproduction for the "two eight-potentials" prerequisite
  (Cayley-Dickson doubling identity check), and documented equation-level reproduction as BLOCKED
  without a legal full-text source (`docs/C034_CHANYAL_2014_REPRODUCTION.md`).
  - Claim-specific sources index: `docs/external_sources/C034_CHANYAL_2014_GRAVI_ELECTROMAGNETISM_SOURCES.md`.
  - Cached open-access background anchors for partial context while the primary PDF is blocked:
    `data/external/papers/arxiv_1502.05293_chanyal_2015_octonionic_gravi_electromagnetism_dark_matter.pdf`,
    `data/external/papers/scirp_2014_mironov_sedeonic_equations_gravitoelectromagnetism.pdf`.
- [x] C-033: added explicit closure criteria for promoting the "sedenion -> SU(5)" mapping beyond
  structural verification (`docs/C033_SU5_MAPPING_CLOSURE.md`).
- [x] C-034: recorded additional blocked-access trace for an alternate host (Academia.edu) that
  requires login; no bypass attempted (`data/external/traces/chanyal_2014_academia_requires_login.txt`).
- [x] C-030: added a concrete decision rule + offline artifact + test showing why sedenion-valued
  actions need explicit parenthesization/representation choices (`docs/C030_SEDENION_LAGRANGIAN_BYPASS.md`,
  `data/csv/c030_sedenion_lagrangian_bypass_checks.csv`).
- [x] C-031: added an offline artifact + test that shows norm multiplicativity holds for
  dims 1/2/4/8 and fails at dim=16, plus a concrete dim=16 zero-divisor example
  (`data/csv/c031_hurwitz_norm_composition_checks.csv`).
- [x] C-039: added deterministic artifact generators and wired them into `make artifacts-exceptional-cosmology`
  (`src/scripts/analysis/c039_spectral_dimension_bigraph_sweep.py`, `data/csv/c039_spectral_dimension_bigraph_*.csv`).
- [x] C-040: added deterministic sweep artifacts and demoted the claim to Refuted under the stated mapping
  (`src/scripts/analysis/c040_primordial_tilt_deff_sweep.py`, `data/csv/c040_primordial_tilt_*.csv`).
- [x] C-041: demoted coincidence claim to Not supported with an explicit mechanism-gap note
  (`docs/C041_F4_STRING_DIMENSION_COINCIDENCE_AUDIT.md`).
- [x] C-043: added an offline artifacts target that builds the unified compact-object catalog
  (`make artifacts-compact-objects`, `data/csv/compact_objects_catalog.csv`).
- [x] C-047: corrected legacy E9/E10/E11 framing with cached open-access anchors + an offline
  Cartan-signature sanity check (`docs/C047_E_SERIES_KAC_MOODY_AUDIT.md`,
  `docs/external_sources/C047_E_SERIES_KAC_MOODY_SOURCES.md`,
  `tests/test_c047_e_series_cartan_signature.py`).
- [x] C-048: anchored the "motivic tower" analogy with a cached open-access reference and
  updated pointers (`docs/external_sources/C048_MOTIVIC_TOWER_SOURCES.md`,
  `data/external/papers/arxiv_0901.1632_dugger_isaksen_2009_motivic_adams_spectral_sequence.pdf`).
- [x] C-050: made the optimization-structure isomorphism explicit as a scope-safe toy LP
  equivalence, with a deterministic artifact + test (`docs/C050_SPACEPLATE_FLOW_ISOMORPHISM.md`,
  `src/scripts/analysis/c050_spaceplate_flow_isomorphism_toy.py`,
  `data/csv/c050_spaceplate_flow_isomorphism_toy.csv`,
  `tests/test_c050_spaceplate_flow_isomorphism_toy.py`).
- [ ] Remaining "programmatic" open claims needing additional work beyond current hooks:
  - C-026 (mass gap mapping still needs algebra->mass mechanism; baseline + falsification plan added)
  - C-033 (Tang & Tang 2023 does not specify a unique sedenion->SU(5) coefficient mapping)
  - C-032/C-034 (literature claims remain partial until a legal full-text source and/or full mapping is available)

## Deliverables

- Range backlog report kept current (regenerate as needed):
  - `PYTHONWARNINGS=error venv/bin/python3 src/scripts/analysis/claims_batch_backlog.py --id-from 1 --id-to 50 --out reports/claims_batch_backlog_C001_C050.md`
- For each open claim in-range:
  - tighten claim wording (separate "observational" vs "interpretive" subclaims when needed)
  - ensure source index coverage (docs/external_sources/*.md)
  - cache primary sources under data/external/ when access is available; otherwise cache traces
  - add at least one offline check hook, or explicitly mark as "Speculative" with symmetric falsification criteria

## Acceptance checks

- `PYTHONWARNINGS=error make smoke`
- `PYTHONWARNINGS=error make check-parallel`
- `PYTHONWARNINGS=error make metadata-hygiene`
