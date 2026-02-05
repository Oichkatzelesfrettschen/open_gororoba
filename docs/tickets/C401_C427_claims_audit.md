# Ticket: Claims audit batch C-401..C-427

Owner: agent
Created: 2026-02-02
Status: IN PROGRESS

## Goal

Make the C-401..C-427 segment of `docs/CLAIMS_EVIDENCE_MATRIX.md` mechanically tractable for
claim-by-claim auditing:
- each open claim has (1) a clear scope boundary, (2) primary sources indexed and cached when possible,
  and (3) an offline check hook (unit test, verifier, or deterministic artifact pipeline) where feasible.

This batch contains:
- method-level claims (C-406..C-408) that must be enforced by verifiers,
- emergence-layer claims (C-403..C-405) that should remain programmatic unless backed by sources,
- and a large "materials/simulation" cluster (C-409..C-427) that must be kept explicit about toy-vs-validated.

## Planning snapshot (auto-generated)

- Backlog report: `reports/claims_batch_backlog_C401_C427.md`

Open claims in-range at time of writing (see report for full details):
- C-401, C-403, C-406, C-407, C-408, C-409, C-410, C-411, C-412,
  C-417, C-418, C-419, C-420, C-421, C-422, C-423, C-424, C-425, C-426, C-427,
  C-404, C-405.

## Progress

- [x] C-403: moved "Where stated" pointer to claim-specific source index
  (`docs/external_sources/C403_SPECTRAL_TRIPLE_RECONSTRUCTION_SOURCES.md`).
- [x] Added convos PDF inventory report (`reports/convos_pdf_inventory.md`) and wired it into
  `make metadata-hygiene` for ongoing triage.
- [x] C-404: added claim-specific source index
  (`docs/external_sources/C404_HOLOGRAPHIC_MODULAR_LOCALITY_SOURCES.md`) and updated the matrix row.
  - Cached the entanglement-wedge reconstruction anchor: `data/external/papers/arxiv_1601.05416_dong_harlow_wall_2016_entanglement_wedge_reconstruction.pdf`.
  - Cached additional core anchors (AdS/CFT, RT/HRT, JLMS, bit threads, HaPPY) and recorded provenance.
- [x] C-405: added claim-specific source index
  (`docs/external_sources/C405_OPEN_SYSTEMS_RECORD_ALGEBRA_SOURCES.md`) and updated the matrix row.
  - Cached open-access anchors (GKSL intros, decoherence/Darwinism, operator-algebra QEC, DFS) and recorded provenance.
- [x] C-406..C-408: confirmed method-hygiene enforcement remains wired into `make smoke`
  (trial-factor ledger consistency and symmetric falsification boundary checks), and is
  supported by opt-in calibration in `make validation-smoke`.
  - Symmetric falsification boundaries: `src/verification/verify_preregistered_falsification_boundaries.py` (C-408).
  - Trial-factor ledger consistency: `src/verification/verify_tscp_prereg_trial_factors.py` + `reports/tscp_trial_factor_ledger.md` (C-407).
  - Embedding sweep hook: `tests/test_tscp_embedding_sweep.py` (C-406).
  - Null calibration (sanity): `make validation-smoke` runs `src/scripts/validation/validate_tscp_null_calibration.py`.
  - Method source index: `docs/external_sources/TSCP_METHOD_SOURCES.md`.

## Deliverables

- Range backlog report kept current (regenerate as needed):
  - `PYTHONWARNINGS=error venv/bin/python3 src/scripts/analysis/claims_batch_backlog.py --id-from 401 --id-to 427 --out reports/claims_batch_backlog_C401_C427.md`
- For each open claim in-range:
  - ensure source index coverage (`docs/external_sources/*.md`, prefer claim-specific indexes)
  - ensure an offline check hook exists (unit tests or `src/verification/`), or explicitly keep the claim Speculative/Theoretical
  - split "simulation result" claims from "real-world feasibility" claims

## Acceptance checks

- `PYTHONWARNINGS=error make smoke`
- `PYTHONWARNINGS=error make check-parallel`
- `PYTHONWARNINGS=error make metadata-hygiene`
