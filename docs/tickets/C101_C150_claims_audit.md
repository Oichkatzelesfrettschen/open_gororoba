# Ticket: Claims audit batch C-101..C-150

Owner: agent
Created: 2026-02-02
Status: DONE

## Goal

Make the C-101..C-150 segment of `docs/CLAIMS_EVIDENCE_MATRIX.md` mechanically tractable for
claim-by-claim auditing:
- each open claim has (1) a clear scope boundary, (2) primary sources indexed and cached when possible,
  and (3) an offline check hook (unit test, verifier, or deterministic artifact pipeline) where feasible.

This batch is dominated by legacy high-dimension Cayley-Dickson experiment summaries. The critical work is
to separate:
- the computable/mathematical invariants (often already reproducible), from
- physics-facing interpretations (which must remain explicitly speculative unless source-backed).

## Planning snapshot (auto-generated)

- Backlog report: `reports/claims_batch_backlog_C101_C150.md`

Open claims in-range at time of writing (see report for full details):
- (none) (all claims in-range are now mechanically checkable and marked Verified).

## Progress

- [x] C-102: reproduced legacy v9 expZ alternativity-ratio convergence into deterministic CSV artifacts + unit test
  (`src/scripts/analysis/c102_alt_ratio_convergence_audit.py`,
  `data/csv/c102_alt_ratio_summary.csv`,
  `tests/test_c102_alt_ratio_convergence_audit.py`).
- [x] C-103: reproduced legacy v9 expAA percolation-style connectivity jump into deterministic CSV artifacts + unit test
  (`src/scripts/analysis/c103_zd_topology_percolation_audit.py`,
  `data/csv/c103_zd_topology_summary.csv`,
  `tests/test_c103_zd_topology_percolation_audit.py`).
- [x] C-108: extracted cached v10 alternativity-ratio fit into deterministic CSV artifacts + unit test
  (`src/scripts/analysis/c108_alt_ratio_convergence_audit.py`,
  `data/csv/c108_alt_ratio_summary.csv`,
  `tests/test_c108_alt_ratio_convergence_audit.py`).
- [x] C-109: extracted cached v10 probing + reproduced lifted diagonal ZD kernel doubling into deterministic CSV artifacts + unit test
  (`src/scripts/analysis/c109_zd_construction_audit.py`,
  `data/csv/c109_zd_construction_summary.csv`,
  `tests/test_c109_zd_construction_audit.py`).
- [x] C-120: extracted cached v12 diagonal/lifted ZD kernel scaling into deterministic CSV artifacts + unit test
  (`src/scripts/analysis/c120_zd_kernel_scaling_audit.py`,
  `data/csv/c120_zd_kernel_scaling_summary.csv`,
  `tests/test_c120_zd_kernel_scaling_audit.py`).
- [x] C-123: extracted cached v12 associator Lie bracket metrics into deterministic CSV artifacts + unit test
  (`src/scripts/analysis/c123_associator_lie_bracket_audit.py`,
  `data/csv/c123_assoc_lie_bracket_summary.csv`,
  `tests/test_c123_associator_lie_bracket_audit.py`).
- [x] C-128: extracted cached v13 conjugate-inverse errors into deterministic CSV artifacts + unit test
  (`src/scripts/analysis/c128_conjugate_inverse_audit.py`,
  `data/csv/c128_conjugate_inverse_summary.csv`,
  `tests/test_c128_conjugate_inverse_audit.py`).
- [x] C-129: extracted cached v13 associator-norm distribution concentration into deterministic CSV artifacts + unit test
  (`src/scripts/analysis/c129_associator_distribution_concentration_audit.py`,
  `data/csv/c129_assoc_norm_dist_summary.csv`,
  `tests/test_c129_associator_distribution_concentration_audit.py`).
- [x] C-130: extracted cached v14 associator norm sqrt(2) metrics into deterministic CSV artifacts + unit test
  (`src/scripts/analysis/c130_associator_norm_sqrt2_audit.py`,
  `data/csv/c130_associator_norm_sqrt2_summary.csv`,
  `tests/test_c130_associator_norm_sqrt2_audit.py`).
- [x] C-132: extracted cached v14 commutator-norm convergence into deterministic CSV artifacts + unit test
  (`src/scripts/analysis/c132_commutator_norm_convergence_audit.py`,
  `data/csv/c132_commutator_norm_summary.csv`,
  `tests/test_c132_commutator_norm_convergence_audit.py`).
- [x] C-135: extracted cached v14 power-norm scaling into deterministic CSV artifacts + unit test
  (`src/scripts/analysis/c135_power_norm_scaling_audit.py`,
  `data/csv/c135_power_norm_summary.csv`,
  `tests/test_c135_power_norm_scaling_audit.py`).

## Deliverables

- Range backlog report kept current (regenerate as needed):
  - `PYTHONWARNINGS=error venv/bin/python3 src/scripts/analysis/claims_batch_backlog.py --id-from 101 --id-to 150 --out reports/claims_batch_backlog_C101_C150.md`
- For each open claim in-range:
  - ensure a dedicated source index exists when the claim cites literature beyond in-repo computations
  - add or reference an offline check hook; if no hook is feasible, add symmetric falsification criteria

## Acceptance checks

- `PYTHONWARNINGS=error make smoke`
- `PYTHONWARNINGS=error make check-parallel`
- `PYTHONWARNINGS=error make metadata-hygiene`
