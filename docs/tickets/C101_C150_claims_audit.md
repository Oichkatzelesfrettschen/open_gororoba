# Ticket: Claims audit batch C-101..C-150

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/claim_tickets.toml -->

Owner: agent
Created: 2026-02-02
Status: DONE

## Goal

Make the C-101..C-150 segment of `docs/CLAIMS_EVIDENCE_MATRIX.md` mechanically tractable for claim-by-claim auditing: - each open claim has (1) a clear scope boundary, (2) primary sources indexed and cached when possible, and (3) an offline check hook (unit test, verifier, or deterministic artifact pipeline) where feasible. This batch is dominated by legacy high-dimension Cayley-Dickson experiment summaries. The critical work is to separate: - the computable/mathematical invariants (often already reproducible), from - physics-facing interpretations (which must remain explicitly speculative unless source-backed).

## Scope

- Ticket ID: `TICKET-C101-C150`
- Kind: `CLAIMS_AUDIT_BATCH`
- Status token: `DONE`
- Claim range: C-101..C-150
- Claims referenced (13): C-101, C-102, C-103, C-108, C-109, C-120, C-123, C-128, C-129, C-130, C-132, C-135, C-150

## Deliverables

- `docs/CLAIMS_EVIDENCE_MATRIX.md`
- `reports/claims_batch_backlog_C101_C150.md`
- `src/scripts/analysis/c102_alt_ratio_convergence_audit.py`
- `data/csv/c102_alt_ratio_summary.csv`
- `tests/test_c102_alt_ratio_convergence_audit.py`
- `src/scripts/analysis/c103_zd_topology_percolation_audit.py`
- `data/csv/c103_zd_topology_summary.csv`
- `tests/test_c103_zd_topology_percolation_audit.py`
- `src/scripts/analysis/c108_alt_ratio_convergence_audit.py`
- `data/csv/c108_alt_ratio_summary.csv`
- `tests/test_c108_alt_ratio_convergence_audit.py`
- `src/scripts/analysis/c109_zd_construction_audit.py`
- `data/csv/c109_zd_construction_summary.csv`
- `tests/test_c109_zd_construction_audit.py`
- `src/scripts/analysis/c120_zd_kernel_scaling_audit.py`
- `data/csv/c120_zd_kernel_scaling_summary.csv`
- `tests/test_c120_zd_kernel_scaling_audit.py`
- `src/scripts/analysis/c123_associator_lie_bracket_audit.py`
- `data/csv/c123_assoc_lie_bracket_summary.csv`
- `tests/test_c123_associator_lie_bracket_audit.py`
- `src/scripts/analysis/c128_conjugate_inverse_audit.py`
- `data/csv/c128_conjugate_inverse_summary.csv`
- `tests/test_c128_conjugate_inverse_audit.py`
- `src/scripts/analysis/c129_associator_distribution_concentration_audit.py`
- `data/csv/c129_assoc_norm_dist_summary.csv`
- `tests/test_c129_associator_distribution_concentration_audit.py`
- `src/scripts/analysis/c130_associator_norm_sqrt2_audit.py`
- `data/csv/c130_associator_norm_sqrt2_summary.csv`
- `tests/test_c130_associator_norm_sqrt2_audit.py`
- `src/scripts/analysis/c132_commutator_norm_convergence_audit.py`
- `data/csv/c132_commutator_norm_summary.csv`
- `tests/test_c132_commutator_norm_convergence_audit.py`
- `src/scripts/analysis/c135_power_norm_scaling_audit.py`
- `data/csv/c135_power_norm_summary.csv`
- `tests/test_c135_power_norm_scaling_audit.py`

## Acceptance checks

- `PYTHONWARNINGS=error make check-parallel`
- `PYTHONWARNINGS=error make metadata-hygiene`
- `PYTHONWARNINGS=error make smoke`

## Progress snapshot

- Completed checkboxes: 11
- Open checkboxes: 0
- Backlog reports:
  - `reports/claims_batch_backlog_C101_C150.md`
