# Ticket: Claims audit batch C-401..C-427

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/claim_tickets.toml -->

Owner: agent
Created: 2026-02-02
Status: IN PROGRESS

## Goal

Make the C-401..C-427 segment of `docs/CLAIMS_EVIDENCE_MATRIX.md` mechanically tractable for claim-by-claim auditing: - each open claim has (1) a clear scope boundary, (2) primary sources indexed and cached when possible, and (3) an offline check hook (unit test, verifier, or deterministic artifact pipeline) where feasible. This batch contains: - method-level claims (C-406..C-408) that must be enforced by verifiers, - emergence-layer claims (C-403..C-405) that should remain programmatic unless backed by sources, - and a large "materials/simulation" cluster (C-409..C-427) that must be kept explicit about toy-vs-validated.

## Scope

- Ticket ID: `TICKET-C401-C427`
- Kind: `CLAIMS_AUDIT_BATCH`
- Status token: `IN_PROGRESS`
- Claim range: C-401..C-427
- Claims referenced (22): C-401, C-403, C-404, C-405, C-406, C-407, C-408, C-409, C-410, C-411, C-412, C-417, C-418, C-419, C-420, C-421, C-422, C-423, C-424, C-425, C-426, C-427

## Deliverables

- `docs/external_sources/*.md`
- `src/verification/`

## Acceptance checks

- `PYTHONWARNINGS=error make check-parallel`
- `PYTHONWARNINGS=error make metadata-hygiene`
- `PYTHONWARNINGS=error make smoke`

## Progress snapshot

- Completed checkboxes: 5
- Open checkboxes: 0
- Backlog reports:
  - `reports/claims_batch_backlog_C401_C427.md`
  - `reports/convos_pdf_inventory.md`
  - `reports/tscp_trial_factor_ledger.md`
