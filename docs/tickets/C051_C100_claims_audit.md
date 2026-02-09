# Ticket: Claims audit batch C-051..C-100

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/claim_tickets.toml -->

Owner: agent
Created: 2026-02-02
Status: IN PROGRESS

## Goal

Make the C-051..C-100 segment of `docs/CLAIMS_EVIDENCE_MATRIX.md` mechanically tractable for claim-by-claim auditing: - each open claim has (1) a clear scope boundary, (2) primary sources indexed and cached when possible, and (3) an offline check hook (unit test, verifier, or deterministic artifact pipeline) where feasible. This ticket is primarily about de-risking legacy/"suggestive" algebra-to-physics mappings by: - splitting observational vs interpretive statements, - keeping any physics links explicitly speculative unless supported by sources + robust statistics, - converting legacy experiment notes into reproducible scripts + tests where possible.

## Scope

- Ticket ID: `TICKET-C051-C100`
- Kind: `CLAIMS_AUDIT_BATCH`
- Status token: `IN_PROGRESS`
- Claim range: C-051..C-100
- Claims referenced (17): C-051, C-053, C-068, C-070, C-074, C-075, C-077, C-078, C-082, C-087, C-090, C-092, C-094, C-096, C-097, C-099, C-100

## Deliverables

- `docs/external_sources/*.md`
- `data/external/`

## Acceptance checks

- `PYTHONWARNINGS=error make check-parallel`
- `PYTHONWARNINGS=error make metadata-hygiene`
- `PYTHONWARNINGS=error make smoke`

## Progress snapshot

- Completed checkboxes: 0
- Open checkboxes: 0
- Backlog reports:
  - `reports/claims_batch_backlog_C051_C100.md`
