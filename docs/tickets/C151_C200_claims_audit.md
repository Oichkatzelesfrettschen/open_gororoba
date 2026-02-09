# Ticket: Claims audit batch C-151..C-200

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/claim_tickets.toml -->

Owner: agent
Created: 2026-02-02
Status: DONE (no open claims in range)

## Goal

Confirm the C-151..C-200 segment of `docs/CLAIMS_EVIDENCE_MATRIX.md` remains mechanically tractable for claim-by-claim auditing. As of 2026-02-02, this range contains no open claims (all rows are Verified/Refuted/Established).

## Scope

- Ticket ID: `TICKET-C151-C200`
- Kind: `CLAIMS_AUDIT_BATCH`
- Status token: `DONE`
- Claim range: C-151..C-200
- Claims referenced (2): C-151, C-200

## Deliverables

- `docs/CLAIMS_EVIDENCE_MATRIX.md`
- `reports/claims_batch_backlog_C151_C200.md`

## Acceptance checks

- `PYTHONWARNINGS=error make metadata-hygiene`
- `PYTHONWARNINGS=error make smoke`

## Progress snapshot

- Completed checkboxes: 0
- Open checkboxes: 0
- Backlog reports:
  - `reports/claims_batch_backlog_C151_C200.md`
