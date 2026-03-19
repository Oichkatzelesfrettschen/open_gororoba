//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/claim_tickets.toml -->
//!
//! # Ticket: Claims audit batch C-101..C-150
//!
//! Owner: agent
//! Created: 2026-02-02
//! Status: IN PROGRESS
//!
//! ## Goal
//!
//! Make the C-101..C-150 segment of `docs/CLAIMS_EVIDENCE_MATRIX.md` mechanically tractable for claim-by-claim auditing: - each open claim has (1) a clear scope boundary, (2) primary sources indexed and cached when possible, and (3) an offline check hook (unit test, verifier, or deterministic artifact pipeline) where feasible. This batch is dominated by legacy high-dimension Cayley-Dickson experiment summaries. The critical work is to separate: - the computable/mathematical invariants (often already reproducible), from - physics-facing interpretations (which must remain explicitly speculative unless source-backed).
//!
//! ## Scope
//!
//! - Ticket ID: `TICKET-C101-C150`
//! - Kind: `CLAIMS_AUDIT_BATCH`
//! - Status token: `IN_PROGRESS`
//! - Claim range: C-101..C-150
//! - Claims referenced (13): C-101, C-102, C-103, C-108, C-109, C-120, C-123, C-128, C-129, C-130, C-132, C-135, C-150
//!
//! ## Deliverables
//!
//! - `docs/CLAIMS_EVIDENCE_MATRIX.md`
//! - `reports/claims_batch_backlog_C101_C150.md`
//!
//! ## Acceptance checks
//!
//! - `PYTHONWARNINGS=error make check-parallel`
//! - `PYTHONWARNINGS=error make metadata-hygiene`
//! - `PYTHONWARNINGS=error make smoke`
//!
//! ## Progress snapshot
//!
//! - Completed checkboxes: 0
//! - Open checkboxes: 0
//! - Backlog reports:
//!   - `reports/claims_batch_backlog_C101_C150.md`
//!
