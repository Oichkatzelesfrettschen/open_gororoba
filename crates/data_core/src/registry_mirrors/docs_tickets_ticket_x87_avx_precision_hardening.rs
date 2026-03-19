//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/claim_tickets.toml -->
//!
//! # Ticket: x87 / AVX Precision Hardening
//!
//! Owner: agent
//! Created: 2026-03-12
//! Status: IN_PROGRESS
//!
//! ## Goal
//!
//! Keep the x87 instruction semantics and measured throughput result verified, but harden the wider dispatch guidance with primary sources, a deterministic crossover test, and a post-AVX benchmark artifact before promoting the `N=2048` heuristic back to Verified.
//!
//! ## Scope
//!
//! - Ticket ID: `TICKET-TICKET-X87-AVX-PRECISION-HARDENING`
//! - Kind: `GENERAL`
//! - Status token: `IN_PROGRESS`
//! - Claim range: none (general ticket)
//! - Claims referenced (4): C-1359, C-1360, C-1361, C-1362
//!
//! ## Deliverables
//!
//! - `crates/algebra_analysis/src/x87_jacobi.rs`
//! - `crates/cd_kernel/src/avx2_primitives.rs`
//! - `crates/algebra_analysis/tests/precision_tier_dispatch.rs`
//! - `data/csv/x87_avx_fma_followup_benchmark_summary.csv`
//! - `docs/external_sources/X87_AVX_ACCUMULATION_SOURCES.md`
//! - `docs/CLAIMS_EVIDENCE_MATRIX.md`
//! - `docs/INSIGHTS.md`
//!
//! ## Acceptance checks
//!
//! - `cargo test -p cd_kernel --lib`
//! - `cargo test -p algebra_analysis --test precision_tier_dispatch`
//! - `cargo bench -p algebra_analysis -- x87`
//!
//! ## Progress snapshot
//!
//! - Completed checkboxes: 0
//! - Open checkboxes: 0
//!
