//! # Roadmap Status (Lambda Workspace)
//!
//! Last updated: 2026-02-13
//!
//! ## Current Phase
//! Phase 6 — Root-module synthesis and merge_in reconciliation
//!
//! ## Progress Snapshot
//! - [x] Sync/pull all three repos from origin.
//! - [x] Normalize workspace directories and adopt `merge_in/`-first staging with root source-of-truth.
//! - [x] Create baseline audit logging and backup snapshots.
//! - [x] Add machine-actionable repo sync script.
//! - [x] Add machine-actionable backup script with manifest.
//! - [x] Establish claims register and license reconciliation report.
//! - [x] Consolidate duplicate roadmap artifacts under one master timeline.
//! - [x] Align all modules to a single requirements matrix + installation contract.
//! - [x] Complete license harmonization memo for GPL-2.0-only policy.
//! - [x] Publish reproducible original-source discovery workflow for all external links and downloads.
//! - [x] Add license boundary matrix and legal decision note for harmonized GPL-2.0-only workspace.
//! - [x] Archive source manifest retention and run review after each consolidation pass.
//! - [x] Refresh repository alignment report for `merge_in/`-first mode.
//! - [x] Refresh TODO/FIXME map and retain source discovery artifacts.
//! - [x] Generate file-by-file consolidation rules for `src/lambda_unified/modules/*` with machine-readable decisions.
//! - [x] Execute archive moves for duplicate docs with hash-logged manifest.
//! - [x] Generate canonical doc/bibliography/topic indices for module navigation.
//! - [x] Add and run automated consolidation verifier with pass/fail artifact logs.
//! - [x] Add one-command consolidation gate orchestration script and integrate optional hook into sync workflow.
//! - [x] Add Python multithreading policy and verifier with explicit exemptions registry.
//! - [x] Complete first threading refactor tranche (`download_papers.py` pair) and reduce active exemptions.
//! - [x] Complete second threading refactor tranche (`search_papers.py` + `update_metadata.py` pairs) and reduce active exemptions.
//! - [x] Publish first-party synthesis corpus under `docs/first_party/` for novel unified architecture and reconciliation.
//! - [ ] Merge all projects into the root-derived canonical layout with full conflict decisions.
//!
//! ## Current Focus (next pass)
//! 1. Complete conflict decisions for shared module-root artifacts and record keep/merge rules.
//! 2. Publish migration guide for module-boundary legal structure and review remaining GPL issues.
//! 3. Resolve top remaining TODO/FIXME high-priority items with owners and evidence-backed acceptance criteria.
//! 4. Convert roadmap completion conditions into executable checks in `scripts/sync_and_audit_workspace.sh` and companion scripts.
//! > 5. Reduce active Python threading exemptions by refactoring next high-impact scripts (`verify_access.py`, `generate_index.py`, `orchestrator.py`) with measurable compliance deltas.
//!
//! ## Exit Gates
//! - All sync/audit scripts run successfully and emit logs.
//! - No destructive operations without explicit approval.
//! - Each module has at least one auditable requirements entry, and licensing is recorded as either known marker or explicit "missing/under review" status.
//! - A reproducible backup manifest exists for each snapshot.
//!
