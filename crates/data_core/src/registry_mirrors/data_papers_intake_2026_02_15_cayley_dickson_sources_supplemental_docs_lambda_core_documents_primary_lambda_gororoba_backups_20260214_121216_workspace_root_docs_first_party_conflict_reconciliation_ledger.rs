//! # Conflict Reconciliation Ledger
//!
//! ## Objective
//! Track how collisions, inconsistencies, and policy tensions are reconciled into additive, first-party outcomes.
//!
//! ## Reconciled Conflicts
//!
//! ### Duplicate docs in `lambda_research`
//! - Conflict:
//! - exact duplicate documents appeared in multiple paths.
//! - Reconciliation:
//! - preserve one canonical in-place source,
//! - move duplicate to archive namespace,
//! - record hash-equality manifest.
//! - Evidence:
//! - `docs/archive/legacy_unified_docs/LATEST.md`,
//! - `logs/legacy_docs_archive_20260213_221719.tsv`.
//!
//! ### Multi-module root collisions (`README.md`, `LICENSE`, `requirements.md`)
//! - Conflict:
//! - same filenames across modules.
//! - Reconciliation:
//! - preserve module-level canonical artifacts,
//! - avoid cross-module overwrite,
//! - harmonize via first-party index and matrix docs.
//! - Evidence:
//! - `docs/archive/legacy_unified_docs/LATEST.md`,
//! - `docs/LICENSES_MATRIX.md`.
//!
//! ### Python threading policy vs legacy scripts
//! - Conflict:
//! - policy requires parallelization, legacy scripts were single-threaded.
//! - Reconciliation:
//! - introduce explicit exemptions registry,
//! - refactor highest-impact scripts first,
//! - enforce with automated verifier and iterative burn-down.
//! - Evidence:
//! - `docs/PYTHON_THREADING_POLICY.md`,
//! - `docs/PYTHON_THREADING_EXEMPTIONS.tsv`,
//! - `docs/python_multithreading_policy_latest.tsv`.
//!
//! ## Active Resolution Loop
//! 1. detect issue,
//! 2. classify severity and scope,
//! 3. choose additive reconciliation pattern,
//! 4. execute with evidence,
//! 5. verify,
//! 6. update ledger and roadmap.
//!
