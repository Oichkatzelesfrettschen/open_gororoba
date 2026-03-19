//! # Integration Interface Contracts
//!
//! ## Contract A: Repository Intake Contract
//! - Producer: origin repositories.
//! - Consumer: `merge_in/*`.
//! - Interface: git clone/fetch/pull with branch/head capture.
//! - Acceptance criteria:
//!   - remote URL captured,
//!   - head hash captured,
//!   - sync audit artifact generated.
//!
//! ## Contract B: Module Synthesis Contract
//! - Producer: `scripts/synthesize_root_layout.sh`.
//! - Consumer: `src/lambda_unified/modules/*`.
//! - Interface:
//!   - module-root artifacts,
//!   - seed docs/source/tests/admin,
//!   - contracts manifest.
//! - Acceptance criteria:
//!   - synthesis report generated,
//!   - integration matrix generated,
//!   - conflict matrix generated.
//!
//! ## Contract C: Consolidation Decision Contract
//! - Producer: `scripts/generate_module_consolidation_rules.sh`.
//! - Consumer: archive/index/verifier pipeline.
//! - Interface: decision table TSV with action classes and canonical mapping.
//! - Acceptance criteria:
//!   - no malformed decision rows,
//!   - deterministic category counts,
//!   - stable latest pointer updated.
//!
//! ## Contract D: Archive Execution Contract
//! - Producer: `scripts/archive_module_duplicate_docs.sh`.
//! - Consumer: module docs archive namespace.
//! - Interface: move-only operations with hash equivalence checks.
//! - Acceptance criteria:
//!   - source and target hashes match,
//!   - canonical target exists,
//!   - archive manifest generated.
//!
//! ## Contract E: Policy Verification Contract
//! - Producer: verifier scripts.
//! - Consumer: quality policy enforcement.
//! - Interface:
//!   - consolidation verification report,
//!   - python threading verification report.
//! - Acceptance criteria:
//!   - pass status with zero blocking failures.
//!
