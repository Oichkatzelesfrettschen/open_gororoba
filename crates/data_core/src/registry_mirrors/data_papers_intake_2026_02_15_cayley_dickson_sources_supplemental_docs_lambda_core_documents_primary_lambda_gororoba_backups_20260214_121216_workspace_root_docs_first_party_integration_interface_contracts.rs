//! # Integration Interface Contracts
//!
//! ## Contract A: Repository Intake Contract
//! - Producer: archived upstream snapshot capture step.
//! - Consumer: `archive/intake_lane_retirement/<TS>/merge_in/*`.
//! - Interface: one-time archival move with branch/head/remote capture logs.
//! - Acceptance criteria:
//! - remote URL captured,
//! - head hash captured,
//! - archive action artifact generated.
//!
//! ## Contract B: Module Synthesis Contract
//! - Producer: `scripts/merge_into_singular_structure.sh`.
//! - Consumer: `research/`, `experiments/`, `learner/`.
//! - Interface:
//! - module-root reconciliation and canonicalization,
//! - root canonical conflict artifacts (`README.md`, `LICENSE`, `requirements.md`),
//! - merge manifest + conflict + duplicate + dedupe logs.
//! - Acceptance criteria:
//! - singular merge report generated,
//! - singular manifest generated,
//! - conflict and dedupe artifacts generated.
//!
//! ## Contract C: Consolidation Decision Contract
//! - Producer: `scripts/generate_module_consolidation_rules.sh`.
//! - Consumer: archive/index/verifier pipeline.
//! - Interface: decision table TSV with action classes and canonical mapping.
//! - Acceptance criteria:
//! - no malformed decision rows,
//! - deterministic category counts,
//! - stable latest pointer updated.
//!
//! ## Contract D: Archive Execution Contract
//! - Producer: `scripts/archive_module_duplicate_docs.sh`.
//! - Consumer: module docs archive namespace.
//! - Interface: move-only operations with hash equivalence checks.
//! - Acceptance criteria:
//! - source and target hashes match,
//! - canonical target exists,
//! - archive manifest generated.
//!
//! ## Contract E: Policy Verification Contract
//! - Producer: verifier scripts.
//! - Consumer: quality policy enforcement.
//! - Interface:
//! - consolidation verification report,
//! - python threading verification report.
//! - Acceptance criteria:
//! - pass status with zero blocking failures.
//!
