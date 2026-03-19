//! # Repository Integration Contract (Derived Layout)
//! Generated: 20260213_211247
//!
//! Merge staging root: /home/eirikr/Github/lambda_gororoba/merge_in
//!
//! ## Canonical module mapping
//! | Repository | Target module dir | Contract | Evidence |
//! |---|---|---|---|
//! | lambda-research | `modules/lambda_research` | Canonical module boundary with full `source_repo/` mirror + seed manifest | /home/eirikr/Github/lambda_gororoba/src/lambda_unified/modules/lambda_research |
//! | lambda-synthesis-experiments | `modules/lambda_synthesis_experiments` | Canonical module boundary with full `source_repo/` mirror + seed manifest | /home/eirikr/Github/lambda_gororoba/src/lambda_unified/modules/lambda_synthesis_experiments |
//! | LambdaLearner | `modules/lambdalearner` | Canonical module boundary with full `source_repo/` mirror + seed manifest | /home/eirikr/Github/lambda_gororoba/src/lambda_unified/modules/lambdalearner |
//!
//! ## Synthesis policy
//! - Keep canonical unified source under `src/lambda_unified/modules/<module>/source_repo`.
//! - Keep repository-specific semantics and provenance in `src/lambda_unified/modules/<module>/contracts`.
//! - Keep synchronized upstream references in each child repository and consume them through deterministic scripts.
//! - Keep module boundaries by directory and dependency domain to avoid cross-repo coupling.
//! - Resolve collisions with explicit "Expected conflict" or "Review" classification.
//!
//! ## Consolidation outcome (deterministic checks)
//! - Run `bash scripts/synthesize_root_layout.sh` after each integration pass.
//! - Treat any conflict classified as "Review" as a blocker until owners assign a resolution owner.
//! - Update claims, licenses, and requirements matrix when module boundaries change.
//!
