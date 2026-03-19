//! # Legacy Lane Retirement
//!
//! ## Objective
//! Retire old multi-lane intake/synthesis paths so the singular project structure is the only active edit path.
//!
//! ## Active paths
//! - Edit and source lane:
//! - `research/`
//! - `experiments/`
//! - `learner/`
//!
//! ## Retired paths
//! - Legacy unified lane:
//! - `src/lambda_unified` (archived)
//! - Intake clone lane:
//! - `merge_in/*` (retired; replaced with note file)
//!
//! ## Archive paths
//! - `archive/legacy_unified_lane/<TS>/src/lambda_unified/`
//! - `archive/intake_lane_retirement/<TS>/merge_in/`
//!
//! ## Default orchestration behavior
//! - `RUN_SYNTHESIS=0` by default in `scripts/sync_and_audit_workspace.sh`
//! - `RUN_CONSOLIDATION_GATE=0` by default in `scripts/sync_and_audit_workspace.sh`
//! - singular reconciliation remains enabled by default:
//! - `RUN_SINGULAR_MERGE=1`
//!
//! ## Evidence files
//! - `logs/legacy_lane_retirement_<TS>.md`
//! - `logs/legacy_lane_retirement_<TS>.tsv`
//! - `logs/manual_singularization_archive_<TS>.md`
//! - `logs/manual_singularization_archive_<TS>.tsv`
//!
