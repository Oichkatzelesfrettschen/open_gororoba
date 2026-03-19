//! # Merge-In Staging Workflow
//!
//! ## Purpose
//! Use `merge_in/` as the deterministic intake lane for repository synchronization and synthesis.
//! Keep `/home/eirikr/Github/lambda_gororoba` as source-of-truth for governance, orchestration, and unified outputs.
//!
//! ## Canonical repositories
//! - `lambda-research`
//! - `lambda-synthesis-experiments`
//! - `LambdaLearner`
//!
//! ## Resolution model
//! 1. If `merge_in/<repo>` exists, use it as the active source path.
//! 2. If `merge_in/<repo>` is missing and `/home/eirikr/Github/lambda_gororoba/<repo>` exists, create a symlink in `merge_in/`.
//! 3. If both are missing, clone from canonical origin into `merge_in/<repo>`.
//!
//! ## Operational commands
//! - Sync and pull:
//! - `bash scripts/sync_and_audit_workspace.sh`
//! - Optional sync + full consolidation quality gate:
//! - `RUN_CONSOLIDATION_GATE=1 bash scripts/sync_and_audit_workspace.sh`
//! - Run full consolidation gate directly:
//! - `bash scripts/run_module_consolidation_gate.sh`
//! - Run source discovery:
//! - `bash scripts/discover_original_sources.sh`
//! - Run integration synthesis:
//! - `bash scripts/synthesize_root_layout.sh`
//! - Backup all workspace docs to `~/Documents`:
//! - `bash scripts/backup_documents_workspace.sh`
//!
//! ## Evidence outputs
//! - Sync audit: `logs/workspace_sync_audit_<TS>.md`
//! - Alignment report: `logs/repo_alignment_<TS>.md`
//! - Source manifest: `logs/source_discovery/source_discovery_<TS>.md`
//! - Synthesis report: `logs/repo_integration_synthesis_<TS>.md`
//! - Conflict matrix: `logs/conflict_matrix_<TS>.tsv`
//! - Backup manifest: `~/Documents/lambda_gororoba_backups/<TS>/backup_manifest.sha256`
//!
//! ## First-party synthesis outputs
//! - First-party corpus root: `docs/first_party/`
//! - First-party source index: `docs/first_party/FIRST_PARTY_SOURCE_INDEX.md`
//! - First-party architecture and reconciliation docs are maintained as additive outputs, not replacements for module provenance snapshots.
//!
