//! # Singular Workspace Workflow
//!
//! ## Purpose
//! Operate `/home/eirikr/Github/lambda_gororoba` as one manual singular repository.
//! `research/`, `experiments/`, and `learner/` are the active sources of truth.
//!
//! ## Canonical module paths
//! - `research/`
//! - `experiments/`
//! - `learner/`
//!
//! ## Intake status
//! - `merge_in/` is retired from active syncing.
//! - Historical intake repositories are preserved under:
//! - `archive/intake_lane_retirement/<TS>/merge_in/`
//!
//! ## Operational commands
//! - Run singular sync + audit:
//! - `bash scripts/sync_and_audit_workspace.sh`
//! - Run singular reconciliation directly:
//! - `bash scripts/merge_into_singular_structure.sh`
//! - Run reproducibility checklist:
//! - `bash scripts/verify_build_test_reproducibility.sh`
//! - Run top-level claims verification:
//! - `bash scripts/verify_top_level_claims.sh`
//! - Run visual docs style verification:
//! - `bash scripts/verify_visual_docs_style.sh`
//! - Generate dark-mode singular merge infographic artifact:
//! - `python3 scripts/generate_singular_merge_infographic.py`
//! - Backup all workspace docs to `~/Documents`:
//! - `bash scripts/backup_documents_workspace.sh`
//!
//! ## Evidence outputs
//! - Sync audit: `logs/workspace_sync_audit_<TS>.md`
//! - Singular merge report: `logs/singular_structure_merge_<TS>.md`
//! - Singular merge manifest: `logs/singular_structure_manifest_<TS>.tsv`
//! - Singular merge conflicts: `logs/singular_structure_conflicts_<TS>.tsv`
//! - Singular merge dedupe actions: `logs/singular_structure_dedupe_actions_<TS>.tsv`
//! - Singular merge duplicates: `logs/singular_structure_duplicates_<TS>.tsv`
//! - Reproducibility checklist report: `logs/build_test_reproducibility_<TS>.md`
//! - Reproducibility checklist table: `logs/build_test_reproducibility_<TS>.tsv`
//! - Backup manifest: `~/Documents/lambda_gororoba_backups/<TS>/backup_manifest.sha256`
//!
//! ## Reproducibility policy notes
//! - `scripts/sync_and_audit_workspace.sh` keeps legacy root synthesis disabled by default (`RUN_SYNTHESIS=0`).
//! - `scripts/sync_and_audit_workspace.sh` runs singular reconciliation by default (`RUN_SINGULAR_MERGE=1`).
//! - singular reconciliation pass normalizes root canonical files (`.gitignore`, `README.md`, `LICENSE`, `requirements.md`) and reconciles module `LICENSE` files to root symlinks.
//! - `scripts/sync_and_audit_workspace.sh` runs `scripts/verify_build_test_reproducibility.sh` by default (`RUN_REPRO_CHECK=1`).
//! - `scripts/sync_and_audit_workspace.sh` runs `scripts/verify_top_level_claims.sh` by default (`RUN_TOP_LEVEL_CLAIMS_VERIFY=1`).
//! - `scripts/sync_and_audit_workspace.sh` runs `scripts/verify_visual_docs_style.sh` by default (`RUN_VISUAL_STYLE_VERIFY=1`).
//!
//! ## First-party synthesis outputs
//! - First-party corpus root: `docs/first_party/`
//! - First-party source index: `docs/first_party/FIRST_PARTY_SOURCE_INDEX.md`
//! - WASM + Python lane design and commands: `docs/WASM_PYTHON_SYNTHESIS.md`
//! - Singular module layout contract: `docs/SINGULAR_REPO_LAYOUT.md`
//! - Source-folder sync scope map (ASCII): `docs/SYNC_SCOPE_FROM_SOURCE_FOLDERS.md`
//! - User-facing visual update format: `docs/ASCII_COMMUNICATION_STYLE.md`
//! - Legacy lane retirement policy: `docs/LEGACY_LANE_RETIREMENT.md`
//! - Dark-mode infographic standard: `docs/VISUAL_DOCUMENTATION_STYLE_GUIDE.md`
//!
