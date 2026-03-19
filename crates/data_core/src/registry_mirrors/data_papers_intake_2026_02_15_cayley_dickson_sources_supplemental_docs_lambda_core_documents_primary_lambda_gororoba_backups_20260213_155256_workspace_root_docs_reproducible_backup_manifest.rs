//! # Backup Manifest Protocol
//!
//! ## Scope
//! Each workspace backup produced by `scripts/backup_documents_workspace.sh` must include:
//!
//! - Timestamped snapshot directory under `~/Documents/lambda_gororoba_backups/<TS>/`.
//! - Deterministic file inclusion rules (docs, admin, scripts, requirement files, root metadata files).
//! - SHA-256 manifest file named `backup_manifest.sha256`.
//!
//! ## Procedure
//! 1. Run: `/home/eirikr/Github/lambda_gororoba/scripts/backup_documents_workspace.sh`
//! 2. Capture printed `path=` and `manifest=` values.
//! 3. Verify manifest integrity for one or more files:
//!    - `sha256sum -c <manifest>` in backup root.
//! 4. Archive workspace log that triggered the backup in `logs/`.
//!
//! ## Non-goal
//! Binary payloads and large runtime directories are intentionally excluded to reduce backup churn.
//!
