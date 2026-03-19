//! # Sync Scope From Source Folders
//!
//! ## Purpose
//! Define exactly what is synchronized in the singular repository now that root modules are authoritative.
//!
//! ## Two-level sync map
//! ```text
//! root authoritative modules/
//! |-- research/
//! |   |-- active: docs/, implementations/, tests/, scripts/, requirements.md, README.md
//! |   `-- excluded from reconciler: .git/, .github/, venv/, .venv/, target/, build/, caches
//! |-- experiments/
//! |   |-- active: src/, tests/, pyproject.toml, requirements.md, README.md
//! |   `-- excluded from reconciler: .git/, .github/, node_modules/, venv/, .venv/, build/, dist/, caches
//! `-- learner/
//!     |-- active: components/, logic/, services/, package.json, requirements.md, README.md
//!     `-- excluded from reconciler: .git/, .github/, node_modules/, build/, dist/, caches
//!
//! intake lane status/
//! |-- merge_in/README.md
//! `-- archive/intake_lane_retirement/<TS>/merge_in/
//! ```
//!
//! ## Root canonical files regenerated every singular reconciliation pass
//! - `.gitignore`
//! - `README.md`
//! - `LICENSE`
//! - `requirements.md`
//!
//! ## Duplicate reconciliation policy
//! - Intra-module exact duplicates: secondary files become symlinks to module-local canonical file.
//! - Cross-module `LICENSE` duplicates: module copies become symlinks to root `LICENSE`.
//! - Cross-module empty placeholders: logged as `skipped_cross_module_empty`.
//! - Other cross-module duplicates: logged as `skipped_cross_module` and kept module-scoped.
//!
//! ## Evidence outputs
//! - `logs/singular_structure_merge_<TS>.md`
//! - `logs/singular_structure_manifest_<TS>.tsv`
//! - `logs/singular_structure_conflicts_<TS>.tsv`
//! - `logs/singular_structure_dedupe_actions_<TS>.tsv`
//! - `logs/singular_structure_duplicates_<TS>.tsv`
//!
