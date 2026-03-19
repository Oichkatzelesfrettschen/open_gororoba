//! # Singular Repository Layout
//!
//! ## Intent
//! Establish `/home/eirikr/Github/lambda_gororoba` as one manual, root-authoritative repository where the three projects are fully merged into a single structure.
//!
//! ## Canonical paths
//! - `research/`
//! - `experiments/`
//! - `learner/`
//!
//! These three directories are the only active source and edit paths.
//!
//! ## Two-level ASCII map
//! ```ignore
//! lambda_gororoba/
//! +-- research/
//! |   +-- admin/
//! |   +-- docs/
//! |   +-- implementations/
//! |   +-- papers-archive/
//! |   +-- scripts/
//! |   `-- tests/
//! +-- experiments/
//! |   +-- src/
//! |   |   +-- data/
//! |   |   +-- experiments/
//! |   |   +-- formal/
//! |   |   `-- kernels/
//! |   `-- tests/
//! |       +-- integration/
//! |       `-- unit/
//! +-- learner/
//! |   +-- components/
//! |   +-- logic/
//! |   `-- services/
//! +-- merge_in/
//! |   `-- README.md (retired)
//! +-- archive/
//! |   `-- intake_lane_retirement/<TS>/merge_in/*
//! +-- README.md
//! +-- LICENSE
//! +-- requirements.md
//! `-- logs/
//! >   +-- singular_structure_merge_<TS>.md
//! >   +-- singular_structure_manifest_<TS>.tsv
//! >   +-- singular_structure_conflicts_<TS>.tsv
//! >   +-- singular_structure_dedupe_actions_<TS>.tsv
//! >   `-- singular_structure_duplicates_<TS>.tsv
//! ```ignore
//!
//! ## Intake lane status
//! - `merge_in/` is retired from active synchronization.
//! - historical git snapshots are preserved under `archive/intake_lane_retirement/<TS>/merge_in/`.
//! - active reconciliation runs directly against `research/`, `experiments/`, and `learner/`.
//!
//! ## Orchestration
//! - `bash scripts/sync_and_audit_workspace.sh`
//! - audits singular root modules
//! - keeps legacy root synthesis disabled (`RUN_SYNTHESIS=0` default)
//! - runs singular reconciliation (`RUN_SINGULAR_MERGE=1` default)
//! - runs reproducibility checks (wasm+python default profile)
//!
//! ## Conflict policy
//! - Shared root filenames (`.gitignore`, `README.md`, `LICENSE`, `requirements.md`) are canonical at workspace root.
//! - Module `LICENSE` files are normalized as symlinks to root `LICENSE`.
//! - Cross-module empty placeholders are tracked as expected (`skipped_cross_module_empty`).
//!
//! ## Communication style
//! - User-facing merge updates use the visual-first style in `docs/ASCII_COMMUNICATION_STYLE.md`.
//!
