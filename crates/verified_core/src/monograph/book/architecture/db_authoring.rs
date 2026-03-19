//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! # DB Authoring Workflow
//!
//! For the migrated control-plane lanes, the canonical operational source of truth
//! is `registry/canonical/control_plane.sqlite3`.
//!
//! That means the normal loop is:
//!
//! ```sh
//! cargo run -p gororoba_cli_provenance --bin provenance -- \
//!   --db registry/canonical/control_plane.sqlite3 \
//!   import-legacy-control-plane
//!
//! cargo run -p gororoba_cli_provenance --bin provenance -- \
//!   --db registry/canonical/control_plane.sqlite3 \
//!   export-control-plane
//!
//! cargo run -p gororoba_cli_provenance --bin provenance -- \
//!   --db registry/canonical/control_plane.sqlite3 \
//!   verify-control-plane
//! ```
//!
//! ## What \u201Cauthoring\u201D means here
//!
//! For now, some lanes still originate from compatibility material and are then
//! re-indexed into SQLite. That is a migration bridge, not the intended steady
//! state.
//!
//! Operationally, contributors should think in this order:
//!
//! 1. update the canonical control-plane state
//! 2. export the compatibility files
//! 3. verify freshness and integrity
//!
//! ## Canonical-path writers that already sync through the DB
//!
//! - `claims-consolidate`
//!   In-place writes to `registry/claims.toml` reindex and re-export through the DB.
//! - `execution-planning`
//!   Canonical writes to `registry/experiments.toml` now replace the experiments
//!   slice in the DB and re-export the compat files.
//!
//! ## Recovery and bootstrap tools
//!
//! These commands still exist, but they are not normal authoring workflows:
//!
//! - `migrate-claims`
//! - `migrate-insights`
//!
//! Treat them as import/recovery utilities only.
//!
//! ## Practical rule
//!
//! If you changed a migrated lane and did not run both export and verify against
//! the canonical DB, the repo is not fully updated yet.
//!
