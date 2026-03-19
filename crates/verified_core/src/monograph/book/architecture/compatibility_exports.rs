//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! # Compatibility Exports
//!
//! Compatibility exports are generated files that remain useful for web browsing,
//! publication assembly, and transitional tooling, but they are not the canonical
//! authoring surface for migrated lanes.
//!
//! Canonical machine-readable inventory:
//!
//! - `registry/control_plane_compatibility_exports.toml`
//!
//! ## Examples
//!
//! - `registry/claims.toml`
//! - `registry/insights.toml`
//! - `registry/experiments.toml`
//! - `registry/binaries.toml`
//! - `docs/CLAIMS_EVIDENCE_MATRIX.md`
//! - `docs/INSIGHTS.md`
//! - `docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md`
//! - `docs/THEOREMS.md`
//!
//! ## Required behavior
//!
//! Compatibility exports must satisfy two constraints:
//!
//! 1. They must be reproducible from the canonical DB.
//! 2. They must not silently become a second authoring source.
//!
//! ## How they are checked
//!
//! - `provenance verify-control-plane`
//!   Verifies DB invariants and freshness for the migrated control-plane exports.
//! - `verify-registry-mirror-freshness`
//!   Verifies generated Markdown mirror freshness.
//! - `registry-check`
//!   Cross-checks counts, identities, execution targets, and canonical DB parity.
//!
//! ## Contributor guidance
//!
//! If you edit a compatibility export directly, you are almost always doing one of
//! two things:
//!
//! - repairing a bootstrap/import path
//! - creating drift that will be overwritten by export
//!
//! If the file is listed in the compatibility-export inventory, prefer changing
//! the canonical source path and regenerating instead.
//!
