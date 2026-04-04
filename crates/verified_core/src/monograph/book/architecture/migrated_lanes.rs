//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! # Migrated Lanes
//!
//! The SQLite-first migration does not apply to the whole repo uniformly. This
//! page is the quick map of which structured lanes are already DB-first and which
//! are still intentionally outside the control-plane cutover.
//!
//! Canonical machine-readable inventory:
//!
//! - `registry/control_plane_migrated_lanes.toml`
//!
//! ## DB-first lanes
//!
//! These lanes are operationally authored in
//! `registry/canonical/control_plane.sqlite3` and exported outward for
//! compatibility:
//!
//! - claims
//! - insights
//! - experiments
//! - binaries
//! - theorems
//!
//! For those lanes:
//!
//! - query live state with `provenance query ...`
//! - verify integrity with `provenance verify-control-plane`
//! - refresh exports with `provenance export-control-plane`
//!
//! ## Hybrid or non-migrated lanes
//!
//! These lanes are still intentionally outside the control-plane migration or only
//! partially synchronized:
//!
//! - `book_docs`
//!   Authored mdBook prose remains outside the SQLite control plane.
//! - roadmap / todo / requirements / module requirements
//!   Planning lanes remain TOML-authored today; SQLite promotion has not landed.
//! - external source governance and dataset alias lanes
//!   These remain TOML-authored compatibility lanes pending a later migration tranche.
//!
//! ## Why this matters
//!
//! When a lane is DB-first, compatibility files are required outputs but not
//! authoring sources. When a lane is still outside the DB cutover, editing its
//! TOML authoring file is expected until promotion lands.
//!
//! That distinction is the difference between:
//!
//! - a normal workflow
//! - a recovery/bootstrap workflow
//! - a stale-export bug
//!
//! ## Related docs
//!
//! - [Canonical Sources](./canonical-sources.md)
//! - [DB Authoring Workflow](./db-authoring.md)
//! - [Compatibility Exports](./compatibility-exports.md)
//!
