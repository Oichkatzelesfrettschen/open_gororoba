//! gororoba_cli library - Shared utilities for CLI tools.
//!
//! # Canonical docs
//!
//! For everyday repo documentation, prefer:
//!
//! - crate rustdoc such as this page and module docs
//! - the mdBook under `docs/book/`
//! - the canonical SQLite control plane at `registry/canonical/control_plane.sqlite3`
//! - generated compatibility exports under `registry/`
//!
//! Generated LaTeX appendices under `docs/latex/` are publication artifacts and
//! should not be treated as the canonical source for CLI behavior.

pub mod claims;
pub mod data_governance;
pub mod thesis_42_support;
pub mod viz;
pub mod warp_forcing_policy;
pub mod warp_gate_policy;
