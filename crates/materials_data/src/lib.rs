//! materials_data: compile-time optical, crystal, and periodic-table data tables.
//!
//! # Architecture
//!
//! Data files live in `data/` as human-readable CSV (n,k tables) and TOML
//! (material parameters, space groups). `build.rs` reads them at compile time
//! and emits Rust `const` arrays into `$OUT_DIR/generated_*.rs`, which are
//! included here via `include!`.
//!
//! Callers (primarily `materials_core`) depend on this crate to access the
//! const tables without pulling in any runtime I/O.
//!
//! # Migration status
//!
//! - Phase 2 (tabulated n,k): COMPLETE -- task #56
//! - Phase 3 (optical params): COMPLETE -- task #57
//! - Phase 4 (crystal tables): pending -- see task #58

// Generated tables are included from OUT_DIR.
// Each include! expands to zero or more pub const declarations.
include!(concat!(env!("OUT_DIR"), "/generated_nk_tables.rs"));
include!(concat!(env!("OUT_DIR"), "/generated_optical_params.rs"));
include!(concat!(env!("OUT_DIR"), "/generated_crystal_tables.rs"));
