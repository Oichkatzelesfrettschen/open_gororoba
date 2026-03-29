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
//! - Phase 4 (crystal tables): COMPLETE -- all 230 ITA space groups -- task #58

// Generated tables are included from OUT_DIR.
// Each include! expands to zero or more pub const declarations.
include!(concat!(env!("OUT_DIR"), "/generated_nk_tables.rs"));
include!(concat!(env!("OUT_DIR"), "/generated_optical_params.rs"));
include!(concat!(env!("OUT_DIR"), "/generated_crystal_tables.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jc_au_tables_nonempty_and_consistent_lengths() {
        assert!(!JC_AU_EV.is_empty(), "JC_AU_EV table must not be empty");
        assert_eq!(JC_AU_EV.len(), JC_AU_N.len());
        assert_eq!(JC_AU_EV.len(), JC_AU_K.len());
    }
}
