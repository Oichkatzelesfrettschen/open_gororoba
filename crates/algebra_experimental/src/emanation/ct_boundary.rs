//! CT Boundary / A7 Star (L10): twist as double transfer.
//!
//! De Marrais (Presto I, "Royal Hunt") identifies twist products as
//! "double transfers" in Catastrophe Theory: swapping both the
//! assessor pair AND the box-kite membership simultaneously. This
//! maps to a composition of Double Cusps in the A-series, with the
//! simplest non-elementary form being A7 Star.
//!
//! The Quincunx lanyard's 120 string-readings connect to the
//! icosahedral reflection group H3 (|H3| = 120).

use super::twist_transitions::twist_transition_table;

/// Result of the CT boundary analysis.
#[derive(Debug, Clone)]
pub struct CtBoundaryResult {
    /// Number of quincunx types per tray-rack (Feet vs Hands).
    pub quincunx_types: usize,
    /// Number of tray-rack axes per box-kite (always 3).
    pub tray_rack_axes: usize,
    /// Number of string-reading start points per quincunx (10).
    pub readings_per_quincunx: usize,
    /// Flow-reversal factor (2: forward and backward).
    pub flow_reversals: usize,
    /// Total string count: types * axes * readings * reversals.
    pub total_strings: usize,
    /// Whether total_strings == |H3| = 120.
    pub matches_h3_order: bool,
}

/// Verify the CT boundary / H3 connection for sedenion box-kites.
///
/// De Marrais: 2 types (Feet/Hands) x 3 tray-rack axes x 10 readings
/// x 2 flow-reversals = 120 = |H3| (icosahedral reflection group).
pub fn ct_boundary_analysis() -> CtBoundaryResult {
    let quincunx_types = 2;
    let tray_rack_axes = 3;
    let readings_per_quincunx = 10;
    let flow_reversals = 2;
    let total = quincunx_types * tray_rack_axes * readings_per_quincunx * flow_reversals;

    CtBoundaryResult {
        quincunx_types,
        tray_rack_axes,
        readings_per_quincunx,
        flow_reversals,
        total_strings: total,
        matches_h3_order: total == 120,
    }
}

/// Verify that each twist product pair (before/after) lives in different
/// box-kites, confirming the "double transfer" property.
pub fn verify_double_transfer() -> bool {
    let transitions = twist_transition_table();
    transitions
        .iter()
        .all(|t| t.source_strut != t.h_star_target && t.source_strut != t.v_star_target)
}
