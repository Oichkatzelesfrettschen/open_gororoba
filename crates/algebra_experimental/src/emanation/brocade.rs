//! Brocade / Slipcover normalization (L18).
//!
//! Any node can be moved to the center of the PSL(2,7) triangle to act
//! as the strut constant, with the main side-effect being broad-based
//! swapping of U-indices (Pathions3).
//!
//! This means `canonical_strut_table()` is correct as a *set* of dyads
//! and strut pairs, but comparing Trip Sync or O-trip alignment to
//! literature diagrams requires a brocade relabeling.
//!
//! The brocade normalization maps a box-kite's L-indices to a standard
//! form where a chosen O-trip serves as the Rule-0 central circle.
//!
//! Each box-kite admits 4 valid relabelings (one per O-trip in its
//! L-set of 6 indices).

use std::collections::HashSet;

use algebra_analysis::boxkites::{BoxKite, O_TRIPS, find_box_kites};

/// A brocade relabeling: maps raw L-indices to standard PSL(2,7) positions.
#[derive(Debug, Clone)]
pub struct BrocadeRelabeling {
    /// Source strut signature.
    pub source_s: usize,
    /// The O-trip chosen as the Rule-0 central circle.
    pub central_trip: [usize; 3],
    /// The 3 remaining L-indices (forming the outer triangle).
    pub outer_indices: [usize; 3],
    /// Whether this relabeling preserves CPO (cyclic positive orientation).
    pub preserves_cpo: bool,
}

/// Compute all valid brocade relabelings for a box-kite.
///
/// Each of the 4 O-trips in the BK's L-set can serve as the central circle,
/// giving 4 possible normalizations.
pub fn brocade_relabelings(bk: &BoxKite) -> Vec<BrocadeRelabeling> {
    let l_set: HashSet<usize> = bk.assessors.iter().map(|a| a.low).collect();
    let available: Vec<[usize; 3]> = O_TRIPS
        .iter()
        .filter(|t| t.iter().all(|&x| l_set.contains(&x)))
        .copied()
        .collect();

    let otrip_set: HashSet<[usize; 3]> = O_TRIPS
        .iter()
        .map(|t| {
            let mut s = *t;
            s.sort();
            s
        })
        .collect();

    let mut relabelings = Vec::new();

    for trip in &available {
        let outer: Vec<usize> = l_set
            .iter()
            .copied()
            .filter(|x| !trip.contains(x))
            .collect();

        if outer.len() != 3 {
            continue;
        }

        // Check CPO preservation: do the outer indices form an O-trip?
        let mut outer_sorted = [outer[0], outer[1], outer[2]];
        outer_sorted.sort();
        let preserves_cpo = otrip_set.contains(&outer_sorted);

        relabelings.push(BrocadeRelabeling {
            source_s: bk.strut_signature,
            central_trip: *trip,
            outer_indices: [outer[0], outer[1], outer[2]],
            preserves_cpo,
        });
    }

    relabelings
}

/// Check that brocade normalization is consistent across all box-kites:
/// each BK has exactly 4 relabelings, and the outer indices always form
/// a well-defined partition of the non-trip L-indices.
pub fn verify_brocade_consistency() -> bool {
    let bks = find_box_kites(16, 1e-10);
    bks.iter().all(|bk| {
        let relabelings = brocade_relabelings(bk);
        relabelings.len() == 4
    })
}
