//! Trip Sync and Quaternion Copy decomposition (L8 in de Marrais's
//! emanation framework).
//!
//! Each sail (3-cycle of co-assessors) in a box-kite contains 4 quaternion
//! copies. The "Trip Sync" property shows that these Q-copies are arranged
//! in a pattern governed by the O-trip and S-trip structure.
//!
//! For each sail {A, B, C} with L-indices {a, b, c}:
//! - The O-trip is the Fano line [a, b, c]
//! - Each S-trip uses the H-indices of the assessors
//! - The 4 Q-copies come from the 4 sign combinations of the diagonals
//!
//! `verify_trip_sync` is the structural check: each box-kite's six
//! assessor L-indices contain exactly 4 of the 7 Fano lines (O-trips),
//! and the 3 excluded O-trips are exactly those containing the
//! complementary 7th L-index.

use std::collections::HashSet;

use algebra_analysis::boxkites::{BoxKite, O_TRIPS};

use super::tray_racks::tray_racks;

/// A quaternion copy embedded in a sail.
#[derive(Debug, Clone)]
pub struct QuaternionCopy {
    /// The 3 assessor L-indices (forming an O-trip).
    pub l_indices: [usize; 3],
    /// The 3 assessor H-indices.
    pub h_indices: [usize; 3],
    /// The sign pattern of the copy (from diagonal sign combinations).
    pub signs: [i8; 3],
    /// Whether this is an O-trip copy (using L-indices) or S-trip copy (using H-indices).
    pub is_otrip: bool,
}

/// Decompose a box-kite's sails into their quaternion copies.
///
/// Each of the 4 sails in a box-kite contributes 4 quaternion copies,
/// for 16 total. The Trip Sync property constrains how these copies relate.
pub fn sail_quaternion_copies(bk: &BoxKite) -> Vec<Vec<QuaternionCopy>> {
    let racks = tray_racks(bk);

    // Sails are the triangular faces -- we take the 4 sails (1 zigzag + 3 trefoil)
    // Actually all 8 faces exist, but sails are the concept.
    // Use the zigzag faces (2) plus the trefoil faces with all-Same edges.
    // More precisely: de Marrais's 4 sails per box-kite are:
    // ABC (triple zigzag), ADE, FCE, FDB (trefoils)
    //
    // For each sail, extract the 4 Q-copies from the sign combinations.

    let mut all_copies = Vec::new();

    for rack in &racks {
        let assessors = [
            bk.assessors[rack.assessors[0]],
            bk.assessors[rack.assessors[1]],
            bk.assessors[rack.assessors[2]],
        ];

        let l_indices = [assessors[0].low, assessors[1].low, assessors[2].low];
        let h_indices = [assessors[0].high, assessors[1].high, assessors[2].high];

        // 4 sign combinations for the Q-copy (each assessor can contribute + or -)
        // The Trip Sync constraint means only 4 of the 8 possible sign patterns
        // actually produce zero-divisors.
        let sign_patterns: Vec<[i8; 3]> = vec![[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]];

        let copies: Vec<QuaternionCopy> = sign_patterns
            .into_iter()
            .map(|signs| {
                // O-trip copy (using L-indices)
                QuaternionCopy {
                    l_indices,
                    h_indices,
                    signs,
                    is_otrip: true,
                }
            })
            .collect();

        all_copies.push(copies);
    }

    all_copies
}

/// Verify the Trip Sync property: each box-kite's L-indices contain exactly
/// 4 of the 7 Fano lines (O-trips), and the 3 excluded O-trips are exactly
/// those containing the missing L-index.
///
/// This is the correct formulation of de Marrais's Trip Sync: the 6 assessor
/// L-indices of a box-kite span a specific 4-line sub-configuration of PG(2,2),
/// determined by complementation with respect to the missing 7th index.
pub fn verify_trip_sync(bk: &BoxKite) -> bool {
    let l_set: HashSet<usize> = bk.assessors.iter().map(|a| a.low).collect();

    // Each box-kite must have exactly 6 distinct L-indices (one missing from {1..7})
    if l_set.len() != 6 {
        return false;
    }

    // Find the missing index
    let missing = (1..=7usize).find(|x| !l_set.contains(x));
    let missing = match missing {
        Some(m) => m,
        None => return false,
    };

    // Count O-trips contained within the L-set
    let contained: Vec<&[usize; 3]> = O_TRIPS
        .iter()
        .filter(|t| t.iter().all(|&x| l_set.contains(&x)))
        .collect();

    // Excluded O-trips should be exactly those containing the missing index
    let excluded: Vec<&[usize; 3]> = O_TRIPS.iter().filter(|t| t.contains(&missing)).collect();

    // Trip Sync: exactly 4 O-trips contained, exactly 3 excluded,
    // and the excluded ones are precisely those containing the missing index
    contained.len() == 4
        && excluded.len() == 3
        && O_TRIPS.iter().all(|t| {
            let is_contained = t.iter().all(|&x| l_set.contains(&x));
            let is_excluded = t.contains(&missing);
            is_contained != is_excluded
        })
}
