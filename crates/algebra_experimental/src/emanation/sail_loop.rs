//! Loop/Box-Kite Duality via Automorpheme Membership (L11).
//!
//! Each box-kite has 8 triangular faces. Exactly 4 of them have
//! L-indices forming an O-trip (Fano plane line). These 4 "O-trip
//! sails" each map to a unique automorpheme (Cawagas loop = deformed
//! octonion copy).
//!
//! The duality:
//!   - Each BK's 4 O-trip sails land in 4 different automorphemes
//!   - Each automorpheme receives sails from exactly 4 different BKs
//!   - Total: 7 BKs x 4 sails = 28 = 7 automorphemes x 4 sails
//!
//! The automorpheme assignment is determined by which of the 7 O-trips
//! matches the sail's sorted L-index triple. Each automorpheme
//! (indexed by its O-trip) contains all assessors whose L-index
//! belongs to the trip and whose H-index is NOT in the exclusion set
//! {8, 8^o1, 8^o2, 8^o3}.

use std::collections::HashSet;

use algebra_analysis::boxkites::{BoxKite, O_TRIPS, find_box_kites};

/// A sail label: (box-kite strut signature, automorpheme O-trip index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SailLabel {
    pub strut_sig: usize,
    /// Index into O_TRIPS (0..7) identifying which automorpheme this sail belongs to.
    pub otrip_idx: usize,
}

/// Result of the sail-to-loop partition analysis.
#[derive(Debug, Clone)]
pub struct SailLoopResult {
    /// The 7 loops (automorphemes), each containing 4 sail labels.
    pub loops: Vec<Vec<SailLabel>>,
    /// Whether each BK's 4 sails land in 4 different loops.
    pub bk_sails_in_different_loops: bool,
    /// Whether each loop has sails from 4 different BKs.
    pub loop_sails_from_different_bks: bool,
    /// Total number of sails classified.
    pub total_sails: usize,
}

/// Get all 8 triangular faces of a box-kite (as assessor-index triples).
pub(crate) fn boxkite_faces(bk: &BoxKite) -> Vec<[usize; 3]> {
    let edge_set: HashSet<(usize, usize)> = bk
        .edges
        .iter()
        .flat_map(|&(a, b)| [(a, b), (b, a)])
        .collect();

    let mut faces = Vec::new();
    for i in 0..6 {
        for j in (i + 1)..6 {
            if !edge_set.contains(&(i, j)) {
                continue;
            }
            for k in (j + 1)..6 {
                if !edge_set.contains(&(i, k)) || !edge_set.contains(&(j, k)) {
                    continue;
                }
                faces.push([i, j, k]);
            }
        }
    }
    faces
}

/// Check if a face's L-indices form an O-trip, and if so return the O-trip index.
pub(crate) fn face_otrip_index(bk: &BoxKite, face: &[usize; 3]) -> Option<usize> {
    let mut l_sorted = [
        bk.assessors[face[0]].low,
        bk.assessors[face[1]].low,
        bk.assessors[face[2]].low,
    ];
    l_sorted.sort();

    O_TRIPS.iter().position(|t| {
        let mut s = *t;
        s.sort();
        s == l_sorted
    })
}

/// Compute the sail-to-loop partition via automorpheme membership.
///
/// Each BK has exactly 4 faces whose L-indices form O-trips. These 28
/// O-trip sails partition into 7 automorphemes (Cawagas loops) of 4 each.
pub fn sail_loop_partition() -> SailLoopResult {
    let bks = find_box_kites(16, 1e-10);

    // For each BK, find its 4 O-trip sails and assign to automorphemes
    let mut loops: Vec<Vec<SailLabel>> = vec![Vec::new(); 7];
    let mut all_sails = Vec::new();

    for bk in &bks {
        let faces = boxkite_faces(bk);
        for face in &faces {
            if let Some(otrip_idx) = face_otrip_index(bk, face) {
                let label = SailLabel {
                    strut_sig: bk.strut_signature,
                    otrip_idx,
                };
                loops[otrip_idx].push(label);
                all_sails.push(label);
            }
        }
    }

    // Sort each loop by strut signature for determinism
    for l in &mut loops {
        l.sort_by_key(|s| s.strut_sig);
    }

    // Check duality properties
    let bk_sails_in_different_loops = bks.iter().all(|bk| {
        let sail_loops: HashSet<usize> = all_sails
            .iter()
            .filter(|s| s.strut_sig == bk.strut_signature)
            .map(|s| s.otrip_idx)
            .collect();
        sail_loops.len() == 4
    });

    let loop_sails_from_different_bks = loops.iter().all(|l| {
        let bk_set: HashSet<usize> = l.iter().map(|s| s.strut_sig).collect();
        bk_set.len() == l.len()
    });

    SailLoopResult {
        loops,
        bk_sails_in_different_loops,
        loop_sails_from_different_bks,
        total_sails: all_sails.len(),
    }
}
