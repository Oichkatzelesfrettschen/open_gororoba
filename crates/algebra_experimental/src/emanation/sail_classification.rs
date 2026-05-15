//! Representation-aware Trip Sync (L15) and full sail decomposition (L15b).
//!
//! Trip Sync is not merely "the sail L-indices form a Fano line"
//! (membership). It is: "there exists a PSL(2,7) embedding in which the
//! zigzag's 4 quaternion copies are co-oriented, while trefoils show
//! controlled desynchronization."
//!
//! The key relationship (from Pathions3):
//!   Zigzag + 3 trefoil L-trips sit at (a,b,c), (a,d,e), (d,b,f), (e,f,c)
//!   forming the 4 faces of a tetrahedron inscribed in the box-kite octahedron.
//!
//! L15b cross-classifies all 8 octahedral faces by twist type
//! (zigzag/trefoil) and O-trip membership (sail/non-sail), producing
//! exactly:
//!   - 1 Zigzag Sail (all-Opposite edges, L-indices form O-trip)
//!   - 3 Trefoil Sails (mixed edges, L-indices form O-trip)
//!   - 1 Vent (all-Opposite edges, L-indices NOT an O-trip)
//!   - 3 non-Sail Trefoils (mixed edges, L-indices NOT an O-trip)
//!
//! De Marrais (2000): the 4 sails carry the quaternion subalgebra
//! copies; the Vent is the "ventilation hole" where trip sync fails
//! locally.

use std::collections::HashSet;

use algebra_analysis::boxkites::{BoxKite, O_TRIPS};

use super::{
    sail_loop::face_otrip_index,
    tray_racks::{TwistType, tray_racks},
};

// ===========================================================================
// L15: Representation-Aware Trip Sync (Orientation Coherence)
// ===========================================================================

/// Result of the orientation-aware Trip Sync check.
#[derive(Debug, Clone)]
pub struct OrientedTripSync {
    /// The box-kite's strut signature.
    pub strut_sig: usize,
    /// The 4 O-trips available in this BK's 6 L-indices.
    pub available_trips: Vec<[usize; 3]>,
    /// For each candidate zigzag trip, whether the shorthand pattern is satisfiable.
    pub candidate_results: Vec<(usize, bool)>,
    /// Whether at least one candidate satisfies Trip Sync.
    pub has_valid_embedding: bool,
}

/// Check orientation-aware Trip Sync for a box-kite.
///
/// For each of the 4 O-trips in the BK's L-indices, try it as the zigzag trip
/// and check whether the remaining L-indices can form the 3 trefoil trips
/// according to the shorthand pattern (a,b,c), (a,d,e), (d,b,f), (e,f,c).
pub fn oriented_trip_sync(bk: &BoxKite) -> OrientedTripSync {
    let l_set: HashSet<usize> = bk.assessors.iter().map(|a| a.low).collect();
    let available: Vec<[usize; 3]> = O_TRIPS
        .iter()
        .filter(|t| t.iter().all(|&x| l_set.contains(&x)))
        .copied()
        .collect();

    let mut candidate_results = Vec::new();
    let mut has_valid = false;

    for (idx, zig_trip) in available.iter().enumerate() {
        // Try zig_trip = (a, b, c) as the zigzag Rule-0 trip.
        // The remaining 3 indices are {d, e, f}.
        let remaining: Vec<usize> = l_set
            .iter()
            .copied()
            .filter(|x| !zig_trip.contains(x))
            .collect();

        if remaining.len() != 3 {
            continue;
        }

        // De Marrais shorthand: trefoils are (a,d,e), (d,b,f), (e,f,c)
        // We need to find an assignment of remaining to {d,e,f} such that
        // all three trefoil triples are also O-trips.
        let valid = try_trefoil_assignment(zig_trip, &remaining);
        candidate_results.push((idx, valid));
        if valid {
            has_valid = true;
        }
    }

    OrientedTripSync {
        strut_sig: bk.strut_signature,
        available_trips: available,
        candidate_results,
        has_valid_embedding: has_valid,
    }
}

/// Try all 6 permutations of remaining indices to find a valid trefoil assignment.
fn try_trefoil_assignment(zig_trip: &[usize; 3], remaining: &[usize]) -> bool {
    let (a, b, c) = (zig_trip[0], zig_trip[1], zig_trip[2]);
    let perms = [
        (remaining[0], remaining[1], remaining[2]),
        (remaining[0], remaining[2], remaining[1]),
        (remaining[1], remaining[0], remaining[2]),
        (remaining[1], remaining[2], remaining[0]),
        (remaining[2], remaining[0], remaining[1]),
        (remaining[2], remaining[1], remaining[0]),
    ];

    let otrip_set: HashSet<[usize; 3]> = O_TRIPS
        .iter()
        .map(|t| {
            let mut s = *t;
            s.sort();
            s
        })
        .collect();

    for (d, e, f) in perms {
        let t1 = {
            let mut t = [a, d, e];
            t.sort();
            t
        };
        let t2 = {
            let mut t = [d, b, f];
            t.sort();
            t
        };
        let t3 = {
            let mut t = [e, f, c];
            t.sort();
            t
        };
        if otrip_set.contains(&t1) && otrip_set.contains(&t2) && otrip_set.contains(&t3) {
            return true;
        }
    }
    false
}

// ===========================================================================
// L15b: Sail Decomposition -- Full face classification per box-kite
// ===========================================================================

/// Classification of a single triangular face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaceRole {
    /// Zigzag face whose L-indices form an O-trip (the unique zigzag sail).
    ZigzagSail,
    /// Trefoil face whose L-indices form an O-trip (one of 3 trefoil sails).
    TrefoilSail,
    /// Zigzag face whose L-indices do NOT form an O-trip (the unique vent).
    Vent,
    /// Trefoil face whose L-indices do NOT form an O-trip.
    NonSailTrefoil,
}

/// A classified face of the box-kite octahedron.
#[derive(Debug, Clone)]
pub struct ClassifiedFace {
    /// The 3 assessor indices (into the box-kite's assessor list).
    pub assessor_indices: [usize; 3],
    /// The 3 L-indices (low parts of the assessors).
    pub l_indices: [usize; 3],
    /// The face's role in the sail decomposition.
    pub role: FaceRole,
    /// If this face is a sail, the index of the O-trip it corresponds to.
    pub otrip_index: Option<usize>,
}

/// Complete sail decomposition of a box-kite.
#[derive(Debug, Clone)]
pub struct SailDecomposition {
    /// The box-kite's strut signature.
    pub strut_sig: usize,
    /// All 8 faces, classified.
    pub faces: Vec<ClassifiedFace>,
    /// The unique zigzag sail (index into `faces`).
    pub zigzag_sail_idx: usize,
    /// The 3 trefoil sail indices (into `faces`).
    pub trefoil_sail_indices: [usize; 3],
    /// The unique vent index (into `faces`).
    pub vent_idx: usize,
    /// The 3 non-sail trefoil indices (into `faces`).
    pub non_sail_trefoil_indices: [usize; 3],
}

/// Compute the full sail decomposition for a box-kite.
///
/// Cross-classifies all 8 octahedral faces by twist type (zigzag/trefoil)
/// and O-trip membership (sail/non-sail), producing exactly:
/// - 1 zigzag sail, 3 trefoil sails, 1 vent, 3 non-sail trefoils.
///
/// Panics if the box-kite does not have the expected 2+6 zigzag/trefoil split
/// or the expected 4+4 sail/non-sail split.
pub fn sail_decomposition(bk: &BoxKite) -> SailDecomposition {
    let racks = tray_racks(bk);
    assert_eq!(racks.len(), 8, "Box-kite must have exactly 8 faces");

    let mut faces = Vec::with_capacity(8);
    for rack in &racks {
        let l_indices = [
            bk.assessors[rack.assessors[0]].low,
            bk.assessors[rack.assessors[1]].low,
            bk.assessors[rack.assessors[2]].low,
        ];
        let otrip_idx = face_otrip_index(bk, &rack.assessors);
        let is_sail = otrip_idx.is_some();
        let is_zigzag = rack.twist_type == TwistType::Zigzag;

        let role = match (is_zigzag, is_sail) {
            (true, true) => FaceRole::ZigzagSail,
            (true, false) => FaceRole::Vent,
            (false, true) => FaceRole::TrefoilSail,
            (false, false) => FaceRole::NonSailTrefoil,
        };

        faces.push(ClassifiedFace {
            assessor_indices: rack.assessors,
            l_indices,
            role,
            otrip_index: otrip_idx,
        });
    }

    // Extract indices by role
    let zigzag_sails: Vec<usize> = faces
        .iter()
        .enumerate()
        .filter(|(_, f)| f.role == FaceRole::ZigzagSail)
        .map(|(i, _)| i)
        .collect();
    let trefoil_sails: Vec<usize> = faces
        .iter()
        .enumerate()
        .filter(|(_, f)| f.role == FaceRole::TrefoilSail)
        .map(|(i, _)| i)
        .collect();
    let vents: Vec<usize> = faces
        .iter()
        .enumerate()
        .filter(|(_, f)| f.role == FaceRole::Vent)
        .map(|(i, _)| i)
        .collect();
    let non_sail_trefoils: Vec<usize> = faces
        .iter()
        .enumerate()
        .filter(|(_, f)| f.role == FaceRole::NonSailTrefoil)
        .map(|(i, _)| i)
        .collect();

    assert_eq!(
        zigzag_sails.len(),
        1,
        "BK S={}: expected 1 zigzag sail, got {}",
        bk.strut_signature,
        zigzag_sails.len()
    );
    assert_eq!(
        trefoil_sails.len(),
        3,
        "BK S={}: expected 3 trefoil sails, got {}",
        bk.strut_signature,
        trefoil_sails.len()
    );
    assert_eq!(
        vents.len(),
        1,
        "BK S={}: expected 1 vent, got {}",
        bk.strut_signature,
        vents.len()
    );
    assert_eq!(
        non_sail_trefoils.len(),
        3,
        "BK S={}: expected 3 non-sail trefoils, got {}",
        bk.strut_signature,
        non_sail_trefoils.len()
    );

    SailDecomposition {
        strut_sig: bk.strut_signature,
        faces,
        zigzag_sail_idx: zigzag_sails[0],
        trefoil_sail_indices: [trefoil_sails[0], trefoil_sails[1], trefoil_sails[2]],
        vent_idx: vents[0],
        non_sail_trefoil_indices: [
            non_sail_trefoils[0],
            non_sail_trefoils[1],
            non_sail_trefoils[2],
        ],
    }
}
