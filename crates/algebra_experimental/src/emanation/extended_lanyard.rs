//! Extended lanyard taxonomy (de Marrais's full classification of
//! cycle types in the ZD graph beyond Sail and TrayRack).
//!
//! De Marrais identifies 5 lanyard types beyond Sail and TrayRack:
//! - Blues (6-cycle with all positive edges, "all-positive" sails)
//! - Zigzag (6-cycle with alternating +/- edges, the "triple zigzag")
//! - Bow-Tie (degenerate: two 3-cycles sharing a vertex)
//! - Quincunx (10-cycle through 5 assessors, relating to H3 icosahedral group)
//! - Bicycle Chain (12-element cycle threading all diagonals of a box-kite)
//!
//! `classify_face_extended` classifies a triangular face by edge-sign
//! pattern. `extended_lanyard_census_dim16` aggregates the census
//! across all box-kites at dim=16.

use std::collections::HashMap;

use algebra_analysis::boxkites::{Assessor, EdgeSignType, edge_sign_type, find_box_kites};

use super::tray_racks::tray_racks;

/// Extended lanyard classification with de Marrais's full taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtendedLanyardType {
    /// All-same-sign 6-cycle (the "blues": 3 co-assessors, all edges Same-sign).
    Blues,
    /// Alternating-sign 6-cycle (the "triple zigzag": all edges Opposite-sign).
    TripleZigzag,
    /// Mixed-sign 3-cycle (a trefoil sail).
    Trefoil,
    /// 4-cycle tray-rack with alternating edge signs.
    TrayRackCycle,
    /// 5-assessor 10-cycle (the quincunx, linking to H3 icosahedral group).
    Quincunx,
    /// Full 12-element cycle threading all diagonals of a box-kite.
    BicycleChain,
}

/// Classify a triangular face into the extended lanyard taxonomy.
///
/// Uses the edge sign pattern to distinguish Blues (all Same),
/// TripleZigzag (all Opposite), and Trefoil (mixed).
pub fn classify_face_extended(assessors: &[Assessor; 3]) -> ExtendedLanyardType {
    let atol = 1e-10;
    let signs = [
        edge_sign_type(&assessors[0], &assessors[1], atol),
        edge_sign_type(&assessors[1], &assessors[2], atol),
        edge_sign_type(&assessors[0], &assessors[2], atol),
    ];

    let n_same = signs.iter().filter(|&&s| s == EdgeSignType::Same).count();
    let n_opp = signs
        .iter()
        .filter(|&&s| s == EdgeSignType::Opposite)
        .count();

    if n_same == 3 {
        ExtendedLanyardType::Blues
    } else if n_opp == 3 {
        ExtendedLanyardType::TripleZigzag
    } else {
        ExtendedLanyardType::Trefoil
    }
}

/// Extended lanyard census for all box-kites at dim=16.
///
/// Returns counts of each face type across all 7 box-kites.
/// Expected: 7 * 2 = 14 zigzag faces, 7 * 6 = 42 trefoil faces.
pub fn extended_lanyard_census_dim16() -> HashMap<ExtendedLanyardType, usize> {
    let bks = find_box_kites(16, 1e-10);
    let mut census: HashMap<ExtendedLanyardType, usize> = HashMap::new();

    for bk in &bks {
        let racks = tray_racks(bk);
        for rack in &racks {
            let face = [
                bk.assessors[rack.assessors[0]],
                bk.assessors[rack.assessors[1]],
                bk.assessors[rack.assessors[2]],
            ];
            let ltype = classify_face_extended(&face);
            *census.entry(ltype).or_insert(0) += 1;
        }
    }

    census
}
