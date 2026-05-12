//! Lanyard (cycle) classification in the zero-divisor graph.
//!
//! De Marrais identifies several cycle types in the ZD graph:
//! - Sail: 3-cycle of co-assessors forming a triangular face with
//!   all Same-sign edges.
//! - TrayRack: 3-cycle (triangular face, any sign pattern not all Same).
//! - Quincunx: 5-element cross-linking pattern.
//! - BicycleChain: longer cycle spanning multiple structures.
//!
//! `classify_lanyard` does the discriminator dispatch (length plus
//! edge-sign predicate for 3-cycles). `lanyard_census_dim16` runs the
//! census over all tray-racks of all sedenion box-kites.

use std::collections::HashMap;

use algebra_analysis::boxkites::{Assessor, EdgeSignType, edge_sign_type, find_box_kites};

use super::tray_racks::tray_racks;

/// Classification of lanyard (cycle) types in the ZD graph.
///
/// De Marrais identifies several cycle types:
/// - Sail: 3-cycle of co-assessors forming a triangular face with same-sign edges
/// - TrayRack: 3-cycle forming a triangular face (any sign pattern)
/// - Quincunx: 5-cycle cross-linking multiple box-kites
/// - BicycleChain: longer cycle spanning multiple box-kites
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LanyardType {
    /// Triangular face with all Same-sign edges (a sail).
    Sail,
    /// Triangular face (any sign pattern, including zigzag).
    TrayRack,
    /// 5-element cross-linking pattern.
    Quincunx,
    /// Longer cycle spanning multiple structures.
    BicycleChain,
}

/// Classify a cycle of assessors into a lanyard type.
///
/// The classification is based on cycle length and edge sign patterns.
pub fn classify_lanyard(cycle: &[Assessor]) -> LanyardType {
    let atol = 1e-10;

    match cycle.len() {
        3 => {
            // Check edge signs to distinguish sail from tray-rack
            let signs = [
                edge_sign_type(&cycle[0], &cycle[1], atol),
                edge_sign_type(&cycle[1], &cycle[2], atol),
                edge_sign_type(&cycle[0], &cycle[2], atol),
            ];
            if signs.iter().all(|&s| s == EdgeSignType::Same) {
                LanyardType::Sail
            } else {
                LanyardType::TrayRack
            }
        }
        5 => LanyardType::Quincunx,
        _ => LanyardType::BicycleChain,
    }
}

/// Census of lanyard types across all box-kites at dim=16.
///
/// Returns counts of each lanyard type found among the triangular faces.
pub fn lanyard_census_dim16() -> HashMap<LanyardType, usize> {
    let bks = find_box_kites(16, 1e-10);
    let mut census: HashMap<LanyardType, usize> = HashMap::new();

    for bk in &bks {
        let racks = tray_racks(bk);
        for rack in &racks {
            let assessors: Vec<Assessor> =
                rack.assessors.iter().map(|&i| bk.assessors[i]).collect();
            let ltype = classify_lanyard(&assessors);
            *census.entry(ltype).or_insert(0) += 1;
        }
    }

    census
}
