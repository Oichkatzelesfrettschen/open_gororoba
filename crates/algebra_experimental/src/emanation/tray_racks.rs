//! Tray-racks (triangular faces of a box-kite octahedron) and their
//! twist-product analysis.
//!
//! - `TwistType` distinguishes Zigzag (all Opposite-sign edges) from
//!   Trefoil (mixed sign edges). A box-kite has exactly 2 Zigzag and
//!   6 Trefoil faces.
//! - `TrayRack` records the three assessor indices into a box-kite's
//!   assessor list plus the face's twist classification.
//! - `tray_racks(bk)` enumerates all 8 triangular faces with classification.
//! - `TwistProductEntry` is one (ordered_pair, sign_solutions) row.
//! - `twist_products(tr, bk)` computes the diagonal zero-product
//!   solutions for every ordered pair of assessors in a tray-rack.

use std::collections::HashSet;

use algebra_analysis::boxkites::{
    BoxKite, EdgeSignType, all_diagonal_zero_products, edge_sign_type,
};

/// Classification of tray-rack twist type.
///
/// In a box-kite octahedron, the 8 triangular faces split into 2 zigzag faces
/// (all Opposite-sign edges) and 6 "trefoil" faces (mixed signs). The 4
/// non-zigzag faces that share an edge with a zigzag face are the tray-racks.
///
/// De Marrais identifies two twist types:
/// - Zigzag: all edges have Opposite sign (the 2 special faces)
/// - Trefoil: mixed Same/Opposite edges (the 6 remaining faces)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TwistType {
    /// All-opposite-sign edges (zigzag face).
    Zigzag,
    /// Mixed sign edges (trefoil face).
    Trefoil,
}

/// A tray-rack: a triangular face of the box-kite octahedron with its twist type.
#[derive(Debug, Clone)]
pub struct TrayRack {
    /// The 3 assessor indices (into the box-kite's assessor list).
    pub assessors: [usize; 3],
    /// Twist classification.
    pub twist_type: TwistType,
}

/// Extract all 8 triangular faces from a box-kite, classified by twist type.
///
/// Returns (zigzag_faces, trefoil_faces). A properly structured box-kite has
/// exactly 2 zigzag and 6 trefoil faces.
pub fn tray_racks(bk: &BoxKite) -> Vec<TrayRack> {
    let n = bk.assessors.len();
    assert_eq!(n, 6);
    let atol = 1e-10;

    let edge_set: HashSet<(usize, usize)> = bk
        .edges
        .iter()
        .flat_map(|&(a, b)| [(a, b), (b, a)])
        .collect();

    let mut racks = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            if !edge_set.contains(&(i, j)) {
                continue;
            }
            for k in (j + 1)..n {
                if !edge_set.contains(&(i, k)) || !edge_set.contains(&(j, k)) {
                    continue;
                }

                // Classify by edge signs
                let signs = [
                    edge_sign_type(&bk.assessors[i], &bk.assessors[j], atol),
                    edge_sign_type(&bk.assessors[j], &bk.assessors[k], atol),
                    edge_sign_type(&bk.assessors[i], &bk.assessors[k], atol),
                ];

                let twist = if signs.iter().all(|&s| s == EdgeSignType::Opposite) {
                    TwistType::Zigzag
                } else {
                    TwistType::Trefoil
                };

                racks.push(TrayRack {
                    assessors: [i, j, k],
                    twist_type: twist,
                });
            }
        }
    }

    racks
}

/// A twist product result: an ordered assessor pair and its ZD sign solutions.
pub type TwistProductEntry = ((usize, usize), Vec<(i8, i8)>);

/// Compute twist products for a tray-rack face.
///
/// For each ordered pair of assessors in the tray-rack (6 ordered pairs from
/// 3 assessors), compute the diagonal zero-product solutions.
/// Returns the sign-pair solutions for each ordered pair.
pub fn twist_products(tr: &TrayRack, bk: &BoxKite) -> Vec<TwistProductEntry> {
    let atol = 1e-10;
    let mut results = Vec::new();

    for &i in &tr.assessors {
        for &j in &tr.assessors {
            if i == j {
                continue;
            }
            let sols = all_diagonal_zero_products(&bk.assessors[i], &bk.assessors[j], atol);
            if !sols.is_empty() {
                results.push(((i, j), sols));
            }
        }
    }

    results
}
