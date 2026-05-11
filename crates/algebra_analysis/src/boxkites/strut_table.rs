//! Canonical A..F strut-table labeling for box-kites.
//!
//! Each box-kite octahedron has exactly 2 zigzag faces (triangles whose
//! three edges all carry Opposite-sign edge classification). The
//! canonical labeling picks the lexicographically smaller zigzag face
//! as (A, B, C) and derives (D, E, F) as the octahedral opposites
//! F = opp(A), E = opp(B), D = opp(C). The strut pairs are then
//! (A, F), (B, E), (C, D).

use std::collections::HashSet;

use super::{Assessor, BoxKite, EdgeSignType, edge_sign_type};

/// Deterministic A..F labeling for a box-kite's strut table.
///
/// - (a, b, c) form a zigzag face (all Opposite-sign edges)
/// - (d, e, f) form the opposite zigzag face
/// - Strut pairs: (a,f), (b,e), (c,d)
#[derive(Debug, Clone)]
pub struct StrutTable {
    pub a: Assessor,
    pub b: Assessor,
    pub c: Assessor,
    pub d: Assessor,
    pub e: Assessor,
    pub f: Assessor,
}

/// Compute the canonical strut table labeling for a box-kite.
///
/// Each box-kite has exactly 2 zigzag faces (triangles with all Opposite-sign
/// edges). We pick the lexicographically smaller one as (A,B,C) and derive
/// (D,E,F) as their octahedral opposites: F=opp(A), E=opp(B), D=opp(C).
///
/// # Panics
/// Panics if the box-kite structure is invalid (wrong face counts or
/// non-zigzag opposite face).
pub fn canonical_strut_table(bk: &BoxKite, atol: f64) -> StrutTable {
    let nodes = &bk.assessors;
    assert_eq!(nodes.len(), 6, "Box-kite must have 6 assessors");

    // Build edge set with both directions for O(1) lookup
    let edge_set: HashSet<(usize, usize)> = bk
        .edges
        .iter()
        .flat_map(|&(a, b)| [(a, b), (b, a)])
        .collect();

    let adjacent = |i: usize, j: usize| -> bool { edge_set.contains(&(i, j)) };

    // Find the unique opposite (non-neighbor) for each vertex
    let mut opposite = [0usize; 6];
    for (i, opp) in opposite.iter_mut().enumerate() {
        let non_neighbors: Vec<usize> = (0..6).filter(|&j| j != i && !adjacent(i, j)).collect();
        assert_eq!(
            non_neighbors.len(),
            1,
            "Expected unique opposite for vertex {}, got {:?}",
            i,
            non_neighbors
        );
        *opp = non_neighbors[0];
    }

    // Find all 8 triangular faces and identify the 2 zigzag faces
    let mut zigzag_faces: Vec<[usize; 3]> = Vec::new();
    for i in 0..6 {
        for j in (i + 1)..6 {
            if !adjacent(i, j) {
                continue;
            }
            for k in (j + 1)..6 {
                if adjacent(i, k) && adjacent(j, k) {
                    let signs = [
                        edge_sign_type(&nodes[i], &nodes[j], atol),
                        edge_sign_type(&nodes[j], &nodes[k], atol),
                        edge_sign_type(&nodes[i], &nodes[k], atol),
                    ];
                    if signs.iter().all(|&s| s == EdgeSignType::Opposite) {
                        zigzag_faces.push([i, j, k]);
                    }
                }
            }
        }
    }

    assert_eq!(
        zigzag_faces.len(),
        2,
        "Expected exactly 2 zigzag faces, got {}",
        zigzag_faces.len()
    );

    // Pick the lexicographically smaller face (by sorted assessor tuples)
    let face_key = |face: &[usize; 3]| -> Vec<(usize, usize)> {
        let mut keys: Vec<(usize, usize)> = face
            .iter()
            .map(|&i| (nodes[i].low, nodes[i].high))
            .collect();
        keys.sort();
        keys
    };

    let abc_face = if face_key(&zigzag_faces[0]) < face_key(&zigzag_faces[1]) {
        zigzag_faces[0]
    } else {
        zigzag_faces[1]
    };

    let a_idx = abc_face[0];
    let b_idx = abc_face[1];
    let c_idx = abc_face[2];
    let f_idx = opposite[a_idx];
    let e_idx = opposite[b_idx];
    let d_idx = opposite[c_idx];

    // Verify the opposite face is also zigzag
    let opp_signs = [
        edge_sign_type(&nodes[d_idx], &nodes[e_idx], atol),
        edge_sign_type(&nodes[e_idx], &nodes[f_idx], atol),
        edge_sign_type(&nodes[d_idx], &nodes[f_idx], atol),
    ];
    assert!(
        opp_signs.iter().all(|&s| s == EdgeSignType::Opposite),
        "Derived opposite face is not a zigzag"
    );

    StrutTable {
        a: nodes[a_idx],
        b: nodes[b_idx],
        c: nodes[c_idx],
        d: nodes[d_idx],
        e: nodes[e_idx],
        f: nodes[f_idx],
    }
}
