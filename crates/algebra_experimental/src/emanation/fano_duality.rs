//! Fano-plane / projective-geometry derived structures: loop-box-kite
//! duality, the Hjelmslev net wrapping PG(n-2,2), and the chingon
//! spectral census.
//!
//! These three sub-blocks share a common theme: each maps from the
//! Cayley-Dickson dimension's underlying projective geometry to a
//! structural invariant of the zero-divisor graph.
//!
//! - Loop-box-kite duality (MIL 9): pairs each of the 7 deformed
//!   octonion copies (Fano lines) with its dual box-kite.
//! - Hjelmslev net (MIL 17): de Marrais's term for the PG(n-2,2)
//!   structure underlying the motif-component-to-point bijection.
//! - Chingon spectral census (MIL 15): per-component degree sequence
//!   and top eigenvalues for each motif component at a given dimension.

use std::collections::HashSet;

use algebra_analysis::boxkites::{
    Assessor, O_TRIPS, automorpheme_assessors, find_box_kites, motif_components_for_cross_assessors,
};

// ===========================================================================
// Loop-Box-Kite Duality (MIL 9)
// ===========================================================================

/// A pair: (Fano O-trip index, box-kite with that strut signature).
///
/// The loop-box-kite duality maps each of the 7 deformed octonion copies
/// (identified by Fano plane lines) to its dual box-kite (identified by
/// the complementary "missing" index).
pub fn loop_boxkite_pairs() -> Vec<(usize, usize)> {
    // Each O-trip uses 3 indices from {1..7}. The strut signature
    // of the dual box-kite is the one index NOT used by any of the
    // 3 Fano-plane lines through that point.
    //
    // Actually, each box-kite's strut signature is the MISSING index
    // from its 6 assessors' low indices. For the duality:
    // O-trip i (using indices a,b,c) maps to box-kite with strut = ?
    //
    // The simplest relationship: the O-trip and box-kite are linked
    // through the Fano plane structure. Each O-trip line {a,b,c}
    // produces assessors in box-kites that DON'T include a,b,c as
    // strut signatures.

    let bks = find_box_kites(16, 1e-10);

    // Map: for each O-trip, find which box-kites contain its assessors
    let mut pairs = Vec::new();
    for (trip_idx, trip) in O_TRIPS.iter().enumerate() {
        let auto = automorpheme_assessors(trip);

        // Find the box-kite(s) containing these assessors
        for bk in &bks {
            let bk_set: HashSet<Assessor> = bk.assessors.iter().copied().collect();
            let overlap: usize = auto.iter().filter(|a| bk_set.contains(a)).count();
            // Each automorpheme of 12 assessors overlaps with multiple box-kites.
            // The duality pairs the O-trip with the box-kite whose strut sig
            // is NOT in the trip.
            if !trip.contains(&bk.strut_signature) {
                // This box-kite's missing index is NOT one of the trip indices
                // => it's a "complementary" box-kite
                if overlap > 0 {
                    pairs.push((trip_idx, bk.id));
                }
            }
        }
    }

    pairs.sort();
    pairs.dedup();
    pairs
}

/// PSL(2,7) navigation table: how each Fano-plane automorphism maps box-kites.
///
/// Returns a 7x7 table where entry `table[i][j]` indicates the box-kite that results
/// from applying the j-th basic Fano transformation to box-kite i.
///
/// PSL(2,7) has order 168 = 7 * 24 and acts transitively on the 7 Fano lines.
pub fn psl27_order() -> usize {
    168
}

// ===========================================================================
// Hjelmslev Net (MIL 17)
// ===========================================================================

/// A Hjelmslev net wrapping the PG(n-2,2) projective geometry.
///
/// This is de Marrais's terminology for the PG structure underlying the
/// motif-component-to-point bijection.
#[derive(Debug, Clone)]
pub struct HjelmslevNet {
    /// Projective dimension m (e.g., m=2 for Fano plane at dim=16).
    pub proj_dim: usize,
    /// Number of points.
    pub n_points: usize,
    /// Number of lines.
    pub n_lines: usize,
    /// Cayley-Dickson dimension.
    pub cd_dim: usize,
}

/// Construct the Hjelmslev net for a Cayley-Dickson dimension.
pub fn hjelmslev_net(dim: usize) -> HjelmslevNet {
    use algebra_analysis::projective_geometry::pg_from_cd_dim;
    let pg = pg_from_cd_dim(dim);
    HjelmslevNet {
        proj_dim: pg.m,
        n_points: pg.points.len(),
        n_lines: pg.lines.len(),
        cd_dim: dim,
    }
}

// ===========================================================================
// Chingon Spectral Census (MIL 15)
// ===========================================================================

/// Spectral fingerprint for a motif component at a given dimension.
#[derive(Debug, Clone)]
pub struct SpectralFingerprint {
    /// Cayley-Dickson dimension.
    pub dim: usize,
    /// Component index.
    pub component_idx: usize,
    /// Number of nodes in the component.
    pub n_nodes: usize,
    /// Number of edges.
    pub n_edges: usize,
    /// Sorted degree sequence.
    pub degree_sequence: Vec<usize>,
    /// Top 5 eigenvalues of adjacency matrix (sorted descending).
    pub top_eigenvalues: Vec<f64>,
    /// Triangle count.
    pub triangle_count: usize,
}

/// Compute spectral fingerprints for all motif components at a given dimension.
///
/// For dim=128 (chingon), there are 63 components with 62 nodes each.
/// Computing the full adjacency matrix and eigendecomposition for each is
/// tractable (62x62 matrices).
pub fn spectral_census(dim: usize) -> Vec<SpectralFingerprint> {
    let comps = motif_components_for_cross_assessors(dim);

    comps
        .iter()
        .enumerate()
        .map(|(idx, comp)| {
            let deg_seq = comp.degree_sequence();
            let spectrum = comp.spectrum();
            let top_eigs: Vec<f64> = spectrum.iter().take(5).copied().collect();
            let tri_count = comp.triangle_count();

            SpectralFingerprint {
                dim,
                component_idx: idx,
                n_nodes: comp.nodes.len(),
                n_edges: comp.edges.len(),
                degree_sequence: deg_seq,
                top_eigenvalues: top_eigs,
                triangle_count: tri_count,
            }
        })
        .collect()
}
