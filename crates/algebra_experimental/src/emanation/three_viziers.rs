//! Three Vizier Relationships (L15b, de Marrais 2007).
//!
//! Each strut in a box-kite connects two non-adjacent assessors. Let
//! the first assessor be (v, V) where v = low index (1..7) and V = high
//! index (8..15), and the second be (z, Z). Then:
//!
//!   VZ1:  v ^ z = V ^ Z = S       (strut constant)
//!   VZ2:  Z ^ v = V ^ z = G       (generator = dim/2)
//!   VZ3:  V ^ v = z ^ Z = G ^ S   (within-assessor invariant)
//!
//! These three XORs form a Klein four-group {0, S, G, G^S} acting on
//! the index space. VZ3 is a per-assessor property (not just per-strut):
//! hi = lo ^ (G ^ S) for every assessor in a box-kite with strut
//! constant S.
//!
//! `vizier_xor_audit` generalizes the check to arbitrary
//! MotifComponents at higher CD dimensions, where the "strut constant"
//! may no longer be a single missing index.

use algebra_analysis::boxkites::BoxKite;

/// Result of verifying the Three Vizier XOR relationships on a box-kite.
#[derive(Debug, Clone)]
pub struct ThreeVizierResult {
    /// The strut constant S.
    pub strut_sig: usize,
    /// The generator G = dim/2.
    pub generator: usize,
    /// Number of struts checked.
    pub n_struts: usize,
    /// Whether VZ1 (v^z = V^Z = S) holds for all struts.
    pub vz1_holds: bool,
    /// Whether VZ2 (Z^v = V^z = G) holds for all struts.
    pub vz2_holds: bool,
    /// Whether VZ3 (V^v = z^Z = G^S) holds for all assessors.
    pub vz3_holds: bool,
    /// The within-assessor lo^hi values (should all be G^S).
    pub lo_hi_xor: usize,
}

/// Verify the Three Vizier XOR relationships on a box-kite.
///
/// For sedenions (dim=16), G=8.  Given a box-kite with strut constant S:
/// - VZ1: for each strut (v,V)--(z,Z), v^z = V^Z = S
/// - VZ2: for each strut (v,V)--(z,Z), Z^v = V^z = G
/// - VZ3: for every assessor (v,V), V^v = G^S
///
/// Returns a `ThreeVizierResult` capturing the verification outcome.
pub fn verify_three_viziers(bk: &BoxKite, dim: usize) -> ThreeVizierResult {
    let g = dim / 2;
    let s = bk.strut_signature;
    let x = g ^ s;

    // VZ3 check: every assessor should have hi ^ lo = G^S
    let vz3_holds = bk.assessors.iter().all(|a| a.low ^ a.high == x);

    // VZ1 + VZ2 check: per-strut
    let mut vz1_holds = true;
    let mut vz2_holds = true;
    for &(i, j) in &bk.struts {
        let a = &bk.assessors[i];
        let b = &bk.assessors[j];
        let v = a.low;
        let big_v = a.high;
        let z = b.low;
        let big_z = b.high;

        // VZ1: v^z = V^Z = S
        if v ^ z != s || big_v ^ big_z != s {
            vz1_holds = false;
        }
        // VZ2: Z^v = V^z = G
        if big_z ^ v != g || big_v ^ z != g {
            vz2_holds = false;
        }
    }

    ThreeVizierResult {
        strut_sig: s,
        generator: g,
        n_struts: bk.struts.len(),
        vz1_holds,
        vz2_holds,
        vz3_holds,
        lo_hi_xor: x,
    }
}

/// Check whether the Vizier XOR relationships hold for zero-product-adjacent
/// cross-assessor pairs in a `MotifComponent` at arbitrary dimension.
///
/// At dim=16 this reduces to the standard Three Vizier check.  At higher
/// dimensions, the "strut constant" is no longer a single missing index
/// but may generalize differently.  This function checks:
///
/// - VZ3 generalization: is lo^hi constant within the component?
/// - If so, what is that constant X, and does X = G^S for some S?
/// - VZ1 generalization: for each ZD-adjacent pair, does lo_a^lo_b = hi_a^hi_b?
///
/// Returns `None` if the component has no edges.
pub fn vizier_xor_audit(
    component: &algebra_analysis::boxkites::MotifComponent,
) -> Option<VizierXorAudit> {
    if component.edges.is_empty() {
        return None;
    }
    let dim = component.dim;
    let g = dim / 2;

    // VZ3 generalization: check if lo^hi is constant
    let lo_hi_xors: std::collections::BTreeSet<usize> =
        component.nodes.iter().map(|&(lo, hi)| lo ^ hi).collect();
    let vz3_constant = lo_hi_xors.len() == 1;
    let lo_hi_xor = if vz3_constant {
        *lo_hi_xors.iter().next().unwrap()
    } else {
        0
    };

    // VZ1 generalization: for each edge, does lo_a^lo_b = hi_a^hi_b?
    let mut vz1_all_match = true;
    let mut lo_xor_values: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for &((lo_a, hi_a), (lo_b, hi_b)) in &component.edges {
        let lo_xor = lo_a ^ lo_b;
        let hi_xor = hi_a ^ hi_b;
        if lo_xor != hi_xor {
            vz1_all_match = false;
        }
        lo_xor_values.insert(lo_xor);
    }
    let vz1_constant = vz1_all_match && lo_xor_values.len() == 1;

    // VZ2 generalization: for each edge, does hi_b^lo_a = hi_a^lo_b?
    let mut vz2_all_match = true;
    let mut vz2_values: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for &((lo_a, hi_a), (lo_b, hi_b)) in &component.edges {
        let cross1 = hi_b ^ lo_a;
        let cross2 = hi_a ^ lo_b;
        if cross1 != cross2 {
            vz2_all_match = false;
        }
        vz2_values.insert(cross1);
    }
    let vz2_constant = vz2_all_match && vz2_values.len() == 1;

    Some(VizierXorAudit {
        dim,
        n_nodes: component.nodes.len(),
        n_edges: component.edges.len(),
        vz3_constant,
        lo_hi_xor,
        inferred_s: if vz3_constant { g ^ lo_hi_xor } else { 0 },
        vz1_lo_eq_hi: vz1_all_match,
        vz1_constant,
        n_distinct_lo_xors: lo_xor_values.len(),
        vz2_cross_eq: vz2_all_match,
        vz2_constant,
        vz2_value: if vz2_constant {
            vz2_values.iter().next().copied()
        } else {
            None
        },
    })
}

/// Result of auditing Vizier XOR structure in a generalized motif component.
#[derive(Debug, Clone)]
pub struct VizierXorAudit {
    /// Cayley-Dickson dimension.
    pub dim: usize,
    /// Number of nodes (cross-assessor pairs) in the component.
    pub n_nodes: usize,
    /// Number of ZD-adjacent edges.
    pub n_edges: usize,
    /// Whether lo^hi is constant across all nodes (VZ3 generalization).
    pub vz3_constant: bool,
    /// The lo^hi value (meaningful only if vz3_constant is true).
    pub lo_hi_xor: usize,
    /// Inferred strut constant S = G ^ lo_hi_xor (meaningful only if vz3_constant).
    pub inferred_s: usize,
    /// Whether lo_a^lo_b = hi_a^hi_b for all edges (VZ1 symmetry).
    pub vz1_lo_eq_hi: bool,
    /// Whether lo_a^lo_b is constant across all edges (full VZ1).
    pub vz1_constant: bool,
    /// Number of distinct lo_a^lo_b values across edges.
    pub n_distinct_lo_xors: usize,
    /// Whether hi_b^lo_a = hi_a^lo_b for all edges (VZ2 symmetry).
    pub vz2_cross_eq: bool,
    /// Whether the VZ2 cross-XOR is constant across all edges.
    pub vz2_constant: bool,
    /// The VZ2 cross-XOR value (if constant).
    pub vz2_value: Option<usize>,
}
