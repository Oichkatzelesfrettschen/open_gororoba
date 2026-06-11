//! Definitions for the canonical subalgebras of the Sedenion algebra.
//!
//! Provides the basis indices for the three canonical octonionic subalgebras
//! and the five disjoint quaternion subalgebras, as described in the literature
//! connecting sedenions to the Standard Model and SU(5) GUT.
//!
//! # References
//! - Tang, Q., & Tang, J. (2023). Sedenion algebra for three lepton/quark
//!   generations and its relations to SU(5). arXiv:2308.14768.

type BasisIndices = Vec<usize>;
type OctonionSubalgebras = (BasisIndices, BasisIndices, BasisIndices);
type QuaternionSubalgebras = (
    BasisIndices,
    BasisIndices,
    BasisIndices,
    BasisIndices,
    BasisIndices,
);

/// Returns the basis indices for the three canonical octonionic subalgebras.
pub fn get_octonion_subalgebras() -> OctonionSubalgebras {
    let o1 = vec![0, 1, 2, 3, 4, 5, 6, 7]; // Standard Octonions
    let o2 = vec![0, 1, 2, 3, 8, 9, 10, 11]; // Second generation
    let o3 = vec![0, 1, 2, 3, 12, 13, 14, 15]; // Third generation
    (o1, o2, o3)
}

/// Returns the basis indices for the five disjoint quaternion subalgebras.
pub fn get_quaternion_subalgebras() -> QuaternionSubalgebras {
    let q_gamma = vec![0, 1, 2, 3]; // Spacetime
    let q_theta = vec![0, 4, 8, 12]; // Pseudo-time / Internal
    let q_u = vec![0, 5, 10, 15]; // 1st Generation (U-type)
    let q_v = vec![0, 6, 11, 13]; // 2nd Generation (V-type)
    let q_w = vec![0, 7, 9, 14]; // 3rd Generation (W-type)
    (q_gamma, q_theta, q_u, q_v, q_w)
}

/// Standard (strict) associator: `[a,b,c] = (a*b)*c - a*(b*c)`.
/// Returns the norm of the associator vector.
pub fn assoc_strict(dim: usize, a: usize, b: usize, c: usize) -> f64 {
    use cd_kernel::cayley_dickson::cd_multiply;
    let mut ea = vec![0.0; dim];
    ea[a] = 1.0;
    let mut eb = vec![0.0; dim];
    eb[b] = 1.0;
    let mut ec = vec![0.0; dim];
    ec[c] = 1.0;
    let ab = cd_multiply(&ea, &eb);
    let ab_c = cd_multiply(&ab, &ec);
    let bc = cd_multiply(&eb, &ec);
    let a_bc = cd_multiply(&ea, &bc);
    ab_c.iter()
        .zip(a_bc.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Wilmot triple associator: `T(b,c,d) = [b,d,c] - [d,c,b] + [c,b,d]`.
///
/// From Wilmot (arXiv:2505.11747, Sec 3): T = 0 defines "associative" triads
/// in Wilmot's classification. This is WEAKER than strict associativity
/// (all individual `[x,y,z] = 0`). A triad can be Wilmot-associative while
/// having nonzero individual associators.
///
/// Returns the norm of the triple associator vector.
pub fn assoc_wilmot(dim: usize, b: usize, c: usize, d: usize) -> f64 {
    use cd_kernel::cayley_dickson::cd_multiply;
    let mut eb = vec![0.0; dim];
    eb[b] = 1.0;
    let mut ec = vec![0.0; dim];
    ec[c] = 1.0;
    let mut ed = vec![0.0; dim];
    ed[d] = 1.0;

    // [b,d,c] = (b*d)*c - b*(d*c)
    let bd = cd_multiply(&eb, &ed);
    let bdc = cd_multiply(&bd, &ec);
    let dc = cd_multiply(&ed, &ec);
    let b_dc = cd_multiply(&eb, &dc);

    // [d,c,b] = (d*c)*b - d*(c*b)
    let dc_b = cd_multiply(&dc, &eb);
    let cb = cd_multiply(&ec, &eb);
    let d_cb = cd_multiply(&ed, &cb);

    // [c,b,d] = (c*b)*d - c*(b*d)
    let cb_d = cd_multiply(&cb, &ed);
    let c_bd = cd_multiply(&ec, &bd);

    // T = [b,d,c] - [d,c,b] + [c,b,d]
    let mut t = vec![0.0; dim];
    for i in 0..dim {
        t[i] = (bdc[i] - b_dc[i]) - (dc_b[i] - d_cb[i]) + (cb_d[i] - c_bd[i]);
    }
    t.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// Test block relocated to sibling sedenion_subalgebras/tests.rs.
// The 2960-line cfg(test) section made the parent file 3062 lines
// with only 100 lines of production code -- moving tests out keeps
// the parent's surface area focused on the actual API.
#[cfg(test)]
mod tests;
