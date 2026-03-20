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
    let o1 = vec![0, 1, 2, 3, 4, 5, 6, 7];    // Standard Octonions
    let o2 = vec![0, 1, 2, 3, 8, 9, 10, 11];   // Second generation
    let o3 = vec![0, 1, 2, 3, 12, 13, 14, 15];// Third generation
    (o1, o2, o3)
}

/// Returns the basis indices for the five disjoint quaternion subalgebras.
pub fn get_quaternion_subalgebras() -> QuaternionSubalgebras {
    let q_gamma = vec![0, 1, 2, 3];        // Spacetime
    let q_theta = vec![0, 4, 8, 12];       // Pseudo-time / Internal
    let q_u = vec![0, 5, 10, 15];          // 1st Generation (U-type)
    let q_v = vec![0, 6, 11, 13];          // 2nd Generation (V-type)
    let q_w = vec![0, 7, 9, 14];           // 3rd Generation (W-type)
    (q_gamma, q_theta, q_u, q_v, q_w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_quaternion_subalgebras_are_disjoint() {
        let (qg, qt, qu, qv, qw) = get_quaternion_subalgebras();
        let all_imaginary: Vec<usize> = qg.iter().skip(1).chain(qt.iter().skip(1))
            .chain(qu.iter().skip(1))
            .chain(qv.iter().skip(1))
            .chain(qw.iter().skip(1))
            .copied().collect();

        assert_eq!(all_imaginary.len(), 15, "There should be 15 total imaginary units.");

        let unique_units: HashSet<usize> = all_imaginary.into_iter().collect();
        assert_eq!(unique_units.len(), 15, "All imaginary units must be unique, proving disjointness.");

        println!("✅ The 5 quaternion subalgebras are disjoint.");
    }
}
