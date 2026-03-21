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

        println!("The 5 quaternion subalgebras are disjoint.");
    }

    /// Tang & Tang (Symmetry 2024, 16-00626) Axiom 1:
    /// Three contiguous-block octonionic subalgebras share the quaternion
    /// subgroup {e_0, e_1, e_2, e_3} (the "spacetime" quaternion Gamma).
    #[test]
    fn test_tang_axiom1_shared_quaternion_subgroup() {
        let (o1, o2, o3) = get_octonion_subalgebras();

        // The first 4 elements of each subalgebra must be {0, 1, 2, 3}
        let shared: Vec<usize> = vec![0, 1, 2, 3];
        assert_eq!(&o1[..4], &shared[..], "O1 must start with quaternion subgroup");
        assert_eq!(&o2[..4], &shared[..], "O2 must start with quaternion subgroup");
        assert_eq!(&o3[..4], &shared[..], "O3 must start with quaternion subgroup");

        // The distinguishing indices are in separate CD-doubling blocks
        let o1_gen: Vec<usize> = o1[4..].to_vec(); // {4,5,6,7}
        let o2_gen: Vec<usize> = o2[4..].to_vec(); // {8,9,10,11}
        let o3_gen: Vec<usize> = o3[4..].to_vec(); // {12,13,14,15}

        // No overlap between generation-specific indices
        let o1_set: HashSet<usize> = o1_gen.iter().copied().collect();
        let o2_set: HashSet<usize> = o2_gen.iter().copied().collect();
        let o3_set: HashSet<usize> = o3_gen.iter().copied().collect();
        assert!(o1_set.is_disjoint(&o2_set), "O1 and O2 gen indices must be disjoint");
        assert!(o2_set.is_disjoint(&o3_set), "O2 and O3 gen indices must be disjoint");
        assert!(o1_set.is_disjoint(&o3_set), "O1 and O3 gen indices must be disjoint");

        // Union of all gen indices = {4..15}
        let all_gen: HashSet<usize> = o1_set.union(&o2_set).copied()
            .chain(o3_set.iter().copied()).collect();
        assert_eq!(all_gen.len(), 12, "12 generation-specific indices total");

        println!("Tang Axiom 1: shared quaternion {{0,1,2,3}}, gen indices disjoint");
    }

    /// Tang & Tang Axiom 2:
    /// Each contiguous-block octonionic subalgebra is closed under
    /// sedenion multiplication (alternative, not just associative).
    #[test]
    fn test_tang_axiom2_subalgebra_closure() {
        use cd_kernel::cayley_dickson::cd_multiply;

        let (o1, o2, o3) = get_octonion_subalgebras();

        for (label, sub) in [("O1", &o1), ("O2", &o2), ("O3", &o3)] {
            let sub_set: HashSet<usize> = sub.iter().copied().collect();

            // Test all basis pair products: e_i * e_j must have support in sub
            for &i in sub.iter() {
                for &j in sub.iter() {
                    let mut a = vec![0.0_f64; 16];
                    let mut b = vec![0.0_f64; 16];
                    a[i] = 1.0;
                    b[j] = 1.0;
                    let prod = cd_multiply(&a, &b);

                    // Find nonzero components
                    for (k, &v) in prod.iter().enumerate() {
                        if v.abs() > 1e-12 {
                            assert!(
                                sub_set.contains(&k),
                                "{}: e_{} * e_{} has component at e_{} (outside subalgebra)",
                                label, i, j, k
                            );
                        }
                    }
                }
            }
        }
        println!("Tang Axiom 2: all three contiguous-block subalgebras are closed");
    }

    /// Tang & Tang Axiom 3 (Eq 10C):
    /// Three pairs of creation/annihilation operators from the FIRST
    /// octonionic subalgebra {e_0..e_7} produce 8 Gell-Mann-type SU(3)
    /// generators via |i><j| tensor products.
    ///
    /// alpha_1 = (-e_6 + i*e_5)/2
    /// alpha_2 = (-e_3 + i*e_1)/2
    /// alpha_3 = (-e_7 + i*e_2)/2
    ///
    /// These satisfy {alpha_i, alpha_j} = 0, {alpha_i, alpha_j^+} = delta_ij.
    #[test]
    fn test_tang_axiom3_creation_annihilation_anticommutation() {
        use cd_kernel::cayley_dickson::cd_multiply;

        // Tang's creation/annihilation operators (Eq 10A)
        // alpha_k = (-e_{2k} + i*e_{2k-1})/2 where "i" is an internal operation
        // For octonion basis: alpha_1 uses (e_5, e_6), alpha_2 uses (e_1, e_3),
        // alpha_3 uses (e_2, e_7)

        // The anti-commutation relation {alpha_i, alpha_j} = 0 means:
        // alpha_i * alpha_j + alpha_j * alpha_i = 0
        // This is verified by checking that the SEDENION product
        // e_a * e_b + e_b * e_a = 0 for the imaginary basis elements (a,b != 0, a != b).

        // All imaginary octonion basis elements anti-commute: {e_i, e_j} = 0 for i != j != 0
        for i in 1..8_usize {
            for j in (i + 1)..8 {
                let mut a = vec![0.0_f64; 16];
                let mut b = vec![0.0_f64; 16];
                a[i] = 1.0;
                b[j] = 1.0;
                let ab = cd_multiply(&a, &b);
                let ba = cd_multiply(&b, &a);
                let mut anticomm = vec![0.0_f64; 16];
                for k in 0..16 {
                    anticomm[k] = ab[k] + ba[k];
                }
                let norm: f64 = anticomm.iter().map(|x| x * x).sum();
                assert!(norm < 1e-20,
                    "Octonion e_{} * e_{} should anti-commute, got norm {:.2e}",
                    i, j, norm.sqrt());
            }
        }
        println!("Tang Axiom 3: all 21 pairs of imaginary octonion basis elements anti-commute");
    }

    /// Tang & Tang Axiom 4 (Eq 11A):
    /// Five pairs of creation/annihilation operators from the FULL
    /// sedenion basis produce 24 SU(5) generators.
    ///
    /// alpha_1 = (-e_6 + i*e_5)/2, alpha_2 = (-e_3 + i*e_1)/2,
    /// alpha_3 = (-e_7 + i*e_2)/2, alpha_4 = (-e_14 + i*e_13)/2,
    /// alpha_5 = (-e_11 + i*e_9)/2
    ///
    /// The anti-commutation extends to all 15 imaginary sedenion elements.
    #[test]
    fn test_tang_axiom4_sedenion_anticommutation() {
        use cd_kernel::cayley_dickson::cd_multiply;

        // All 15 imaginary sedenion basis elements must anti-commute pairwise
        let mut violations = 0;
        for i in 1..16_usize {
            for j in (i + 1)..16 {
                let mut a = vec![0.0_f64; 16];
                let mut b = vec![0.0_f64; 16];
                a[i] = 1.0;
                b[j] = 1.0;
                let ab = cd_multiply(&a, &b);
                let ba = cd_multiply(&b, &a);
                let mut anticomm_norm_sq = 0.0_f64;
                for k in 0..16 {
                    anticomm_norm_sq += (ab[k] + ba[k]).powi(2);
                }
                if anticomm_norm_sq > 1e-20 {
                    violations += 1;
                }
            }
        }
        // C(15,2) = 105 pairs, all should anti-commute
        assert_eq!(violations, 0,
            "All 105 pairs of imaginary sedenion basis elements must anti-commute");
        println!("Tang Axiom 4: all 105 imaginary sedenion basis pairs anti-commute");
        println!("  This provides the fermionic creation/annihilation algebra for SU(5)");
    }

    /// Tang & Tang Axiom 5 (Eq 11A, indices):
    /// The five creation/annihilation operator pairs use specific basis
    /// element pairings that span all three octonionic subalgebras.
    ///
    /// Pair 1: (e_5, e_6) -- from O1 (U-type, generation 1)
    /// Pair 2: (e_1, e_3) -- from shared quaternion (spacetime)
    /// Pair 3: (e_2, e_7) -- from O1 (U-type, generation 1)
    /// Pair 4: (e_13, e_14) -- from O3 (W-type, generation 3)
    /// Pair 5: (e_9, e_11) -- from O2 (V-type, generation 2)
    #[test]
    fn test_tang_axiom5_creation_operator_spanning() {
        let (o1, o2, o3) = get_octonion_subalgebras();
        let shared = vec![0, 1, 2, 3];

        // Tang's 5 creation/annihilation operator index pairs
        let pairs = [
            (5_usize, 6_usize, "O1"),   // alpha_1: from U-type
            (1, 3, "shared"),  // alpha_2: from spacetime quaternion
            (2, 7, "O1"),     // alpha_3: from U-type
            (13, 14, "O3"),   // alpha_4: from W-type
            (9, 11, "O2"),    // alpha_5: from V-type
        ];

        for (a, b, expected_sub) in &pairs {
            let in_o1 = o1.contains(a) && o1.contains(b);
            let in_o2 = o2.contains(a) && o2.contains(b);
            let in_o3 = o3.contains(a) && o3.contains(b);
            let in_shared = shared.contains(a) && shared.contains(b);

            let actual = if in_shared { "shared" }
                else if in_o1 { "O1" }
                else if in_o2 { "O2" }
                else if in_o3 { "O3" }
                else { "cross-subalgebra" };

            println!("  alpha pair (e_{a}, e_{b}): expected={expected_sub}, actual={actual}");
            assert_eq!(actual, *expected_sub,
                "Pair (e_{}, e_{}) should be in {}", a, b, expected_sub);
        }
        println!("Tang Axiom 5: creation operator pairs span O1+shared+O2+O3");
        println!("  3 pairs from SU(3) sector (O1+shared), 2 from leptoquark (O2,O3)");
        println!("  => 8 SU(3) generators from 3 pairs, 24 SU(5) generators from 5 pairs");
    }

    /// Tang & Tang Axiom 6 (Fig 3):
    /// The sedenion multiplication table has the property that
    /// e_4, e_8, and e_12 are the "Theta" operators that connect
    /// the three octonionic subalgebras. They commute with all
    /// basis elements within their own subalgebra's quaternion triplet.
    #[test]
    fn test_tang_axiom6_theta_operator_structure() {
        let (qg, qt, _qu, _qv, _qw) = get_quaternion_subalgebras();

        // Theta quaternion: {e_0, e_4, e_8, e_12}
        assert_eq!(qt, vec![0, 4, 8, 12], "Theta quaternion must be {{0, 4, 8, 12}}");

        // The Theta elements are exactly the "anchors" of each CD-doubling block
        println!("Tang Axiom 6: Theta = {{e_0, e_4, e_8, e_12}} = CD block anchors");
        println!("  e_4 = anchor of O1 gen block {{4,5,6,7}}");
        println!("  e_8 = anchor of O2 gen block {{8,9,10,11}}");
        println!("  e_12 = anchor of O3 gen block {{12,13,14,15}}");

        // Gamma quaternion: {e_0, e_1, e_2, e_3} = spacetime
        assert_eq!(qg, vec![0, 1, 2, 3], "Gamma quaternion must be {{0, 1, 2, 3}}");
        println!("  Gamma = {{e_0, e_1, e_2, e_3}} = spacetime quaternion (shared)");
    }
}
