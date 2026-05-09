//! `F_4` root system: 24 positive roots in 4D Euclidean space.
//!
//! `F_4` is the smallest non-simply-laced exceptional Lie algebra (rank 4,
//! dim 52, 48 roots; 24 positive). The Dynkin diagram has a double bond
//! between `alpha_2` (long) and `alpha_3` (short):
//!
//! ```text
//!   alpha_1 (long) -- alpha_2 (long) ==> alpha_3 (short) -- alpha_4 (short)
//! ```
//!
//! Long roots have squared length 2, short roots have squared length 1
//! (math normalization). The 24 positive roots split as:
//! - 12 long: `e_i + e_j` and `e_i - e_j` for `1 <= i < j <= 4`
//! - 4 short: `e_i` for `1 <= i <= 4`
//! - 8 short: `(1/2)(e_1 +/- e_2 +/- e_3 +/- e_4)` (with `e_1` always positive)

/// Returns the 24 positive roots of `F_4` in 4D Euclidean coordinates.
pub fn f4_positive_roots() -> Vec<[f64; 4]> {
    let mut positive_roots: Vec<[f64; 4]> = Vec::with_capacity(24);

    // 12 long roots: e_i ± e_j with i < j
    for i in 0..4 {
        for j in (i + 1)..4 {
            let mut r1 = [0.0; 4];
            r1[i] = 1.0;
            r1[j] = 1.0;
            positive_roots.push(r1);

            let mut r2 = [0.0; 4];
            r2[i] = 1.0;
            r2[j] = -1.0;
            positive_roots.push(r2);
        }
    }

    // 4 short roots: e_i
    for i in 0..4 {
        let mut r = [0.0; 4];
        r[i] = 1.0;
        positive_roots.push(r);
    }

    // 8 short roots: 1/2 (e_1 +/- e_2 +/- e_3 +/- e_4) with e_1 = +1/2
    for s2 in [-1.0, 1.0] {
        for s3 in [-1.0, 1.0] {
            for s4 in [-1.0, 1.0] {
                positive_roots.push([0.5, 0.5 * s2, 0.5 * s3, 0.5 * s4]);
            }
        }
    }

    debug_assert_eq!(positive_roots.len(), 24);
    positive_roots
}

/// Four `F_4` simple roots (Bourbaki labeling).
///
/// `(alpha_1, alpha_2)` are long (squared length 2); `(alpha_3, alpha_4)` are
/// short (squared length 1). The double bond between `alpha_2` and `alpha_3`
/// makes the Cartan asymmetric.
pub fn f4_simple_roots() -> [[f64; 4]; 4] {
    [
        [0.0, 1.0, -1.0, 0.0],   // alpha_1 (long)
        [0.0, 0.0, 1.0, -1.0],   // alpha_2 (long)
        [0.0, 0.0, 0.0, 1.0],    // alpha_3 (short)
        [0.5, -0.5, -0.5, -0.5], // alpha_4 (short)
    ]
}

/// `F_4` Cartan matrix (Bourbaki). Asymmetric due to the double bond at
/// `(alpha_2, alpha_3)`: `A_23 = -2`, `A_32 = -1`.
pub fn f4_cartan_matrix() -> [[i32; 4]; 4] {
    [
        [2, -1, 0, 0],
        [-1, 2, -2, 0],
        [0, -1, 2, -1],
        [0, 0, -1, 2],
    ]
}

/// Order of the Weyl group `|W(F_4)| = 1152 = 2^7 * 3^2`.
pub fn f4_weyl_group_order() -> u64 {
    1152
}

/// Number of positive roots: 24.
pub fn f4_positive_root_count() -> usize {
    24
}

/// Total root count: 48.
pub fn f4_root_count() -> usize {
    48
}

/// Dimension of `F_4`: rank + #roots = 4 + 48 = 52.
pub fn f4_dim() -> usize {
    52
}

/// Weyl vector `rho = (1/2) sum of positive roots`. For `F_4` this is
/// `(11/2, 5/2, 3/2, 1/2)`.
pub fn weyl_vector() -> [f64; 4] {
    let mut sum = [0.0; 4];
    for r in f4_positive_roots() {
        for i in 0..4 {
            sum[i] += r[i];
        }
    }
    let mut rho = [0.0; 4];
    for i in 0..4 {
        rho[i] = sum[i] / 2.0;
    }
    rho
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f4_has_24_positive_roots() {
        assert_eq!(f4_positive_roots().len(), 24);
        assert_eq!(f4_positive_root_count(), 24);
    }

    #[test]
    fn long_roots_have_squared_length_two_short_have_one() {
        let roots = f4_positive_roots();
        for (i, r) in roots.iter().enumerate() {
            let n2: f64 = r.iter().map(|x| x * x).sum();
            let is_long_or_short = (n2 - 2.0).abs() < 1e-12 || (n2 - 1.0).abs() < 1e-12;
            assert!(
                is_long_or_short,
                "root {i} has squared length {n2}, expected 1 or 2",
            );
        }
    }

    #[test]
    fn long_root_count_is_twelve_short_count_is_twelve() {
        let roots = f4_positive_roots();
        let long: usize = roots
            .iter()
            .filter(|r| (r.iter().map(|x| x * x).sum::<f64>() - 2.0).abs() < 1e-12)
            .count();
        let short: usize = roots
            .iter()
            .filter(|r| (r.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() < 1e-12)
            .count();
        assert_eq!(long, 12);
        assert_eq!(short, 12);
    }

    /// Cartan derivation `A_ij = 2 (alpha_i, alpha_j) / (alpha_j, alpha_j)`,
    /// with the asymmetric denominator `(alpha_j, alpha_j)` accounting for the
    /// non-simply-laced double bond.
    #[test]
    fn cartan_matrix_derives_from_simple_roots() {
        let simple = f4_simple_roots();
        let cartan = f4_cartan_matrix();
        for i in 0..4 {
            let alpha_i = &simple[i];
            for j in 0..4 {
                let alpha_j = &simple[j];
                let dot_jj: f64 = alpha_j.iter().map(|x| x * x).sum();
                let dot_ij: f64 = alpha_i.iter().zip(alpha_j.iter()).map(|(a, b)| a * b).sum();
                let derived = (2.0 * dot_ij / dot_jj).round() as i32;
                assert_eq!(
                    derived, cartan[i][j],
                    "F4 Cartan mismatch at ({i},{j})",
                );
            }
        }
    }

    #[test]
    fn double_bond_at_alpha_2_alpha_3() {
        let cartan = f4_cartan_matrix();
        assert_eq!(cartan[1][2], -2, "A_23 should be -2 (double bond)");
        assert_eq!(cartan[2][1], -1, "A_32 should be -1");
    }

    #[test]
    fn weyl_group_order_factorizes() {
        let n = f4_weyl_group_order();
        assert_eq!(n, 1152);
        assert_eq!(n, 2u64.pow(7) * 3u64.pow(2));
    }

    #[test]
    fn algebra_dimension_is_52() {
        assert_eq!(f4_dim(), 52);
    }

    #[test]
    fn weyl_vector_matches_textbook() {
        // F4: rho = (11/2, 5/2, 3/2, 1/2).
        let rho = weyl_vector();
        assert!((rho[0] - 11.0 / 2.0).abs() < 1e-12);
        assert!((rho[1] - 5.0 / 2.0).abs() < 1e-12);
        assert!((rho[2] - 3.0 / 2.0).abs() < 1e-12);
        assert!((rho[3] - 1.0 / 2.0).abs() < 1e-12);
    }
}
