//! `E_6` root system: 72 roots in 8D Euclidean space.
//!
//! Construction: the rank-6 sub-root-system of `E_8` orthogonal to the two
//! E8 simple roots `alpha_0 = e_0 - e_1` and `alpha_1 = e_1 - e_2`, leaving
//! six remaining E8 simple roots that span a 6D subspace of `R^8`. This
//! preserves the branch-at-node-4 numbering of
//! [`crate::lie::e8::root_system`].
//!
//! After dropping `alpha_0, alpha_1`, the remaining E8 simple roots
//! `alpha_2, ..., alpha_7` form an E6 Dynkin diagram with branch at
//! `alpha_4` (the original E8 branch node):
//!
//! ```text
//!   alpha_2 - alpha_3 - alpha_4 - alpha_5
//!                       |
//!                       alpha_6 - alpha_7
//! ```
//!
//! Counts (filtered from the 240 E8 roots by the two orthogonality conditions):
//! - 40 roots of E8 type 1 with positions in `{3, 4, 5, 6, 7}` (untouched by
//!   the constraint, so all `(±e_i ± e_j, 0_3)` permutations are kept).
//! - 32 roots of E8 type 2 with `c_0 = c_1 = c_2` and adjusted parity.
//! - Total: 72.

use crate::lie::e8::root_system::{E8Root, generate_e8_roots};

/// Generate all 72 E6 roots as a filtered subset of the E8 roots.
///
/// An E8 root is in E6 iff its inner product with both
/// `alpha_0 = e_0 - e_1` and `alpha_1 = e_1 - e_2` vanishes, i.e.
/// `c_0 = c_1` and `c_1 = c_2`.
pub fn generate_e6_roots() -> Vec<E8Root> {
    let roots: Vec<E8Root> = generate_e8_roots()
        .into_iter()
        .filter(|r| {
            (r.coords[0] - r.coords[1]).abs() < 1e-12
                && (r.coords[1] - r.coords[2]).abs() < 1e-12
        })
        .collect();
    assert_eq!(roots.len(), 72, "E6 must have exactly 72 roots");
    roots
}

/// Six E6 simple roots, taken as the E8 simple roots `alpha_2, ..., alpha_7`
/// (relabelled `beta_1, ..., beta_6` to keep the conventional 1-based naming).
pub fn e6_simple_roots() -> [E8Root; 6] {
    [
        E8Root::new([0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0]), // beta_1 (= alpha_2)
        E8Root::new([0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0]), // beta_2 (= alpha_3)
        E8Root::new([0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0]), // beta_3 (= alpha_4, branch)
        E8Root::new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0]), // beta_4 (= alpha_5)
        E8Root::new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]),  // beta_5 (= alpha_6)
        E8Root::new([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5]), // beta_6 (= alpha_7)
    ]
}

/// E6 Cartan matrix derived from the simple roots.
///
/// Branch at `beta_3` (degree 3, connects to `beta_2, beta_4, beta_5`).
/// Dynkin: `beta_1 - beta_2 - beta_3 - beta_4` with `beta_3 - beta_5 - beta_6`.
pub fn e6_cartan_matrix() -> [[i32; 6]; 6] {
    [
        [2, -1, 0, 0, 0, 0],
        [-1, 2, -1, 0, 0, 0],
        [0, -1, 2, -1, -1, 0], // branch (degree 3)
        [0, 0, -1, 2, 0, 0],
        [0, 0, -1, 0, 2, -1],
        [0, 0, 0, 0, -1, 2],
    ]
}

/// Order of the Weyl group `|W(E_6)| = 51840 = 2^7 * 3^4 * 5`.
pub fn e6_weyl_group_order() -> u64 {
    51_840
}

/// Number of positive roots: 36 (= 72 / 2).
pub fn e6_positive_root_count() -> usize {
    36
}

/// Dimension of `E_6` as a Lie algebra: `rank + #roots = 6 + 72 = 78`.
pub fn e6_dim() -> usize {
    78
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e6_has_72_roots() {
        assert_eq!(generate_e6_roots().len(), 72);
    }

    #[test]
    fn every_e6_root_lives_in_e6_subspace() {
        for r in generate_e6_roots() {
            assert!((r.coords[0] - r.coords[1]).abs() < 1e-12);
            assert!((r.coords[1] - r.coords[2]).abs() < 1e-12);
        }
    }

    #[test]
    fn every_e6_root_has_norm_squared_two() {
        for r in generate_e6_roots() {
            assert!(r.is_valid_root());
        }
    }

    #[test]
    fn simple_roots_have_norm_squared_two() {
        for (i, r) in e6_simple_roots().iter().enumerate() {
            assert!(r.is_valid_root(), "simple root {i} has bad norm");
        }
    }

    /// Cartan matrix derives from the simple roots.
    #[test]
    fn cartan_matrix_derives_from_simple_roots() {
        let simple = e6_simple_roots();
        let cartan = e6_cartan_matrix();
        for (i, alpha_i) in simple.iter().enumerate() {
            let dot_ii = alpha_i.inner_product(alpha_i);
            for (j, alpha_j) in simple.iter().enumerate() {
                let dot_ij = alpha_i.inner_product(alpha_j);
                let derived = (2.0 * dot_ij / dot_ii).round() as i32;
                assert_eq!(
                    derived, cartan[i][j],
                    "E6 Cartan mismatch at ({i},{j})",
                );
            }
        }
    }

    #[test]
    fn weyl_group_order_factorizes() {
        let n = e6_weyl_group_order();
        assert_eq!(n, 51_840);
        assert_eq!(n, 2u64.pow(7) * 3u64.pow(4) * 5);
    }

    #[test]
    fn algebra_dimension_is_78() {
        assert_eq!(e6_dim(), 78);
    }

    #[test]
    fn positive_root_count_is_half_of_total() {
        assert_eq!(e6_positive_root_count(), 36);
        assert_eq!(generate_e6_roots().len() / 2, e6_positive_root_count());
    }

    /// Every E6 root is also an E8 root (E6 is a sub-root-system).
    #[test]
    fn e6_is_subset_of_e8() {
        let e8 = generate_e8_roots();
        let e8_set: std::collections::HashSet<_> = e8
            .iter()
            .map(|r| {
                r.coords
                    .iter()
                    .map(|x| (x * 2.0).round() as i32)
                    .collect::<Vec<_>>()
            })
            .collect();
        for r in generate_e6_roots() {
            let key: Vec<i32> = r.coords.iter().map(|x| (x * 2.0).round() as i32).collect();
            assert!(
                e8_set.contains(&key),
                "E6 root {:?} not found in E8 root set",
                r.coords
            );
        }
    }
}
