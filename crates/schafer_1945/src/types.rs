//! Core types for Schafer's division algebra theory (Steps 11-14).
//!
//! Builds on hurwitz_1898 (composition theory) and dickson_1919 (CD construction)
//! to provide division algebra testing and the modified CD product.
//!
//! Mirrors: SchaferDivAlg16.v Parts I-IV.

use std::fmt;

use cd_kernel::cayley_dickson::cd_norm_sq;
use dickson_1919::CayleyDicksonLevel;

/// Result of testing whether a CD algebra is a division algebra.
///
/// Mirrors: SchaferDivAlg16.v schafer_negative_zd + schafer_eq2_singular_witness.
#[derive(Debug, Clone)]
pub struct DivisionTest {
    /// The CD level tested.
    pub level: CayleyDicksonLevel,
    /// Whether the algebra is a division algebra over R.
    pub is_division: bool,
    /// If not division: the zero-divisor witness pair (if found).
    pub zd_witness: Option<(Vec<f64>, Vec<f64>)>,
    /// Maximum norm violation found.
    pub norm_violation: f64,
    /// The Hurwitz classification for context.
    pub hurwitz: hurwitz_1898::HurwitzClassification,
}

impl fmt::Display for DivisionTest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} (norm_viol={:.2e}, hurwitz={})",
            self.level,
            if self.is_division {
                "DIVISION ALGEBRA"
            } else {
                "NOT division (has ZDs)"
            },
            self.norm_violation,
            if self.hurwitz.is_composition {
                "composition"
            } else {
                "not composition"
            },
        )
    }
}

/// A modified Cayley-Dickson algebra with non-scalar parameter g.
///
/// Schafer (1945) eq.1: (a+vb)(x+vy) = (ax + g*(y*bS)) + v(aS*y + xb)
/// where S is conjugation and g is an element of the base algebra.
///
/// Mirrors: SchaferDivAlg16.v Part I.
#[derive(Debug, Clone)]
pub struct ModifiedCDAlgebra {
    /// Dimension of the base algebra (half of the full algebra).
    pub base_dim: usize,
    /// The fixed element g in the base algebra.
    pub element_g: Vec<f64>,
}

impl ModifiedCDAlgebra {
    /// Create a new modified CD algebra.
    pub fn new(base_dim: usize, element_g: Vec<f64>) -> Self {
        assert_eq!(
            element_g.len(),
            base_dim,
            "g must have same dimension as base algebra"
        );
        Self {
            base_dim,
            element_g,
        }
    }

    /// The full (doubled) dimension.
    pub fn full_dim(&self) -> usize {
        2 * self.base_dim
    }

    /// Multiply two elements using the modified CD formula.
    pub fn multiply(&self, lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
        crate::modified_cd::modified_cd_multiply(
            &lhs[..self.base_dim],
            &lhs[self.base_dim..],
            &rhs[..self.base_dim],
            &rhs[self.base_dim..],
            &self.element_g,
        )
    }

    /// Norm of the g element.
    pub fn g_norm_sq(&self) -> f64 {
        cd_norm_sq(&self.element_g)
    }
}

/// Test whether a CD algebra at a given dimension is a division algebra over R.
///
/// For dims 1, 2, 4, 8: returns is_division = true (norm multiplicativity).
/// For dims >= 16: searches for zero-divisor witnesses among basis-element sums.
///
/// Mirrors: SchaferDivAlg16.v schafer_not_division_over_R.
pub fn test_division(dim: usize) -> DivisionTest {
    let level = CayleyDicksonLevel::from_dim(dim)
        .unwrap_or(CayleyDicksonLevel::Higher(dim));
    let hurwitz = hurwitz_1898::classify(dim);

    if dim <= 8 {
        // Division algebra: verify by checking inverse existence
        let mut max_viol = 0.0_f64;
        for i in 0..dim {
            let mut x = vec![0.0; dim];
            x[i] = 1.0;
            if let Some(err) = dickson_1919::verify_inverse(&x) {
                max_viol = max_viol.max(err);
            }
        }
        return DivisionTest {
            level,
            is_division: true,
            zd_witness: None,
            norm_violation: max_viol,
            hurwitz,
        };
    }

    // Dim >= 16: search for ZD pair
    let zd = crate::division_test::find_basis_sum_zd(dim);
    let has_zd = zd.is_some();

    let witness = if has_zd {
        // Reconstruct the ZD pair from the indices
        let (a_pair, b_pair, sign) = zd.unwrap();
        let mut a = vec![0.0; dim];
        a[a_pair[0]] = 1.0;
        a[a_pair[1]] = 1.0;
        let mut b = vec![0.0; dim];
        b[b_pair[0]] = 1.0;
        b[b_pair[1]] = sign;
        Some((a, b))
    } else {
        None
    };

    DivisionTest {
        level,
        is_division: !has_zd,
        zd_witness: witness,
        norm_violation: if has_zd { f64::INFINITY } else { 0.0 },
        hurwitz,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cd_kernel::cayley_dickson::cd_multiply;

    #[test]
    fn test_division_dims_1_2_4_8() {
        for d in [1, 2, 4, 8] {
            let t = test_division(d);
            assert!(t.is_division, "dim {d} should be division");
            assert!(t.zd_witness.is_none());
        }
    }

    #[test]
    fn test_not_division_dim_16() {
        let t = test_division(16);
        assert!(!t.is_division, "dim 16 should NOT be division");
        assert!(t.zd_witness.is_some());
        let (a, b) = t.zd_witness.unwrap();
        let ab = cd_multiply(&a, &b);
        assert!(cd_norm_sq(&ab) < 1e-10, "witness product should be zero");
    }

    #[test]
    fn test_division_display() {
        let t = test_division(8);
        let s = format!("{t}");
        assert!(s.contains("DIVISION ALGEBRA"));
    }

    #[test]
    fn test_modified_cd_algebra() {
        // Create a modified CD algebra at dim 8 with g = e_1
        let mut g = vec![0.0; 4];
        g[1] = 1.0;
        let alg = ModifiedCDAlgebra::new(4, g);
        assert_eq!(alg.full_dim(), 8);
        assert!((alg.g_norm_sq() - 1.0).abs() < 1e-10);

        // Multiply two elements
        let x = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let y = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let result = alg.multiply(&x, &y);
        assert_eq!(result.len(), 8);
        assert!(cd_norm_sq(&result) > 0.1, "product should be nonzero");
    }
}
