//! Generalized Cayley-Dickson construction (Brown eq.2).
//!
//! The generalized product with parameter gamma:
//!   (a + bu)(c + du) = (ac + gamma*d*b*) + (da + bc*)u
//!
//! Standard CD uses gamma = -1. Different gamma values can produce
//! different algebraic properties (e.g., division algebras over non-real fields).
//!
//! Mirrors: BrownGeneralizedCD.v, schafer_1945::modified_cd.

use cd_kernel::cayley_dickson::{cd_conjugate, cd_multiply, cd_norm_sq};

/// A generalized Cayley-Dickson algebra with parameter gamma.
///
/// Doubles a base algebra of dimension `base_dim` to produce
/// an algebra of dimension `2 * base_dim` using the parameter `gamma`.
///
/// Standard CD uses gamma = -1.
///
/// Mirrors: Brown (1967) eq.2.
#[derive(Debug, Clone)]
pub struct GeneralizedCD {
    /// Dimension of the base algebra.
    pub base_dim: usize,
    /// The doubling parameter gamma (nonzero scalar).
    pub gamma: f64,
}

impl GeneralizedCD {
    /// Create a new generalized CD algebra.
    pub fn new(base_dim: usize, gamma: f64) -> Self {
        assert!(gamma.abs() > 1e-15, "gamma must be nonzero");
        Self { base_dim, gamma }
    }

    /// Standard CD construction (gamma = -1).
    pub fn standard(base_dim: usize) -> Self {
        Self::new(base_dim, -1.0)
    }

    /// Full (doubled) dimension.
    pub fn full_dim(&self) -> usize {
        2 * self.base_dim
    }

    /// Multiply two elements using the generalized CD formula.
    ///
    /// (a + bu)(c + du) = (ac + gamma*d**b) + (da + bc*)u
    ///
    /// where * denotes conjugation in the base algebra.
    pub fn multiply(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        let n = self.base_dim;
        assert_eq!(x.len(), 2 * n);
        assert_eq!(y.len(), 2 * n);

        let a = &x[..n];
        let b = &x[n..];
        let c = &y[..n];
        let d = &y[n..];

        // lo = ac + gamma * conj(d) * b
        let ac = cd_multiply(a, c);
        let conj_d = cd_conjugate(d);
        let conj_d_b = cd_multiply(&conj_d, b);
        let lo: Vec<f64> = ac
            .iter()
            .zip(conj_d_b.iter())
            .map(|(ac_i, db_i)| ac_i + self.gamma * db_i)
            .collect();

        // hi = da + b * conj(c)
        let da = cd_multiply(d, a);
        let conj_c = cd_conjugate(c);
        let b_conj_c = cd_multiply(b, &conj_c);
        let hi: Vec<f64> = da
            .iter()
            .zip(b_conj_c.iter())
            .map(|(da_i, bc_i)| da_i + bc_i)
            .collect();

        let mut result = lo;
        result.extend_from_slice(&hi);
        result
    }

    /// Compute the generalized norm: n(x) = x * conj(x).
    ///
    /// For the generalized CD, the norm decomposes as:
    /// n(a + bu) = n(a) - gamma * n(b)
    ///
    /// Mirrors: Brown (1967) eq.1.
    pub fn norm_sq(&self, x: &[f64]) -> f64 {
        let n = self.base_dim;
        let a = &x[..n];
        let b = &x[n..];
        cd_norm_sq(a) - self.gamma * cd_norm_sq(b)
    }

    /// Verify that the standard CD (gamma = -1) matches cd_multiply.
    pub fn verify_standard_consistency(&self) -> f64 {
        assert!(
            (self.gamma - (-1.0)).abs() < 1e-15,
            "only valid for standard CD"
        );
        let dim = self.full_dim();
        let mut max_err = 0.0_f64;
        for i in 0..dim {
            for j in 0..dim {
                let mut x = vec![0.0; dim];
                let mut y = vec![0.0; dim];
                x[i] = 1.0;
                y[j] = 1.0;
                let generalized = self.multiply(&x, &y);
                let standard = cd_multiply(&x, &y);
                let err: f64 = generalized
                    .iter()
                    .zip(standard.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                max_err = max_err.max(err);
            }
        }
        max_err
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_cd_dim4() {
        let cd = GeneralizedCD::standard(2);
        assert!(
            cd.verify_standard_consistency() < 1e-10,
            "standard CD at dim 4 should match cd_multiply"
        );
    }

    #[test]
    fn test_standard_cd_dim8() {
        let cd = GeneralizedCD::standard(4);
        assert!(
            cd.verify_standard_consistency() < 1e-10,
            "standard CD at dim 8 should match cd_multiply"
        );
    }

    #[test]
    fn test_standard_cd_dim16() {
        let cd = GeneralizedCD::standard(8);
        assert!(
            cd.verify_standard_consistency() < 1e-10,
            "standard CD at dim 16 should match cd_multiply"
        );
    }

    #[test]
    fn test_generalized_norm_standard() {
        // For gamma = -1: n(x) = n(a) + n(b) = standard norm
        let cd = GeneralizedCD::standard(4);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let generalized_norm = cd.norm_sq(&x);
        let std_norm = cd_norm_sq(&x);
        assert!(
            (generalized_norm - std_norm).abs() < 1e-10,
            "generalized norm with gamma=-1 should match standard: {generalized_norm} vs {std_norm}"
        );
    }

    #[test]
    fn test_generalized_norm_positive_gamma() {
        // For gamma = +1: n(x) = n(a) - n(b) (indefinite form!)
        let cd = GeneralizedCD::new(4, 1.0);
        let x = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let norm = cd.norm_sq(&x);
        // n(a) = 1, n(b) = 1, so n(x) = 1 - 1 = 0
        assert!((norm - 0.0).abs() < 1e-10, "indefinite norm gives zero");
    }

    #[test]
    fn test_different_gamma_different_algebra() {
        // gamma = -1 vs gamma = -2 should give different products
        let cd1 = GeneralizedCD::new(4, -1.0);
        let cd2 = GeneralizedCD::new(4, -2.0);
        let x = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let p1 = cd1.multiply(&x, &y);
        let p2 = cd2.multiply(&x, &y);
        let diff: f64 = p1
            .iter()
            .zip(p2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(diff > 0.1, "different gamma should give different products");
    }
}
