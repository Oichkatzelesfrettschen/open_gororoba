//! Norm defect formula (Brown Theorem 7.1, Corollary 7.2).
//!
//! N(A)N(B) - N(AB) measures how far the norm is from being multiplicative.
//! At dim <= 8: defect = 0 (Hurwitz). At dim >= 16: defect can be nonzero.
//!
//! Theorem 7.1: N(A)N(B) - N(AB) = sum of cross-norm terms + trace term
//! Corollary 7.2: In A_4, simplified to a single trace expression.

use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};

/// Compute the norm defect: N(A)*N(B) - N(AB).
///
/// This is zero for dims 1, 2, 4, 8 (composition algebras).
/// For dim >= 16, it can be positive, negative, or zero.
///
/// Mirrors: Brown (1972) Theorem 7.1.
pub fn norm_defect(a: &[f64], b: &[f64]) -> f64 {
    let ab = cd_multiply(a, b);
    cd_norm_sq(a) * cd_norm_sq(b) - cd_norm_sq(&ab)
}

/// Compute the norm defect decomposed into half-components.
///
/// For A = a1 + ea2, B = b1 + eb2 (each half of dimension n/2):
/// Returns (defect, n_a1, n_a2, n_b1, n_b2, n_ab).
pub fn norm_defect_decomposed(a: &[f64], b: &[f64]) -> NormDefectReport {
    let dim = a.len();
    let half = dim / 2;
    let a1 = &a[..half];
    let a2 = &a[half..];
    let b1 = &b[..half];
    let b2 = &b[half..];
    let ab = cd_multiply(a, b);

    NormDefectReport {
        dim,
        defect: norm_defect(a, b),
        norm_a: cd_norm_sq(a),
        norm_b: cd_norm_sq(b),
        norm_ab: cd_norm_sq(&ab),
        norm_a1: cd_norm_sq(a1),
        norm_a2: cd_norm_sq(a2),
        norm_b1: cd_norm_sq(b1),
        norm_b2: cd_norm_sq(b2),
    }
}

/// Report of norm defect decomposition.
#[derive(Debug, Clone)]
pub struct NormDefectReport {
    pub dim: usize,
    pub defect: f64,
    pub norm_a: f64,
    pub norm_b: f64,
    pub norm_ab: f64,
    pub norm_a1: f64,
    pub norm_a2: f64,
    pub norm_b1: f64,
    pub norm_b2: f64,
}

impl std::fmt::Display for NormDefectReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dim={}: N(A)N(B)-N(AB) = {:.6} [N(A)={:.4}, N(B)={:.4}, N(AB)={:.4}]",
            self.dim, self.defect, self.norm_a, self.norm_b, self.norm_ab
        )
    }
}

/// Check Brown's example from p.43: A=e1+e10, B=e0+e1+e4-e15.
/// N(A)=2, N(B)=4, N(AB)=4, so defect = 2*4 - 4 = 4 > 0.
/// This shows N(AB) < N(A)N(B) is possible.
pub fn brown_example_defect_positive() -> NormDefectReport {
    let mut a = vec![0.0; 16];
    a[1] = 1.0;
    a[10] = 1.0;
    let mut b = vec![0.0; 16];
    b[0] = 1.0;
    b[1] = 1.0;
    b[4] = 1.0;
    b[15] = -1.0;
    norm_defect_decomposed(&a, &b)
}

/// Check Brown's second example: A=e1-e10, B=e0+e1+e4-e15.
/// N(A)=2, N(B)=4, N(AB)=12, so defect = 8 - 12 = -4 < 0.
/// This shows N(AB) > N(A)N(B) is also possible!
pub fn brown_example_defect_negative() -> NormDefectReport {
    let mut a = vec![0.0; 16];
    a[1] = 1.0;
    a[10] = -1.0;
    let mut b = vec![0.0; 16];
    b[0] = 1.0;
    b[1] = 1.0;
    b[4] = 1.0;
    b[15] = -1.0;
    norm_defect_decomposed(&a, &b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defect_zero_dim8() {
        // Octonions: norm is multiplicative, defect = 0
        for i in 0..8 {
            for j in 0..8 {
                let mut a = vec![0.0; 8];
                let mut b = vec![0.0; 8];
                a[i] = 1.0;
                b[j] = 1.0;
                assert!(
                    norm_defect(&a, &b).abs() < 1e-10,
                    "octonion defect should be 0 for ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_defect_positive_example() {
        let r = brown_example_defect_positive();
        assert!((r.norm_a - 2.0).abs() < 1e-10);
        assert!((r.norm_b - 4.0).abs() < 1e-10);
        assert!(r.defect > 0.0, "defect should be positive: {}", r.defect);
    }

    #[test]
    fn test_defect_negative_example() {
        let r = brown_example_defect_negative();
        assert!((r.norm_a - 2.0).abs() < 1e-10);
        assert!((r.norm_b - 4.0).abs() < 1e-10);
        assert!(r.defect < 0.0, "defect should be negative: {}", r.defect);
    }

    #[test]
    fn test_defect_zd_pair() {
        // For the ZD pair: defect = N(A)*N(B) - 0 = N(A)*N(B) > 0
        let (a, b) = schafer_1945::division_test::zd_witness_16();
        let d = norm_defect(&a, &b);
        assert!(
            (d - 4.0).abs() < 1e-10,
            "ZD pair defect should be N(A)*N(B) = 2*2 = 4: {d}"
        );
    }

    #[test]
    fn test_report_display() {
        let r = brown_example_defect_positive();
        let s = format!("{r}");
        assert!(s.contains("dim=16"));
    }
}
