//! Cayley-Dickson Construction as an Endofunctor.
//!
//! Maps a composition algebra (A, *, ||.||) to its double (A', *', ||.||').
//! CD: CompAlg -> CompAlg

use crate::traits::Hypercomplex;

/// Cayley-Dickson construction as an endofunctor.
pub struct CayleyDicksonFunctor;

impl CayleyDicksonFunctor {
    /// Lift an element a in A to (a, 0) in CD(A).
    pub fn lift<H: Hypercomplex>(a: Vec<f64>) -> Vec<f64> {
        let dim = a.len();
        let mut res = vec![0.0; dim * 2];
        res[..dim].copy_from_slice(&a);
        res
    }

    /// The norm preservation property: ||CD(a, b)||^2 = ||a||^2 + ||b||^2.
    pub fn verify_norm_preservation(a: &[f64], b: &[f64]) -> bool {
        let dim_a = a.len();
        let dim_b = b.len();
        assert_eq!(dim_a, dim_b);

        let mut combined = vec![0.0; dim_a * 2];
        combined[..dim_a].copy_from_slice(a);
        combined[dim_a..].copy_from_slice(b);

        let norm_sq_a: f64 = a.iter().map(|x| x * x).sum();
        let norm_sq_b: f64 = b.iter().map(|x| x * x).sum();
        let norm_sq_combined: f64 = combined.iter().map(|x| x * x).sum();

        (norm_sq_combined - (norm_sq_a + norm_sq_b)).abs() < 1e-10
    }

    /// CD is NOT a monad because composition does not yield a natural associative
    /// algebra structure for all N (fails at N=8).
    pub fn is_monad() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_preservation() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        assert!(CayleyDicksonFunctor::verify_norm_preservation(&a, &b));
    }
}
