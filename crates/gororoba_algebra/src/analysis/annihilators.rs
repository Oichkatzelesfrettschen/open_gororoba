//! Annihilator Dimension Explorer.
//!
//! Investigates the Biss-Dugger-Isaksen upper bound for annihilator dimensions:
//! dim Ann(x) <= 2^n - 4n + 4
//!
//! # References
//! - Biss, Dugger, Isaksen (2005): "Annihilators of zero divisors in Cayley-Dickson algebras"
//! - Moreno (1997): "The Zero Divisors of the Cayley-Dickson Algebras"

use crate::construction::hypercomplex::AlgebraDim;
use cd_kernel::cayley_dickson::cd_multiply;
use nalgebra::DMatrix;

/// Calculate the annihilator dimension of an element x in A_n.
/// Ann(x) = { y : x * y = 0 }
pub fn annihilator_dimension(dim: usize, x: &[f64], atol: f64) -> usize {
    // We construct the matrix representing multiplication by x: M_x
    // The dimension of the nullspace of M_x is the annihilator dimension.
    
    let mut matrix_data = Vec::with_capacity(dim * dim);
    for j in 0..dim {
        let mut e_j = vec![0.0; dim];
        e_j[j] = 1.0;
        let col = cd_multiply(x, &e_j);
        matrix_data.extend_from_slice(&col);
    }
    
    let m = DMatrix::from_column_slice(dim, dim, &matrix_data);
    let svd = m.svd(false, false);
    
    // Count singular values below tolerance
    svd.singular_values.iter().filter(|&&s| s < atol).count()
}

/// The Biss-Dugger-Isaksen theoretical upper bound for dim Ann(x).
pub fn biss_dugger_isaksen_bound(dim: usize) -> usize {
    let n = (dim as f64).log2() as usize;
    if n < 4 {
        0
    } else {
        (1 << n) - 4 * n + 4
    }
}

/// Moreno Corollary 1.17 upper bound on dim Ker(L_a) for doubly-pure unit-norm a.
///
/// Returns `2^n - 4` where `n = log2(dim)`.
///
/// # Distinction from BDI bound
///
/// - Moreno Cor 1.17: `dim Ker(L_a) <= 2^n - 4` for doubly-pure unit-norm a.
/// - Biss-Dugger-Isaksen: `dim Ann(x) <= 2^n - 4n + 4` for any nonzero x (all n >= 4).
///
/// For n >= 4: BDI gives a TIGHTER bound (2^n - 4n + 4 < 2^n - 4 since 4n > 8 for n > 2).
/// Both are 0 mod 4, consistent with Theorem 1.16 (eigenspaces have dim = 0 mod 4).
pub fn moreno_corollary_1_17_bound(dim: usize) -> usize {
    let n = (dim as f64).log2() as usize;
    (1 << n) - 4
}

/// Analyze annihilator dimensions for a suite of known zero divisors.
pub fn analyze_annihilators(dim_enum: AlgebraDim) -> Vec<(usize, bool)> {
    let dim = dim_enum.dim();
    let n = (dim as f64).log2() as usize;
    let bound = biss_dugger_isaksen_bound(dim);

    if n < 4 {
        return vec![(0, true)]; // No ZDs below 16D
    }

    // Use known 2-blade ZDs from the kernel.
    // x = e_i + e_j is the LEFT factor of a ZD pair (i, j, k, l, ...).
    // x IS a zero divisor: it has a nontrivial left annihilator (Ann(x) != {0}).
    // We measure dim Ann(x) and verify it stays within the BDI bound.
    use cd_kernel::cayley_dickson::find_zero_divisors;
    let zds = find_zero_divisors(dim, 1e-10);

    let mut results = Vec::new();
    for (i, j, _, _, _) in zds.iter().take(10) {
        let mut x = vec![0.0; dim];
        x[*i] = 1.0;
        x[*j] = 1.0;
        // x = e_i + e_j is a ZD; Ann(x) = {y : x*y = 0} has dim >= 1.
        let d = annihilator_dimension(dim, &x, 1e-8);
        results.push((d, d <= bound));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_annihilator_bound_16d() {
        // n=4: 16 - 16 + 4 = 4
        assert_eq!(biss_dugger_isaksen_bound(16), 4);
        let results = analyze_annihilators(AlgebraDim::Sedenion);
        for (dim, within_bound) in results {
            assert!(within_bound, "Annihilator dimension {} exceeded bound 4", dim);
        }
    }
}
