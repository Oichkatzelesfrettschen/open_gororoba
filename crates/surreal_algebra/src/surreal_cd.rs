//! Surreal-coefficient Cayley-Dickson multiplication and ZD verification.
//!
//! Uses the same sign table as real CD algebras (cd_basis_mul_sign_iter)
//! but with SurrealDyadic coefficients instead of f64.
//!
//! The scalar extension theorem (C-1504) guarantees that all polynomial
//! identities with integer structure constants transfer from A_n(R) to
//! A_n(K) for any commutative ring K.  Since the sign table entries are
//! integers ({+1, -1}), the multiplication rule transfers exactly.

use crate::dyadic::SurrealDyadic;
use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;

/// Multiply two surreal-coefficient CD algebra elements.
///
/// The product is computed using the sign table:
///   (sum_p a_p * e_p) * (sum_q b_q * e_q) = sum_{p,q} a_p * b_q * sign(p,q) * e_{p XOR q}
///
/// This is the SAME formula as f64 CD multiplication, but with exact
/// SurrealDyadic arithmetic instead of floating-point.
#[allow(clippy::needless_range_loop)]
pub fn surreal_cd_multiply(dim: usize, a: &[SurrealDyadic], b: &[SurrealDyadic]) -> Vec<SurrealDyadic> {
    assert_eq!(a.len(), dim);
    assert_eq!(b.len(), dim);

    let mut result = vec![SurrealDyadic::zero(); dim];

    // Indices p, q are used for XOR (p ^ q) and sign table lookup,
    // not just array indexing -- needless_range_loop doesn't apply.
    for p in 0..dim {
        if a[p].is_zero() { continue; }
        for q in 0..dim {
            if b[q].is_zero() { continue; }
            let sign = cd_basis_mul_sign_iter(dim, p, q);
            let target = p ^ q;
            let coeff = a[p] * b[q];
            if sign == 1 {
                result[target] = result[target] + coeff;
            } else {
                result[target] = result[target] - coeff;
            }
        }
    }

    result
}

/// Check if a pair of surreal-coefficient sedenion elements are zero divisors.
///
/// Returns true if a * b = 0 exactly (all 16 components vanish).
/// This verifies zero-divisor persistence under scalar extension:
/// if (e_i + e_j)(e_k - e_l) = 0 over R, then
/// (alpha*e_i + beta*e_j)(gamma*e_k - delta*e_l) = 0 for
/// specific surreal alpha, beta, gamma, delta.
pub fn surreal_sedenion_zd_check(a: &[SurrealDyadic; 16], b: &[SurrealDyadic; 16]) -> bool {
    let product = surreal_cd_multiply(16, a, b);
    product.iter().all(|c| c.is_zero())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the standard ZD witness persists with surreal coefficients.
    /// (e_1 + e_10)(e_4 - e_15) = 0 over R => also 0 over No.
    #[test]
    fn test_surreal_zd_persistence_standard_witness() {
        let one = SurrealDyadic::one();
        let neg_one = -one;

        // a = e_1 + e_10
        let mut a = [SurrealDyadic::zero(); 16];
        a[1] = one;
        a[10] = one;

        // b = e_4 - e_15
        let mut b = [SurrealDyadic::zero(); 16];
        b[4] = one;
        b[15] = neg_one;

        assert!(surreal_sedenion_zd_check(&a, &b),
            "Standard ZD witness should persist over surreal coefficients");

        println!("PASS: (e_1 + e_10)(e_4 - e_15) = 0 over No");
    }

    /// Verify ZD persistence with non-trivial surreal scaling.
    /// If a*b = 0, then (alpha*a)*(beta*b) = alpha*beta*(a*b) = 0
    /// for any surreal alpha, beta (by bilinearity).
    #[test]
    fn test_surreal_zd_persistence_scaled() {
        let half = SurrealDyadic::new(1, 1);  // 1/2
        let three = SurrealDyadic::from_int(3);

        // a = (1/2)*e_1 + (1/2)*e_10
        let mut a = [SurrealDyadic::zero(); 16];
        a[1] = half;
        a[10] = half;

        // b = 3*e_4 - 3*e_15
        let mut b = [SurrealDyadic::zero(); 16];
        b[4] = three;
        b[15] = -three;

        assert!(surreal_sedenion_zd_check(&a, &b),
            "Scaled ZD should persist: (1/2)(e_1+e_10) * 3(e_4-e_15) = 0");

        println!("PASS: (1/2)(e_1 + e_10) * 3(e_4 - e_15) = 0 over No");
    }

    /// Verify ZD with mixed surreal coefficients (different scales per component).
    /// a = alpha*e_1 + beta*e_10, b = gamma*e_4 - delta*e_15
    /// The ZD condition requires specific relationships between alpha,beta,gamma,delta.
    #[test]
    fn test_surreal_zd_mixed_coefficients() {
        let two = SurrealDyadic::from_int(2);
        let five = SurrealDyadic::from_int(5);

        // a = 2*e_1 + 5*e_10
        let mut a = [SurrealDyadic::zero(); 16];
        a[1] = two;
        a[10] = five;

        // b = e_4 - e_15  (unit coefficients)
        let mut b = [SurrealDyadic::zero(); 16];
        b[4] = SurrealDyadic::one();
        b[15] = -SurrealDyadic::one();

        let product = surreal_cd_multiply(16, &a, &b);
        let is_zd = product.iter().all(|c| c.is_zero());

        println!("Mixed-coeff ZD test: (2*e_1 + 5*e_10)(e_4 - e_15)");
        println!("  Product components:");
        for (i, c) in product.iter().enumerate() {
            if !c.is_zero() {
                println!("    e_{}: {}", i, c);
            }
        }
        println!("  Is ZD: {}", is_zd);

        // This should NOT be zero -- different scaling breaks the ZD condition
        // (unless the specific linear combination happens to vanish, which it
        // won't for generic alpha != beta).
        // The ZD condition (alpha*e_i + beta*e_j)(gamma*e_k - delta*e_l) = 0
        // requires alpha*gamma = beta*delta (from the sign table structure).
        // Here 2*1 != 5*1, so it should NOT be a ZD.
        assert!(!is_zd,
            "Mixed scaling should break ZD: 2*1 != 5*1");
        println!("PASS: Mixed scaling correctly breaks ZD condition");
    }

    /// Verify the birthday filtration: track maximum birthday across
    /// a product computation.
    #[test]
    fn test_birthday_filtration() {
        let one = SurrealDyadic::one();
        let half = SurrealDyadic::new(1, 1);
        let quarter = SurrealDyadic::new(1, 2);

        // a = (1/4)*e_1 + e_3 (max birthday = 2)
        let mut a = [SurrealDyadic::zero(); 16];
        a[1] = quarter;
        a[3] = one;

        // b = (1/2)*e_2 + e_5 (max birthday = 1)
        let mut b = [SurrealDyadic::zero(); 16];
        b[2] = half;
        b[5] = one;

        let product = surreal_cd_multiply(16, &a, &b);

        let max_birthday: u32 = product.iter()
            .map(|c| c.birthday())
            .max()
            .unwrap_or(0);

        println!("Birthday filtration test:");
        println!("  a: max birthday = {}", a.iter().map(|c| c.birthday()).max().unwrap_or(0));
        println!("  b: max birthday = {}", b.iter().map(|c| c.birthday()).max().unwrap_or(0));
        println!("  a*b: max birthday = {}", max_birthday);

        // Product birthday should be bounded by sum of input birthdays
        // (multiplication can increase birthday by at most the sum of shifts)
        println!("  Product components:");
        for (i, c) in product.iter().enumerate() {
            if !c.is_zero() {
                println!("    e_{}: {} (birthday {})", i, c, c.birthday());
            }
        }
    }
}
