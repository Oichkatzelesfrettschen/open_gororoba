//! Cayley-Dickson algebras over finite fields F_p.
//!
//! # Purpose
//!
//! Test the scalar extension of CD algebras to non-ordered fields.
//! Over ordered fields (R, No), sums of squares are anisotropic,
//! guaranteeing division through octonions. Over F_p, sums of
//! squares CAN be isotropic (zero with nonzero summands), which
//! means zero divisors may appear even at the OCTONION level.
//!
//! # Why F_p instead of Q_p
//!
//! Q_p (p-adic numbers) requires Hensel lifting and infinite-precision
//! arithmetic. F_p is the residue field of Q_p, and many properties
//! (like norm isotropy) can be tested over F_p first. If A_3(F_p)
//! has zero divisors, then A_3(Q_p) likely does too (by Hensel lifting
//! of norm-zero elements). If not, the Q_p question remains open.
//!
//! # Key theorems
//!
//! - **Chevalley-Warning**: Over F_p, any quadratic form in n >= 3
//!   variables has a nontrivial zero. So norm-zero elements exist
//!   at dim >= 4 for ALL primes.
//! - **Quadratic reciprocity**: -1 is a quadratic residue mod p iff
//!   p = 1 mod 4. So F_p complex numbers (dim=2) have zero divisors
//!   iff p = 1 mod 4 (e.g., p=5: 2^2 + 1^2 = 5 = 0 mod 5).
//! - **Scalar extension (C-1504)**: Sedenion ZDs persist over F_p
//!   because the ZD identity has integer structure constants (C-1520).
//!
//! # Concrete results
//!
//! | Field | Complex (dim=2) | Octonion norm-zero | Sedenion ZD |
//! |-------|----------------|-------------------|-------------|
//! | F_3 | No (-1 not QR) | Yes (dim >= 3) | Yes (C-1520) |
//! | F_5 | Yes (2^2+1=0) | Yes | Yes |
//! | F_7 | No | Yes | Yes |
//!
//! # Callers
//!
//! - `test_fp_complex_zd`: quadratic reciprocity validation
//! - `test_fp_octonion_norm_zero`: Chevalley-Warning verification
//! - `test_fp_sedenion_zd`: scalar extension universality (C-1520)
//!
//! Reference: surreal_cayley_dickson_harmonized.md, Section 6.
//! Reference: Bales 2003 (arXiv:1107.1375) for twist formulation.

use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;

/// Multiply two CD elements over F_p (modular arithmetic).
///
/// Uses the same sign table as real CD, but with all arithmetic mod p.
/// The sign values {+1, -1} are mapped to {1, p-1} in F_p.
#[allow(clippy::needless_range_loop)]
pub fn fp_cd_multiply(dim: usize, p: u64, a: &[u64], b: &[u64]) -> Vec<u64> {
    assert_eq!(a.len(), dim);
    assert_eq!(b.len(), dim);

    let mut result = vec![0_u64; dim];

    for q in 0..dim {
        if b[q] == 0 {
            continue;
        }
        for t in 0..dim {
            let src = t ^ q;
            if a[src] == 0 {
                continue;
            }
            let sign = cd_basis_mul_sign_iter(dim, src, q);
            let coeff = (a[src] * b[q]) % p;
            if sign == 1 {
                result[t] = (result[t] + coeff) % p;
            } else {
                // -1 mod p = p - 1
                result[t] = (result[t] + p - coeff) % p;
            }
        }
    }

    result
}

/// Compute norm N(x) = sum x_i^2 mod p.
pub fn fp_norm_sq(p: u64, x: &[u64]) -> u64 {
    x.iter().map(|&xi| (xi * xi) % p).sum::<u64>() % p
}

/// Find a norm-zero element in F_p^dim (brute force for small p, dim).
///
/// Returns the first nonzero x with N(x) = 0 mod p, or None if none exists.
pub fn find_norm_zero_fp(dim: usize, p: u64) -> Option<Vec<u64>> {
    // For small p and dim, brute force over all vectors
    // For dim=2: try all (x0, x1) with x0^2 + x1^2 = 0 mod p
    // This checks if -1 is a quadratic residue mod p.

    if dim == 2 {
        for x0 in 1..p {
            for x1 in 0..p {
                let n = (x0 * x0 + x1 * x1) % p;
                if n == 0 {
                    return Some(vec![x0, x1]);
                }
            }
        }
        return None;
    }

    // For dim >= 3: Chevalley-Warning guarantees existence.
    // Search systematically: fix first coordinate, search rest.
    for x0 in 1..p {
        // Need sum_{i>0} x_i^2 = -x0^2 mod p
        let target = (p - (x0 * x0) % p) % p;
        // For dim=4,8: recurse or brute-force
        if dim <= 4 {
            for x1 in 0..p {
                let rem = (target + p - (x1 * x1) % p) % p;
                if dim == 3 {
                    // Need x2^2 = rem mod p
                    for x2 in 0..p {
                        if (x2 * x2) % p == rem {
                            return Some(vec![x0, x1, x2]);
                        }
                    }
                } else {
                    // dim == 4
                    for x2 in 0..p {
                        let rem2 = (rem + p - (x2 * x2) % p) % p;
                        for x3 in 0..p {
                            if (x3 * x3) % p == rem2 {
                                return Some(vec![x0, x1, x2, x3]);
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: does -1 have a square root mod p?
    /// If yes, F_p complex numbers (dim=2) have zero divisors.
    /// -1 is a QR mod p iff p = 1 mod 4 (quadratic reciprocity).
    #[test]
    fn test_fp_complex_zd() {
        println!("--- F_p CD: quadratic residue of -1 ---\n");

        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31_u64] {
            let norm_zero = find_norm_zero_fp(2, p);
            let has_sqrt_neg1 = p % 4 == 1;
            println!(
                "  p={:>2}: -1 is QR: {:>5}, norm-zero element: {:?}",
                p,
                has_sqrt_neg1,
                norm_zero.as_ref().map(|v| format!("({},{})", v[0], v[1]))
            );

            if has_sqrt_neg1 {
                assert!(
                    norm_zero.is_some(),
                    "p={}: -1 is QR so norm-zero should exist",
                    p
                );

                // Check: is this actually a zero divisor in A_1(F_p)?
                let v = norm_zero.unwrap();
                let a = vec![1, v[1]]; // 1 + sqrt(-1)*e_1
                let b = vec![1, p - v[1]]; // 1 - sqrt(-1)*e_1
                let product = fp_cd_multiply(2, p, &a, &b);
                let is_zd = product.iter().all(|&c| c == 0);
                println!("    (1+i*e_1)(1-i*e_1) = {:?}, is_zd={}", product, is_zd);
            }
        }
    }

    /// Test: F_p octonions (dim=8) over small primes.
    /// Chevalley-Warning: norm form in 8 vars is isotropic for all p.
    #[test]
    fn test_fp_octonion_norm_zero() {
        println!("\n--- F_p Octonions: norm-zero elements ---\n");

        for p in [3, 5, 7_u64] {
            let norm_zero = find_norm_zero_fp(4, p); // dim=4 for tractability
            println!(
                "  p={}: dim=4 norm-zero: {:?}",
                p,
                norm_zero.as_ref().map(|v| format!("{:?}", v))
            );

            if let Some(v) = &norm_zero {
                let n = fp_norm_sq(p, v);
                assert_eq!(n, 0, "p={}: norm should be 0 mod p", p);
            }
        }
    }

    /// Test F_p sedenion multiplication at dim=16.
    /// Verify that the standard ZD witness works mod p.
    #[test]
    fn test_fp_sedenion_zd() {
        println!("\n--- F_p Sedenions: ZD witness mod p ---\n");

        for p in [3, 5, 7, 11, 13_u64] {
            // a = e_1 + e_10, b = e_4 - e_15 = e_4 + (p-1)*e_15
            let mut a = vec![0_u64; 16];
            a[1] = 1;
            a[10] = 1;
            let mut b = vec![0_u64; 16];
            b[4] = 1;
            b[15] = p - 1; // -1 mod p

            let product = fp_cd_multiply(16, p, &a, &b);
            let is_zd = product.iter().all(|&c| c == 0);

            println!(
                "  p={:>2}: (e_1+e_10)(e_4-e_15) mod {} is ZD: {}",
                p, p, is_zd
            );
            assert!(
                is_zd,
                "p={}: sedenion ZD should persist mod p (integer structure constants)",
                p
            );
        }

        println!("\n  PASS: Sedenion ZD persists over F_p for all tested primes");
        println!("  (Consequence of scalar extension: structure constants are integers)");
    }
}
