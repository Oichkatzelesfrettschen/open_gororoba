//! Sedenion (16D) specialized operations and triad structures.
//!
//! This module provides the complete explicit multiplication structure for the 
//! 16-dimensional sedenion algebra, resolving the 35 triads that dictate its
//! non-associative, non-alternative geometry.

/// The 35 defining triads {i, j, k} for the sedenion multiplication table.
/// Each triad indicates that e_i * e_j = e_k (and cyclically e_j * e_k = e_i, 
/// e_k * e_i = e_j), with anti-commutativity e_j * e_i = -e_k.
pub const SEDENION_TRIADS: [[usize; 3]; 35] = [
    [1, 2, 3], [1, 4, 5], [1, 7, 6], [1, 8, 9], [1, 11, 10], [1, 13, 12], [1, 14, 15],
    [2, 4, 6], [2, 5, 7], [2, 8, 10], [2, 9, 11], [2, 14, 12], [2, 15, 13],
    [3, 4, 7], [3, 6, 5], [3, 8, 11], [3, 10, 9], [3, 13, 14], [3, 15, 12],
    [4, 8, 12], [4, 9, 13], [4, 10, 14], [4, 11, 15],
    [5, 8, 13], [5, 10, 15], [5, 12, 9], [5, 14, 11],
    [6, 8, 14], [6, 11, 13], [6, 12, 10], [6, 15, 9],
    [7, 8, 15], [7, 9, 14], [7, 12, 11], [7, 13, 10],
];

/// Multiplies two 16D sedenions using the explicit 35-triad structure.
/// This approach significantly reduces operations compared to O(N^2) loops
/// by explicitly mapping the non-zero cross terms.
pub fn sedenion_multiply_explicit(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut res = [0.0; 16];

    // e_0 (real part) interactions
    for i in 0..16 {
        res[i] += a[0] * b[i] + a[i] * b[0];
    }
    // Remove the double-counted e_0 * e_0 term
    res[0] -= a[0] * b[0];

    // e_i * e_i = -1 for i > 0
    for i in 1..16 {
        res[0] -= a[i] * b[i];
    }

    // Apply the 35 triads
    for &[i, j, k] in SEDENION_TRIADS.iter() {
        // e_i * e_j = e_k
        res[k] += a[i] * b[j] - a[j] * b[i];
        
        // e_j * e_k = e_i
        res[i] += a[j] * b[k] - a[k] * b[j];
        
        // e_k * e_i = e_j
        res[j] += a[k] * b[i] - a[i] * b[k];
    }

    res
}

/// Helper function to check if a sedenion is a zero divisor within a numerical tolerance.
/// In the sedenion algebra, zero divisors form a space homeomorphic to the exceptional Lie group G2.
pub fn is_sedenion_zero_divisor(a: &[f64; 16], b: &[f64; 16], tol: f64) -> bool {
    let prod = sedenion_multiply_explicit(a, b);
    let norm_a: f64 = a.iter().map(|x| x * x).sum();
    let norm_b: f64 = b.iter().map(|x| x * x).sum();
    let norm_prod: f64 = prod.iter().map(|x| x * x).sum();

    // If ||a|| != 0 and ||b|| != 0, but ||a*b|| == 0, it's a zero divisor.
    norm_a > tol && norm_b > tol && norm_prod < tol
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sedenion_multiply_matches_standard() {
        let mut a = [0.0; 16];
        let mut b = [0.0; 16];
        
        // Random-like assignment
        for i in 0..16 {
            a[i] = (i as f64) * 0.1;
            b[i] = (16 - i) as f64 * 0.1;
        }

        let res_explicit = sedenion_multiply_explicit(&a, &b);
        let res_standard = crate::cayley_dickson::arith::cd_multiply(&a, &b);

        for i in 0..16 {
            assert!((res_explicit[i] - res_standard[i]).abs() < 1e-10, "Mismatch at index {}: explicit {} vs standard {}", i, res_explicit[i], res_standard[i]);
        }
    }
}
