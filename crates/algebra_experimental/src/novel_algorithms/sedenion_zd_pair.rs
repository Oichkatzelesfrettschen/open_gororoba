//! Sedenion zero-divisor pair detector.
//!
//! Cayley-Dickson dimension 16 (sedenions) is the smallest doubling that
//! contains zero-divisors. This module tests a fundamental ZD relation
//! `(e_i + e_j)(e_k +/- e_l) = 0` *blade-wise*: given two 16D vectors,
//! it computes their sedenion product and reports whether the result is
//! within numerical tolerance of zero. A nonzero pair satisfying this is, by
//! definition, a zero-divisor pair.
//!
//! # Naming history
//!
//! Originally shipped as `e8_sieve.rs` with function `is_e8_root_candidate`.
//! The name was metaphorical -- the test computes a sedenion product, not an
//! E8 lattice membership check, and 16D vectors are not E8 roots (which live
//! in 8D). Renamed 2026-05-08 for accuracy.

use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};

/// Numerical tolerance below which a product is treated as the zero sedenion.
const ZERO_TOLERANCE: f64 = 1e-4;

/// True iff the sedenion product `a * b` has norm-squared below
/// [`ZERO_TOLERANCE`].
///
/// A nonzero `(a, b)` returning `true` is a sedenion zero-divisor pair.
pub fn is_sedenion_zd_pair(a: &[f64; 16], b: &[f64; 16]) -> bool {
    let product: [f64; 16] = cd_multiply(a, b).try_into().expect("16D sedenion product");
    cd_norm_sq(&product) < ZERO_TOLERANCE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_blade_pair_is_zero_divisor() {
        // The pair (e_1 + e_10, e_15 - e_4) is one of the canonical sedenion
        // ZD pairs surfaced by the box-kite enumeration. Their CD-product
        // collapses to zero.
        let mut a = [0.0; 16];
        a[1] = 1.0;
        a[10] = 1.0;
        let mut b = [0.0; 16];
        b[15] = 1.0;
        b[4] = -1.0;

        assert!(is_sedenion_zd_pair(&a, &b));
    }

    #[test]
    fn generic_pair_is_not_a_zero_divisor() {
        // A pair of distinct unit basis elements does not zero-out: e_1 * e_2
        // = e_3 (or similar nonzero basis), well above tolerance.
        let mut a = [0.0; 16];
        a[1] = 1.0;
        let mut b = [0.0; 16];
        b[2] = 1.0;

        assert!(!is_sedenion_zd_pair(&a, &b));
    }
}
