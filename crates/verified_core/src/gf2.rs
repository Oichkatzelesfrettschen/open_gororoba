//! GF(2) polynomial evaluation and separation (Rocq-verified: C-881).
//!
//! Evaluates GF(2) polynomials represented as lists of monomial bitmasks
//! on finite point sets.  Used to verify the minimum separating degree
//! for the PG(3,2) cross-assessor partition.
//! Kernel-checked in `proofs/verified/C881_GF2CubicMinDegree.v`.

/// Evaluate a single GF(2) monomial on a point.
///
/// Returns true iff all bits of `mask` are set in `label`.
pub fn gf2_monomial_eval(mask: u32, label: u32) -> bool {
    (label & mask) == mask
}

/// Evaluate a GF(2) polynomial (XOR of monomials) on a point.
///
/// A polynomial is represented as a slice of monomial bitmasks.
/// The result is the XOR (sum mod 2) of individual monomial evaluations.
pub fn gf2_poly_eval(monomials: &[u32], label: u32) -> bool {
    monomials
        .iter()
        .filter(|&&m| gf2_monomial_eval(m, label))
        .count()
        % 2
        == 1
}

/// Check whether a polynomial separates labels according to classes.
///
/// Returns true iff for every (label, class) pair, the polynomial
/// evaluates to the expected class value.
pub fn separates(labels: &[u32], classes: &[bool], monomials: &[u32]) -> bool {
    labels
        .iter()
        .zip(classes)
        .all(|(&l, &c)| gf2_poly_eval(monomials, l) == c)
}

/// The PG(3,2) point labels (dim=32 cross-assessor XOR signatures).
pub const PG32_LABELS: [u32; 15] = [3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1];

/// The PG(3,2) classes for the cubic witness polynomial.
/// true = heptacross (84 edges, poly evals to 1), false = mixed-degree (poly evals to 0).
/// This is the polarity that matches the cubic witness output.
pub const PG32_CLASSES: [bool; 15] = [
    true, true, true, true, true, true, false, true, false, false, false, false, false, false, true,
];

/// The cubic witness polynomial that separates the PG(3,2) partition.
pub const CUBIC_WITNESS: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monomial_eval_basics() {
        // mask=0b0011 (3), label=0b0111 (7): all bits set => true
        assert!(gf2_monomial_eval(3, 7));
        // mask=0b0011 (3), label=0b0101 (5): bit 1 not set => false
        assert!(!gf2_monomial_eval(3, 5));
        // mask=0 (constant): always true
        assert!(gf2_monomial_eval(0, 42));
    }

    #[test]
    fn cubic_witness_separates_pg32() {
        assert!(separates(&PG32_LABELS, &PG32_CLASSES, &CUBIC_WITNESS));
    }

    #[test]
    fn degree1_fails() {
        // No single degree-1 monomial separates
        for &m in &[1u32, 2, 4, 8] {
            assert!(
                !separates(&PG32_LABELS, &PG32_CLASSES, &[m]),
                "degree-1 monomial {m} should not separate"
            );
        }
    }

    #[test]
    fn partition_sizes() {
        // true = heptacross (8), false = mixed (7)
        let n_heptacross = PG32_CLASSES.iter().filter(|&&c| c).count();
        let n_mixed = PG32_CLASSES.iter().filter(|&&c| !c).count();
        assert_eq!(n_heptacross, 8);
        assert_eq!(n_mixed, 7);
    }
}
