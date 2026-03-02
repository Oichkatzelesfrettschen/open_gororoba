//! Shannon binary entropy (Rocq-verified: C-915..C-917).
//!
//! H(p) = -(p*ln(p) + (1-p)*ln(1-p)).
//! Kernel-checked in `proofs/theories/BinaryEntropy.v`.

/// Binary entropy function H(p).
///
/// # Panics
///
/// Panics if p is not in the open interval (0, 1).
pub fn binary_entropy(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "p must be in (0,1), got {p}");
    -(p * p.ln() + (1.0 - p) * (1.0 - p).ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn symmetric() {
        for &p in &[0.1, 0.2, 0.3, 0.4, 0.49] {
            let h_p = binary_entropy(p);
            let h_1mp = binary_entropy(1.0 - p);
            assert!(
                (h_p - h_1mp).abs() < TOL,
                "H({p}) = {h_p} should equal H({}) = {h_1mp}",
                1.0 - p
            );
        }
    }

    #[test]
    fn max_at_half() {
        let h = binary_entropy(0.5);
        assert!(
            (h - 2.0_f64.ln()).abs() < TOL,
            "H(0.5) = {h} should equal ln(2) = {}",
            2.0_f64.ln()
        );
    }

    #[test]
    fn nonnegative() {
        for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            assert!(binary_entropy(p) >= 0.0, "H({p}) should be >= 0");
        }
    }

    #[test]
    #[should_panic]
    fn zero_panics() {
        binary_entropy(0.0);
    }

    #[test]
    #[should_panic]
    fn one_panics() {
        binary_entropy(1.0);
    }
}
