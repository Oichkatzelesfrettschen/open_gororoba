//! Calcagni spectral dimension (Rocq-verified: C-883).
//!
//! d_S(s) = 4 - 2/(1+s) for s > 0.  Range is (2, 4), strictly increasing.
//! Kernel-checked in `proofs/verified/C883_CalcagniMonotoneRange.v`.

/// Calcagni's running spectral dimension.
///
/// # Panics
///
/// Panics if `s <= 0.0`.
pub fn calcagni_d_s(s: f64) -> f64 {
    assert!(s > 0.0, "s must be positive, got {s}");
    4.0 - 2.0 / (1.0 + s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_bounds() {
        for &s in &[0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1e6] {
            let d = calcagni_d_s(s);
            assert!(d > 2.0, "d_s({s}) = {d} should be > 2");
            assert!(d < 4.0, "d_s({s}) = {d} should be < 4");
        }
    }

    #[test]
    fn monotone_increasing() {
        let values: Vec<f64> = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            .iter()
            .map(|&s| calcagni_d_s(s))
            .collect();
        for w in values.windows(2) {
            assert!(w[1] > w[0], "d_s should be increasing");
        }
    }

    #[test]
    fn known_value() {
        // s = 1: d_s = 4 - 2/2 = 3
        assert!((calcagni_d_s(1.0) - 3.0).abs() < 1e-15);
    }

    #[test]
    #[should_panic]
    fn negative_s_panics() {
        calcagni_d_s(-1.0);
    }
}
