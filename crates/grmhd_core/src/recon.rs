//! Piecewise-linear reconstruction with minmod slope limiter.
//!
//! Reconstructs left and right states at cell interfaces for the
//! Riemann solver. The minmod limiter prevents oscillations at
//! discontinuities (TVD property).

/// Minmod function: returns the value with smallest magnitude if both
/// have the same sign, zero otherwise.
#[inline]
pub fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 {
        0.0
    } else if a.abs() < b.abs() {
        a
    } else {
        b
    }
}

/// Reconstruct left and right states at the i+1/2 interface.
///
/// Given cell-centered values q[i-1], q[i], q[i+1]:
///   q_L = q[i] + 0.5 * minmod(q[i]-q[i-1], q[i+1]-q[i])
///   q_R = q[i+1] - 0.5 * minmod(q[i+1]-q[i], q[i+2]-q[i+1])
///
/// Returns (q_left, q_right) at the i+1/2 face.
#[inline]
pub fn plm_lr(qim1: f64, qi: f64, qip1: f64, qip2: f64) -> (f64, f64) {
    let slope_l = minmod(qi - qim1, qip1 - qi);
    let slope_r = minmod(qip1 - qi, qip2 - qip1);
    let q_left = qi + 0.5 * slope_l;
    let q_right = qip1 - 0.5 * slope_r;
    (q_left, q_right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minmod_same_sign() {
        assert_eq!(minmod(2.0, 3.0), 2.0);
        assert_eq!(minmod(-1.0, -5.0), -1.0);
    }

    #[test]
    fn test_minmod_opposite_sign() {
        assert_eq!(minmod(1.0, -1.0), 0.0);
        assert_eq!(minmod(-2.0, 3.0), 0.0);
    }

    #[test]
    fn test_plm_smooth() {
        // Linear profile: q = 1, 2, 3, 4
        let (ql, qr) = plm_lr(1.0, 2.0, 3.0, 4.0);
        // PLM should exactly reconstruct linear profiles
        assert!((ql - 2.5).abs() < 1e-10);
        assert!((qr - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_plm_discontinuity() {
        // Step function: q = 1, 1, 10, 10
        let (ql, qr) = plm_lr(1.0, 1.0, 10.0, 10.0);
        // Minmod should limit: slope_l = minmod(0, 9) = 0
        // So q_left = 1.0 (no extrapolation across discontinuity)
        assert!((ql - 1.0).abs() < 1e-10);
        assert!((qr - 10.0).abs() < 1e-10);
    }
}
