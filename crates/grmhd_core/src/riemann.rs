//! HLL approximate Riemann solver for GRMHD.
//!
//! The Harten-Lax-van Leer (HLL) solver approximates the Riemann problem
//! with two waves (fastest left-going and right-going). Simple, robust,
//! and first-order at contact discontinuities (acceptable for MRI turbulence).
//!
//! F_HLL = (s_R * F_L - s_L * F_R + s_L * s_R * (U_R - U_L)) / (s_R - s_L)

use crate::cons::NCONS;

/// Compute HLL flux from left and right states.
///
/// Arguments:
/// - `f_l`: flux computed from left state
/// - `f_r`: flux computed from right state
/// - `u_l`: conservative variables from left state
/// - `u_r`: conservative variables from right state
/// - `s_l`: minimum wave speed (leftmost, typically negative)
/// - `s_r`: maximum wave speed (rightmost, typically positive)
///
/// Returns the HLL numerical flux (NCONS components).
pub fn hll_flux(
    f_l: &[f64; NCONS],
    f_r: &[f64; NCONS],
    u_l: &[f64; NCONS],
    u_r: &[f64; NCONS],
    s_l: f64,
    s_r: f64,
) -> [f64; NCONS] {
    let mut f_hll = [0.0f64; NCONS];

    if s_l >= 0.0 {
        // All waves move right: use left flux
        f_hll.copy_from_slice(f_l);
    } else if s_r <= 0.0 {
        // All waves move left: use right flux
        f_hll.copy_from_slice(f_r);
    } else {
        // Waves straddle: SIMD f64x4 for 8 components in 2 passes
        use wide::f64x4;
        let denom = f64x4::from([1.0 / (s_r - s_l); 4]);
        let sr4 = f64x4::from([s_r; 4]);
        let sl4 = f64x4::from([s_l; 4]);
        let slsr4 = f64x4::from([s_l * s_r; 4]);

        // First 4 components (mass, energy, 2 momenta)
        let fl_lo = f64x4::from([f_l[0], f_l[1], f_l[2], f_l[3]]);
        let fr_lo = f64x4::from([f_r[0], f_r[1], f_r[2], f_r[3]]);
        let ul_lo = f64x4::from([u_l[0], u_l[1], u_l[2], u_l[3]]);
        let ur_lo = f64x4::from([u_r[0], u_r[1], u_r[2], u_r[3]]);
        let hll_lo = (sr4 * fl_lo - sl4 * fr_lo + slsr4 * (ur_lo - ul_lo)) * denom;
        let lo = hll_lo.to_array();
        f_hll[0] = lo[0];
        f_hll[1] = lo[1];
        f_hll[2] = lo[2];
        f_hll[3] = lo[3];

        // Last 4 components (1 momentum + 3 B-field)
        let fl_hi = f64x4::from([f_l[4], f_l[5], f_l[6], f_l[7]]);
        let fr_hi = f64x4::from([f_r[4], f_r[5], f_r[6], f_r[7]]);
        let ul_hi = f64x4::from([u_l[4], u_l[5], u_l[6], u_l[7]]);
        let ur_hi = f64x4::from([u_r[4], u_r[5], u_r[6], u_r[7]]);
        let hll_hi = (sr4 * fl_hi - sl4 * fr_hi + slsr4 * (ur_hi - ul_hi)) * denom;
        let hi = hll_hi.to_array();
        f_hll[4] = hi[0];
        f_hll[5] = hi[1];
        f_hll[6] = hi[2];
        f_hll[7] = hi[3];
    }

    f_hll
}

/// Estimate maximum wave speeds for GRMHD.
///
/// For ideal MHD in GR, the fastest wave is the fast magnetosonic speed.
/// Simplified estimate: s = alpha * v +/- alpha * c_fast
/// where alpha is the lapse, v is the normal velocity, and c_fast is
/// the fast magnetosonic speed.
///
/// Returns (s_l, s_r) = (min wave speed, max wave speed).
pub fn wave_speeds(
    v_normal_l: f64,
    v_normal_r: f64,
    cs2_l: f64,
    cs2_r: f64,
    va2_l: f64,
    va2_r: f64,
    alpha: f64,
) -> (f64, f64) {
    // Fast magnetosonic speed squared: cf^2 ~ cs^2 + va^2 (simplified)
    let cf2_l = (cs2_l + va2_l).min(1.0); // cap at speed of light
    let cf2_r = (cs2_r + va2_r).min(1.0);

    let cf_l = cf2_l.sqrt();
    let cf_r = cf2_r.sqrt();

    // Signal speeds (in coordinate frame)
    let s_l = (alpha * (v_normal_l - cf_l)).min(alpha * (v_normal_r - cf_r));
    let s_r = (alpha * (v_normal_l + cf_l)).max(alpha * (v_normal_r + cf_r));

    (s_l, s_r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hll_upwind_left() {
        // All waves going right: should return left flux
        let f_l = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let f_r = [10.0; NCONS];
        let u_l = [0.0; NCONS];
        let u_r = [0.0; NCONS];
        let f = hll_flux(&f_l, &f_r, &u_l, &u_r, 0.1, 0.5);
        assert_eq!(f, f_l);
    }

    #[test]
    fn test_hll_upwind_right() {
        // All waves going left: should return right flux
        let f_l = [10.0; NCONS];
        let f_r = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let u_l = [0.0; NCONS];
        let u_r = [0.0; NCONS];
        let f = hll_flux(&f_l, &f_r, &u_l, &u_r, -0.5, -0.1);
        assert_eq!(f, f_r);
    }

    #[test]
    fn test_hll_symmetric() {
        // Equal states: flux should equal the common flux
        let f_common = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let u_common = [1.0; NCONS];
        let f = hll_flux(&f_common, &f_common, &u_common, &u_common, -0.5, 0.5);
        for k in 0..NCONS {
            assert!(
                (f[k] - f_common[k]).abs() < 1e-10,
                "Component {}: {} != {}",
                k,
                f[k],
                f_common[k]
            );
        }
    }

    #[test]
    fn test_wave_speeds() {
        let (sl, sr) = wave_speeds(0.0, 0.0, 0.01, 0.01, 0.0, 0.0, 1.0);
        // For static fluid with cs~0.1: speeds should be ~+/-0.1
        assert!(sl < 0.0 && sl > -0.2);
        assert!(sr > 0.0 && sr < 0.2);
    }
}
