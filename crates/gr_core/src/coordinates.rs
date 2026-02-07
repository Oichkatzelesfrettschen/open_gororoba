//! Coordinate transformations for black hole spacetimes.
//!
//! Provides conversions between:
//! - Boyer-Lindquist (BL): Standard spherical-like coordinates for Kerr
//! - Modified Kerr-Schild (MKS): Horizon-penetrating with logarithmic radial
//! - Cartesian: Observer coordinates
//!
//! Modified Kerr-Schild coordinates use:
//!   X^0 = t
//!   X^1 = ln(r/R0)  (logarithmic radial)
//!   X^2 = theta/pi + derefining correction
//!   X^3 = phi/(2 pi)
//!
//! The logarithmic radial mapping concentrates resolution near the horizon,
//! and the "derefining" function concentrates angular resolution near the
//! equator where accretion disks live.
//!
//! References:
//!   - McKinney & Gammie (2004): ApJ 611, 977 (MKS coordinates)
//!   - Gammie, McKinney, Toth (2003): ApJ 589, 444 (HARM formalism)

use std::f64::consts::PI;

/// Parameters for the Modified Kerr-Schild (MKS) coordinate system.
///
/// `r = R0 * exp(X1)` gives logarithmic radial spacing.
/// `theta = pi * X2 + (1 - hslope) * sin(2*pi*X2) / 2` gives adaptive
/// polar spacing, concentrating resolution near the equator.
#[derive(Debug, Clone, Copy)]
pub struct MksParams {
    /// Radial scale factor (default 1.0).
    pub r0: f64,
    /// Theta derefining parameter: 0 = maximum equatorial concentration,
    /// 1 = uniform (no derefining).
    pub hslope: f64,
}

impl Default for MksParams {
    fn default() -> Self {
        Self {
            r0: 1.0,
            hslope: 0.0,
        }
    }
}

/// Boyer-Lindquist coordinates (r, theta, phi).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlCoord {
    pub r: f64,
    pub theta: f64,
    pub phi: f64,
}

// ============================================================================
// MKS <-> Boyer-Lindquist
// ============================================================================

/// Convert MKS code coordinates (X1, X2, X3) to Boyer-Lindquist.
///
/// Mapping:
///   r = R0 * exp(X1)
///   theta = pi * X2 + (1 - hslope) * sin(2*pi*X2) / 2
///   phi = X3
pub fn mks_to_bl(x1: f64, x2: f64, x3: f64, params: &MksParams) -> BlCoord {
    let r = params.r0 * x1.exp();
    let theta = th_of_x2(x2, params.hslope);
    BlCoord {
        r,
        theta,
        phi: x3,
    }
}

/// Compute r from MKS radial coordinate X1.
///
/// r = R0 * exp(X1)
pub fn r_of_x1(x1: f64, r0: f64) -> f64 {
    r0 * x1.exp()
}

/// Compute X1 from BL radial coordinate r.
///
/// X1 = ln(r / R0)
pub fn x1_of_r(r: f64, r0: f64) -> f64 {
    (r / r0).ln()
}

/// Compute theta from MKS polar coordinate X2.
///
/// theta = pi * X2 + (1 - hslope) * sin(2*pi*X2) / 2
///
/// hslope = 0: Maximum derefining (concentrate resolution at equator).
/// hslope = 1: No derefining (uniform/linear spacing).
///
/// Result is clamped to (epsilon, pi - epsilon) to avoid coordinate poles.
pub fn th_of_x2(x2: f64, hslope: f64) -> f64 {
    let theta = if (hslope - 1.0).abs() < 1e-10 {
        PI * x2
    } else {
        PI * x2 + (1.0 - hslope) * (2.0 * PI * x2).sin() / 2.0
    };
    theta.clamp(1e-10, PI - 1e-10)
}

/// Derivative dr/dX1.
///
/// For r = R0 * exp(X1): dr/dX1 = r.
pub fn dr_dx1(x1: f64, r0: f64) -> f64 {
    r0 * x1.exp()
}

/// Derivative dtheta/dX2.
///
/// dtheta/dX2 = pi * (1 + (1 - hslope) * cos(2*pi*X2))
///
/// Constant pi when hslope = 1 (uniform spacing).
pub fn dth_dx2(x2: f64, hslope: f64) -> f64 {
    if (hslope - 1.0).abs() < 1e-10 {
        return PI;
    }
    PI * (1.0 + (1.0 - hslope) * (2.0 * PI * x2).cos())
}

// ============================================================================
// Spherical <-> Cartesian
// ============================================================================

/// Convert spherical (r, theta, phi) to Cartesian (x, y, z).
///
///   x = r sin(theta) cos(phi)
///   y = r sin(theta) sin(phi)
///   z = r cos(theta)
pub fn spherical_to_cartesian(r: f64, theta: f64, phi: f64) -> [f64; 3] {
    let st = theta.sin();
    let ct = theta.cos();
    let sp = phi.sin();
    let cp = phi.cos();
    [r * st * cp, r * st * sp, r * ct]
}

/// Convert Cartesian (x, y, z) to spherical (r, theta, phi).
///
/// phi is normalized to [0, 2*pi).
pub fn cartesian_to_spherical(x: f64, y: f64, z: f64) -> BlCoord {
    let r = (x * x + y * y + z * z).sqrt();
    if r < 1e-30 {
        return BlCoord {
            r: 0.0,
            theta: 0.0,
            phi: 0.0,
        };
    }
    let theta = (z / r).clamp(-1.0, 1.0).acos();
    let mut phi = y.atan2(x);
    if phi < 0.0 {
        phi += 2.0 * PI;
    }
    BlCoord { r, theta, phi }
}

// ============================================================================
// Kerr-Schild <-> Boyer-Lindquist
// ============================================================================

/// Convert Kerr-Schild r to Boyer-Lindquist r (they are identical).
///
/// In Kerr-Schild (KS) and Boyer-Lindquist (BL) coordinates, the radial
/// and polar coordinates are the same: r_BL = r_KS, theta_BL = theta_KS.
/// Only the time and azimuthal coordinates differ by integral terms that
/// depend on the geodesic history.
pub fn ks_r_to_bl_r(r_ks: f64) -> f64 {
    r_ks
}

/// The BL-to-KS phi correction (simplified for distant observers).
///
/// The full transformation phi_BL = phi_KS - integral(a / Delta dr) requires
/// the geodesic history. For distant observers (r >> M), the correction
/// vanishes. This returns the approximate correction.
pub fn ks_phi_correction(r: f64, mass: f64, spin: f64) -> f64 {
    let delta = r * r - 2.0 * mass * r + spin * spin;
    if delta.abs() < 1e-30 {
        return 0.0;
    }
    // At large r, the integrand a/Delta ~ a/r^2 -> 0
    // Near the horizon this diverges; the simplified version returns 0
    // (full version needs numerical integration along the ray)
    let _ = delta;
    0.0
}

// ============================================================================
// Jacobian for MKS -> BL
// ============================================================================

/// Diagonal Jacobian elements for MKS -> BL coordinate transformation.
///
/// Returns (dr/dX1, dtheta/dX2, dphi/dX3).
///
/// Since the MKS transformation is diagonal (each BL coordinate depends
/// on only one MKS coordinate), the Jacobian is diagonal. This is used
/// for transforming vectors and tensors between coordinate systems.
pub fn mks_to_bl_jacobian(x1: f64, x2: f64, params: &MksParams) -> (f64, f64, f64) {
    let dr = dr_dx1(x1, params.r0);
    let dth = dth_dx2(x2, params.hslope);
    let dphi = 1.0; // phi = X3 directly
    (dr, dth, dphi)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn test_r_round_trip() {
        let r0 = 1.0;
        for &r in &[1.0, 5.0, 10.0, 100.0, 0.5] {
            let x1 = x1_of_r(r, r0);
            let r_back = r_of_x1(x1, r0);
            assert!((r_back - r).abs() < TOL * r, "r={r}: got {r_back}");
        }
    }

    #[test]
    fn test_r_round_trip_nonunit_r0() {
        let r0 = 2.5;
        let r = 10.0;
        let x1 = x1_of_r(r, r0);
        let r_back = r_of_x1(x1, r0);
        assert!((r_back - r).abs() < TOL * r);
    }

    #[test]
    fn test_th_uniform_hslope_1() {
        // hslope=1: theta = pi * X2 (uniform)
        for &x2 in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let theta = th_of_x2(x2, 1.0);
            let expected = (PI * x2).clamp(1e-10, PI - 1e-10);
            assert!(
                (theta - expected).abs() < TOL,
                "x2={x2}: got {theta}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_th_derefine_hslope_0() {
        // hslope=0: theta = pi * X2 + sin(2*pi*X2) / 2
        // At X2=0.5 (equator): sin(pi)=0, so theta = pi/2 exactly
        let theta = th_of_x2(0.5, 0.0);
        assert!((theta - PI / 2.0).abs() < TOL);

        // At X2=0.25: theta = pi/4 + sin(pi/2)/2 = pi/4 + 0.5
        let theta = th_of_x2(0.25, 0.0);
        let expected = PI * 0.25 + 0.5;
        assert!((theta - expected).abs() < TOL);
    }

    #[test]
    fn test_dth_dx2_uniform() {
        // hslope=1: dtheta/dX2 = pi (constant)
        assert!((dth_dx2(0.0, 1.0) - PI).abs() < TOL);
        assert!((dth_dx2(0.5, 1.0) - PI).abs() < TOL);
    }

    #[test]
    fn test_dth_dx2_derefine() {
        // hslope=0: dtheta/dX2 = pi * (1 + cos(2*pi*X2))
        // At X2=0: cos(0) = 1, so dth = 2*pi (maximum stretching at poles)
        assert!((dth_dx2(0.0, 0.0) - 2.0 * PI).abs() < TOL);

        // At X2=0.5: cos(pi) = -1, so dth = 0 (maximum compression at equator)
        assert!(dth_dx2(0.5, 0.0).abs() < TOL);
    }

    #[test]
    fn test_dr_dx1_equals_r() {
        let r0 = 1.0;
        let x1 = 2.0;
        let r = r_of_x1(x1, r0);
        let dr = dr_dx1(x1, r0);
        assert!((dr - r).abs() < TOL * r);
    }

    #[test]
    fn test_spherical_cartesian_round_trip() {
        let r = 5.0;
        let theta = 1.0; // ~57 degrees
        let phi = 2.0; // ~115 degrees
        let [x, y, z] = spherical_to_cartesian(r, theta, phi);
        let bl = cartesian_to_spherical(x, y, z);
        assert!((bl.r - r).abs() < TOL * r);
        assert!((bl.theta - theta).abs() < TOL);
        assert!((bl.phi - phi).abs() < TOL);
    }

    #[test]
    fn test_cartesian_axes() {
        // Along +z: theta=0
        let bl = cartesian_to_spherical(0.0, 0.0, 1.0);
        assert!((bl.r - 1.0).abs() < TOL);
        assert!(bl.theta.abs() < TOL);

        // Along +x: theta=pi/2, phi=0
        let bl = cartesian_to_spherical(1.0, 0.0, 0.0);
        assert!((bl.r - 1.0).abs() < TOL);
        assert!((bl.theta - PI / 2.0).abs() < TOL);
        assert!(bl.phi.abs() < TOL);

        // Along +y: theta=pi/2, phi=pi/2
        let bl = cartesian_to_spherical(0.0, 1.0, 0.0);
        assert!((bl.r - 1.0).abs() < TOL);
        assert!((bl.theta - PI / 2.0).abs() < TOL);
        assert!((bl.phi - PI / 2.0).abs() < TOL);
    }

    #[test]
    fn test_cartesian_origin() {
        let bl = cartesian_to_spherical(0.0, 0.0, 0.0);
        assert!(bl.r < 1e-30);
    }

    #[test]
    fn test_mks_to_bl_default() {
        let params = MksParams::default(); // r0=1, hslope=0
        let bl = mks_to_bl(0.0, 0.5, 1.0, &params);
        // X1=0 -> r = 1*exp(0) = 1
        assert!((bl.r - 1.0).abs() < TOL);
        // X2=0.5 with hslope=0 -> theta = pi/2
        assert!((bl.theta - PI / 2.0).abs() < TOL);
        // phi = X3 = 1.0
        assert!((bl.phi - 1.0).abs() < TOL);
    }

    #[test]
    fn test_mks_jacobian() {
        let params = MksParams {
            r0: 1.0,
            hslope: 1.0,
        };
        let x1 = 1.0_f64.ln(); // r = 1
        let (dr, dth, dphi) = mks_to_bl_jacobian(x1, 0.5, &params);
        assert!((dr - 1.0).abs() < TOL); // dr/dX1 = r = 1
        assert!((dth - PI).abs() < TOL); // uniform: dth/dX2 = pi
        assert!((dphi - 1.0).abs() < TOL);
    }

    #[test]
    fn test_ks_r_identity() {
        assert!((ks_r_to_bl_r(10.0) - 10.0).abs() < TOL);
    }

    #[test]
    fn test_th_poles_clamped() {
        // X2=0 maps to theta~0 but clamped to 1e-10
        let theta = th_of_x2(0.0, 1.0);
        assert!(theta > 0.0);
        assert!(theta < 1e-9);

        // X2=1 maps to theta~pi but clamped to pi - 1e-10
        let theta = th_of_x2(1.0, 1.0);
        assert!(theta < PI);
        assert!((theta - PI).abs() < 1e-9);
    }
}
