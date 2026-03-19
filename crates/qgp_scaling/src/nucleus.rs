//! Nuclear density profiles and thickness functions.
//!
//! Implements hard-sphere uniform density for nuclei used in heavy-ion
//! collisions: Pb-208, Au-197, Xe-129.  The hard-sphere approximation
//! gives analytic thickness functions, avoiding numerical integration
//! of the density along the beam axis.
//!
//! Reference: Arleo & Falmagne arXiv:2411.13258, Sec. II.

use std::f64::consts::PI;

/// Radius parameter r_0 in the hard-sphere nuclear radius R_A = r_0 * A^(1/3).
/// Standard value from nuclear physics: 1.12 fm.
const R0_FM: f64 = 1.12;

/// Parameters for a nucleus used in Glauber calculations.
#[derive(Debug, Clone)]
pub struct NucleusParams {
    /// Mass number A.
    pub a: u32,
    /// Atomic number Z.
    pub z: u32,
    /// Hard-sphere radius R_A = r_0 * A^(1/3) in fm.
    pub r_a: f64,
    /// Nucleus name (e.g. "Pb-208").
    pub name: &'static str,
}

impl NucleusParams {
    /// Create nucleus parameters from mass and atomic numbers.
    /// Computes hard-sphere radius R_A = r_0 * A^(1/3).
    #[must_use]
    pub fn new(a: u32, z: u32, name: &'static str) -> Self {
        let r_a = R0_FM * (a as f64).cbrt();
        Self { a, z, r_a, name }
    }

    /// Pb-208 (lead).
    #[must_use]
    pub fn pb208() -> Self {
        Self::new(208, 82, "Pb-208")
    }

    /// Au-197 (gold).
    #[must_use]
    pub fn au197() -> Self {
        Self::new(197, 79, "Au-197")
    }

    /// Xe-129 (xenon).
    #[must_use]
    pub fn xe129() -> Self {
        Self::new(129, 54, "Xe-129")
    }

    /// O-16 (oxygen).
    #[must_use]
    pub fn o16() -> Self {
        Self::new(16, 8, "O-16")
    }

    /// Ne-20 (neon).
    /// Note: Ne-20 is significantly prolate ($\beta_2 \approx 0.72$).
    /// This hard-sphere model ignores deformation; use MC Glauber for shape studies.
    #[must_use]
    pub fn ne20() -> Self {
        Self::new(20, 10, "Ne-20")
    }

    /// Uniform (hard-sphere) nuclear density rho_0 = 3 / (4 * pi * R_A^3).
    /// Units: fm^{-3}.
    #[must_use]
    pub fn rho0(&self) -> f64 {
        3.0 / (4.0 * PI * self.r_a.powi(3))
    }
}

/// Hard-sphere nuclear density at radial distance r (fm).
///
/// Returns rho_0 for r <= R_A, 0 otherwise.
/// Normalized so that integral rho(r) d^3r = 1 (probability density).
#[must_use]
pub fn hard_sphere_density(r: f64, nuc: &NucleusParams) -> f64 {
    if r <= nuc.r_a { nuc.rho0() } else { 0.0 }
}

/// Nuclear thickness function T_A(s) for a hard-sphere nucleus.
///
/// T_A(s) = integral_{-inf}^{+inf} rho_A(sqrt(s^2 + z^2)) dz
///
/// For a uniform sphere of radius R_A, this is analytic:
///   T_A(s) = 2 * rho_0 * sqrt(R_A^2 - s^2)   for s <= R_A
///   T_A(s) = 0                                  for s >  R_A
///
/// where s is the transverse distance from the nuclear center.
/// Units: fm^{-2} (after integrating fm^{-3} over fm).
#[must_use]
pub fn thickness_function(s: f64, nuc: &NucleusParams) -> f64 {
    let s2 = s * s;
    let ra2 = nuc.r_a * nuc.r_a;
    if s2 < ra2 {
        2.0 * nuc.rho0() * (ra2 - s2).sqrt()
    } else {
        0.0
    }
}

/// Thickness function integrated over 2D transverse plane.
///
/// integral T_A(s) d^2s = 1  (by normalization of rho).
/// This is verified in tests as a consistency check.
#[must_use]
pub fn integrated_thickness(nuc: &NucleusParams, n_points: usize) -> f64 {
    // Numerical integration in polar coordinates:
    // integral_0^{R_A} T_A(s) * 2*pi*s ds
    let ds = nuc.r_a / n_points as f64;
    let mut sum = 0.0;
    for i in 0..n_points {
        let s = (i as f64 + 0.5) * ds;
        sum += thickness_function(s, nuc) * 2.0 * PI * s * ds;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pb208_radius() {
        let pb = NucleusParams::pb208();
        assert_eq!(pb.a, 208);
        assert_eq!(pb.z, 82);
        // R_A = 1.12 * 208^(1/3) = 1.12 * 5.928 = 6.639 fm
        assert!((pb.r_a - 6.639).abs() < 0.01, "Pb-208 R_A = {}", pb.r_a);
    }

    #[test]
    fn test_au197_radius() {
        let au = NucleusParams::au197();
        // R_A = 1.12 * 197^(1/3) = 1.12 * 5.819 = 6.517 fm
        assert!((au.r_a - 6.517).abs() < 0.01, "Au-197 R_A = {}", au.r_a);
    }

    #[test]
    fn test_xe129_radius() {
        let xe = NucleusParams::xe129();
        // R_A = 1.12 * 129^(1/3) = 1.12 * 5.052 = 5.658 fm
        assert!((xe.r_a - 5.66).abs() < 0.02, "Xe-129 R_A = {}", xe.r_a);
    }

    #[test]
    fn test_o16_radius() {
        let o16 = NucleusParams::o16();
        assert_eq!(o16.a, 16);
        assert!((o16.r_a - 2.82).abs() < 0.05);
    }

    #[test]
    fn test_hard_sphere_density_normalization() {
        // integral rho(r) 4*pi*r^2 dr from 0 to R_A should equal 1
        let pb = NucleusParams::pb208();
        let n = 10_000;
        let dr = pb.r_a / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let r = (i as f64 + 0.5) * dr;
            integral += hard_sphere_density(r, &pb) * 4.0 * PI * r * r * dr;
        }
        assert!(
            (integral - 1.0).abs() < 1e-4,
            "density normalization = {} (expected 1.0)",
            integral
        );
    }

    #[test]
    fn test_hard_sphere_density_outside() {
        let pb = NucleusParams::pb208();
        assert_eq!(hard_sphere_density(pb.r_a + 0.01, &pb), 0.0);
        assert_eq!(hard_sphere_density(100.0, &pb), 0.0);
    }

    #[test]
    fn test_thickness_function_center() {
        let pb = NucleusParams::pb208();
        // T_A(0) = 2 * rho_0 * R_A
        let expected = 2.0 * pb.rho0() * pb.r_a;
        let actual = thickness_function(0.0, &pb);
        assert!(
            (actual - expected).abs() < 1e-10,
            "T_A(0) = {} (expected {})",
            actual,
            expected
        );
    }

    #[test]
    fn test_thickness_function_edge() {
        let pb = NucleusParams::pb208();
        // T_A(R_A) = 0 (edge of nucleus)
        let val = thickness_function(pb.r_a, &pb);
        assert!(val.abs() < 1e-10, "T_A(R_A) = {} (expected 0)", val);
    }

    #[test]
    fn test_integrated_thickness_normalization() {
        // integral T_A(s) d^2s = 1
        let pb = NucleusParams::pb208();
        let integral = integrated_thickness(&pb, 10_000);
        assert!(
            (integral - 1.0).abs() < 1e-3,
            "integrated thickness = {} (expected 1.0)",
            integral
        );
    }
}
