//! Schwarzschild spacetime: metric, Christoffel symbols, and orbital properties.
//!
//! The Schwarzschild metric in Schwarzschild coordinates (G = c = 1):
//!
//!   ds^2 = -(1 - 2M/r) dt^2 + (1 - 2M/r)^{-1} dr^2
//!          + r^2 dtheta^2 + r^2 sin^2(theta) dphi^2
//!
//! This module provides:
//! - Exact closed-form metric components and Christoffel symbols
//! - Implementation of the `SpacetimeMetric` trait
//! - ISCO, photon sphere, and unstable circular orbit radii
//! - Effective potential for timelike and null geodesics
//!
//! # Literature
//! - Schwarzschild, K. (1916): Sitzungsber. Preuss. Akad. Wiss., 189
//! - Misner, Thorne, Wheeler (1973): Gravitation, Ch. 25
//! - Chandrasekhar (1983): The Mathematical Theory of Black Holes, Ch. 3

use crate::metric::{
    ChristoffelComponents, MetricComponents, SpacetimeMetric, DIM, PHI, R, T, THETA,
};

/// Schwarzschild black hole spacetime.
///
/// Parameterized by mass M in natural units (G = c = 1).
/// The metric is diagonal in Schwarzschild coordinates [t, r, theta, phi].
#[derive(Debug, Clone, Copy)]
pub struct Schwarzschild {
    /// Mass parameter (G = c = 1)
    pub mass: f64,
}

impl Schwarzschild {
    /// Create a Schwarzschild black hole with mass M.
    pub fn new(mass: f64) -> Self {
        assert!(mass > 0.0, "mass must be positive");
        Self { mass }
    }

    /// The metric function f(r) = 1 - 2M/r.
    pub fn f(&self, r: f64) -> f64 {
        1.0 - 2.0 * self.mass / r
    }

    /// df/dr = 2M/r^2
    pub fn df_dr(&self, r: f64) -> f64 {
        2.0 * self.mass / (r * r)
    }

    /// Schwarzschild radius r_s = 2M.
    pub fn schwarzschild_radius(&self) -> f64 {
        2.0 * self.mass
    }

    /// Effective potential for massive particles (timelike geodesics).
    ///
    /// V_eff(r) = (1 - 2M/r)(1 + L^2/r^2)
    ///
    /// where L is specific angular momentum (per unit rest mass).
    pub fn effective_potential_timelike(&self, r: f64, angular_momentum: f64) -> f64 {
        let f_r = self.f(r);
        let l2 = angular_momentum * angular_momentum;
        f_r * (1.0 + l2 / (r * r))
    }

    /// Effective potential for photons (null geodesics).
    ///
    /// V_eff(r) = (1 - 2M/r) / r^2
    ///
    /// The impact parameter b = L/E satisfies b^2 = r^2 / f(r) at turning points.
    pub fn effective_potential_null(&self, r: f64) -> f64 {
        self.f(r) / (r * r)
    }

    /// Radius of the unstable circular photon orbit.
    /// r_ph = 3M (only unstable orbit exists for Schwarzschild)
    pub fn photon_sphere(&self) -> f64 {
        3.0 * self.mass
    }

    /// Radius of the innermost stable circular orbit (ISCO).
    /// r_ISCO = 6M for Schwarzschild.
    pub fn isco(&self) -> f64 {
        6.0 * self.mass
    }

    /// Radius of the innermost bound circular orbit (IBCO).
    /// r_IBCO = 4M for Schwarzschild.
    pub fn ibco(&self) -> f64 {
        4.0 * self.mass
    }

    /// Critical impact parameter for photon capture.
    /// b_crit = 3*sqrt(3)*M
    pub fn critical_impact_parameter(&self) -> f64 {
        3.0 * 3.0_f64.sqrt() * self.mass
    }

    /// Surface gravity at the event horizon.
    /// kappa = 1/(4M)
    pub fn surface_gravity(&self) -> f64 {
        1.0 / (4.0 * self.mass)
    }

    /// Gravitational light deflection angle (weak-field approximation).
    ///
    /// delta_phi = 4M / b
    ///
    /// Valid when b >> r_s = 2M. The Sun deflects light by ~1.75 arcsec.
    /// This was the first experimental confirmation of general relativity
    /// (Eddington 1919).
    pub fn gravitational_deflection(&self, impact_param: f64) -> f64 {
        4.0 * self.mass / impact_param
    }

    /// Check if a photon with given impact parameter is captured.
    ///
    /// Returns true if b < b_crit = 3 sqrt(3) M.
    pub fn is_photon_captured(&self, impact_param: f64) -> bool {
        impact_param < self.critical_impact_parameter()
    }

    /// Shapiro time delay for light passing near a mass.
    ///
    /// Delta_t = 2M ln((r1 + r2 + d) / (r1 + r2 - d))
    ///
    /// where d = sqrt((r1 + r2)^2 - b^2) is the straight-line distance
    /// between source and observer projected along the line of sight.
    ///
    /// All quantities in natural units (G = c = 1). The delay is the
    /// excess travel time compared to a flat-space path.
    ///
    /// Irwin Shapiro proposed this "fourth test of GR" in 1964, first
    /// confirmed by radar ranging to Mercury and Venus (Shapiro et al. 1968).
    pub fn shapiro_delay(&self, r1: f64, r2: f64, impact_param: f64) -> f64 {
        let r_sum = r1 + r2;
        let d_sq = r_sum * r_sum - impact_param * impact_param;
        if d_sq <= 0.0 {
            return 0.0;
        }
        let d = d_sq.sqrt();
        2.0 * self.mass * ((r_sum + d) / (r_sum - d)).ln()
    }

    /// Gravitational time dilation factor at radius r.
    ///
    /// d_tau / dt = sqrt(1 - 2M/r) = sqrt(f(r))
    ///
    /// This is the ratio of proper time (clock at r) to coordinate time
    /// (clock at infinity). Approaches 0 at the horizon (r = 2M) and
    /// 1 at spatial infinity.
    pub fn time_dilation_factor(&self, r: f64) -> f64 {
        let f_r = self.f(r);
        if f_r <= 0.0 {
            return 0.0;
        }
        f_r.sqrt()
    }

    /// Radial orbital equation for null geodesics.
    ///
    /// (dr/dphi)^2 = r^4 / b^2 - r^2 (1 - 2M/r)
    ///
    /// where b is the impact parameter. Returns the value of (dr/dphi)^2.
    /// At turning points (closest approach), this equals zero.
    pub fn null_radial_equation(&self, r: f64, impact_param: f64) -> f64 {
        let b2 = impact_param * impact_param;
        let r2 = r * r;
        r2 * r2 / b2 - r2 * self.f(r)
    }

    /// Turning point radius for a photon with given impact parameter.
    ///
    /// The turning point is where dr/dphi = 0, i.e., the closest approach.
    /// For b > b_crit = 3 sqrt(3) M, a turning point exists outside the
    /// photon sphere. For b < b_crit, the photon is captured.
    ///
    /// Uses bisection between r_s (horizon) and b (far field) to find
    /// the root of the null radial equation.
    ///
    /// Returns None if the photon is captured (b < b_crit).
    pub fn photon_turning_point(&self, impact_param: f64) -> Option<f64> {
        if self.is_photon_captured(impact_param) {
            return None;
        }
        // Bisection: turning point is between photon sphere and b
        let r_ph = self.photon_sphere();
        let mut r_lo = r_ph;
        let mut r_hi = impact_param;
        for _ in 0..100 {
            let r_mid = 0.5 * (r_lo + r_hi);
            let val = self.null_radial_equation(r_mid, impact_param);
            if val < 0.0 {
                // Inside the forbidden region (between turning point and photon sphere)
                r_lo = r_mid;
            } else {
                r_hi = r_mid;
            }
        }
        Some(0.5 * (r_lo + r_hi))
    }
}

impl SpacetimeMetric for Schwarzschild {
    fn metric_components(&self, x: &[f64; DIM]) -> MetricComponents {
        let r = x[R];
        let theta = x[THETA];

        let f_r = self.f(r);
        let sin2 = theta.sin().powi(2);

        let mut g = [[0.0; DIM]; DIM];
        g[T][T] = -f_r;
        g[R][R] = 1.0 / f_r;
        g[THETA][THETA] = r * r;
        g[PHI][PHI] = r * r * sin2;
        g
    }

    fn inverse_metric(&self, x: &[f64; DIM]) -> MetricComponents {
        let r = x[R];
        let theta = x[THETA];

        let f_r = self.f(r);
        let sin2 = theta.sin().powi(2);

        let mut g_inv = [[0.0; DIM]; DIM];
        g_inv[T][T] = -1.0 / f_r;
        g_inv[R][R] = f_r;
        g_inv[THETA][THETA] = 1.0 / (r * r);
        g_inv[PHI][PHI] = 1.0 / (r * r * sin2);
        g_inv
    }

    /// Exact Christoffel symbols for Schwarzschild metric.
    ///
    /// The 9 non-zero symbols (out of 40 independent) are:
    ///
    /// Gamma^t_{tr} = M / (r(r-2M))
    /// Gamma^r_{tt} = M(r-2M) / r^3
    /// Gamma^r_{rr} = -M / (r(r-2M))
    /// Gamma^r_{theta theta} = -(r-2M)
    /// Gamma^r_{phi phi} = -(r-2M) sin^2(theta)
    /// Gamma^theta_{r theta} = 1/r
    /// Gamma^theta_{phi phi} = -sin(theta) cos(theta)
    /// Gamma^phi_{r phi} = 1/r
    /// Gamma^phi_{theta phi} = cos(theta) / sin(theta)
    fn christoffel(&self, x: &[f64; DIM]) -> ChristoffelComponents {
        let r = x[R];
        let theta = x[THETA];
        let m = self.mass;

        let sin_th = theta.sin();
        let cos_th = theta.cos();
        let sin2 = sin_th * sin_th;
        let r_2m = r - 2.0 * m;

        let mut gamma = [[[0.0; DIM]; DIM]; DIM];

        // Gamma^t_{tr} = Gamma^t_{rt} = M / (r * (r - 2M))
        let g_t_tr = m / (r * r_2m);
        gamma[T][T][R] = g_t_tr;
        gamma[T][R][T] = g_t_tr;

        // Gamma^r_{tt} = M * (r - 2M) / r^3
        gamma[R][T][T] = m * r_2m / (r * r * r);

        // Gamma^r_{rr} = -M / (r * (r - 2M))
        gamma[R][R][R] = -m / (r * r_2m);

        // Gamma^r_{theta theta} = -(r - 2M)
        gamma[R][THETA][THETA] = -r_2m;

        // Gamma^r_{phi phi} = -(r - 2M) * sin^2(theta)
        gamma[R][PHI][PHI] = -r_2m * sin2;

        // Gamma^theta_{r theta} = Gamma^theta_{theta r} = 1/r
        let g_th_rth = 1.0 / r;
        gamma[THETA][R][THETA] = g_th_rth;
        gamma[THETA][THETA][R] = g_th_rth;

        // Gamma^theta_{phi phi} = -sin(theta) * cos(theta)
        gamma[THETA][PHI][PHI] = -sin_th * cos_th;

        // Gamma^phi_{r phi} = Gamma^phi_{phi r} = 1/r
        let g_phi_rphi = 1.0 / r;
        gamma[PHI][R][PHI] = g_phi_rphi;
        gamma[PHI][PHI][R] = g_phi_rphi;

        // Gamma^phi_{theta phi} = Gamma^phi_{phi theta} = cos(theta) / sin(theta)
        let g_phi_thphi = cos_th / sin_th;
        gamma[PHI][THETA][PHI] = g_phi_thphi;
        gamma[PHI][PHI][THETA] = g_phi_thphi;

        gamma
    }

    fn event_horizon_radius(&self) -> Option<f64> {
        Some(self.schwarzschild_radius())
    }

    fn isco_radius(&self) -> Option<f64> {
        Some(self.isco())
    }

    fn photon_sphere_radius(&self) -> Option<f64> {
        Some(self.photon_sphere())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{
        full_curvature, ricci_from_riemann, riemann_from_christoffel,
    };

    const TOL: f64 = 1e-10;

    fn bh() -> Schwarzschild {
        Schwarzschild::new(1.0) // M = 1
    }

    // -- Metric tests --

    #[test]
    fn test_metric_at_r10() {
        let s = bh();
        let x = [0.0, 10.0, std::f64::consts::FRAC_PI_2, 0.0];
        let g = s.metric_components(&x);
        // g_tt = -(1 - 2/10) = -0.8
        assert!((g[T][T] - (-0.8)).abs() < TOL);
        // g_rr = 1 / (1 - 2/10) = 1.25
        assert!((g[R][R] - 1.25).abs() < TOL);
        // g_theta_theta = r^2 = 100
        assert!((g[THETA][THETA] - 100.0).abs() < TOL);
        // g_phi_phi = r^2 sin^2(pi/2) = 100
        assert!((g[PHI][PHI] - 100.0).abs() < TOL);
    }

    #[test]
    fn test_metric_diagonal() {
        let s = bh();
        let x = [0.0, 5.0, 1.0, 0.5];
        let g = s.metric_components(&x);
        // Off-diagonal should be zero
        for i in 0..DIM {
            for j in 0..DIM {
                if i != j {
                    assert!(
                        g[i][j].abs() < TOL,
                        "g[{}][{}] = {} (expected 0)",
                        i, j, g[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn test_inverse_metric_consistency() {
        let s = bh();
        let x = [0.0, 8.0, 1.2, 0.5];
        let g = s.metric_components(&x);
        let g_inv = s.inverse_metric(&x);
        // g * g_inv = I for diagonal metric
        for i in 0..DIM {
            let product = g[i][i] * g_inv[i][i];
            assert!(
                (product - 1.0).abs() < TOL,
                "g[{}][{}] * g_inv[{}][{}] = {} (expected 1)",
                i, i, i, i, product
            );
        }
    }

    // -- Christoffel symbol tests --

    #[test]
    fn test_christoffel_gamma_t_tr() {
        let s = bh();
        let r = 10.0;
        let x = [0.0, r, std::f64::consts::FRAC_PI_2, 0.0];
        let gamma = s.christoffel(&x);
        // Gamma^t_{tr} = M / (r(r-2M)) = 1 / (10 * 8) = 0.0125
        let expected = 1.0 / (r * (r - 2.0));
        assert!(
            (gamma[T][T][R] - expected).abs() < TOL,
            "Gamma^t_tr = {} (expected {})",
            gamma[T][T][R],
            expected
        );
        // Symmetric in lower indices
        assert!((gamma[T][R][T] - expected).abs() < TOL);
    }

    #[test]
    fn test_christoffel_gamma_r_tt() {
        let s = bh();
        let r = 10.0;
        let x = [0.0, r, std::f64::consts::FRAC_PI_2, 0.0];
        let gamma = s.christoffel(&x);
        // Gamma^r_{tt} = M(r-2M)/r^3 = 1*8/1000 = 0.008
        let expected = (r - 2.0) / (r * r * r);
        assert!(
            (gamma[R][T][T] - expected).abs() < TOL,
            "Gamma^r_tt = {} (expected {})",
            gamma[R][T][T],
            expected
        );
    }

    #[test]
    fn test_christoffel_gamma_r_rr() {
        let s = bh();
        let r = 10.0;
        let x = [0.0, r, std::f64::consts::FRAC_PI_2, 0.0];
        let gamma = s.christoffel(&x);
        // Gamma^r_{rr} = -M / (r(r-2M)) = -1/(10*8) = -0.0125
        let expected = -1.0 / (r * (r - 2.0));
        assert!(
            (gamma[R][R][R] - expected).abs() < TOL,
            "Gamma^r_rr = {} (expected {})",
            gamma[R][R][R],
            expected
        );
    }

    #[test]
    fn test_christoffel_symmetry() {
        let s = bh();
        let x = [0.0, 7.0, 1.0, 0.5];
        let gamma = s.christoffel(&x);
        // All Christoffel symbols should be symmetric in lower indices
        for a in 0..DIM {
            for m in 0..DIM {
                for n in 0..DIM {
                    assert!(
                        (gamma[a][m][n] - gamma[a][n][m]).abs() < TOL,
                        "Gamma^{}_{}{} != Gamma^{}_{}{} ({} vs {})",
                        a, m, n, a, n, m, gamma[a][m][n], gamma[a][n][m]
                    );
                }
            }
        }
    }

    #[test]
    fn test_christoffel_exact_vs_numerical() {
        let s = bh();
        let x = [0.0, 10.0, 1.0, 0.5];
        let exact = s.christoffel(&x);
        let numerical = crate::metric::christoffel_numerical(&s, &x);

        for a in 0..DIM {
            for m in 0..DIM {
                for n in 0..DIM {
                    let diff = (exact[a][m][n] - numerical[a][m][n]).abs();
                    let scale = exact[a][m][n].abs().max(1e-10);
                    assert!(
                        diff / scale < 1e-4,
                        "Gamma^{}_{}{}: exact={}, numerical={}, diff={}",
                        a, m, n, exact[a][m][n], numerical[a][m][n], diff
                    );
                }
            }
        }
    }

    // -- Vacuum Einstein equation test --

    #[test]
    fn test_vacuum_einstein_equation() {
        // Schwarzschild is a vacuum solution: R_{mu nu} = 0
        let s = bh();
        let x = [0.0, 10.0, std::f64::consts::FRAC_PI_2, 0.0];
        let riemann = riemann_from_christoffel(&s, &x);
        let ricci = ricci_from_riemann(&riemann);

        for mu in 0..DIM {
            for nu in 0..DIM {
                assert!(
                    ricci[mu][nu].abs() < 0.01,
                    "R_{}{} = {} (expected 0 for vacuum)",
                    mu, nu, ricci[mu][nu]
                );
            }
        }
    }

    // -- Kretschner scalar test --

    #[test]
    fn test_kretschner_scalar_schwarzschild() {
        // K = 48 M^2 / r^6
        let s = bh();
        let r = 10.0;
        let x = [0.0, r, std::f64::consts::FRAC_PI_2, 0.0];
        let result = full_curvature(&s, &x);

        let expected = 48.0 / r.powi(6);
        let relative_error = (result.kretschner - expected).abs() / expected;
        assert!(
            relative_error < 0.05,
            "Kretschner at r={}: got {}, expected {}, relative error={}",
            r, result.kretschner, expected, relative_error
        );
    }

    // -- Orbital property tests --

    #[test]
    fn test_event_horizon() {
        let s = bh();
        assert!((s.schwarzschild_radius() - 2.0).abs() < TOL);
        assert_eq!(s.event_horizon_radius(), Some(2.0));
    }

    #[test]
    fn test_photon_sphere() {
        let s = bh();
        assert!((s.photon_sphere() - 3.0).abs() < TOL);
        assert_eq!(s.photon_sphere_radius(), Some(3.0));
    }

    #[test]
    fn test_isco() {
        let s = bh();
        assert!((s.isco() - 6.0).abs() < TOL);
        assert_eq!(s.isco_radius(), Some(6.0));
    }

    #[test]
    fn test_ibco() {
        let s = bh();
        assert!((s.ibco() - 4.0).abs() < TOL);
    }

    #[test]
    fn test_critical_impact_parameter() {
        let s = bh();
        // b_crit = 3*sqrt(3) ~ 5.196
        let expected = 3.0 * 3.0_f64.sqrt();
        assert!((s.critical_impact_parameter() - expected).abs() < TOL);
    }

    #[test]
    fn test_surface_gravity() {
        let s = bh();
        // kappa = 1/(4M) = 0.25
        assert!((s.surface_gravity() - 0.25).abs() < TOL);
    }

    #[test]
    fn test_effective_potential_null_peak_at_photon_sphere() {
        let s = bh();
        let r_ph = s.photon_sphere();
        // dV/dr = 0 at photon sphere
        let dr = 1e-6;
        let v_minus = s.effective_potential_null(r_ph - dr);
        let v_plus = s.effective_potential_null(r_ph + dr);
        let deriv = (v_plus - v_minus) / (2.0 * dr);
        assert!(
            deriv.abs() < 1e-4,
            "dV_null/dr at r_ph = {} (expected ~0)",
            deriv
        );
    }

    #[test]
    fn test_mass_scaling() {
        // r_ISCO scales linearly with M
        let s1 = Schwarzschild::new(1.0);
        let s10 = Schwarzschild::new(10.0);
        assert!((s10.isco() / s1.isco() - 10.0).abs() < TOL);
        assert!((s10.photon_sphere() / s1.photon_sphere() - 10.0).abs() < TOL);
        assert!(
            (s10.schwarzschild_radius() / s1.schwarzschild_radius() - 10.0).abs() < TOL
        );
    }

    // -- Gravitational deflection --

    #[test]
    fn test_deflection_at_photon_sphere() {
        let s = bh();
        // At b = b_crit, deflection is large (not reliable from weak-field formula,
        // but should be finite and positive)
        let b = s.critical_impact_parameter();
        let delta = s.gravitational_deflection(b);
        assert!(delta > 0.0);
    }

    #[test]
    fn test_deflection_weak_field() {
        let s = bh();
        // At b = 1000 M, delta_phi = 4M/b = 4/1000 = 0.004 rad
        let b = 1000.0;
        let delta = s.gravitational_deflection(b);
        assert!((delta - 0.004).abs() < 1e-10);
    }

    #[test]
    fn test_deflection_decreases_with_distance() {
        let s = bh();
        let d1 = s.gravitational_deflection(10.0);
        let d2 = s.gravitational_deflection(100.0);
        assert!(d1 > d2, "Deflection should decrease with impact parameter");
    }

    // -- Photon capture --

    #[test]
    fn test_photon_captured_inside_critical() {
        let s = bh();
        let b_crit = s.critical_impact_parameter();
        assert!(s.is_photon_captured(b_crit * 0.9));
        assert!(!s.is_photon_captured(b_crit * 1.1));
    }

    // -- Shapiro delay --

    #[test]
    fn test_shapiro_delay_positive() {
        let s = bh();
        // Source and observer at r=100, impact parameter b=10
        let dt = s.shapiro_delay(100.0, 100.0, 10.0);
        assert!(dt > 0.0, "Shapiro delay = {dt}");
    }

    #[test]
    fn test_shapiro_delay_scales_with_mass() {
        let s1 = Schwarzschild::new(1.0);
        let s2 = Schwarzschild::new(2.0);
        let dt1 = s1.shapiro_delay(100.0, 100.0, 10.0);
        let dt2 = s2.shapiro_delay(100.0, 100.0, 10.0);
        assert!((dt2 / dt1 - 2.0).abs() < 1e-10, "Delay should scale linearly with M");
    }

    #[test]
    fn test_shapiro_delay_symmetric() {
        let s = bh();
        let dt1 = s.shapiro_delay(50.0, 100.0, 10.0);
        let dt2 = s.shapiro_delay(100.0, 50.0, 10.0);
        assert!((dt1 - dt2).abs() < TOL, "Delay should be symmetric in r1, r2");
    }

    // -- Time dilation --

    #[test]
    fn test_time_dilation_at_infinity() {
        let s = bh();
        let td = s.time_dilation_factor(1e10);
        assert!((td - 1.0).abs() < 1e-8, "Time dilation at infinity = {td}");
    }

    #[test]
    fn test_time_dilation_at_horizon() {
        let s = bh();
        let td = s.time_dilation_factor(s.schwarzschild_radius());
        assert!(td.abs() < TOL, "Time dilation at horizon = {td}");
    }

    #[test]
    fn test_time_dilation_at_r10() {
        let s = bh();
        // sqrt(1 - 2/10) = sqrt(0.8) ~ 0.894
        let td = s.time_dilation_factor(10.0);
        let expected = (0.8_f64).sqrt();
        assert!((td - expected).abs() < TOL, "td(r=10) = {td}, expected {expected}");
    }

    // -- Null radial equation --

    #[test]
    fn test_null_radial_equation_at_turning_point() {
        let s = bh();
        let b = 10.0; // b > b_crit ~ 5.196
        if let Some(r_turn) = s.photon_turning_point(b) {
            let val = s.null_radial_equation(r_turn, b);
            assert!(val.abs() < 1e-6, "Radial eq at turning point = {val}");
        }
    }

    // -- Photon turning point --

    #[test]
    fn test_turning_point_exists_above_critical() {
        let s = bh();
        let b = s.critical_impact_parameter() * 1.5;
        assert!(s.photon_turning_point(b).is_some());
    }

    #[test]
    fn test_turning_point_none_below_critical() {
        let s = bh();
        let b = s.critical_impact_parameter() * 0.5;
        assert!(s.photon_turning_point(b).is_none());
    }

    #[test]
    fn test_turning_point_outside_photon_sphere() {
        let s = bh();
        let b = 20.0;
        let r_turn = s.photon_turning_point(b).unwrap();
        assert!(
            r_turn > s.photon_sphere(),
            "Turning point r={r_turn} should be outside photon sphere r={}",
            s.photon_sphere()
        );
    }

    #[test]
    fn test_turning_point_approaches_photon_sphere_at_critical() {
        let s = bh();
        // As b -> b_crit from above, r_turn -> r_ph = 3M
        let b = s.critical_impact_parameter() * 1.001;
        let r_turn = s.photon_turning_point(b).unwrap();
        let r_ph = s.photon_sphere();
        assert!(
            (r_turn - r_ph).abs() < 0.1,
            "At b ~ b_crit, turning point {r_turn} should be near photon sphere {r_ph}"
        );
    }
}
