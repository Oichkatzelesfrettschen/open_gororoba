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
        full_curvature, kretschner_scalar, ricci_from_riemann, riemann_from_christoffel,
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
}
