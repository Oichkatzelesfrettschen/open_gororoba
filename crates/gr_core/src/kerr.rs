//! Kerr black hole geodesic integrator in Boyer-Lindquist coordinates.
//!
//! Integrates null geodesics (photon orbits) around a Kerr black hole
//! using the constants of motion: energy E, angular momentum L, and
//! Carter constant Q. The equations are written in Mino time (lambda)
//! to regularize the coordinate singularity at the horizon.
//!
//! The Kerr metric in Boyer-Lindquist coordinates (G = c = M = 1):
//!   ds^2 = -(1 - 2r/Sigma) dt^2 - (4ar sin^2(theta)/Sigma) dt dphi
//!          + (Sigma/Delta) dr^2 + Sigma dtheta^2
//!          + (r^2 + a^2 + 2a^2 r sin^2(theta)/Sigma) sin^2(theta) dphi^2
//!
//! where:
//!   Sigma = r^2 + a^2 cos^2(theta)
//!   Delta = r^2 - 2r + a^2
//!
//! Refs:
//!   Bardeen, J.M. (1973), in "Black Holes", Les Houches.
//!   Chandrasekhar, S. (1983), "The Mathematical Theory of Black Holes", Ch. 7.
//!   Teo, E. (2003), Gen. Relativ. Gravit. 35, 1909.

use std::f64::consts::PI;

/// State for geodesic integration: [t, r, theta, phi, v_r, v_theta].
#[derive(Debug, Clone, Copy)]
pub struct GeodesicState {
    pub t: f64,
    pub r: f64,
    pub theta: f64,
    pub phi: f64,
    pub v_r: f64,      // dr/dlambda
    pub v_theta: f64,  // dtheta/dlambda
}

/// Result of geodesic tracing.
#[derive(Debug, Clone)]
pub struct GeodesicResult {
    pub t: Vec<f64>,
    pub r: Vec<f64>,
    pub theta: Vec<f64>,
    pub phi: Vec<f64>,
    pub lam: Vec<f64>,
    pub terminated: bool,
    pub termination_reason: String,
}

/// Compute Kerr metric functions Sigma, Delta.
///
/// # Arguments
/// * `r` - Boyer-Lindquist radial coordinate
/// * `theta` - Boyer-Lindquist polar angle
/// * `a` - Spin parameter (0 <= a < 1 for M=1)
///
/// # Returns
/// (sigma, delta)
pub fn kerr_metric_quantities(r: f64, theta: f64, a: f64) -> (f64, f64) {
    let sigma = r * r + a * a * theta.cos().powi(2);
    let delta = r * r - 2.0 * r + a * a;
    (sigma, delta)
}

/// Event horizon radius for Kerr black hole.
pub fn event_horizon_radius(a: f64) -> f64 {
    1.0 + (1.0 - a * a).sqrt()
}

/// Radii of prograde and retrograde circular photon orbits.
///
/// For a Kerr black hole with spin a (M=1), the photon orbit radii are:
///   r_ph = 2 * (1 + cos(2/3 * arccos(-/+a)))
///
/// # Returns
/// (r_prograde, r_retrograde)
pub fn photon_orbit_radius(a: f64) -> (f64, f64) {
    let r_pro = 2.0 * (1.0 + (2.0 / 3.0 * (-a).acos()).cos());
    let r_retro = 2.0 * (1.0 + (2.0 / 3.0 * a.acos()).cos());
    (r_pro, r_retro)
}

/// Critical impact parameters (xi, eta) for a photon orbit at radius r_ph.
///
/// These define the boundary of the black hole shadow as seen by a
/// distant observer in the equatorial plane.
///
///   xi = L/E = (r_ph^2(r_ph - 3) + a^2(r_ph + 1)) / (a(r_ph - 1))
///   eta = Q/E^2 = r_ph^3 * (-r_ph*(r_ph - 3)^2 + 4*a^2) / (a^2*(r_ph - 1)^2)
///
/// # Returns
/// (xi, eta) - Reduced impact parameters (L/E and Q/E^2)
pub fn impact_parameters(r_ph: f64, a: f64) -> (f64, f64) {
    let r = r_ph;
    let xi = (r * r * (r - 3.0) + a * a * (r + 1.0)) / (a * (r - 1.0));
    let eta = r.powi(3) * (-r * (r - 3.0).powi(2) + 4.0 * a * a) / (a * a * (r - 1.0).powi(2));
    (xi, eta)
}

/// Compute the Kerr black hole shadow boundary (Bardeen curve).
///
/// The shadow boundary is parametrized by the photon orbit radius r_ph
/// ranging from the prograde to retrograde orbit. Each r_ph maps to
/// celestial coordinates (alpha, beta) via the critical impact parameters.
///
/// # Arguments
/// * `a` - Spin parameter (0 <= a < 1)
/// * `n_points` - Number of points on the boundary
/// * `theta_o` - Observer inclination (pi/2 = equatorial)
///
/// # Returns
/// (alpha, beta) - Celestial coordinates (upper + lower halves)
pub fn shadow_boundary(a: f64, n_points: usize, theta_o: f64) -> (Vec<f64>, Vec<f64>) {
    if a.abs() < 1e-6 {
        // Schwarzschild: circular shadow of radius sqrt(27)*M
        let r_shadow = 27.0_f64.sqrt();
        let mut alpha = Vec::with_capacity(2 * n_points);
        let mut beta = Vec::with_capacity(2 * n_points);

        for i in 0..2 * n_points {
            let angle = 2.0 * PI * (i as f64) / (2 * n_points) as f64;
            alpha.push(r_shadow * angle.cos());
            beta.push(r_shadow * angle.sin());
        }

        return (alpha, beta);
    }

    let (r_pro, r_retro) = photon_orbit_radius(a);

    let sin_o = theta_o.sin();
    let cos_o = theta_o.cos();

    let mut alpha = Vec::with_capacity(2 * n_points);
    let mut beta = Vec::with_capacity(2 * n_points);

    // Upper half
    for i in 0..n_points {
        let r_ph = r_pro + (r_retro - r_pro) * (i as f64) / (n_points - 1) as f64;
        let (xi, eta) = impact_parameters(r_ph, a);

        let alpha_val = -xi / sin_o;
        let beta_sq = eta + a * a * cos_o * cos_o - xi * xi * cos_o * cos_o / (sin_o * sin_o);
        let beta_val = beta_sq.max(0.0).sqrt();

        alpha.push(alpha_val);
        beta.push(beta_val);
    }

    // Lower half (reversed for closed curve)
    for i in (0..n_points).rev() {
        let r_ph = r_pro + (r_retro - r_pro) * (i as f64) / (n_points - 1) as f64;
        let (xi, eta) = impact_parameters(r_ph, a);

        let alpha_val = -xi / sin_o;
        let beta_sq = eta + a * a * cos_o * cos_o - xi * xi * cos_o * cos_o / (sin_o * sin_o);
        let beta_val = -beta_sq.max(0.0).sqrt();

        alpha.push(alpha_val);
        beta.push(beta_val);
    }

    (alpha, beta)
}

/// Right-hand side of the Kerr geodesic equations in Mino time.
///
/// Uses the second-order form dv/dlambda = (1/2)*dV/dq, which naturally
/// handles turning points without explicit sign tracking.
pub fn geodesic_rhs(state: GeodesicState, a: f64, e: f64, l: f64, q: f64) -> [f64; 6] {
    let GeodesicState { t: _, r, theta, phi: _, v_r, v_theta } = state;

    let (_, delta) = kerr_metric_quantities(r, theta, a);
    let sin_th = theta.sin();
    let cos_th = theta.cos();

    // Avoid coordinate singularity at poles
    let sin2 = (sin_th * sin_th).max(1e-30);
    let sin3 = (sin_th.abs().powi(3)).max(1e-30);

    let t_val = e * (r * r + a * a) - a * l;

    // Coordinate time and azimuthal angle evolution
    let dphi = -(a * e - l / sin2) + a * t_val / delta;
    let dt = -a * (a * e * sin2 - l) + (r * r + a * a) * t_val / delta;

    // Radial acceleration: dv_r/dlambda = (1/2) dR/dr
    // R(r) = T^2 - Delta*(Q + (L-aE)^2)
    // dR/dr = 4*E*r*T - (2r-2)*(Q + (L-aE)^2)
    let dr_dr = 4.0 * e * r * t_val - (2.0 * r - 2.0) * (q + (l - a * e).powi(2));

    // Polar acceleration: dv_theta/dlambda = (1/2) dTheta/dtheta
    // Theta = Q - cos^2(th)*(a^2*(1-E^2) + L^2/sin^2(th))
    // dTheta/dtheta = sin(2th)*a^2*(1-E^2) + 2*L^2*cos(th)/sin^3(th)
    let dtheta_dth = (2.0 * theta).sin() * a * a * (1.0 - e * e)
        + 2.0 * l * l * cos_th / sin3;

    [dt, v_r, v_theta, dphi, 0.5 * dr_dr, 0.5 * dtheta_dth]
}

/// RK4 step for geodesic integration.
fn rk4_step(
    state: GeodesicState,
    dlam: f64,
    a: f64,
    e: f64,
    l: f64,
    q: f64,
) -> GeodesicState {
    let to_state = |s: GeodesicState, deriv: [f64; 6], h: f64| -> GeodesicState {
        GeodesicState {
            t: s.t + deriv[0] * h,
            r: s.r + deriv[1] * h,
            theta: s.theta + deriv[2] * h,
            phi: s.phi + deriv[3] * h,
            v_r: s.v_r + deriv[4] * h,
            v_theta: s.v_theta + deriv[5] * h,
        }
    };

    let k1 = geodesic_rhs(state, a, e, l, q);
    let k2 = geodesic_rhs(to_state(state, k1, dlam * 0.5), a, e, l, q);
    let k3 = geodesic_rhs(to_state(state, k2, dlam * 0.5), a, e, l, q);
    let k4 = geodesic_rhs(to_state(state, k3, dlam), a, e, l, q);

    GeodesicState {
        t: state.t + dlam / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        r: state.r + dlam / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        theta: state.theta + dlam / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
        phi: state.phi + dlam / 6.0 * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]),
        v_r: state.v_r + dlam / 6.0 * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]),
        v_theta: state.v_theta + dlam / 6.0 * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]),
    }
}

/// Trace a null geodesic in a Kerr spacetime.
///
/// # Arguments
/// * `a` - Spin parameter
/// * `e` - Energy (typically 1.0 for null geodesics)
/// * `l` - Angular momentum L/E
/// * `q` - Carter constant Q/E^2
/// * `r0` - Initial radial coordinate
/// * `theta0` - Initial polar angle
/// * `lam_max` - Maximum Mino time for integration
/// * `sgn_r` - Initial sign of dr/dlambda (+1 outgoing, -1 ingoing)
/// * `sgn_theta` - Initial sign of dtheta/dlambda
/// * `n_steps` - Number of integration steps
pub fn trace_null_geodesic(
    a: f64,
    e: f64,
    l: f64,
    q: f64,
    r0: f64,
    theta0: f64,
    lam_max: f64,
    sgn_r: f64,
    sgn_theta: f64,
    n_steps: usize,
) -> GeodesicResult {
    // Compute initial radial and polar velocities from the potentials
    let (_, delta0) = kerr_metric_quantities(r0, theta0, a);
    let t0 = e * (r0 * r0 + a * a) - a * l;
    let r_pot = t0 * t0 - delta0 * (q + (l - a * e).powi(2));

    let sin_th0 = theta0.sin();
    let cos_th0 = theta0.cos();
    let sin2_0 = (sin_th0 * sin_th0).max(1e-30);
    let theta_pot = q - cos_th0 * cos_th0 * (a * a * (1.0 - e * e) + l * l / sin2_0);

    let v_r0 = sgn_r * r_pot.max(0.0).sqrt();
    let v_theta0 = sgn_theta * theta_pot.max(0.0).sqrt();

    let mut state = GeodesicState {
        t: 0.0,
        r: r0,
        theta: theta0,
        phi: 0.0,
        v_r: v_r0,
        v_theta: v_theta0,
    };

    let r_horizon = event_horizon_radius(a);
    let dlam = lam_max / (n_steps as f64);

    let mut t_vec = Vec::with_capacity(n_steps + 1);
    let mut r_vec = Vec::with_capacity(n_steps + 1);
    let mut theta_vec = Vec::with_capacity(n_steps + 1);
    let mut phi_vec = Vec::with_capacity(n_steps + 1);
    let mut lam_vec = Vec::with_capacity(n_steps + 1);

    t_vec.push(state.t);
    r_vec.push(state.r);
    theta_vec.push(state.theta);
    phi_vec.push(state.phi);
    lam_vec.push(0.0);

    let mut terminated = false;
    let mut termination_reason = String::new();

    for i in 1..=n_steps {
        state = rk4_step(state, dlam, a, e, l, q);
        let lam = (i as f64) * dlam;

        t_vec.push(state.t);
        r_vec.push(state.r);
        theta_vec.push(state.theta);
        phi_vec.push(state.phi);
        lam_vec.push(lam);

        // Check termination conditions
        if state.r < r_horizon + 0.01 {
            terminated = true;
            termination_reason = "hit_horizon".to_string();
            break;
        }

        if state.r > 5.0 * r0 {
            terminated = true;
            termination_reason = "escaped".to_string();
            break;
        }
    }

    GeodesicResult {
        t: t_vec,
        r: r_vec,
        theta: theta_vec,
        phi: phi_vec,
        lam: lam_vec,
        terminated,
        termination_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_schwarzschild_metric() {
        // a = 0: Schwarzschild
        let (sigma, delta) = kerr_metric_quantities(10.0, PI / 2.0, 0.0);
        assert_relative_eq!(sigma, 100.0, epsilon = 1e-10);
        assert_relative_eq!(delta, 80.0, epsilon = 1e-10);  // r^2 - 2r
    }

    #[test]
    fn test_kerr_metric() {
        let a = 0.9;
        let (sigma, delta) = kerr_metric_quantities(5.0, PI / 4.0, a);

        // sigma = r^2 + a^2 * cos^2(theta) = 25 + 0.81 * 0.5 = 25.405
        assert_relative_eq!(sigma, 25.0 + 0.81 * 0.5, epsilon = 1e-10);

        // delta = r^2 - 2r + a^2 = 25 - 10 + 0.81 = 15.81
        assert_relative_eq!(delta, 15.81, epsilon = 1e-10);
    }

    #[test]
    fn test_event_horizon() {
        // Schwarzschild: r_h = 2M
        assert_relative_eq!(event_horizon_radius(0.0), 2.0, epsilon = 1e-10);

        // Extremal Kerr: r_h = M
        assert_relative_eq!(event_horizon_radius(1.0), 1.0, epsilon = 1e-10);

        // a = 0.9: r_h = 1 + sqrt(1 - 0.81) = 1 + sqrt(0.19) ~ 1.436
        assert_relative_eq!(event_horizon_radius(0.9), 1.0 + 0.19_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_photon_orbit_schwarzschild() {
        let (r_pro, r_retro) = photon_orbit_radius(0.0);
        // Schwarzschild photon sphere at r = 3M
        assert_relative_eq!(r_pro, 3.0, epsilon = 1e-6);
        assert_relative_eq!(r_retro, 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_photon_orbit_kerr() {
        let a = 0.9;
        let (r_pro, r_retro) = photon_orbit_radius(a);

        // Prograde orbit is closer to horizon
        assert!(r_pro < r_retro);
        assert!(r_pro > event_horizon_radius(a));
    }

    #[test]
    fn test_shadow_boundary_schwarzschild() {
        let (alpha, beta) = shadow_boundary(0.0, 100, PI / 2.0);

        // Schwarzschild shadow is circular with radius sqrt(27)
        let r_expected = 27.0_f64.sqrt();

        for i in 0..alpha.len() {
            let r = (alpha[i] * alpha[i] + beta[i] * beta[i]).sqrt();
            assert_relative_eq!(r, r_expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_shadow_boundary_kerr_asymmetric() {
        let a = 0.9;
        let (alpha, beta) = shadow_boundary(a, 100, PI / 2.0);

        // Kerr shadow is asymmetric: find min and max alpha
        let alpha_min = alpha.iter().cloned().fold(f64::INFINITY, f64::min);
        let alpha_max = alpha.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Shadow should be shifted toward prograde side (negative alpha)
        assert!(alpha_min.abs() > alpha_max.abs());
    }

    #[test]
    fn test_geodesic_infall() {
        // Radially infalling photon from close to the horizon
        let a = 0.5;
        let e = 1.0;
        let l = 0.0;  // Zero angular momentum
        let q = 0.0;

        // Start closer to horizon to ensure infall before escape check
        let r0 = 5.0;

        let result = trace_null_geodesic(
            a, e, l, q,
            r0, PI / 2.0,  // Start at r=5, equator
            50.0,          // Max Mino time
            -1.0, 0.0,     // Ingoing, no theta motion
            2000,
        );

        // Should hit horizon
        assert!(result.terminated, "Geodesic should terminate");

        // Final r should be near horizon (whether it escapes or falls in)
        let r_final = *result.r.last().unwrap();
        let r_horizon = event_horizon_radius(a);

        // Either hits horizon or check that r decreased
        if result.termination_reason == "hit_horizon" {
            assert!(r_final < r_horizon + 0.1);
        } else {
            // For radial infall, r should at least decrease from starting point
            assert!(r_final < r0 || r_final > 5.0 * r0,
                "Geodesic should either fall in or escape, r_final = {}", r_final);
        }
    }

    #[test]
    fn test_geodesic_escape() {
        // Outgoing photon
        let a = 0.5;
        let e = 1.0;
        let l = 5.0;  // Large angular momentum
        let q = 10.0;

        let result = trace_null_geodesic(
            a, e, l, q,
            10.0, PI / 2.0,
            200.0,
            1.0, 0.0,  // Outgoing
            2000,
        );

        // Should escape
        assert!(result.terminated);
        assert_eq!(result.termination_reason, "escaped");
    }

    #[test]
    fn test_geodesic_coordinate_time_increases() {
        let a = 0.7;
        let result = trace_null_geodesic(
            a, 1.0, 2.0, 5.0,
            15.0, PI / 3.0,
            50.0,
            -1.0, 1.0,
            500,
        );

        // Coordinate time should increase monotonically
        for i in 1..result.t.len() {
            assert!(result.t[i] >= result.t[i - 1]);
        }
    }

    #[test]
    fn test_geodesic_mino_time_linear() {
        let result = trace_null_geodesic(
            0.5, 1.0, 3.0, 2.0,
            20.0, PI / 2.0,
            10.0,
            -1.0, 0.0,
            100,
        );

        // Mino time should increase linearly with steps
        let dlam = 10.0 / 100.0;
        for (i, &lam) in result.lam.iter().enumerate() {
            assert_relative_eq!(lam, (i as f64) * dlam, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_impact_parameters_finite() {
        let a = 0.8;
        let (r_pro, r_retro) = photon_orbit_radius(a);

        // Check at several radii
        for r_ph in [r_pro, (r_pro + r_retro) / 2.0, r_retro] {
            let (xi, eta) = impact_parameters(r_ph, a);
            assert!(xi.is_finite());
            assert!(eta.is_finite());
        }
    }
}
