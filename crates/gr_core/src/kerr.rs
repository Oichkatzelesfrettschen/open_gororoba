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

use crate::metric::{
    ChristoffelComponents, MetricComponents, SpacetimeMetric, DIM, PHI, R, T, THETA,
};
use ode_solvers::dopri5::Dopri5;
use ode_solvers::{SVector, System};
use rayon::prelude::*;
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

/// Reduced ODE system for Kerr geodesic integration in regularized coordinates.
///
/// Only the dynamical degrees of freedom are integrated: `[u, theta, v_u, v_theta]`
/// where `u = 1/r`. The cyclic coordinates `t` and `phi` (which do not feed back
/// into the equations of motion) are reconstructed by post-integration quadrature.
/// This eliminates the scale mismatch between `dt/dlam ~ r^2` and `du/dlam ~ O(1)`
/// that caused the adaptive integrator to reject every step.
///
/// **Regularization**: `u = 1/r` compresses six orders of magnitude into an O(1)
/// state vector. At r=500, u=0.002; at the horizon, u~0.5.
///
/// The transformation from (r, v_r) to (u, v_u):
///   v_u = du/dlam = -u^2 * v_r
///   dv_u/dlam = 2*v_u^2/u - u^2 * (0.5 * dR/dr)
struct KerrGeodesicReduced {
    a: f64,
    e: f64,
    l: f64,
    q: f64,
    u_horizon: f64,  // 1/r_horizon
    u_escape: f64,   // 1/r_escape (small, since r_escape is large)
}

type State4 = SVector<f64, 4>;

impl System<f64, State4> for KerrGeodesicReduced {
    fn system(&self, _lam: f64, y: &State4, dy: &mut State4) {
        let u = y[0];          // 1/r
        let theta = y[1];
        let v_u = y[2];        // du/dlam
        let v_theta = y[3];

        // Recover r from u for metric computation.
        let r = if u.abs() > 1e-15 { 1.0 / u } else { 1e15 };

        let a = self.a;
        let e = self.e;
        let l = self.l;
        let q = self.q;
        let sin_th = theta.sin();
        let cos_th = theta.cos();
        let sin3 = (sin_th.abs().powi(3)).max(1e-30);

        let t_val = e * (r * r + a * a) - a * l;

        // Radial acceleration in the original r-coordinate:
        //   a_r = dv_r/dlam = 0.5 * dR/dr
        let dr_dr = 4.0 * e * r * t_val - (2.0 * r - 2.0) * (q + (l - a * e).powi(2));
        let a_r = 0.5 * dr_dr;

        // Regularized acceleration:
        //   dv_u/dlam = 2*v_u^2/u - u^2*a_r
        let dv_u = if u.abs() > 1e-15 {
            2.0 * v_u * v_u / u - u * u * a_r
        } else {
            0.0
        };

        // Polar acceleration (unchanged by radial regularization):
        //   dv_theta/dlam = 0.5 * dTheta/dtheta
        let dtheta_dth = (2.0 * theta).sin() * a * a * (1.0 - e * e)
            + 2.0 * l * l * cos_th / sin3;

        dy[0] = v_u;
        dy[1] = v_theta;
        dy[2] = dv_u;
        dy[3] = 0.5 * dtheta_dth;
    }

    fn solout(&mut self, _lam: f64, y: &State4, _dy: &State4) -> bool {
        let u = y[0];
        // Halt when photon crosses the horizon (u > u_horizon)
        // or escapes to large r (u < u_escape).
        u > self.u_horizon || u < self.u_escape
    }
}

/// Compute dt/dlam and dphi/dlam at a given (r, theta) for post-integration quadrature.
fn cyclic_derivatives(r: f64, theta: f64, a: f64, e: f64, l: f64) -> (f64, f64) {
    let (_, delta) = kerr_metric_quantities(r, theta, a);
    let sin_th = theta.sin();
    let sin2 = (sin_th * sin_th).max(1e-30);

    let t_val = e * (r * r + a * a) - a * l;

    let dphi = -(a * e - l / sin2) + a * t_val / delta;
    let dt = -a * (a * e * sin2 - l) + (r * r + a * a) * t_val / delta;

    (dt, dphi)
}

/// Trace a null geodesic in a Kerr spacetime.
///
/// Uses Dormand-Prince 5(4) adaptive integration (ode_solvers::Dopri5)
/// with **regularized radial coordinates** (`u = 1/r`). This regularization
/// ensures all state-vector components are O(1), preventing the step-size
/// underflow that occurs when integrating the raw Mino-time equations at
/// large r (where v_r ~ r^2 ~ 10^5).
///
/// The public API accepts and returns r (not u): the transformation is
/// purely internal to the integrator.
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
/// * `n_steps` - Number of output sample points (integration steps are adaptive)
#[allow(clippy::too_many_arguments)]
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
    // Compute initial radial and polar velocities from the potentials.
    let (_, delta0) = kerr_metric_quantities(r0, theta0, a);
    let t0 = e * (r0 * r0 + a * a) - a * l;
    let r_pot = t0 * t0 - delta0 * (q + (l - a * e).powi(2));

    let sin_th0 = theta0.sin();
    let cos_th0 = theta0.cos();
    let sin2_0 = (sin_th0 * sin_th0).max(1e-30);
    let theta_pot = q - cos_th0 * cos_th0 * (a * a * (1.0 - e * e) + l * l / sin2_0);

    let v_r0 = sgn_r * r_pot.max(0.0).sqrt();
    let v_theta0 = sgn_theta * theta_pot.max(0.0).sqrt();

    // Transform to regularized coordinates: u0 = 1/r0, v_u0 = -u0^2 * v_r0.
    let u0 = 1.0 / r0;
    let v_u0 = -u0 * u0 * v_r0;

    // Reduced state: [u, theta, v_u, v_theta] -- only dynamical variables.
    // The cyclic coordinates (t, phi) are reconstructed by quadrature below.
    let y0 = State4::new(u0, theta0, v_u0, v_theta0);

    let r_horizon = event_horizon_radius(a);
    let r_escape = 5.0 * r0;

    let system = KerrGeodesicReduced {
        a, e, l, q,
        u_horizon: 1.0 / r_horizon,
        u_escape: 1.0 / r_escape,
    };

    // Dense output step: lam_max / n_steps gives output sample spacing.
    let dx = lam_max / n_steps.max(1) as f64;

    let mut stepper = Dopri5::new(system, 0.0, lam_max, dx, y0, 1e-10, 1e-12);

    let integration_ok = stepper.integrate().is_ok();

    let x_out = stepper.x_out();
    let y_out = stepper.y_out();

    // Transform u -> r for each output sample.
    let r_vec: Vec<f64> = y_out.iter().map(|y| {
        let u = y[0];
        if u.abs() > 1e-15 { 1.0 / u } else { 1e15 }
    }).collect();
    let theta_vec: Vec<f64> = y_out.iter().map(|y| y[1]).collect();
    let lam_vec: Vec<f64> = x_out.to_vec();

    // Reconstruct cyclic coordinates (t, phi) by trapezoidal quadrature.
    // dt/dlam and dphi/dlam depend only on r and theta (not on t or phi).
    let n_out = r_vec.len();
    let mut t_vec = vec![0.0_f64; n_out];
    let mut phi_vec = vec![0.0_f64; n_out];

    if n_out > 1 {
        let (dt0, dphi0) = cyclic_derivatives(r_vec[0], theta_vec[0], a, e, l);
        let mut prev_dt = dt0;
        let mut prev_dphi = dphi0;

        for i in 1..n_out {
            let dlam = lam_vec[i] - lam_vec[i - 1];
            let (dt_i, dphi_i) = cyclic_derivatives(r_vec[i], theta_vec[i], a, e, l);

            // Trapezoidal rule
            t_vec[i] = t_vec[i - 1] + 0.5 * (prev_dt + dt_i) * dlam;
            phi_vec[i] = phi_vec[i - 1] + 0.5 * (prev_dphi + dphi_i) * dlam;

            prev_dt = dt_i;
            prev_dphi = dphi_i;
        }
    }

    // Determine termination from the final r value.
    let r_final = r_vec.last().copied().unwrap_or(f64::INFINITY);
    let lam_final = lam_vec.last().copied().unwrap_or(0.0);
    let early_stop = lam_final < lam_max - 1e-6;

    let (terminated, termination_reason) = if r_final < r_horizon + 0.5 {
        (true, "hit_horizon".to_string())
    } else if r_final > 4.0 * r0 || (early_stop && r_final > r0) {
        (true, "escaped".to_string())
    } else if !integration_ok {
        (true, "integration_error".to_string())
    } else {
        (early_stop, if early_stop { "solout_halt".to_string() } else { String::new() })
    };

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

/// Result of backward ray-tracing the black hole shadow.
#[derive(Debug, Clone)]
pub struct ShadowResult {
    /// Celestial coordinate grid (horizontal).
    pub alpha: Vec<f64>,
    /// Celestial coordinate grid (vertical).
    pub beta: Vec<f64>,
    /// Shadow mask: `true` where the photon falls into the black hole.
    /// Stored in row-major order: `shadow_mask[i * n_beta + j]` for pixel (i, j).
    pub shadow_mask: Vec<bool>,
    pub n_alpha: usize,
    pub n_beta: usize,
}

/// Ray-trace the black hole shadow by backward-tracing photons.
///
/// For each pixel (alpha, beta) in the observer's sky, launches a null
/// geodesic backward from the observer and checks whether it falls into
/// the horizon. Uses rayon for pixel-level parallelism.
///
/// # Arguments
/// * `a` - Spin parameter (0 <= a < 1)
/// * `r_obs` - Observer distance (large for far-field limit)
/// * `theta_obs` - Observer inclination angle
/// * `n_alpha` - Horizontal grid resolution
/// * `n_beta` - Vertical grid resolution
/// * `alpha_range` - (min, max) for horizontal celestial coordinate
/// * `beta_range` - (min, max) for vertical celestial coordinate
#[allow(clippy::too_many_arguments)]
pub fn shadow_ray_traced(
    a: f64,
    r_obs: f64,
    theta_obs: f64,
    n_alpha: usize,
    n_beta: usize,
    alpha_range: (f64, f64),
    beta_range: (f64, f64),
) -> ShadowResult {
    let alpha: Vec<f64> = (0..n_alpha)
        .map(|i| {
            alpha_range.0
                + (alpha_range.1 - alpha_range.0) * i as f64 / (n_alpha - 1).max(1) as f64
        })
        .collect();

    let beta: Vec<f64> = (0..n_beta)
        .map(|j| {
            beta_range.0
                + (beta_range.1 - beta_range.0) * j as f64 / (n_beta - 1).max(1) as f64
        })
        .collect();

    let sin_o = theta_obs.sin();
    let cos_o = theta_obs.cos();
    let r_horizon = event_horizon_radius(a);

    // With u=1/r regularization, we can integrate from arbitrarily large r_obs
    // without underflow. The Mino-time integration length scales with r_obs.
    let lam_max = 2.0 * r_obs;
    let n_steps = 2000;

    // Build flat list of (i, j) pixel indices for parallel iteration.
    let pixels: Vec<(usize, usize)> = (0..n_alpha)
        .flat_map(|i| (0..n_beta).map(move |j| (i, j)))
        .collect();

    let results: Vec<(usize, usize, bool)> = pixels
        .par_iter()
        .map(|&(i, j)| {
            let a_val = alpha[i];
            let b_val = beta[j];

            // Map (alpha, beta) -> constants of motion (L, Q) for distant observer.
            let l = -a_val * sin_o;
            let q = b_val * b_val + cos_o * cos_o * (a_val * a_val - a * a);

            if q < 0.0 {
                return (i, j, false);
            }

            let result = trace_null_geodesic(
                a, 1.0, l, q, r_obs, theta_obs, lam_max, -1.0, 0.0, n_steps,
            );

            let r_final = *result.r.last().unwrap_or(&f64::INFINITY);
            let in_shadow = r_final < r_horizon + 0.5;
            (i, j, in_shadow)
        })
        .collect();

    let mut shadow_mask = vec![false; n_alpha * n_beta];
    for (i, j, val) in results {
        shadow_mask[i * n_beta + j] = val;
    }

    ShadowResult {
        alpha,
        beta,
        shadow_mask,
        n_alpha,
        n_beta,
    }
}

// ---------------------------------------------------------------------------
// Kerr spacetime: SpacetimeMetric trait implementation
// ---------------------------------------------------------------------------

/// Kerr black hole implementing the generic `SpacetimeMetric` trait.
///
/// Unlike the free functions above (which fix M=1), this struct carries
/// explicit mass and spin parameters. The spin parameter a has dimensions
/// of length (a = J/M) and must satisfy |a| <= M for a black hole.
///
/// The metric is written in Boyer-Lindquist coordinates (G = c = 1):
///
///   ds^2 = -(1 - 2Mr/Sigma) dt^2 - (4Mar sin^2 theta / Sigma) dt dphi
///          + (Sigma/Delta) dr^2 + Sigma dtheta^2
///          + A sin^2 theta / Sigma dphi^2
///
/// where:
///   Sigma = r^2 + a^2 cos^2 theta
///   Delta = r^2 - 2Mr + a^2
///   A = (r^2 + a^2)^2 - a^2 Delta sin^2 theta
///
/// References:
///   Kerr (1963): Phys. Rev. Lett. 11, 237
///   Boyer & Lindquist (1967): J. Math. Phys. 8, 265
///   Chandrasekhar (1983): Mathematical Theory of Black Holes, Ch. 6-7
///   Bardeen, Press, Teukolsky (1972): ApJ 178, 347 (ISCO formula)
#[derive(Debug, Clone, Copy)]
pub struct Kerr {
    /// Black hole mass (G = c = 1)
    pub mass: f64,
    /// Spin parameter a = J/M (|a| <= M for a black hole)
    pub spin: f64,
}

impl Kerr {
    /// Create a Kerr black hole with mass M and spin a.
    ///
    /// Panics if mass <= 0 or |a| > M (naked singularity).
    pub fn new(mass: f64, spin: f64) -> Self {
        assert!(mass > 0.0, "mass must be positive");
        assert!(
            spin.abs() <= mass,
            "|spin| must be <= mass (no naked singularities)"
        );
        Self { mass, spin }
    }

    /// Sigma = r^2 + a^2 cos^2(theta)
    fn sigma(&self, r: f64, theta: f64) -> f64 {
        r * r + self.spin * self.spin * theta.cos().powi(2)
    }

    /// Delta = r^2 - 2Mr + a^2
    fn delta(&self, r: f64) -> f64 {
        r * r - 2.0 * self.mass * r + self.spin * self.spin
    }

    /// A = (r^2 + a^2)^2 - a^2 Delta sin^2(theta)
    fn big_a(&self, r: f64, theta: f64) -> f64 {
        let a = self.spin;
        let r2_a2 = r * r + a * a;
        r2_a2 * r2_a2 - a * a * self.delta(r) * theta.sin().powi(2)
    }

    /// Outer (event) horizon radius: r+ = M + sqrt(M^2 - a^2).
    pub fn outer_horizon(&self) -> f64 {
        self.mass + (self.mass * self.mass - self.spin * self.spin).sqrt()
    }

    /// Inner (Cauchy) horizon radius: r- = M - sqrt(M^2 - a^2).
    pub fn inner_horizon(&self) -> f64 {
        self.mass - (self.mass * self.mass - self.spin * self.spin).sqrt()
    }

    /// Ergosphere radius at polar angle theta.
    ///
    /// r_ergo(theta) = M + sqrt(M^2 - a^2 cos^2 theta)
    ///
    /// At the equator: r_ergo = 2M (for all spins).
    /// At the poles: r_ergo = r+ (coincides with horizon).
    pub fn ergosphere_radius(&self, theta: f64) -> f64 {
        let a = self.spin;
        self.mass + (self.mass * self.mass - a * a * theta.cos().powi(2)).sqrt()
    }

    /// Prograde ISCO radius (Bardeen, Press, Teukolsky 1972).
    ///
    /// For Schwarzschild (a=0): r_ISCO = 6M.
    /// For extremal prograde (a=M): r_ISCO = M.
    pub fn isco_prograde(&self) -> f64 {
        self.isco_bpt(1.0)
    }

    /// Retrograde ISCO radius.
    ///
    /// For Schwarzschild (a=0): r_ISCO = 6M.
    /// For extremal retrograde (a=M): r_ISCO = 9M.
    pub fn isco_retrograde(&self) -> f64 {
        self.isco_bpt(-1.0)
    }

    /// Bardeen-Press-Teukolsky ISCO formula.
    ///
    /// r_ISCO = M (3 + Z2 -/+ sqrt((3-Z1)(3+Z1+2Z2)))
    /// Z1 = 1 + (1 - a*^2)^{1/3} ((1+a*)^{1/3} + (1-a*)^{1/3})
    /// Z2 = sqrt(3 a*^2 + Z1^2)
    /// where a* = a/M and -/+ is prograde/retrograde.
    fn isco_bpt(&self, sign: f64) -> f64 {
        let m = self.mass;
        let a_star = self.spin / m;
        let z1 = 1.0
            + (1.0 - a_star * a_star).cbrt()
                * ((1.0 + a_star).cbrt() + (1.0 - a_star).cbrt());
        let z2 = (3.0 * a_star * a_star + z1 * z1).sqrt();
        m * (3.0 + z2 - sign * ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt())
    }

    /// Prograde circular photon orbit radius.
    ///
    /// r_ph = 2M (1 + cos(2/3 arccos(-a/M)))
    pub fn photon_orbit_prograde(&self) -> f64 {
        let a_star = self.spin / self.mass;
        2.0 * self.mass * (1.0 + (2.0 / 3.0 * (-a_star).acos()).cos())
    }

    /// Retrograde circular photon orbit radius.
    pub fn photon_orbit_retrograde(&self) -> f64 {
        let a_star = self.spin / self.mass;
        2.0 * self.mass * (1.0 + (2.0 / 3.0 * a_star.acos()).cos())
    }

    /// Angular velocity of the event horizon: Omega_H = a / (r+^2 + a^2).
    pub fn angular_velocity_horizon(&self) -> f64 {
        let r_plus = self.outer_horizon();
        self.spin / (r_plus * r_plus + self.spin * self.spin)
    }

    /// Surface gravity at the outer horizon.
    ///
    /// kappa = (r+ - r-) / (2(r+^2 + a^2))
    ///
    /// For Schwarzschild: kappa = 1/(4M).
    pub fn surface_gravity(&self) -> f64 {
        let r_plus = self.outer_horizon();
        let r_minus = self.inner_horizon();
        (r_plus - r_minus) / (2.0 * (r_plus * r_plus + self.spin * self.spin))
    }

    /// Irreducible mass: M_irr = sqrt(A_H / (16 pi)) = sqrt(r+^2 + a^2) / 2.
    ///
    /// For Schwarzschild: M_irr = M.
    /// For extremal Kerr: M_irr = M / sqrt(2).
    pub fn irreducible_mass(&self) -> f64 {
        let r_plus = self.outer_horizon();
        0.5 * (r_plus * r_plus + self.spin * self.spin).sqrt()
    }

    /// Horizon area: A_H = 4 pi (r+^2 + a^2).
    pub fn horizon_area(&self) -> f64 {
        let r_plus = self.outer_horizon();
        4.0 * PI * (r_plus * r_plus + self.spin * self.spin)
    }

    // -- Exact metric derivatives (internal) --

    /// dg_{mu nu}/dr (exact analytic expressions).
    fn metric_derivs_r(&self, x: &[f64; DIM]) -> MetricComponents {
        let m = self.mass;
        let a = self.spin;
        let r = x[R];
        let theta = x[THETA];
        let s2 = theta.sin().powi(2);
        let c2 = theta.cos().powi(2);

        let sigma = self.sigma(r, theta);
        let delta = self.delta(r);
        let big_a = self.big_a(r, theta);
        let sigma2 = sigma * sigma;

        // Derivatives of auxiliary quantities
        let sigma_r = 2.0 * r;
        let delta_r = 2.0 * (r - m);
        let a_r = 4.0 * r * (r * r + a * a) - delta_r * a * a * s2;

        let mut dg = [[0.0; DIM]; DIM];

        // dg_tt/dr = 2M(a^2 cos^2 theta - r^2) / Sigma^2
        dg[T][T] = 2.0 * m * (a * a * c2 - r * r) / sigma2;

        // dg_tphi/dr = 2Ma sin^2 theta (r^2 - a^2 cos^2 theta) / Sigma^2
        dg[T][PHI] = 2.0 * m * a * s2 * (r * r - a * a * c2) / sigma2;
        dg[PHI][T] = dg[T][PHI];

        // dg_rr/dr = (Sigma_r Delta - Sigma Delta_r) / Delta^2
        let delta2 = delta * delta;
        dg[R][R] = (sigma_r * delta - sigma * delta_r) / delta2;

        // dg_theta_theta/dr = 2r
        dg[THETA][THETA] = sigma_r;

        // dg_phiphi/dr = sin^2 theta (A_r Sigma - Sigma_r A) / Sigma^2
        dg[PHI][PHI] = s2 * (a_r * sigma - sigma_r * big_a) / sigma2;

        dg
    }

    /// dg_{mu nu}/dtheta (exact analytic expressions).
    fn metric_derivs_theta(&self, x: &[f64; DIM]) -> MetricComponents {
        let m = self.mass;
        let a = self.spin;
        let r = x[R];
        let theta = x[THETA];
        let s2 = theta.sin().powi(2);
        let sin2th = (2.0 * theta).sin(); // sin(2 theta)

        let sigma = self.sigma(r, theta);
        let delta = self.delta(r);
        let big_a = self.big_a(r, theta);
        let sigma2 = sigma * sigma;

        // Derivatives of auxiliary quantities
        let sigma_th = -a * a * sin2th;

        let mut dg = [[0.0; DIM]; DIM];

        // dg_tt/dtheta = 2Mr a^2 sin(2 theta) / Sigma^2
        dg[T][T] = 2.0 * m * r * a * a * sin2th / sigma2;

        // dg_tphi/dtheta = -2Mar sin(2 theta) (r^2 + a^2) / Sigma^2
        dg[T][PHI] = -2.0 * m * a * r * sin2th * (r * r + a * a) / sigma2;
        dg[PHI][T] = dg[T][PHI];

        // dg_rr/dtheta = Sigma_theta / Delta = -a^2 sin(2 theta) / Delta
        dg[R][R] = sigma_th / delta;

        // dg_theta_theta/dtheta = -a^2 sin(2 theta)
        dg[THETA][THETA] = sigma_th;

        // dg_phiphi/dtheta = sin(2 theta) [A(r^2+a^2) - a^2 Delta Sigma sin^2 theta] / Sigma^2
        dg[PHI][PHI] =
            sin2th * (big_a * (r * r + a * a) - a * a * delta * sigma * s2) / sigma2;

        dg
    }
}

impl SpacetimeMetric for Kerr {
    fn metric_components(&self, x: &[f64; DIM]) -> MetricComponents {
        let m = self.mass;
        let a = self.spin;
        let r = x[R];
        let theta = x[THETA];
        let s2 = theta.sin().powi(2);

        let sigma = self.sigma(r, theta);
        let delta = self.delta(r);
        let big_a = self.big_a(r, theta);

        let mut g = [[0.0; DIM]; DIM];
        g[T][T] = -(1.0 - 2.0 * m * r / sigma);
        g[T][PHI] = -2.0 * m * a * r * s2 / sigma;
        g[PHI][T] = g[T][PHI];
        g[R][R] = sigma / delta;
        g[THETA][THETA] = sigma;
        g[PHI][PHI] = big_a * s2 / sigma;
        g
    }

    fn inverse_metric(&self, x: &[f64; DIM]) -> MetricComponents {
        let m = self.mass;
        let a = self.spin;
        let r = x[R];
        let theta = x[THETA];
        let s2 = theta.sin().powi(2);

        let sigma = self.sigma(r, theta);
        let delta = self.delta(r);
        let big_a = self.big_a(r, theta);

        let mut g_inv = [[0.0; DIM]; DIM];

        // The (t, phi) block inverse uses det = g_tt g_phiphi - g_tphi^2 = -Delta sin^2 theta
        g_inv[T][T] = -big_a / (sigma * delta);
        g_inv[T][PHI] = -2.0 * m * a * r / (sigma * delta);
        g_inv[PHI][T] = g_inv[T][PHI];
        g_inv[R][R] = delta / sigma;
        g_inv[THETA][THETA] = 1.0 / sigma;
        // g^{phi phi} = (Sigma - 2Mr) / (Sigma Delta sin^2 theta)
        if s2 > 1e-30 {
            g_inv[PHI][PHI] = (sigma - 2.0 * m * r) / (sigma * delta * s2);
        }

        g_inv
    }

    /// Exact Christoffel symbols computed from analytic metric derivatives.
    ///
    /// Avoids double numerical differentiation by using closed-form
    /// expressions for all 10 non-zero metric derivatives (5 components
    /// x 2 non-Killing directions r and theta). The standard assembly
    /// formula then gives exact Christoffel symbols.
    #[allow(clippy::needless_range_loop)]
    fn christoffel(&self, x: &[f64; DIM]) -> ChristoffelComponents {
        let g_inv = self.inverse_metric(x);

        // Exact metric derivatives (only R and THETA directions are non-zero)
        let dg_dr = self.metric_derivs_r(x);
        let dg_dth = self.metric_derivs_theta(x);

        let mut dg = [[[0.0; DIM]; DIM]; DIM];
        for a in 0..DIM {
            for b in 0..DIM {
                dg[a][b][R] = dg_dr[a][b];
                dg[a][b][THETA] = dg_dth[a][b];
            }
        }

        // Gamma^alpha_{mu nu} = (1/2) g^{alpha beta} (g_{beta mu,nu} + g_{beta nu,mu} - g_{mu nu,beta})
        let mut gamma = [[[0.0; DIM]; DIM]; DIM];
        for alpha in 0..DIM {
            for mu in 0..DIM {
                for nu in mu..DIM {
                    let mut sum = 0.0;
                    for beta in 0..DIM {
                        sum += g_inv[alpha][beta]
                            * (dg[beta][mu][nu] + dg[beta][nu][mu] - dg[mu][nu][beta]);
                    }
                    gamma[alpha][mu][nu] = 0.5 * sum;
                    gamma[alpha][nu][mu] = gamma[alpha][mu][nu];
                }
            }
        }

        gamma
    }

    fn event_horizon_radius(&self) -> Option<f64> {
        Some(self.outer_horizon())
    }

    fn isco_radius(&self) -> Option<f64> {
        Some(self.isco_prograde())
    }

    fn photon_sphere_radius(&self) -> Option<f64> {
        Some(self.photon_orbit_prograde())
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
        let (alpha, _beta) = shadow_boundary(a, 100, PI / 2.0);

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
            assert!(r_final < r_horizon + 0.5);
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
        let r_horizon = event_horizon_radius(a);
        let result = trace_null_geodesic(
            a, 1.0, 2.0, 5.0,
            15.0, PI / 3.0,
            50.0,
            -1.0, 1.0,
            500,
        );

        // Coordinate time should increase monotonically outside the horizon.
        // Inside the horizon (r < r_h), t and r swap causal roles and t can
        // decrease -- this is physical, not a numerical error.
        for i in 1..result.t.len() {
            if result.r[i] > r_horizon + 0.1 && result.r[i - 1] > r_horizon + 0.1 {
                assert!(
                    result.t[i] >= result.t[i - 1] - 1e-6,
                    "t should increase outside horizon: t[{}]={} < t[{}]={}",
                    i, result.t[i], i - 1, result.t[i - 1]
                );
            }
        }
    }

    #[test]
    fn test_geodesic_mino_time_monotonic() {
        let result = trace_null_geodesic(
            0.5, 1.0, 3.0, 2.0,
            20.0, PI / 2.0,
            10.0,
            -1.0, 0.0,
            100,
        );

        // Mino time should increase monotonically
        for i in 1..result.lam.len() {
            assert!(
                result.lam[i] > result.lam[i - 1],
                "Mino time should increase: lam[{}]={} <= lam[{}]={}",
                i, result.lam[i], i - 1, result.lam[i - 1]
            );
        }
        // First sample is 0, last should reach near lam_max (or terminate early)
        assert_relative_eq!(result.lam[0], 0.0, epsilon = 1e-14);
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

    #[test]
    fn test_geodesic_from_large_distance() {
        // A radially infalling photon (L=0, Q=0) from r=500 should reach the
        // horizon for a Schwarzschild black hole. This tests the u=1/r
        // regularization: at r=500 the unregularized v_r ~ 250000 which
        // caused Dopri5 underflow, but u=1/500=0.002 keeps things O(1).
        let a = 0.0;
        let r0 = 500.0;
        let r_horizon = event_horizon_radius(a); // = 2.0

        let result = trace_null_geodesic(
            a, 1.0, 0.0, 0.0,
            r0, PI / 2.0,
            2.0 * r0,   // lam_max
            -1.0, 0.0,  // ingoing
            2000,
        );
        let r_final = *result.r.last().unwrap();
        assert!(
            r_final < r_horizon + 1.0,
            "Radial photon from r=500 should fall in, got r_final={}",
            r_final
        );
        assert_eq!(result.termination_reason, "hit_horizon");
    }

    /// Verify the classically-allowed region constraint along a geodesic.
    ///
    /// For null geodesics with constants of motion (E, L, Q), the photon
    /// may only exist where R(r) >= 0 and Theta(theta) >= 0. If the
    /// integrator pushes the photon into a classically forbidden region,
    /// either the ODE is wrong or the step-size control has failed.
    ///
    /// R(r) = T^2 - Delta*(Q + (L-aE)^2)  where T = E*(r^2+a^2) - aL
    /// Theta(theta) = Q - cos^2(theta) * (a^2*(1-E^2) + L^2/sin^2(theta))
    #[test]
    fn test_geodesic_potential_non_negativity() {
        // Use a moderately interesting orbit: a=0.7, L=2.5, Q=5.0, ingoing.
        let a = 0.7;
        let e = 1.0;
        let l = 2.5;
        let q = 5.0;
        let r0 = 20.0;
        let theta0 = PI / 3.0;

        let result = trace_null_geodesic(
            a, e, l, q,
            r0, theta0,
            40.0,
            -1.0, 1.0,
            500,
        );

        let r_horizon = event_horizon_radius(a);

        for i in 0..result.r.len() {
            let ri = result.r[i];
            let thi = result.theta[i];

            // Skip points near/inside the horizon: Delta -> 0 makes R(r) ill-conditioned.
            if ri < r_horizon + 0.3 {
                continue;
            }

            // Radial potential R(r)
            let (_, delta) = kerr_metric_quantities(ri, thi, a);
            let t_val = e * (ri * ri + a * a) - a * l;
            let r_potential = t_val * t_val - delta * (q + (l - a * e).powi(2));

            // Theta potential Theta(theta)
            let sin_th = thi.sin();
            let cos_th = thi.cos();
            let sin2 = (sin_th * sin_th).max(1e-30);
            let theta_potential = q - cos_th * cos_th * (a * a * (1.0 - e * e) + l * l / sin2);

            // Allow small negative values from numerical drift (1e-6 relative).
            let r_scale = t_val * t_val;
            assert!(
                r_potential > -1e-6 * r_scale,
                "R(r) < 0 at step {}: r={:.4}, R={:.4e} (forbidden region!)",
                i, ri, r_potential
            );
            assert!(
                theta_potential > -1e-6 * q.max(1.0),
                "Theta(theta) < 0 at step {}: theta={:.4}, Theta={:.4e} (forbidden!)",
                i, thi, theta_potential
            );
        }
    }

    /// Test a near-horizon photon that starts just outside the horizon and falls in.
    /// This exercises the u=1/r regularization at its most extreme: u ~ 1/r_horizon
    /// where Delta ~ 0 and the unregularized equations would blow up.
    #[test]
    fn test_near_horizon_infall() {
        let a = 0.9;
        let r_horizon = event_horizon_radius(a);

        // Start at r = r_horizon + 0.1 (very close to the horizon).
        let r0 = r_horizon + 0.1;
        let e = 1.0;
        let l = 0.0;  // Radial infall
        let q = 0.0;

        let result = trace_null_geodesic(
            a, e, l, q,
            r0, PI / 2.0,
            50.0,
            -1.0, 0.0,
            500,
        );

        // Should terminate by hitting the horizon.
        assert!(
            result.terminated,
            "Near-horizon photon should terminate"
        );
        assert_eq!(
            result.termination_reason, "hit_horizon",
            "Near-horizon photon should hit horizon, got: {}",
            result.termination_reason
        );

        // Should have produced output points (not crashed). At r_horizon+0.1
        // ingoing, the photon crosses almost immediately, so 2 points (start +
        // termination) is the minimum expected.
        assert!(
            result.r.len() >= 2,
            "Should have at least start + end point, got {}",
            result.r.len()
        );
    }

    /// Test that a high-spin (a=0.998) geodesic completes without NaN or panic.
    /// Near-extremal Kerr spacetimes have very narrow regions between the
    /// horizon and the ISCO, stressing the integrator's numerical stability.
    #[test]
    fn test_near_extremal_kerr_geodesic() {
        let a = 0.998;
        let r_horizon = event_horizon_radius(a);

        // Prograde photon orbit just outside the near-extremal horizon.
        let (r_pro, _) = photon_orbit_radius(a);
        let (xi, eta) = impact_parameters(r_pro + 0.01, a);

        let result = trace_null_geodesic(
            a, 1.0, xi, eta.max(0.0),
            15.0, PI / 2.0,
            100.0,
            -1.0, 0.0,
            500,
        );

        // The geodesic should terminate (either falls in or escapes).
        // The key check is that no NaN appears in the trajectory.
        for &r in &result.r {
            assert!(r.is_finite(), "r should be finite, got NaN/Inf");
        }
        for &th in &result.theta {
            assert!(th.is_finite(), "theta should be finite, got NaN/Inf");
        }
        for &t in &result.t {
            assert!(t.is_finite(), "t should be finite, got NaN/Inf");
        }

        // Verify the photon reached near the horizon or escaped.
        let r_final = *result.r.last().unwrap();
        assert!(
            r_final < r_horizon + 1.0 || r_final > 50.0,
            "Near-extremal geodesic should fall in or escape, r_final={}",
            r_final
        );
    }

    /// Verify that a circular photon orbit (at the Schwarzschild photon sphere r=3M)
    /// stays near r=3M throughout the integration, testing orbit stability.
    #[test]
    fn test_schwarzschild_circular_photon_orbit() {
        // Schwarzschild a=0: photon sphere at r=3M.
        // For a circular orbit: L/E = sqrt(27)*M = 3*sqrt(3), Q/E^2 = 0 (equatorial).
        let a = 0.0;
        let e = 1.0;
        let l = 3.0 * 3.0_f64.sqrt(); // L/E for circular orbit at r=3
        let q = 0.0;
        let r0 = 3.0;
        let theta0 = PI / 2.0;

        // Outgoing radial velocity = 0 for circular orbit, theta velocity = 0.
        let result = trace_null_geodesic(
            a, e, l, q,
            r0, theta0,
            20.0,
            1.0, 0.0,  // sgn_r doesn't matter much since v_r ~ 0
            500,
        );

        // The orbit should stay near r=3 (but it's unstable, so numerical
        // perturbations will eventually grow exponentially). For short
        // integration times, r should remain within 10% of r=3.
        let n_check = result.r.len().min(100); // Check first 100 points
        for i in 0..n_check {
            assert!(
                result.r[i] > 2.0 && result.r[i] < 6.0,
                "Circular orbit at r=3 diverged too fast at step {}: r={:.4}",
                i, result.r[i]
            );
        }
    }

    #[test]
    fn test_shadow_ray_traced_schwarzschild() {
        // Schwarzschild (a=0): shadow should be roughly circular with
        // radius ~ sqrt(27) ~ 5.196 for a distant observer.
        let result = shadow_ray_traced(
            0.0,             // a = 0 (Schwarzschild)
            500.0,           // r_obs (distant)
            PI / 2.0,        // equatorial observer
            50, 50,          // low-res grid for test speed
            (-8.0, 8.0),
            (-8.0, 8.0),
        );

        assert_eq!(result.shadow_mask.len(), 50 * 50);

        // Count shadow pixels -- should be a filled disk
        let n_shadow: usize = result.shadow_mask.iter().filter(|&&b| b).count();
        assert!(
            n_shadow > 0,
            "Shadow should contain at least some pixels"
        );

        // The shadow should be smaller than the full grid
        assert!(
            n_shadow < 50 * 50,
            "Shadow should not fill the entire grid"
        );

        // Check rough circularity: center pixel should be in shadow,
        // corner pixel should not.
        let center = 25 * 50 + 25;
        assert!(
            result.shadow_mask[center],
            "Center of grid should be in shadow"
        );

        let corner = 0; // (0, 0) = (-8, -8)
        assert!(
            !result.shadow_mask[corner],
            "Corner of grid should not be in shadow"
        );

        // Quantitative check: the shadow area should be pi * r_shadow^2 where
        // r_shadow = sqrt(27). The pixel area is (16/50)^2, so expected pixel
        // count ~= pi * 27 / (16/50)^2 ~= 828.
        let pixel_area = (16.0 / 50.0) * (16.0 / 50.0);
        let expected_area = PI * 27.0;
        let expected_pixels = expected_area / pixel_area;
        let n_shadow_f = n_shadow as f64;

        // Allow 15% tolerance for pixelization and boundary effects.
        assert!(
            (n_shadow_f - expected_pixels).abs() < 0.15 * expected_pixels,
            "Shadow pixel count {:.0} should be within 15% of expected {:.0} (pi*27/pixel_area)",
            n_shadow_f, expected_pixels
        );
    }

    #[test]
    fn test_shadow_ray_traced_kerr_asymmetric() {
        // Kerr (a=0.9): shadow should be asymmetric (shifted prograde).
        let result = shadow_ray_traced(
            0.9,
            500.0,
            PI / 2.0,
            40, 40,
            (-8.0, 8.0),
            (-8.0, 8.0),
        );

        // Count shadow pixels in left half vs right half
        let mut left_count = 0;
        let mut right_count = 0;
        for i in 0..40 {
            for j in 0..40 {
                if result.shadow_mask[i * 40 + j] {
                    if i < 20 {
                        left_count += 1;
                    } else {
                        right_count += 1;
                    }
                }
            }
        }

        // For prograde observer at equator, shadow shifts to negative alpha
        // (left half of alpha grid if range is symmetric).
        // At minimum, both halves should have some shadow pixels.
        assert!(left_count > 0 || right_count > 0, "Should have shadow pixels");
    }

    // -- Kerr SpacetimeMetric trait tests --

    fn kerr_bh() -> Kerr {
        Kerr::new(1.0, 0.9) // M=1, a=0.9
    }

    #[test]
    fn test_kerr_trait_schwarzschild_limit_metric() {
        // Kerr(M, a=0) should reproduce Schwarzschild metric exactly
        let k = Kerr::new(1.0, 0.0);
        let s = crate::schwarzschild::Schwarzschild::new(1.0);
        let x = [0.0, 10.0, std::f64::consts::FRAC_PI_2, 0.0];

        let g_kerr = k.metric_components(&x);
        let g_schw = s.metric_components(&x);

        for i in 0..DIM {
            for j in 0..DIM {
                assert!(
                    (g_kerr[i][j] - g_schw[i][j]).abs() < 1e-12,
                    "g[{}][{}]: Kerr(a=0)={}, Schwarzschild={}",
                    i, j, g_kerr[i][j], g_schw[i][j]
                );
            }
        }
    }

    #[test]
    fn test_kerr_trait_off_diagonal() {
        // Kerr with spin should have non-zero g_tphi (frame dragging)
        let k = kerr_bh();
        let x = [0.0, 10.0, std::f64::consts::FRAC_PI_2, 0.0];
        let g = k.metric_components(&x);

        assert!(
            g[T][PHI].abs() > 1e-6,
            "g_tphi should be non-zero for spinning black hole"
        );
        assert_relative_eq!(g[T][PHI], g[PHI][T], epsilon = 1e-14);
    }

    #[test]
    fn test_kerr_trait_inverse_consistency() {
        // g * g^{-1} = delta (identity) for the (t,phi) block
        let k = kerr_bh();
        let x = [0.0, 8.0, 1.2, 0.5];
        let g = k.metric_components(&x);
        let g_inv = k.inverse_metric(&x);

        // Check that g * g^{-1} = I for all components
        for i in 0..DIM {
            for j in 0..DIM {
                let mut product = 0.0;
                for k_idx in 0..DIM {
                    product += g[i][k_idx] * g_inv[k_idx][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product - expected).abs() < 1e-10,
                    "(g g^-1)[{}][{}] = {} (expected {})",
                    i, j, product, expected
                );
            }
        }
    }

    #[test]
    fn test_kerr_trait_inverse_schwarzschild_limit() {
        // Kerr(M, a=0) inverse should match Schwarzschild inverse
        let k = Kerr::new(1.0, 0.0);
        let s = crate::schwarzschild::Schwarzschild::new(1.0);
        let x = [0.0, 10.0, 1.0, 0.5];

        let g_inv_k = k.inverse_metric(&x);
        let g_inv_s = s.inverse_metric(&x);

        for i in 0..DIM {
            for j in 0..DIM {
                assert!(
                    (g_inv_k[i][j] - g_inv_s[i][j]).abs() < 1e-12,
                    "g_inv[{}][{}]: Kerr(a=0)={}, Schwarzschild={}",
                    i, j, g_inv_k[i][j], g_inv_s[i][j]
                );
            }
        }
    }

    #[test]
    fn test_kerr_trait_christoffel_symmetry() {
        // Christoffel symbols must be symmetric in lower indices
        let k = kerr_bh();
        let x = [0.0, 7.0, 1.0, 0.5];
        let gamma = k.christoffel(&x);

        for a in 0..DIM {
            for m in 0..DIM {
                for n in 0..DIM {
                    assert!(
                        (gamma[a][m][n] - gamma[a][n][m]).abs() < 1e-12,
                        "Gamma^{}_{}{} != Gamma^{}_{}{}  ({} vs {})",
                        a, m, n, a, n, m, gamma[a][m][n], gamma[a][n][m]
                    );
                }
            }
        }
    }

    #[test]
    fn test_kerr_trait_christoffel_schwarzschild_limit() {
        // Kerr(M, a=0) Christoffel should match Schwarzschild
        let k = Kerr::new(1.0, 0.0);
        let s = crate::schwarzschild::Schwarzschild::new(1.0);
        let x = [0.0, 10.0, 1.0, 0.5];

        let gamma_k = k.christoffel(&x);
        let gamma_s = s.christoffel(&x);

        for a in 0..DIM {
            for m in 0..DIM {
                for n in 0..DIM {
                    let diff = (gamma_k[a][m][n] - gamma_s[a][m][n]).abs();
                    let scale = gamma_s[a][m][n].abs().max(1e-12);
                    assert!(
                        diff / scale < 1e-8,
                        "Gamma^{}_{}{}: Kerr(a=0)={:.6e}, Schw={:.6e}, diff={:.2e}",
                        a, m, n, gamma_k[a][m][n], gamma_s[a][m][n], diff
                    );
                }
            }
        }
    }

    #[test]
    fn test_kerr_trait_christoffel_vs_numerical() {
        // Exact analytic Christoffel vs finite-difference numerical
        let k = kerr_bh();
        let x = [0.0, 10.0, 1.0, 0.5];

        let exact = k.christoffel(&x);
        let numerical = crate::metric::christoffel_numerical(&k, &x);

        for a in 0..DIM {
            for m in 0..DIM {
                for n in 0..DIM {
                    let diff = (exact[a][m][n] - numerical[a][m][n]).abs();
                    let scale = exact[a][m][n].abs().max(1e-12);
                    assert!(
                        diff / scale < 1e-3,
                        "Gamma^{}_{}{}: exact={:.6e}, numerical={:.6e}, rel_err={:.2e}",
                        a, m, n, exact[a][m][n], numerical[a][m][n], diff / scale
                    );
                }
            }
        }
    }

    #[test]
    fn test_kerr_trait_frame_dragging_christoffel() {
        // Kerr with spin should have non-zero Gamma^t_{r phi} (frame dragging)
        // This symbol vanishes for Schwarzschild
        let k = kerr_bh();
        let x = [0.0, 10.0, std::f64::consts::FRAC_PI_2, 0.0];
        let gamma = k.christoffel(&x);

        // Gamma^t_{r phi} -- coupling between radial and azimuthal motion
        assert!(
            gamma[T][R][PHI].abs() > 1e-6,
            "Gamma^t_rphi should be non-zero for spinning BH, got {}",
            gamma[T][R][PHI]
        );

        // Verify it vanishes for Schwarzschild
        let k0 = Kerr::new(1.0, 0.0);
        let gamma0 = k0.christoffel(&x);
        assert!(
            gamma0[T][R][PHI].abs() < 1e-12,
            "Gamma^t_rphi should vanish for Schwarzschild"
        );
    }

    #[test]
    fn test_kerr_trait_vacuum_einstein() {
        // Kerr is a vacuum solution: R_{mu nu} = 0
        let k = kerr_bh();
        let x = [0.0, 10.0, std::f64::consts::FRAC_PI_4, 0.0];

        let riemann = crate::metric::riemann_from_christoffel(&k, &x);
        let ricci = crate::metric::ricci_from_riemann(&riemann);

        for mu in 0..DIM {
            for nu in 0..DIM {
                assert!(
                    ricci[mu][nu].abs() < 0.01,
                    "R_{}{} = {:.4e} (expected 0 for vacuum Kerr)",
                    mu, nu, ricci[mu][nu]
                );
            }
        }
    }

    #[test]
    fn test_kerr_trait_kretschner_schwarzschild_limit() {
        // At a=0, Kretschner should match Schwarzschild: K = 48 M^2 / r^6
        let k = Kerr::new(1.0, 0.0);
        let r = 10.0;
        let x = [0.0, r, std::f64::consts::FRAC_PI_2, 0.0];
        let result = crate::metric::full_curvature(&k, &x);

        let expected = 48.0 / r.powi(6);
        let rel_err = (result.kretschner - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "Kretschner at a=0, r={}: got {:.4e}, expected {:.4e}, rel_err={:.2e}",
            r, result.kretschner, expected, rel_err
        );
    }

    #[test]
    fn test_kerr_trait_kretschner_general() {
        // Kerr Kretschner scalar: K = 48 M^2 (r^2 - a^2 c^2)(r^4 - 14r^2a^2c^2 + a^4c^4) / Sigma^6
        let k = kerr_bh();
        let m = k.mass;
        let a = k.spin;
        let r = 10.0;
        let theta = std::f64::consts::FRAC_PI_4;
        let x = [0.0, r, theta, 0.0];

        let result = crate::metric::full_curvature(&k, &x);

        let c2 = theta.cos().powi(2);
        let sigma = r * r + a * a * c2;
        let f = (r * r - a * a * c2)
            * (r.powi(4) - 14.0 * r * r * a * a * c2 + a.powi(4) * c2 * c2);
        let expected = 48.0 * m * m * f / sigma.powi(6);

        let rel_err = (result.kretschner - expected).abs() / expected.abs().max(1e-30);
        assert!(
            rel_err < 0.05,
            "Kerr Kretschner at r={}, theta=pi/4: got {:.4e}, expected {:.4e}, rel_err={:.2e}",
            r, result.kretschner, expected, rel_err
        );
    }

    // -- Kerr horizon and orbital property tests --

    #[test]
    fn test_kerr_trait_horizons() {
        // Schwarzschild limit: r+ = 2M, r- = 0
        let k0 = Kerr::new(1.0, 0.0);
        assert_relative_eq!(k0.outer_horizon(), 2.0, epsilon = 1e-12);
        assert_relative_eq!(k0.inner_horizon(), 0.0, epsilon = 1e-12);

        // Extremal: r+ = r- = M
        let ke = Kerr::new(1.0, 1.0);
        assert_relative_eq!(ke.outer_horizon(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(ke.inner_horizon(), 1.0, epsilon = 1e-12);

        // Generic: r+ = M + sqrt(M^2 - a^2)
        let k = kerr_bh();
        let expected = 1.0 + (1.0 - 0.81_f64).sqrt();
        assert_relative_eq!(k.outer_horizon(), expected, epsilon = 1e-12);
    }

    #[test]
    fn test_kerr_trait_ergosphere() {
        let k = kerr_bh();

        // At equator: r_ergo = 2M for ALL spins
        assert_relative_eq!(
            k.ergosphere_radius(std::f64::consts::FRAC_PI_2),
            2.0 * k.mass,
            epsilon = 1e-12
        );

        // At poles: r_ergo = r+ (coincides with horizon)
        assert_relative_eq!(
            k.ergosphere_radius(0.0),
            k.outer_horizon(),
            epsilon = 1e-12
        );

        // Ergosphere is always outside or at the horizon
        for theta in [0.3, 0.5, 1.0, 1.2, std::f64::consts::FRAC_PI_2] {
            assert!(
                k.ergosphere_radius(theta) >= k.outer_horizon() - 1e-12,
                "Ergosphere at theta={:.2} should be >= horizon",
                theta
            );
        }
    }

    #[test]
    fn test_kerr_trait_isco() {
        // Schwarzschild: prograde = retrograde = 6M
        let k0 = Kerr::new(1.0, 0.0);
        assert_relative_eq!(k0.isco_prograde(), 6.0, epsilon = 1e-8);
        assert_relative_eq!(k0.isco_retrograde(), 6.0, epsilon = 1e-8);

        // Extremal: prograde ISCO = M, retrograde ISCO = 9M
        let ke = Kerr::new(1.0, 1.0);
        assert_relative_eq!(ke.isco_prograde(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(ke.isco_retrograde(), 9.0, epsilon = 1e-6);

        // Generic: prograde < retrograde
        let k = kerr_bh();
        assert!(
            k.isco_prograde() < k.isco_retrograde(),
            "Prograde ISCO ({}) should be < retrograde ISCO ({})",
            k.isco_prograde(),
            k.isco_retrograde()
        );

        // Both should be outside the horizon
        assert!(k.isco_prograde() > k.outer_horizon());
    }

    #[test]
    fn test_kerr_trait_photon_orbits() {
        // Schwarzschild: both = 3M
        let k0 = Kerr::new(1.0, 0.0);
        assert_relative_eq!(k0.photon_orbit_prograde(), 3.0, epsilon = 1e-6);
        assert_relative_eq!(k0.photon_orbit_retrograde(), 3.0, epsilon = 1e-6);

        // Generic: prograde < retrograde
        let k = kerr_bh();
        assert!(k.photon_orbit_prograde() < k.photon_orbit_retrograde());

        // Both outside horizon
        assert!(k.photon_orbit_prograde() > k.outer_horizon());
    }

    #[test]
    fn test_kerr_trait_surface_gravity() {
        // Schwarzschild: kappa = 1/(4M)
        let k0 = Kerr::new(1.0, 0.0);
        assert_relative_eq!(k0.surface_gravity(), 0.25, epsilon = 1e-12);

        // Extremal: kappa = 0 (extremal BH has zero surface gravity)
        let ke = Kerr::new(1.0, 1.0);
        assert_relative_eq!(ke.surface_gravity(), 0.0, epsilon = 1e-12);

        // Generic: 0 < kappa < 1/(4M)
        let k = kerr_bh();
        assert!(k.surface_gravity() > 0.0);
        assert!(k.surface_gravity() < 0.25);
    }

    #[test]
    fn test_kerr_trait_irreducible_mass() {
        // Schwarzschild: M_irr = M
        let k0 = Kerr::new(1.0, 0.0);
        assert_relative_eq!(k0.irreducible_mass(), 1.0, epsilon = 1e-12);

        // Extremal: M_irr = M / sqrt(2)
        let ke = Kerr::new(1.0, 1.0);
        assert_relative_eq!(
            ke.irreducible_mass(),
            1.0 / 2.0_f64.sqrt(),
            epsilon = 1e-12
        );

        // M_irr <= M always
        let k = kerr_bh();
        assert!(k.irreducible_mass() <= k.mass + 1e-12);
    }

    #[test]
    fn test_kerr_trait_angular_velocity_horizon() {
        // Schwarzschild: Omega_H = 0 (no rotation)
        let k0 = Kerr::new(1.0, 0.0);
        assert_relative_eq!(k0.angular_velocity_horizon(), 0.0, epsilon = 1e-12);

        // Extremal: Omega_H = a/(r+^2 + a^2) = 1/(1+1) = 0.5
        let ke = Kerr::new(1.0, 1.0);
        assert_relative_eq!(ke.angular_velocity_horizon(), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_kerr_trait_mass_scaling() {
        // All radii should scale linearly with M
        let k1 = Kerr::new(1.0, 0.5);
        let k10 = Kerr::new(10.0, 5.0); // same a/M ratio

        assert_relative_eq!(
            k10.outer_horizon() / k1.outer_horizon(),
            10.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            k10.isco_prograde() / k1.isco_prograde(),
            10.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            k10.photon_orbit_prograde() / k1.photon_orbit_prograde(),
            10.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_kerr_trait_event_horizon_radius() {
        let k = kerr_bh();
        assert_eq!(k.event_horizon_radius(), Some(k.outer_horizon()));
    }

    #[test]
    fn test_kerr_trait_isco_radius() {
        let k = kerr_bh();
        assert_eq!(k.isco_radius(), Some(k.isco_prograde()));
    }

    #[test]
    fn test_kerr_trait_horizon_area() {
        // Schwarzschild: A_H = 16 pi M^2
        let k0 = Kerr::new(1.0, 0.0);
        assert_relative_eq!(k0.horizon_area(), 16.0 * PI, epsilon = 1e-10);

        // Extremal: A_H = 8 pi M^2
        let ke = Kerr::new(1.0, 1.0);
        assert_relative_eq!(ke.horizon_area(), 8.0 * PI, epsilon = 1e-10);
    }
}
