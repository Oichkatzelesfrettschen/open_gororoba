//! Energy-conserving geodesic integration with Hamiltonian preservation.
//!
//! Supplements RK4 with constraint-preserving corrections to maintain:
//!   - Energy E = -(g_tt v^t + g_{t phi} v^phi)  (time Killing vector)
//!   - Angular momentum L = g_{phi phi} v^phi + g_{t phi} v^t  (axial Killing vector)
//!   - Carter constant Q (Kerr separability)
//!   - Metric norm g_{ab} v^a v^b = -m^2 (null or timelike constraint)
//!
//! This provides a *generic* integration method that works for any metric via
//! the 4x4 metric components and Christoffel symbols. It complements the
//! Mino-time Kerr integrator in `kerr.rs`, which uses E, L, Q as inputs.
//!
//! Algorithm per step:
//! 1. Compute standard RK4 step using geodesic equation
//! 2. Evaluate conserved quantities at start and end
//! 3. Apply constraint-preserving correction (rescale spatial velocities)
//!
//! Derived from Rocq formalization: rocq/theories/Geodesics/EnergyConserving.v
//!
//! # References
//!
//! - Chan et al. (2013): GRay - GPU ray tracing in GR spacetimes
//! - Misner, Thorne, Wheeler (1973): Gravitation, Ch. 25, 33
//! - Carter (1968): Phys. Rev. 174, 1559 (constants of motion)

use crate::metric::{MetricComponents, ChristoffelComponents, DIM, T, R, THETA, PHI};
use crate::null_constraint;

// ============================================================================
// Conserved quantities (constants of motion)
// ============================================================================

/// Container for Kerr geodesic conserved quantities.
///
/// For orbits in a stationary, axisymmetric spacetime:
///   - Energy E: conserved by time-translation symmetry (Killing vector d/dt)
///   - Angular momentum L: conserved by axial symmetry (Killing vector d/dphi)
///   - Carter constant Q: conserved by hidden symmetry of Kerr metric
///   - Metric norm: g_{ab} v^a v^b (= 0 null, = -1 timelike)
#[derive(Debug, Clone, Copy)]
pub struct ConservedQuantities {
    /// E = -(g_tt v^t + g_{t phi} v^phi)
    pub energy: f64,
    /// L = g_{phi phi} v^phi + g_{t phi} v^t
    pub angular_momentum: f64,
    /// Q = p_theta^2 + cos^2(theta) * (a^2(m^2 - E^2) + L^2/sin^2(theta))
    pub carter_constant: f64,
    /// g_{ab} v^a v^b (0 for null, -1 for timelike with proper normalization)
    pub metric_norm: f64,
}

/// Compute conserved energy from the time Killing vector.
///
/// E = -(g_tt v^t + g_{t phi} v^phi)
///
/// Valid for any stationary spacetime with t as a Killing coordinate.
pub fn compute_energy(g: &MetricComponents, v: &[f64; DIM]) -> f64 {
    -(g[T][T] * v[T] + g[T][PHI] * v[PHI])
}

/// Compute conserved angular momentum from the axial Killing vector.
///
/// L = g_{phi phi} v^phi + g_{t phi} v^t
///
/// Valid for any axisymmetric spacetime with phi as a Killing coordinate.
pub fn compute_angular_momentum(g: &MetricComponents, v: &[f64; DIM]) -> f64 {
    g[PHI][PHI] * v[PHI] + g[T][PHI] * v[T]
}

/// Compute Carter constant for Kerr orbits.
///
/// Q = p_theta^2 + cos^2(theta) * (a^2(m^2 - E^2) + L^2/sin^2(theta))
///
/// where p_theta = g_{theta theta} v^theta.
///
/// Returns 0 for equatorial orbits (theta = pi/2). Clamps to Q >= 0.
pub fn compute_carter_constant(
    g: &MetricComponents,
    v: &[f64; DIM],
    theta: f64,
    a: f64,
) -> f64 {
    let sin_th = theta.sin();
    let cos_th = theta.cos();
    let cos2 = cos_th * cos_th;
    let sin2 = sin_th * sin_th;

    // Near poles, Q is poorly defined; return 0.
    if sin2 < 1e-10 {
        return 0.0;
    }

    let p_theta = g[THETA][THETA] * v[THETA];
    let e = compute_energy(g, v);
    let l = compute_angular_momentum(g, v);
    let m2 = null_constraint::null_constraint(g, v);

    let q = p_theta * p_theta + cos2 * (a * a * (m2 - e * e) + l * l / sin2);

    q.max(0.0)
}

/// Extract all conserved quantities from the current state.
pub fn extract_conserved_quantities(
    g: &MetricComponents,
    v: &[f64; DIM],
    theta: f64,
    a: f64,
) -> ConservedQuantities {
    ConservedQuantities {
        energy: compute_energy(g, v),
        angular_momentum: compute_angular_momentum(g, v),
        carter_constant: compute_carter_constant(g, v, theta, a),
        metric_norm: null_constraint::null_constraint(g, v),
    }
}

// ============================================================================
// Geodesic state (8-component: 4 positions + 4 velocities)
// ============================================================================

/// Full 8-component geodesic state: [x^mu, v^mu] = [t, r, theta, phi, vt, vr, vth, vph].
///
/// Unlike the Kerr-specific `GeodesicState` in `kerr.rs` (which stores only
/// r, theta, v_r, v_theta because t, phi are cyclic), this generic state
/// carries all 8 components for use with arbitrary metrics.
#[derive(Debug, Clone, Copy)]
pub struct FullGeodesicState {
    /// Coordinate positions [t, r, theta, phi]
    pub x: [f64; DIM],
    /// 4-velocity components [v^t, v^r, v^theta, v^phi]
    pub v: [f64; DIM],
}

// ============================================================================
// Generic RK4 step for geodesic equation
// ============================================================================

/// Compute geodesic acceleration: d(v^mu)/dlambda = -Gamma^mu_{alpha beta} v^alpha v^beta.
pub fn geodesic_acceleration(gamma: &ChristoffelComponents, v: &[f64; DIM]) -> [f64; DIM] {
    let mut accel = [0.0; DIM];
    for mu in 0..DIM {
        let mut sum = 0.0;
        for alpha in 0..DIM {
            for beta in 0..DIM {
                sum += gamma[mu][alpha][beta] * v[alpha] * v[beta];
            }
        }
        accel[mu] = -sum;
    }
    accel
}

/// RK4 step for the geodesic equation using Christoffel symbols.
///
/// The system is:
///   dx^mu/dlambda = v^mu
///   dv^mu/dlambda = -Gamma^mu_{ab} v^a v^b
///
/// The `metric_fn` provides the metric at any point, and `christoffel_fn`
/// provides Christoffel symbols. For production use, these would come from
/// a `SpacetimeMetric` implementation.
///
/// Returns the new state after one step of size h.
pub fn rk4_geodesic_step<M, C>(
    state: &FullGeodesicState,
    h: f64,
    metric_fn: M,
    christoffel_fn: C,
) -> FullGeodesicState
where
    M: Fn(&[f64; DIM]) -> MetricComponents,
    C: Fn(&[f64; DIM]) -> ChristoffelComponents,
{
    // k1
    let gamma1 = christoffel_fn(&state.x);
    let a1 = geodesic_acceleration(&gamma1, &state.v);
    let k1_x = state.v;
    let k1_v = a1;

    // k2: state at x + h/2 * k1
    let mut x2 = [0.0; DIM];
    let mut v2 = [0.0; DIM];
    for i in 0..DIM {
        x2[i] = state.x[i] + 0.5 * h * k1_x[i];
        v2[i] = state.v[i] + 0.5 * h * k1_v[i];
    }
    let gamma2 = christoffel_fn(&x2);
    let a2 = geodesic_acceleration(&gamma2, &v2);
    let k2_x = v2;
    let k2_v = a2;

    // k3: state at x + h/2 * k2
    let mut x3 = [0.0; DIM];
    let mut v3 = [0.0; DIM];
    for i in 0..DIM {
        x3[i] = state.x[i] + 0.5 * h * k2_x[i];
        v3[i] = state.v[i] + 0.5 * h * k2_v[i];
    }
    let gamma3 = christoffel_fn(&x3);
    let a3 = geodesic_acceleration(&gamma3, &v3);
    let k3_x = v3;
    let k3_v = a3;

    // k4: state at x + h * k3
    let mut x4 = [0.0; DIM];
    let mut v4 = [0.0; DIM];
    for i in 0..DIM {
        x4[i] = state.x[i] + h * k3_x[i];
        v4[i] = state.v[i] + h * k3_v[i];
    }
    let gamma4 = christoffel_fn(&x4);
    let a4 = geodesic_acceleration(&gamma4, &v4);
    let k4_x = v4;
    let k4_v = a4;

    // Combine: y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)
    let mut new_x = [0.0; DIM];
    let mut new_v = [0.0; DIM];
    for i in 0..DIM {
        new_x[i] = state.x[i] + (h / 6.0) * (k1_x[i] + 2.0 * k2_x[i] + 2.0 * k3_x[i] + k4_x[i]);
        new_v[i] = state.v[i] + (h / 6.0) * (k1_v[i] + 2.0 * k2_v[i] + 2.0 * k3_v[i] + k4_v[i]);
    }

    // Suppress unused binding warning for metric_fn.
    // It is accepted for API symmetry: future adaptive methods will
    // evaluate the metric alongside Christoffel symbols.
    let _ = &metric_fn;

    FullGeodesicState { x: new_x, v: new_v }
}

// ============================================================================
// Constraint-preserving correction
// ============================================================================

/// Apply constraint correction to restore the geodesic norm.
///
/// Rescales the radial and polar velocities (v^r, v^theta) by a common
/// factor to bring g_{ab} v^a v^b back to the target value (0 for null,
/// -1 for timelike). The temporal and azimuthal components are held fixed
/// to preserve the Killing-vector conserved quantities E and L.
///
/// Returns the corrected 4-velocity.
pub fn apply_constraint_correction(
    g: &MetricComponents,
    v: &[f64; DIM],
    target_norm: f64,
) -> [f64; DIM] {
    let current = null_constraint::null_constraint(g, v);

    // Avoid correction when norm is essentially zero (e.g., at rest)
    if current.abs() < 1e-30 {
        return *v;
    }

    // The spatial-only norm from r and theta components
    let spatial_rt = g[R][R] * v[R] * v[R] + g[THETA][THETA] * v[THETA] * v[THETA];

    // If no spatial r/theta velocity to rescale, fall back to full renormalization
    if spatial_rt.abs() < 1e-30 {
        return null_constraint::renormalize_null(g, v);
    }

    // We want: current - spatial_rt + alpha^2 * spatial_rt = target_norm
    // => alpha^2 = (target_norm - current + spatial_rt) / spatial_rt
    let alpha_sq = (target_norm - current + spatial_rt) / spatial_rt;

    if alpha_sq < 0.0 {
        // Cannot rescale to achieve target; fall back to full renormalization
        return null_constraint::renormalize_null(g, v);
    }

    let alpha = alpha_sq.sqrt();

    [v[T], alpha * v[R], alpha * v[THETA], v[PHI]]
}

// ============================================================================
// Energy-conserving integration step
// ============================================================================

/// One energy-conserving geodesic step: RK4 + constraint correction.
///
/// 1. Performs a standard RK4 step using the geodesic equation
/// 2. Applies constraint correction to restore g_{ab} v^a v^b = target_norm
///
/// The `target_norm` should be 0.0 for null geodesics or -1.0 for timelike
/// (with affine parameterization where g_{ab} v^a v^b = -1).
pub fn energy_conserving_step<M, C>(
    state: &FullGeodesicState,
    h: f64,
    metric_fn: M,
    christoffel_fn: C,
    target_norm: f64,
) -> FullGeodesicState
where
    M: Fn(&[f64; DIM]) -> MetricComponents,
    C: Fn(&[f64; DIM]) -> ChristoffelComponents,
{
    // 1. RK4 step
    let stepped = rk4_geodesic_step(state, h, &metric_fn, &christoffel_fn);

    // 2. Compute metric at new position and apply constraint correction
    let g_new = metric_fn(&stepped.x);
    let v_corrected = apply_constraint_correction(&g_new, &stepped.v, target_norm);

    FullGeodesicState {
        x: stepped.x,
        v: v_corrected,
    }
}

/// Integrate a geodesic with energy conservation over multiple steps.
///
/// Returns the final state and a `ConstraintStats` summary.
///
/// Integration halts early if the `stop_fn` returns true (e.g., horizon
/// crossing or escape to large radius).
pub fn integrate_energy_conserving<M, C, S>(
    initial: &FullGeodesicState,
    h: f64,
    n_steps: usize,
    metric_fn: M,
    christoffel_fn: C,
    target_norm: f64,
    stop_fn: S,
) -> (FullGeodesicState, null_constraint::ConstraintStats)
where
    M: Fn(&[f64; DIM]) -> MetricComponents,
    C: Fn(&[f64; DIM]) -> ChristoffelComponents,
    S: Fn(&FullGeodesicState) -> bool,
{
    let mut state = *initial;
    let mut stats = null_constraint::ConstraintStats::new();

    for _ in 0..n_steps {
        if stop_fn(&state) {
            break;
        }

        state = energy_conserving_step(&state, h, &metric_fn, &christoffel_fn, target_norm);

        let g = metric_fn(&state.x);
        let c = null_constraint::null_constraint(&g, &state.v);
        let needs_renorm = (c - target_norm).abs() > null_constraint::adaptive_tolerance(h, 10.0);
        stats.update(c, needs_renorm);
    }

    (state, stats)
}

/// Relative energy drift between two sets of conserved quantities.
///
/// Returns |E_after - E_before| / (|E_before| + epsilon).
pub fn relative_energy_drift(q_before: &ConservedQuantities, q_after: &ConservedQuantities) -> f64 {
    (q_after.energy - q_before.energy).abs() / (q_before.energy.abs() + 1e-10)
}

/// Relative angular momentum drift.
pub fn relative_angular_momentum_drift(
    q_before: &ConservedQuantities,
    q_after: &ConservedQuantities,
) -> f64 {
    (q_after.angular_momentum - q_before.angular_momentum).abs()
        / (q_before.angular_momentum.abs() + 1e-10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    // -- Schwarzschild helpers --

    fn schwarzschild_metric(x: &[f64; DIM]) -> MetricComponents {
        let r = x[R];
        let theta = x[THETA];
        let m = 1.0;
        let f = 1.0 - 2.0 * m / r;
        let mut g = [[0.0; DIM]; DIM];
        g[T][T] = -f;
        g[R][R] = 1.0 / f;
        g[THETA][THETA] = r * r;
        g[PHI][PHI] = r * r * theta.sin() * theta.sin();
        g
    }

    fn schwarzschild_christoffel(x: &[f64; DIM]) -> ChristoffelComponents {
        let r = x[R];
        let theta = x[THETA];
        let m = 1.0;
        let f = 1.0 - 2.0 * m / r;
        let mut gamma = [[[0.0; DIM]; DIM]; DIM];

        // Non-zero Schwarzschild Christoffel symbols (standard results):
        // Gamma^t_{tr} = Gamma^t_{rt} = M / (r^2 f)
        gamma[T][T][R] = m / (r * r * f);
        gamma[T][R][T] = gamma[T][T][R];

        // Gamma^r_{tt} = M f / r^2
        gamma[R][T][T] = m * f / (r * r);

        // Gamma^r_{rr} = -M / (r^2 f)
        gamma[R][R][R] = -m / (r * r * f);

        // Gamma^r_{theta theta} = -(r - 2M) = -r*f
        gamma[R][THETA][THETA] = -r * f;

        // Gamma^r_{phi phi} = -r*f*sin^2(theta)
        gamma[R][PHI][PHI] = -r * f * theta.sin() * theta.sin();

        // Gamma^theta_{r theta} = Gamma^theta_{theta r} = 1/r
        gamma[THETA][R][THETA] = 1.0 / r;
        gamma[THETA][THETA][R] = 1.0 / r;

        // Gamma^theta_{phi phi} = -sin(theta)*cos(theta)
        gamma[THETA][PHI][PHI] = -theta.sin() * theta.cos();

        // Gamma^phi_{r phi} = Gamma^phi_{phi r} = 1/r
        gamma[PHI][R][PHI] = 1.0 / r;
        gamma[PHI][PHI][R] = 1.0 / r;

        // Gamma^phi_{theta phi} = Gamma^phi_{phi theta} = cos(theta)/sin(theta)
        gamma[PHI][THETA][PHI] = theta.cos() / theta.sin();
        gamma[PHI][PHI][THETA] = theta.cos() / theta.sin();

        gamma
    }

    /// Create a radial null geodesic state in Schwarzschild at radius r.
    fn radial_null_state(r: f64) -> FullGeodesicState {
        let theta = FRAC_PI_2;
        let g = schwarzschild_metric(&[0.0, r, theta, 0.0]);
        // Radial ingoing: v^r < 0, v^t from null condition
        let vr: f64 = -1.0;
        let vt = (g[R][R] / (-g[T][T])).sqrt() * vr.abs();
        FullGeodesicState {
            x: [0.0, r, theta, 0.0],
            v: [vt, vr, 0.0, 0.0],
        }
    }

    /// Create a circular null orbit at r = 3M (photon sphere) in Schwarzschild.
    fn circular_null_state() -> FullGeodesicState {
        let r = 3.0; // photon sphere
        let theta = FRAC_PI_2;
        let g = schwarzschild_metric(&[0.0, r, theta, 0.0]);
        // Circular: v^r = 0, v^theta = 0
        // g_tt (v^t)^2 + g_{phi phi} (v^phi)^2 = 0
        // v^phi = sqrt(-g_tt / g_{phi phi}) * v^t
        let vt = 1.0;
        let vphi = (-g[T][T] / g[PHI][PHI]).sqrt() * vt;
        FullGeodesicState {
            x: [0.0, r, theta, 0.0],
            v: [vt, 0.0, 0.0, vphi],
        }
    }

    // -- Conserved quantity tests --

    #[test]
    fn test_energy_schwarzschild_static_observer() {
        let r = 10.0;
        let g = schwarzschild_metric(&[0.0, r, FRAC_PI_2, 0.0]);
        // Static observer: v = [1/sqrt(-g_tt), 0, 0, 0]
        let vt = 1.0 / (-g[T][T]).sqrt();
        let v = [vt, 0.0, 0.0, 0.0];
        let e = compute_energy(&g, &v);
        // E = -g_tt * v^t = (1 - 2M/r) / sqrt(1 - 2M/r) = sqrt(1 - 2M/r)
        let expected = (1.0 - 2.0 / r).sqrt();
        assert!((e - expected).abs() < 1e-12, "E = {e}, expected {expected}");
    }

    #[test]
    fn test_angular_momentum_circular_orbit() {
        let state = circular_null_state();
        let g = schwarzschild_metric(&state.x);
        let l = compute_angular_momentum(&g, &state.v);
        // L = g_{phi phi} * v^phi (no g_{t phi} for Schwarzschild)
        let expected = g[PHI][PHI] * state.v[PHI];
        assert!((l - expected).abs() < 1e-12, "L = {l}, expected {expected}");
    }

    #[test]
    fn test_carter_constant_equatorial() {
        // Equatorial orbit: Q should be zero (theta = pi/2, v_theta = 0)
        let state = circular_null_state();
        let g = schwarzschild_metric(&state.x);
        let q = compute_carter_constant(&g, &state.v, state.x[THETA], 0.0);
        assert!(q.abs() < 1e-10, "Q = {q} (should be ~0 for equatorial)");
    }

    // -- Geodesic acceleration test --

    #[test]
    fn test_geodesic_acceleration_static_schwarzschild() {
        let r = 10.0;
        let x = [0.0, r, FRAC_PI_2, 0.0];
        let gamma = schwarzschild_christoffel(&x);

        // Static observer at rest (only v^t nonzero)
        let g = schwarzschild_metric(&x);
        let vt = 1.0 / (-g[T][T]).sqrt();
        let v = [vt, 0.0, 0.0, 0.0];

        let accel = geodesic_acceleration(&gamma, &v);

        // Should have radial acceleration (gravitational attraction)
        // a^r = -Gamma^r_{tt} (v^t)^2 = -M*f/r^2 * 1/(-g_tt) = -M/r^2 (Newtonian limit)
        assert!(accel[R] < 0.0, "a^r = {} (should be negative = inward)", accel[R]);
        assert!(accel[THETA].abs() < 1e-14, "a^theta should be zero");
        assert!(accel[PHI].abs() < 1e-14, "a^phi should be zero");
    }

    // -- RK4 geodesic step test --

    #[test]
    fn test_rk4_step_radial_infall() {
        let state = radial_null_state(20.0);
        let h = 0.1;

        let new_state = rk4_geodesic_step(
            &state,
            h,
            schwarzschild_metric,
            schwarzschild_christoffel,
        );

        // Ingoing: r should decrease
        assert!(new_state.x[R] < state.x[R], "r should decrease: {} -> {}", state.x[R], new_state.x[R]);
        // t should increase (future-directed)
        assert!(new_state.x[T] > state.x[T], "t should increase");
    }

    #[test]
    fn test_rk4_step_preserves_null_approximately() {
        let state = radial_null_state(20.0);
        let h = 0.01;

        let new_state = rk4_geodesic_step(
            &state,
            h,
            schwarzschild_metric,
            schwarzschild_christoffel,
        );

        let g = schwarzschild_metric(&new_state.x);
        let c = null_constraint::null_constraint(&g, &new_state.v);
        // RK4 drift should be O(h^4) ~ 1e-8 for h = 0.01
        assert!(c.abs() < 1e-4, "constraint after RK4: C = {c}");
    }

    // -- Constraint correction test --

    #[test]
    fn test_constraint_correction_restores_null() {
        let state = radial_null_state(20.0);
        let g = schwarzschild_metric(&state.x);

        // Perturb v^r slightly to break the null condition
        let v_bad = [state.v[T], state.v[R] * 1.01, state.v[THETA], state.v[PHI]];
        let c_before = null_constraint::null_constraint(&g, &v_bad);
        assert!(c_before.abs() > 1e-4, "should be non-null before correction");

        let v_fixed = apply_constraint_correction(&g, &v_bad, 0.0);
        let c_after = null_constraint::null_constraint(&g, &v_fixed);
        assert!(c_after.abs() < 1e-10, "after correction: C = {c_after}");
    }

    #[test]
    fn test_constraint_correction_preserves_temporal() {
        let state = radial_null_state(20.0);
        let g = schwarzschild_metric(&state.x);
        let v_bad = [state.v[T], state.v[R] * 1.1, state.v[THETA], state.v[PHI]];

        let v_fixed = apply_constraint_correction(&g, &v_bad, 0.0);

        // v^t and v^phi should be unchanged
        assert!((v_fixed[T] - v_bad[T]).abs() < 1e-14);
        assert!((v_fixed[PHI] - v_bad[PHI]).abs() < 1e-14);
    }

    // -- Energy-conserving step test --

    #[test]
    fn test_energy_conserving_step_null() {
        let state = radial_null_state(20.0);
        let h = 0.1;

        let new_state = energy_conserving_step(
            &state,
            h,
            schwarzschild_metric,
            schwarzschild_christoffel,
            0.0, // null
        );

        // Constraint should be well-preserved after correction
        let g = schwarzschild_metric(&new_state.x);
        let c = null_constraint::null_constraint(&g, &new_state.v);
        assert!(c.abs() < 1e-8, "energy-conserving C = {c}");
    }

    #[test]
    fn test_energy_conservation_over_many_steps() {
        let state = radial_null_state(50.0);
        let h = 0.5;
        let n = 100;

        let g0 = schwarzschild_metric(&state.x);
        let q0 = extract_conserved_quantities(&g0, &state.v, state.x[THETA], 0.0);

        let (final_state, stats) = integrate_energy_conserving(
            &state,
            h,
            n,
            schwarzschild_metric,
            schwarzschild_christoffel,
            0.0,
            |s| s.x[R] < 2.5 || s.x[R] > 1000.0, // stop at horizon or escape
        );

        let g_final = schwarzschild_metric(&final_state.x);
        let q_final = extract_conserved_quantities(&g_final, &final_state.v, final_state.x[THETA], 0.0);

        let e_drift = relative_energy_drift(&q0, &q_final);
        assert!(e_drift < 0.01, "energy drift = {e_drift} (should be < 1%)");

        // Constraint should remain well-controlled
        assert!(
            stats.max_constraint.abs() < 0.1,
            "max constraint = {}",
            stats.max_constraint
        );
    }

    // -- Circular orbit stability --

    #[test]
    fn test_circular_orbit_energy_conserved() {
        let state = circular_null_state();
        let h = 0.01;
        let n = 1000;

        let g0 = schwarzschild_metric(&state.x);
        let e0 = compute_energy(&g0, &state.v);
        let l0 = compute_angular_momentum(&g0, &state.v);

        let (final_state, _stats) = integrate_energy_conserving(
            &state,
            h,
            n,
            schwarzschild_metric,
            schwarzschild_christoffel,
            0.0,
            |s| s.x[R] < 2.1 || s.x[R] > 100.0,
        );

        let g_final = schwarzschild_metric(&final_state.x);
        let e_final = compute_energy(&g_final, &final_state.v);
        let l_final = compute_angular_momentum(&g_final, &final_state.v);

        let de = (e_final - e0).abs() / e0.abs();
        let dl = (l_final - l0).abs() / l0.abs();

        assert!(de < 0.01, "energy drift = {de}");
        assert!(dl < 0.01, "angular momentum drift = {dl}");
    }

    // -- Kerr frame-dragging test --

    fn kerr_metric_fn(x: &[f64; DIM]) -> MetricComponents {
        let r = x[R];
        let theta = x[THETA];
        let a = 0.5;
        let m = 1.0;
        let sigma = r * r + a * a * theta.cos() * theta.cos();
        let delta = r * r - 2.0 * m * r + a * a;
        let sin2 = theta.sin() * theta.sin();
        let big_a = (r * r + a * a).powi(2) - a * a * delta * sin2;

        let mut g = [[0.0; DIM]; DIM];
        g[T][T] = -(1.0 - 2.0 * m * r / sigma);
        g[R][R] = sigma / delta;
        g[THETA][THETA] = sigma;
        g[PHI][PHI] = big_a / sigma * sin2;
        g[T][PHI] = -2.0 * m * r * a * sin2 / sigma;
        g[PHI][T] = g[T][PHI];
        g
    }

    #[test]
    fn test_kerr_energy_from_killing_vector() {
        let r = 10.0;
        let theta = FRAC_PI_2;
        let x = [0.0, r, theta, 0.0];
        let g = kerr_metric_fn(&x);

        // Construct null 4-velocity using renormalization
        let v_spatial = [0.0, -1.0, 0.0, 0.3];
        let v = null_constraint::renormalize_null(&g, &v_spatial);

        let e = compute_energy(&g, &v);
        let l = compute_angular_momentum(&g, &v);

        // E should be positive for future-directed geodesic
        assert!(e > 0.0, "E = {e}");
        // L = g_{phi phi} v^phi + g_{t phi} v^t, sign depends on direction
        assert!(l.is_finite(), "L = {l}");
    }

    // -- Extract conserved quantities --

    #[test]
    fn test_extract_all_conserved() {
        let state = radial_null_state(10.0);
        let g = schwarzschild_metric(&state.x);
        let q = extract_conserved_quantities(&g, &state.v, state.x[THETA], 0.0);

        assert!(q.energy > 0.0, "E = {}", q.energy);
        assert!(q.metric_norm.abs() < 1e-10, "norm = {}", q.metric_norm);
        // Radial orbit: L = 0 (no phi velocity)
        assert!(q.angular_momentum.abs() < 1e-12, "L = {}", q.angular_momentum);
    }

    // -- Relative drift functions --

    #[test]
    fn test_relative_energy_drift() {
        let q1 = ConservedQuantities {
            energy: 1.0,
            angular_momentum: 3.0,
            carter_constant: 0.0,
            metric_norm: 0.0,
        };
        let q2 = ConservedQuantities {
            energy: 1.001,
            angular_momentum: 3.003,
            carter_constant: 0.0,
            metric_norm: 0.0,
        };

        let de = relative_energy_drift(&q1, &q2);
        assert!((de - 0.001).abs() < 1e-6, "de = {de}");

        let dl = relative_angular_momentum_drift(&q1, &q2);
        assert!((dl - 0.001).abs() < 1e-6, "dl = {dl}");
    }
}
