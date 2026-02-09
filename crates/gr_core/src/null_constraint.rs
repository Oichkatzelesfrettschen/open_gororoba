//! Null geodesic constraint preservation and renormalization.
//!
//! For null geodesics (light rays), the constraint g_{ab} v^a v^b = 0 must be
//! preserved during numerical integration. For massive particles, the mass-shell
//! condition g_{ab} v^a v^b = -m^2 (in c=1 units) applies instead.
//!
//! RK4 integration introduces truncation error O(h^4) per step in the constraint.
//! Over N steps of size h, the accumulated drift is bounded by N * C * h^4 =
//! (Lambda/h) * C * h^4 = C * Lambda * h^3, which converges as h -> 0.
//!
//! When constraint drift exceeds a tolerance, renormalization recomputes v^t
//! from the spatial velocity components to exactly restore the constraint.
//!
//! Derived from Rocq formalization: rocq/theories/Geodesics/NullConstraint.v
//!
//! # References
//!
//! - Misner, Thorne, Wheeler (1973): Gravitation, Ch. 25 (geodesic integration)
//! - Press (2007): Numerical Recipes 3rd ed., Ch. 17 (ODE error control)

use crate::metric::{MetricComponents, DIM, PHI, R, T, THETA};

// ============================================================================
// Constraint evaluation
// ============================================================================

/// Compute the null constraint: C = g_{ab} v^a v^b.
///
/// For null geodesics, C should be zero.
/// For timelike geodesics (with proper normalization), C = -1.
///
/// Works with the full 4x4 metric, including off-diagonal terms
/// (e.g., g_{t phi} for Kerr frame dragging).
pub fn null_constraint(g: &MetricComponents, v: &[f64; DIM]) -> f64 {
    let mut c = 0.0;
    for a in 0..DIM {
        for b in 0..DIM {
            c += g[a][b] * v[a] * v[b];
        }
    }
    c
}

/// Check if the 4-velocity satisfies the null condition within tolerance.
pub fn is_null(g: &MetricComponents, v: &[f64; DIM], tol: f64) -> bool {
    null_constraint(g, v).abs() < tol
}

/// Check if the 4-velocity satisfies the massive particle condition.
///
/// Mass-shell: g_{ab} v^a v^b = -m^2 (geometric units, c=1).
pub fn mass_shell_constraint(g: &MetricComponents, v: &[f64; DIM], m: f64) -> f64 {
    null_constraint(g, v) + m * m
}

/// Check if the 4-velocity satisfies the timelike condition for mass m.
pub fn is_timelike(g: &MetricComponents, v: &[f64; DIM], m: f64, tol: f64) -> bool {
    mass_shell_constraint(g, v, m).abs() < tol
}

// ============================================================================
// Renormalization (diagonal metric)
// ============================================================================

/// Renormalize v^t to restore the null condition for a diagonal metric.
///
/// Solves g_{tt} (v^t)^2 + g_{rr} (v^r)^2 + g_{thth} (v^th)^2 + g_{phph} (v^ph)^2 = 0
/// for v^t, keeping spatial velocities fixed.
///
/// Requires g_{tt} < 0 (Lorentzian signature) and at least one nonzero
/// spatial velocity.
///
/// Returns the renormalized 4-velocity.
pub fn renormalize_null_diagonal(g: &MetricComponents, v: &[f64; DIM]) -> [f64; DIM] {
    let spatial_norm = g[R][R] * v[R] * v[R]
        + g[THETA][THETA] * v[THETA] * v[THETA]
        + g[PHI][PHI] * v[PHI] * v[PHI];

    // g_tt v0^2 = -spatial_norm => v0 = sqrt(spatial_norm / (-g_tt))
    let new_vt = (spatial_norm / (-g[T][T])).sqrt();

    [new_vt, v[R], v[THETA], v[PHI]]
}

/// Renormalize v^t to restore the null condition for a metric with frame dragging.
///
/// For Kerr-family metrics with g_{t phi} != 0, the null condition becomes:
///   g_{tt} (v^t)^2 + 2 g_{t phi} v^t v^phi + g_{rr} (v^r)^2
///     + g_{th th} (v^th)^2 + g_{phi phi} (v^phi)^2 = 0
///
/// This is a quadratic in v^t: a (v^t)^2 + b v^t + c = 0, solved via the
/// quadratic formula. We take the future-directed root (v^t > 0).
pub fn renormalize_null_kerr(g: &MetricComponents, v: &[f64; DIM]) -> [f64; DIM] {
    let spatial = g[R][R] * v[R] * v[R]
        + g[THETA][THETA] * v[THETA] * v[THETA]
        + g[PHI][PHI] * v[PHI] * v[PHI];

    let a = g[T][T];
    let b = 2.0 * g[T][PHI] * v[PHI];
    let c = spatial;

    let discriminant = b * b - 4.0 * a * c;
    // Discriminant is always positive for physical metrics:
    // a = g_tt < 0 and c >= 0, so -4ac >= 0, thus b^2 - 4ac >= b^2.
    let sqrt_disc = discriminant.sqrt();

    // Take the future-directed root (v^t > 0).
    // With a = g_tt < 0: numerator (-b - sqrt(D)) < 0 and denominator 2a < 0,
    // so their quotient is positive.  The other root (-b + sqrt(D))/(2a)
    // gives a negative (past-directed) v^t.
    let new_vt = (-b - sqrt_disc) / (2.0 * a);

    [new_vt, v[R], v[THETA], v[PHI]]
}

/// Renormalize v^t to restore the null condition for any metric.
///
/// Automatically selects diagonal or Kerr renormalization based on
/// whether the off-diagonal g_{t phi} component is significant.
pub fn renormalize_null(g: &MetricComponents, v: &[f64; DIM]) -> [f64; DIM] {
    if g[T][PHI].abs() < 1e-15 {
        renormalize_null_diagonal(g, v)
    } else {
        renormalize_null_kerr(g, v)
    }
}

/// Renormalize v^t for a massive particle: g_{ab} v^a v^b = -m^2.
///
/// For diagonal metrics: g_{tt} (v^t)^2 + spatial = -m^2
/// => v^t = sqrt((spatial + m^2) / (-g_{tt}))
pub fn renormalize_massive(g: &MetricComponents, v: &[f64; DIM], m: f64) -> [f64; DIM] {
    let spatial = g[R][R] * v[R] * v[R]
        + g[THETA][THETA] * v[THETA] * v[THETA]
        + g[PHI][PHI] * v[PHI] * v[PHI];

    let new_vt = ((spatial + m * m) / (-g[T][T])).sqrt();

    [new_vt, v[R], v[THETA], v[PHI]]
}

// ============================================================================
// Drift bounds and monitoring
// ============================================================================

/// Estimated maximum constraint drift per RK4 step: C * h^4.
///
/// The RK4 local truncation error is O(h^5) for state variables, but the
/// constraint (quadratic in velocity) accumulates error as O(h^4) per step.
pub fn constraint_drift_bound(bound_constant: f64, h: f64) -> f64 {
    let h2 = h * h;
    bound_constant * h2 * h2
}

/// Estimated total constraint drift after N steps: N * C * h^4.
///
/// For total affine parameter Lambda = N*h:
///   drift ~ (Lambda/h) * C * h^4 = C * Lambda * h^3
///
/// This decreases as h decreases, confirming convergence.
pub fn global_drift_bound(bound_constant: f64, h: f64, n: usize) -> f64 {
    n as f64 * constraint_drift_bound(bound_constant, h)
}

/// Adaptive tolerance based on step size.
///
/// Reasonable tolerance is proportional to expected per-step drift: O(h^4).
/// A safety factor of ~10 allows some accumulation before triggering renorm.
pub fn adaptive_tolerance(h: f64, safety_factor: f64) -> f64 {
    let h2 = h * h;
    safety_factor * h2 * h2
}

/// Whether renormalization is needed based on constraint violation.
pub fn needs_renormalization(g: &MetricComponents, v: &[f64; DIM], tol: f64) -> bool {
    null_constraint(g, v).abs() > tol
}

// ============================================================================
// Constraint monitoring statistics
// ============================================================================

/// Statistics for constraint monitoring during geodesic integration.
#[derive(Clone, Debug)]
pub struct ConstraintStats {
    /// Maximum |C| observed during integration
    pub max_constraint: f64,
    /// Accumulated total |C| (for computing average)
    pub total_drift: f64,
    /// Number of renormalization events
    pub renorm_count: usize,
    /// Total integration steps
    pub step_count: usize,
}

impl ConstraintStats {
    /// New empty statistics.
    pub fn new() -> Self {
        Self {
            max_constraint: 0.0,
            total_drift: 0.0,
            renorm_count: 0,
            step_count: 0,
        }
    }

    /// Update statistics after one integration step.
    pub fn update(&mut self, constraint: f64, renormalized: bool) {
        let abs_c = constraint.abs();
        if abs_c > self.max_constraint {
            self.max_constraint = abs_c;
        }
        self.total_drift += abs_c;
        if renormalized {
            self.renorm_count += 1;
        }
        self.step_count += 1;
    }

    /// Average constraint violation per step.
    pub fn average_constraint(&self) -> f64 {
        if self.step_count > 0 {
            self.total_drift / self.step_count as f64
        } else {
            0.0
        }
    }

    /// Fraction of steps that required renormalization.
    pub fn renorm_frequency(&self) -> f64 {
        if self.step_count > 0 {
            self.renorm_count as f64 / self.step_count as f64
        } else {
            0.0
        }
    }
}

impl Default for ConstraintStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Schwarzschild metric at (r, theta) in geometric units (M=1).
    fn schwarzschild_metric(r: f64, theta: f64) -> MetricComponents {
        let mut g = [[0.0; DIM]; DIM];
        let m = 1.0;
        g[T][T] = -(1.0 - 2.0 * m / r);
        g[R][R] = 1.0 / (1.0 - 2.0 * m / r);
        g[THETA][THETA] = r * r;
        g[PHI][PHI] = r * r * theta.sin() * theta.sin();
        g
    }

    /// Kerr metric components at (r, theta) for spin a, in BL coordinates.
    fn kerr_metric(r: f64, theta: f64, a: f64) -> MetricComponents {
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

    // -- Null constraint evaluation --

    #[test]
    fn test_null_constraint_schwarzschild() {
        let r = 10.0;
        let theta = std::f64::consts::FRAC_PI_2;
        let g = schwarzschild_metric(r, theta);

        // Radial null geodesic: g_tt (v^t)^2 + g_rr (v^r)^2 = 0
        // v^t = sqrt(g_rr / (-g_tt)) * v^r
        let vr = 1.0;
        let vt = (g[R][R] / (-g[T][T])).sqrt() * vr;
        let v = [vt, vr, 0.0, 0.0];

        let c = null_constraint(&g, &v);
        assert!(c.abs() < 1e-14, "C = {c}");
        assert!(is_null(&g, &v, 1e-10));
    }

    #[test]
    fn test_null_constraint_nonzero_for_timelike() {
        let g = schwarzschild_metric(10.0, std::f64::consts::FRAC_PI_2);
        // Static observer: v = [1/sqrt(-g_tt), 0, 0, 0]
        let vt = 1.0 / (-g[T][T]).sqrt();
        let v = [vt, 0.0, 0.0, 0.0];

        let c = null_constraint(&g, &v);
        // Should be -1 for properly normalized timelike
        assert!((c - (-1.0)).abs() < 1e-10, "C = {c}");
    }

    #[test]
    fn test_null_constraint_kerr_with_frame_dragging() {
        let r = 5.0;
        let theta = std::f64::consts::FRAC_PI_2;
        let a = 0.9;
        let g = kerr_metric(r, theta, a);

        // Construct a null 4-velocity using renormalization
        let v_spatial = [0.0, 1.0, 0.0, 0.5]; // arbitrary spatial + phi
        let v = renormalize_null_kerr(&g, &[0.0, v_spatial[1], v_spatial[2], v_spatial[3]]);

        let c = null_constraint(&g, &v);
        assert!(c.abs() < 1e-10, "C = {c}");
    }

    // -- Mass-shell constraint --

    #[test]
    fn test_mass_shell_constraint() {
        let g = schwarzschild_metric(10.0, std::f64::consts::FRAC_PI_2);
        let vt = 1.0 / (-g[T][T]).sqrt();
        let v = [vt, 0.0, 0.0, 0.0];

        // g_ab v^a v^b = -1, so mass_shell with m=1 should be 0
        let ms = mass_shell_constraint(&g, &v, 1.0);
        assert!(ms.abs() < 1e-10, "mass_shell = {ms}");
        assert!(is_timelike(&g, &v, 1.0, 1e-8));
    }

    // -- Renormalization (diagonal) --

    #[test]
    fn test_renormalize_null_diagonal() {
        let g = schwarzschild_metric(10.0, std::f64::consts::FRAC_PI_2);
        // Start with arbitrary spatial velocities
        let v_bad = [999.0, 1.0, 0.5, 0.3]; // wrong v^t

        let v_fixed = renormalize_null_diagonal(&g, &v_bad);
        let c = null_constraint(&g, &v_fixed);
        assert!(c.abs() < 1e-12, "After renorm: C = {c}");
    }

    #[test]
    fn test_renormalize_preserves_spatial_velocities() {
        let g = schwarzschild_metric(10.0, std::f64::consts::FRAC_PI_2);
        let v = [999.0, 1.0, 0.5, 0.3];

        let v_fixed = renormalize_null_diagonal(&g, &v);
        assert!((v_fixed[R] - v[R]).abs() < 1e-14);
        assert!((v_fixed[THETA] - v[THETA]).abs() < 1e-14);
        assert!((v_fixed[PHI] - v[PHI]).abs() < 1e-14);
    }

    // -- Renormalization (Kerr) --

    #[test]
    fn test_renormalize_null_kerr() {
        let g = kerr_metric(5.0, std::f64::consts::FRAC_PI_2, 0.9);
        let v_bad = [999.0, 1.0, 0.0, 2.0]; // wrong v^t

        let v_fixed = renormalize_null_kerr(&g, &v_bad);
        let c = null_constraint(&g, &v_fixed);
        assert!(c.abs() < 1e-10, "After Kerr renorm: C = {c}");
    }

    #[test]
    fn test_renormalize_auto_selects() {
        // Schwarzschild (diagonal): should use diagonal renorm
        let g_s = schwarzschild_metric(10.0, std::f64::consts::FRAC_PI_2);
        let v_bad = [999.0, 1.0, 0.0, 0.5];
        let v_fixed = renormalize_null(&g_s, &v_bad);
        assert!(null_constraint(&g_s, &v_fixed).abs() < 1e-12);

        // Kerr (off-diagonal): should use Kerr renorm
        let g_k = kerr_metric(5.0, std::f64::consts::FRAC_PI_2, 0.9);
        let v_fixed = renormalize_null(&g_k, &v_bad);
        assert!(null_constraint(&g_k, &v_fixed).abs() < 1e-10);
    }

    // -- Renormalization (massive) --

    #[test]
    fn test_renormalize_massive() {
        let g = schwarzschild_metric(10.0, std::f64::consts::FRAC_PI_2);
        let v_bad = [999.0, 0.1, 0.0, 0.05]; // wrong v^t
        let m = 1.0;

        let v_fixed = renormalize_massive(&g, &v_bad, m);
        let ms = mass_shell_constraint(&g, &v_fixed, m);
        assert!(ms.abs() < 1e-12, "After massive renorm: ms = {ms}");
    }

    // -- Drift bounds --

    #[test]
    fn test_drift_bound_scales_as_h4() {
        let c = 1.0;
        let h1 = 0.1;
        let h2 = 0.01;
        let d1 = constraint_drift_bound(c, h1);
        let d2 = constraint_drift_bound(c, h2);
        // h2 = h1/10, so d2 = d1/10^4
        let ratio = d1 / d2;
        assert!((ratio - 1e4).abs() < 1e-6, "ratio = {ratio}");
    }

    #[test]
    fn test_global_drift_bound() {
        let c = 1.0;
        let h = 0.01;
        let n = 1000;
        let total = global_drift_bound(c, h, n);
        let per_step = constraint_drift_bound(c, h);
        assert!((total - n as f64 * per_step).abs() < 1e-14);
    }

    #[test]
    fn test_adaptive_tolerance() {
        let h = 0.01;
        let tol = adaptive_tolerance(h, 10.0);
        // 10 * (0.01)^4 = 10 * 1e-8 = 1e-7
        assert!((tol - 1e-7).abs() < 1e-14, "tol = {tol}");
    }

    // -- Constraint statistics --

    #[test]
    fn test_constraint_stats() {
        let mut stats = ConstraintStats::new();
        stats.update(1e-6, false);
        stats.update(2e-6, false);
        stats.update(3e-6, true); // renormalized

        assert_eq!(stats.step_count, 3);
        assert_eq!(stats.renorm_count, 1);
        assert!((stats.max_constraint - 3e-6).abs() < 1e-20);
        assert!((stats.average_constraint() - 2e-6).abs() < 1e-20);
        assert!((stats.renorm_frequency() - 1.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_constraint_stats_empty() {
        let stats = ConstraintStats::new();
        assert_eq!(stats.average_constraint(), 0.0);
        assert_eq!(stats.renorm_frequency(), 0.0);
    }

    // -- Needs renormalization --

    #[test]
    fn test_needs_renormalization() {
        let g = schwarzschild_metric(10.0, std::f64::consts::FRAC_PI_2);

        // Good null ray: should NOT need renorm
        let vr = 1.0;
        let vt = (g[R][R] / (-g[T][T])).sqrt() * vr;
        let v_good = [vt, vr, 0.0, 0.0];
        assert!(!needs_renormalization(&g, &v_good, 1e-10));

        // Bad velocity: SHOULD need renorm
        let v_bad = [1.0, 1.0, 0.0, 0.0]; // not null
        assert!(needs_renormalization(&g, &v_bad, 1e-10));
    }

    // -- Kerr reduces to Schwarzschild --

    #[test]
    fn test_kerr_a0_matches_schwarzschild() {
        let r = 10.0;
        let theta = std::f64::consts::FRAC_PI_2;
        let g_s = schwarzschild_metric(r, theta);
        let g_k = kerr_metric(r, theta, 0.0);

        // Compare all diagonal components
        for i in 0..DIM {
            assert!(
                (g_s[i][i] - g_k[i][i]).abs() < 1e-10,
                "g[{i}][{i}]: Schw={}, Kerr(a=0)={}",
                g_s[i][i],
                g_k[i][i]
            );
        }
        // Off-diagonal should be zero
        assert!(g_k[T][PHI].abs() < 1e-14);
    }
}
