//! Constrained direction computation and Gauss-Newton 2D optimizer.
//!
//! # Purpose
//!
//! These are the numerical tools for finding the optimal (t_solar, t_atmo)
//! parameters in the V_6 perturbation space.  The constrained directions
//! use Gram-Schmidt orthogonalization to isolate solar and atmospheric
//! sensitivity while zeroing first-order reactor leakage.
//!
//! # Why constrained directions instead of unconstrained 6D optimization
//!
//! The 6D V_6 space has 6 degrees of freedom, but the PMNS matrix has only
//! 3 angles (+ 1 phase).  Unconstrained optimization in 6D is under-determined
//! and numerically unstable.  The constrained-direction approach reduces to 2D
//! (t_solar, t_atmo) by:
//!
//! 1. Computing gradient vectors g_12, g_13, g_23 (how each V_6 direction
//!    affects each PMNS angle).
//! 2. Projecting g_12 orthogonal to {g_13, g_23} -> solar direction u_solar
//!    (maximal theta_12 sensitivity with zero reactor/atmospheric leakage).
//! 3. Projecting g_23 orthogonal to {g_13, u_solar} -> atmospheric direction
//!    u_atmo (maximal theta_23 sensitivity, orthogonal to solar).
//!
//! This gives a 2D affine model: M_nu(t1, t2) = M_nu_base + t1*A + t2*B
//! where A, B are the mass matrix perturbations along u_solar, u_atmo.
//!
//! # Gauss-Newton with Levenberg-Marquardt damping
//!
//! The [`gauss_newton_2d`] solver minimizes weighted relative residuals:
//!
//! ```text
//! r_i = w_i * (theta_i - pdg_i) / pdg_i
//! ```
//!
//! using a 3x2 finite-difference Jacobian, Levenberg-Marquardt damped
//! normal equations (lambda = 0.01), and backtracking line search.
//! Convergence typically in 5-10 iterations for well-conditioned gradients.
//!
//! # Callers
//!
//! - `neutrino_sector::test_pmns_gauss_newton_regression` (C-1492)
//! - `neutrino_sector::test_cp_violation_phase_only` (C-1494)
//! - `neutrino_sector::test_cp_violation_joint_3d_scan` (C-1497)

/// Compute the constrained solar direction in V_6 space.
///
/// Projects g_12 orthogonal to the g_13 and g_23 constraint planes using
/// Gram-Schmidt orthogonalization. The result is a unit vector u such that:
///   g_13 . u = 0  (zero first-order reactor leakage)
///   g_23 . u = 0  (zero first-order atmospheric leakage)
///   g_12 . u is maximized (maximal solar sensitivity)
pub fn compute_constrained_solar_direction(
    g_12: &[f64; 6],
    g_13: &[f64; 6],
    g_23: &[f64; 6],
) -> [f64; 6] {
    let dot = |a: &[f64; 6], b: &[f64; 6]| -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    };

    let mut u1 = *g_13;
    let norm_u1 = dot(&u1, &u1).sqrt();
    if norm_u1 < 1e-15 {
        let mut u2 = *g_23;
        let norm_u2 = dot(&u2, &u2).sqrt();
        if norm_u2 < 1e-15 {
            let norm_12 = dot(g_12, g_12).sqrt();
            if norm_12 < 1e-15 { return [0.0; 6]; }
            let mut out = *g_12;
            for x in &mut out { *x /= norm_12; }
            return out;
        }
        for x in &mut u2 { *x /= norm_u2; }
        let proj = dot(g_12, &u2);
        let mut out = [0.0_f64; 6];
        for i in 0..6 { out[i] = g_12[i] - proj * u2[i]; }
        let norm = dot(&out, &out).sqrt();
        if norm < 1e-15 { return [0.0; 6]; }
        for x in &mut out { *x /= norm; }
        return out;
    }
    for x in &mut u1 { *x /= norm_u1; }

    let proj_23_on_1 = dot(g_23, &u1);
    let mut u2 = [0.0_f64; 6];
    for i in 0..6 {
        u2[i] = g_23[i] - proj_23_on_1 * u1[i];
    }
    let norm_u2 = dot(&u2, &u2).sqrt();
    if norm_u2 > 1e-15 {
        for x in &mut u2 { *x /= norm_u2; }
    }

    let proj_12_on_1 = dot(g_12, &u1);
    let proj_12_on_2 = dot(g_12, &u2);

    let mut optimal = [0.0_f64; 6];
    for i in 0..6 {
        optimal[i] = g_12[i] - proj_12_on_1 * u1[i] - proj_12_on_2 * u2[i];
    }

    let norm = dot(&optimal, &optimal).sqrt();
    if norm < 1e-15 { return [0.0; 6]; }
    for x in &mut optimal { *x /= norm; }

    optimal
}

/// Compute a constrained atmospheric direction orthogonal to the solar direction.
///
/// Projects g_23 orthogonal to {g_13, u_solar} using Gram-Schmidt.
pub fn compute_constrained_atmospheric_direction(
    g_23: &[f64; 6],
    g_13: &[f64; 6],
    u_solar: &[f64; 6],
) -> [f64; 6] {
    let dot = |a: &[f64; 6], b: &[f64; 6]| -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    };

    let mut u1 = *g_13;
    let norm_u1 = dot(&u1, &u1).sqrt();
    if norm_u1 < 1e-15 { return [0.0; 6]; }
    for x in &mut u1 { *x /= norm_u1; }

    let proj = dot(u_solar, &u1);
    let mut u2 = *u_solar;
    for i in 0..6 { u2[i] -= proj * u1[i]; }
    let norm_u2 = dot(&u2, &u2).sqrt();
    if norm_u2 < 1e-15 { return [0.0; 6]; }
    for x in &mut u2 { *x /= norm_u2; }

    let proj_1 = dot(g_23, &u1);
    let proj_2 = dot(g_23, &u2);
    let mut optimal = [0.0_f64; 6];
    for i in 0..6 {
        optimal[i] = g_23[i] - proj_1 * u1[i] - proj_2 * u2[i];
    }

    let norm = dot(&optimal, &optimal).sqrt();
    if norm < 1e-15 { return [0.0; 6]; }
    for x in &mut optimal { *x /= norm; }

    optimal
}

/// Gauss-Newton solver for 2D (t_solar, t_atmo) optimization.
///
/// Minimizes the weighted residual ||r(t)||^2 where
///   r = [w_i * (theta_i - pdg_i) / pdg_i]
/// using Levenberg-Marquardt damped normal equations with backtracking
/// line search.
///
/// Returns (best_t1, best_t2, best_angles, score).
pub fn gauss_newton_2d<F>(
    angles_fn: &F,
    t1_init: f64,
    t2_init: f64,
    pdg: (f64, f64, f64),
    weights: (f64, f64, f64),
    max_iter: usize,
) -> (f64, f64, (f64, f64, f64), f64)
where
    F: Fn(f64, f64) -> (f64, f64, f64),
{
    let eps = 0.01_f64;
    let mut t1 = t1_init;
    let mut t2 = t2_init;

    for _iter in 0..max_iter {
        let (a12, a13, a23) = angles_fn(t1, t2);
        let r = [
            weights.0 * (a12 - pdg.0) / pdg.0,
            weights.1 * (a13 - pdg.1) / pdg.1,
            weights.2 * (a23 - pdg.2) / pdg.2,
        ];

        let (a12_p1, a13_p1, a23_p1) = angles_fn(t1 + eps, t2);
        let (a12_m1, a13_m1, a23_m1) = angles_fn(t1 - eps, t2);
        let (a12_p2, a13_p2, a23_p2) = angles_fn(t1, t2 + eps);
        let (a12_m2, a13_m2, a23_m2) = angles_fn(t1, t2 - eps);

        let j = [
            [weights.0 * (a12_p1 - a12_m1) / (2.0 * eps * pdg.0),
             weights.0 * (a12_p2 - a12_m2) / (2.0 * eps * pdg.0)],
            [weights.1 * (a13_p1 - a13_m1) / (2.0 * eps * pdg.1),
             weights.1 * (a13_p2 - a13_m2) / (2.0 * eps * pdg.1)],
            [weights.2 * (a23_p1 - a23_m1) / (2.0 * eps * pdg.2),
             weights.2 * (a23_p2 - a23_m2) / (2.0 * eps * pdg.2)],
        ];

        let jtj = [
            [j[0][0]*j[0][0] + j[1][0]*j[1][0] + j[2][0]*j[2][0],
             j[0][0]*j[0][1] + j[1][0]*j[1][1] + j[2][0]*j[2][1]],
            [j[0][1]*j[0][0] + j[1][1]*j[1][0] + j[2][1]*j[2][0],
             j[0][1]*j[0][1] + j[1][1]*j[1][1] + j[2][1]*j[2][1]],
        ];
        let jtr = [
            j[0][0]*r[0] + j[1][0]*r[1] + j[2][0]*r[2],
            j[0][1]*r[0] + j[1][1]*r[1] + j[2][1]*r[2],
        ];

        let lambda = 0.01;
        let a11 = jtj[0][0] + lambda;
        let a12_m = jtj[0][1];
        let a22 = jtj[1][1] + lambda;
        let det = a11 * a22 - a12_m * a12_m;
        if det.abs() < 1e-30 { break; }

        let dt1 = -(a22 * jtr[0] - a12_m * jtr[1]) / det;
        let dt2 = -(a11 * jtr[1] - a12_m * jtr[0]) / det;

        let mut alpha = 1.0_f64;
        let current_cost: f64 = r.iter().map(|x| x * x).sum();
        for _ in 0..10 {
            let new_t1 = t1 + alpha * dt1;
            let new_t2 = t2 + alpha * dt2;
            let (na12, na13, na23) = angles_fn(new_t1, new_t2);
            let nr = [
                weights.0 * (na12 - pdg.0) / pdg.0,
                weights.1 * (na13 - pdg.1) / pdg.1,
                weights.2 * (na23 - pdg.2) / pdg.2,
            ];
            let new_cost: f64 = nr.iter().map(|x| x * x).sum();
            if new_cost < current_cost {
                t1 = new_t1;
                t2 = new_t2;
                break;
            }
            alpha *= 0.5;
        }

        if dt1.abs() < 1e-6 && dt2.abs() < 1e-6 { break; }
    }

    let (a12, a13, a23) = angles_fn(t1, t2);
    let score = ((a12 - pdg.0) / pdg.0).powi(2)
              + ((a13 - pdg.1) / pdg.1).powi(2)
              + ((a23 - pdg.2) / pdg.2).powi(2);
    (t1, t2, (a12, a13, a23), score)
}
