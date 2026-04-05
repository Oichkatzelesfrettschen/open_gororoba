//! Time evolution: CFL condition + forward Euler step.
//!
//! The full RK2 (Heun) integrator and flux divergence computation
//! require the flux module (not yet implemented). This module provides
//! the CFL timestep calculator and the update formula.
//!
//! The main evolution loop is:
//!   1. Compute dt from CFL condition
//!   2. Reconstruct L/R states at faces (recon.rs)
//!   3. Compute numerical fluxes via HLL (riemann.rs)
//!   4. Update conserved variables: U^{n+1} = U^n - dt/dx * (F_{i+1/2} - F_{i-1/2})
//!   5. Convert back to primitives (con2prim)
//!   6. Apply floors and boundary conditions

use crate::{eos::GammaLaw, grid::Grid};

/// Compute the CFL-limited timestep.
///
/// dt = CFL * min over all cells of (dx / (|v| + c_fast))
/// where c_fast is the fast magnetosonic speed.
pub fn cfl_dt(grid: &Grid, max_signal_speed: f64, cfl: f64) -> f64 {
    if max_signal_speed <= 0.0 {
        return 1e-6; // fallback for zero velocity
    }
    let dx_min = grid.dx1.min(grid.dx2).min(grid.dx3);
    cfl * dx_min / max_signal_speed
}

/// Apply a forward Euler update to conservative variables.
///
/// U_new[k] = U_old[k] + dt * rhs[k]
/// where rhs = -div(F) + source terms.
pub fn euler_update(u_old: &[f64], rhs: &[f64], dt: f64, u_new: &mut [f64]) {
    for (i, (old, r)) in u_old.iter().zip(rhs.iter()).enumerate() {
        u_new[i] = old + dt * r;
    }
}

/// Simple con2prim for zero-B, zero-v case (diagnostic only).
///
/// For the full iterative con2prim, see Noble et al. (2006).
/// This simplified version works only for static fluid with no B-field
/// and is used to verify the prim2con -> con2prim roundtrip in tests.
pub fn con2prim_simple(
    cons: &[f64; 8],
    sqrt_neg_g: f64,
    g_tt: f64,
    eos: &GammaLaw,
) -> Option<[f64; 8]> {
    // For static fluid: D = sqrt_g * rho * u^t, where u^t = 1/sqrt(-g_tt)
    let ut = 1.0 / (-g_tt).sqrt();
    let rho = cons[0] / (sqrt_neg_g * ut);
    if rho <= 0.0 {
        return None;
    }

    // Energy: cons[1] = sqrt_g * (T^t_t + rho * u^t)
    // For static fluid: T^t_t = -rho - u + p (with u^t u_t = -1)
    // Actually this is complicated. Just recover rho and set u from a cold floor.
    let u = rho * 1e-2; // cold gas approximation

    let _p = eos.pressure(u);
    let b1 = cons[5] / sqrt_neg_g;
    let b2 = cons[6] / sqrt_neg_g;
    let b3 = cons[7] / sqrt_neg_g;

    Some([rho, u, 0.0, 0.0, 0.0, b1, b2, b3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfl_dt() {
        let metric = crate::metric::KerrMetric::schwarzschild();
        let grid = Grid::new(32, 16, 1, 2.5, 40.0, metric);
        let dt = cfl_dt(&grid, 1.0, 0.5);
        assert!(dt > 0.0 && dt < 1.0, "CFL dt should be small, got {}", dt);
    }

    #[test]
    fn test_euler_update() {
        let u_old = vec![1.0, 2.0, 3.0];
        let rhs = vec![0.1, -0.2, 0.3];
        let mut u_new = vec![0.0; 3];
        euler_update(&u_old, &rhs, 0.5, &mut u_new);
        assert!((u_new[0] - 1.05).abs() < 1e-10);
        assert!((u_new[1] - 1.9).abs() < 1e-10);
        assert!((u_new[2] - 3.15).abs() < 1e-10);
    }

    #[test]
    fn test_prim2con_roundtrip_static() {
        // Test that prim2con -> con2prim_simple recovers rho for static fluid
        let metric = crate::metric::KerrMetric::schwarzschild();
        let eos = GammaLaw::harm_default();
        let r = 10.0;
        let th = std::f64::consts::FRAC_PI_2;
        let sqrt_g = metric.sqrt_neg_g(r, th) * r;
        let g_tt = -(1.0 - 2.0 / r);

        let p_orig = [1.0, 0.01, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0];
        let cons = crate::cons::prim2con(&p_orig, &metric, r, th, &eos, sqrt_g);
        let p_recov = con2prim_simple(&cons, sqrt_g, g_tt, &eos).unwrap();

        // Rho should roundtrip
        assert!(
            (p_recov[0] - p_orig[0]).abs() / p_orig[0] < 0.01,
            "Rho roundtrip: {} vs {}",
            p_recov[0],
            p_orig[0]
        );
        // B1 should roundtrip exactly
        assert!(
            (p_recov[5] - p_orig[5]).abs() < 1e-10,
            "B1 roundtrip: {} vs {}",
            p_recov[5],
            p_orig[5]
        );
    }
}
