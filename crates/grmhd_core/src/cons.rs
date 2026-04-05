//! Conservative variables and primitive-to-conservative conversion.
//!
//! The GRMHD system conserves:
//!   U[0] = sqrt(-g) * rho * u^t                    (mass)
//!   U[1] = sqrt(-g) * (T^t_t + rho * u^t)          (energy, with rest-mass subtracted)
//!   U[2..4] = sqrt(-g) * T^t_i                      (momentum, covariant)
//!   U[5..7] = sqrt(-g) * B^i                         (magnetic field)
//!
//! where T^mu_nu is the MHD stress-energy tensor:
//!   T^mu_nu = (rho + u + p + b^2) u^mu u_nu + (p + b^2/2) delta^mu_nu - b^mu b_nu
//!
//! and b^mu is the magnetic 4-vector (field in fluid frame).

use crate::{
    eos::GammaLaw,
    metric::KerrMetric,
    prims::{self, NPRIM, Prim},
};

/// Number of conserved variables (same as primitives).
pub const NCONS: usize = NPRIM;

/// Convert primitive variables to conservative variables at a single cell.
///
/// This is the forward direction (cheap, O(1) per cell).
/// The inverse (con2prim) requires iterative Newton-Raphson and is much harder.
pub fn prim2con(
    p: &Prim,
    metric: &KerrMetric,
    r: f64,
    theta: f64,
    eos: &GammaLaw,
    sqrt_neg_g: f64,
) -> [f64; NCONS] {
    let gcov = metric.gcov(r, theta);
    prim2con_cached(p, &gcov, eos, sqrt_neg_g)
}

/// Cached variant: takes precomputed metric components to avoid repeated trig.
/// This is the hot-path version called from the flux sweep inner loop.
#[inline]
pub fn prim2con_cached(p: &Prim, gcov: &[f64; 5], eos: &GammaLaw, sqrt_neg_g: f64) -> [f64; NCONS] {
    let rho = p[prims::RHO];
    let u = p[prims::UU];
    let v1 = p[prims::V1];
    let v2 = p[prims::V2];
    let v3 = p[prims::V3];
    let b1 = p[prims::B1];
    let b2 = p[prims::B2];
    let b3 = p[prims::B3];

    let pressure = eos.pressure(u);

    // Metric components (from cache -- no trig)
    let [g_tt, g_rr, g_thth, g_phph, g_tph] = *gcov;

    // 3-velocity squared: v^2 = g_ij v^i v^j
    let vsq = g_rr * v1 * v1 + g_thth * v2 * v2 + g_phph * v3 * v3;

    // Lorentz factor: u^t = 1 / sqrt(-(g_tt + 2*g_tph*v3 + g_ij*v^i*v^j))
    let alpha_sq = -(g_tt + 2.0 * g_tph * v3 + vsq);
    if alpha_sq <= 0.0 {
        // Superluminal -- return floor state
        return [
            sqrt_neg_g * rho,
            0.0,
            0.0,
            0.0,
            0.0,
            sqrt_neg_g * b1,
            sqrt_neg_g * b2,
            sqrt_neg_g * b3,
        ];
    }
    let ut = 1.0 / alpha_sq.sqrt();

    // Covariant 4-velocity: u_mu = g_mu_nu u^nu
    // u^mu = u^t (1, v^1, v^2, v^3)
    let u_cov_t = g_tt * ut + g_tph * ut * v3;
    let u_cov_r = g_rr * ut * v1;
    let u_cov_th = g_thth * ut * v2;
    let u_cov_ph = g_phph * ut * v3 + g_tph * ut;

    // Magnetic 4-vector b^mu (in fluid frame)
    // b^t = B^i u_i / u^t  (where u_i is covariant spatial velocity)
    let bt = (b1 * u_cov_r + b2 * u_cov_th + b3 * u_cov_ph) / ut;
    // b^i = (B^i + b^t * u^t * v^i) / u^t
    let b_up_1 = (b1 + bt * ut * v1) / ut;
    let b_up_2 = (b2 + bt * ut * v2) / ut;
    let b_up_3 = (b3 + bt * ut * v3) / ut;

    // b^2 = b_mu b^mu = b^t^2 * g_tt + ... (simplified)
    let bsq = (b1 * b1 * g_rr + b2 * b2 * g_thth + b3 * b3 * g_phph) / (ut * ut)
        + bt * bt * (-1.0 / (ut * ut) + vsq);
    let bsq = bsq.max(0.0); // numerical floor

    // Total enthalpy density with magnetic contribution
    let w = rho + u + pressure + bsq;
    let ptot = pressure + 0.5 * bsq;

    // Conservative variables
    let mut cons = [0.0f64; NCONS];

    // Mass conservation: D = sqrt(-g) * rho * u^t
    cons[0] = sqrt_neg_g * rho * ut;

    // Energy: U_energy = sqrt(-g) * (T^t_t + rho * u^t)
    // T^t_t = w * u^t * u_t + ptot - b^t * b_t
    let ttt = w * ut * u_cov_t + ptot - bt * (g_tt * bt + g_tph * b_up_3);
    cons[1] = sqrt_neg_g * (ttt + rho * ut);

    // Momentum: U_i = sqrt(-g) * T^t_i
    // T^t_r = w * u^t * u_r - b^t * b_r
    cons[2] = sqrt_neg_g * (w * ut * u_cov_r - bt * (g_rr * b_up_1));
    cons[3] = sqrt_neg_g * (w * ut * u_cov_th - bt * (g_thth * b_up_2));
    cons[4] = sqrt_neg_g * (w * ut * u_cov_ph - bt * (g_phph * b_up_3 + g_tph * bt));

    // Magnetic field: just sqrt(-g) * B^i
    cons[5] = sqrt_neg_g * b1;
    cons[6] = sqrt_neg_g * b2;
    cons[7] = sqrt_neg_g * b3;

    cons
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prim2con_static_fluid() {
        // Static fluid (v=0, B=0) in Schwarzschild at r=10
        let metric = KerrMetric::schwarzschild();
        let eos = GammaLaw::harm_default();
        let r = 10.0;
        let th = std::f64::consts::FRAC_PI_2;
        let sqrt_g = metric.sqrt_neg_g(r, th) * r; // approximate

        let p: Prim = [1.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let u = prim2con(&p, &metric, r, th, &eos, sqrt_g);

        // Mass: D = sqrt_g * rho * u^t. For static fluid, u^t = 1/sqrt(-g_tt)
        let g_tt = -(1.0 - 2.0 / r);
        let ut_expected = 1.0 / (-g_tt).sqrt();
        let d_expected = sqrt_g * 1.0 * ut_expected;
        assert!(
            (u[0] - d_expected).abs() / d_expected < 0.01,
            "Mass D: got {}, expected {}",
            u[0],
            d_expected
        );
    }

    #[test]
    fn test_prim2con_preserves_b() {
        // B-field should be preserved as sqrt(-g) * B^i
        let metric = KerrMetric::schwarzschild();
        let eos = GammaLaw::harm_default();
        let r = 10.0;
        let th = std::f64::consts::FRAC_PI_2;
        let sqrt_g = metric.sqrt_neg_g(r, th) * r;

        let p: Prim = [1.0, 0.01, 0.0, 0.0, 0.0, 0.1, 0.2, 0.05];
        let u = prim2con(&p, &metric, r, th, &eos, sqrt_g);

        assert!((u[5] - sqrt_g * 0.1).abs() < 1e-10);
        assert!((u[6] - sqrt_g * 0.2).abs() < 1e-10);
        assert!((u[7] - sqrt_g * 0.05).abs() < 1e-10);
    }
}
