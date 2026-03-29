//! Fishbone-Moncrief equilibrium torus initial condition.
//!
//! Constructs a thick, pressure-supported torus orbiting a Kerr black hole.
//! The torus is the standard test problem for GRMHD codes because it is
//! unstable to the magnetorotational instability (MRI), generating turbulence
//! that feeds the accretion flow.
//!
//! Parameters:
//! - r_in: inner edge of the torus (typically 6M for Schwarzschild)
//! - r_max: radius of maximum pressure (typically 12M)
//! - The specific angular momentum l is determined by r_max.
//!
//! Reference: Fishbone & Moncrief (1976) ApJ 207, 962.

use crate::eos::GammaLaw;
use crate::grid::Grid;
use crate::metric::KerrMetric;
use crate::prims::{B1, B2, B3, PrimGrid, RHO, UU, V1, V2, V3};

/// Fishbone-Moncrief torus parameters.
pub struct FMTorus {
    /// Inner edge of the torus.
    pub r_in: f64,
    /// Radius of pressure maximum.
    pub r_max: f64,
    /// Specific angular momentum (computed from r_max).
    pub l: f64,
    /// Maximum density (normalization).
    pub rho_max: f64,
    /// Adiabatic index.
    pub gamma: f64,
}

impl FMTorus {
    /// Create a Fishbone-Moncrief torus for a Schwarzschild (a=0) black hole.
    /// This simplified version avoids the full Kerr angular momentum calculation.
    pub fn schwarzschild(r_in: f64, r_max: f64) -> Self {
        // For Schwarzschild, the covariant specific angular momentum at circular orbit r:
        // l = sqrt(M * r^2 / (r/M - 3))  (Gammie+ 2003 appendix)
        // We use l at r_max to define the torus
        let l = (r_max * r_max / (r_max - 3.0)).sqrt();
        Self {
            r_in,
            r_max,
            l,
            rho_max: 1.0,
            gamma: 13.0 / 9.0,
        }
    }

    /// Effective potential W(r, theta) for the Fishbone-Moncrief solution.
    /// In Schwarzschild: W = -1/(2 * ln_term) where ln_term involves the metric.
    /// Actually for Schwarzschild, the FM potential is:
    ///   W(r, theta) = ln|u_t| where u_t is computed from the metric and l.
    fn potential(&self, metric: &KerrMetric, r: f64, theta: f64) -> f64 {
        let sth = theta.sin();
        if sth.abs() < 1e-10 {
            return -1e10; // on axis, outside torus
        }

        // Use CONTRAVARIANT metric: g^tt, g^phph, g^tph
        let [gtt_up, _, _, gphph_up, gtph_up] = metric.gcon(r, theta);

        // From g^{mu nu} u_mu u_nu = -1 with u_mu = (u_t, 0, 0, l*u_t):
        // u_t^2 (g^tt + 2*l*g^tph + l^2*g^phph) = -1
        let denom = gtt_up + 2.0 * self.l * gtph_up + self.l * self.l * gphph_up;
        if denom >= 0.0 {
            return -1e10; // no valid orbit here
        }
        let ut_cov_sq = -1.0 / denom;
        let u_t = -(ut_cov_sq.sqrt()); // u_t is negative (future-directed)

        // W = ln(-u_t)
        (-u_t).ln()
    }

    /// Initialize the primitive variables on the grid.
    pub fn initialize(&self, grid: &Grid) -> PrimGrid {
        let mut prims = PrimGrid::new(grid.n_total());
        let _eos = GammaLaw::new(self.gamma);

        // Compute potential at r_in (equatorial plane) to set the zero level
        let w_in = self.potential(&grid.metric, self.r_in, std::f64::consts::FRAC_PI_2);
        let w_max = self.potential(&grid.metric, self.r_max, std::f64::consts::FRAC_PI_2);

        let kappa = (self.gamma - 1.0) / self.gamma; // polytropic constant
        let hm1_max = (w_max - w_in).exp() - 1.0; // max enthalpy - 1

        if hm1_max <= 0.0 {
            eprintln!("WARNING: FM torus hm1_max = {} <= 0, torus may be empty", hm1_max);
            return prims;
        }

        // Determine rho_max normalization
        // hm1 = kappa * K * rho^(gamma-1) where K is the polytropic constant
        // We set K so that rho_max = self.rho_max at the pressure maximum
        let k_poly = hm1_max / (kappa * self.rho_max.powf(self.gamma - 1.0));

        for i in 0..grid.n1_total() {
            for j in 0..grid.n2_total() {
                let r = grid.r(i);
                let th = grid.theta(j);
                let w = self.potential(&grid.metric, r, th);
                let hm1 = (w - w_in).exp() - 1.0;

                for k in 0..grid.n3_total() {
                    let idx = grid.idx(i, j, k);
                    let p = prims.get_mut(idx);

                    if hm1 > 0.0 && r > self.r_in {
                        // Inside the torus
                        let rho = (hm1 / (kappa * k_poly)).powf(1.0 / (self.gamma - 1.0));
                        let u = k_poly * rho.powf(self.gamma) / (self.gamma - 1.0);

                        p[RHO] = rho;
                        p[UU] = u;
                        // Keplerian velocity: v^phi = l / (g_phph + l * g_tph)
                        let [_, _, _, g_phph, g_tph] = grid.metric.gcov(r, th);
                        let denom = g_phph + self.l * g_tph;
                        p[V3] = if denom.abs() > 1e-20 { self.l / denom } else { 0.0 };
                        p[V1] = 0.0;
                        p[V2] = 0.0;
                    } else {
                        // Atmosphere (vacuum floor)
                        p[RHO] = 1e-6 * self.rho_max;
                        p[UU] = 1e-8 * self.rho_max;
                        p[V1] = 0.0;
                        p[V2] = 0.0;
                        p[V3] = 0.0;
                    }

                    // No magnetic field yet (added separately as vector potential)
                    p[B1] = 0.0;
                    p[B2] = 0.0;
                    p[B3] = 0.0;
                }
            }
        }

        prims
    }

    /// Add a weak poloidal magnetic field loop inside the torus.
    /// A_phi = max(rho/rho_max - 0.2, 0) * beta_factor
    /// This seeds the MRI without dominating the dynamics.
    pub fn add_magnetic_loop(&self, grid: &Grid, prims: &mut PrimGrid, beta_inv: f64) {
        // Simple approach: set B^r proportional to d(A_phi)/dtheta
        // and B^theta proportional to -d(A_phi)/dr
        // Using finite differences on the grid.

        let n1t = grid.n1_total();
        let n2t = grid.n2_total();
        let n3t = grid.n3_total();

        // First compute vector potential A_phi at cell centers
        let mut a_phi = vec![0.0f64; n1t * n2t];
        let mut max_p = 0.0f64;

        for i in 0..n1t {
            for j in 0..n2t {
                let idx0 = grid.idx(i, j, 0);
                let rho = prims.get(idx0)[RHO];
                let q = rho / self.rho_max - 0.2;
                a_phi[i * n2t + j] = if q > 0.0 { q } else { 0.0 };

                let p_gas = GammaLaw::new(self.gamma).pressure(prims.get(idx0)[UU]);
                if p_gas > max_p {
                    max_p = p_gas;
                }
            }
        }

        // Normalize: beta = p_gas / p_mag => p_mag = p_gas / beta => B^2 / 2 = max_p * beta_inv
        let b_norm = (2.0 * max_p * beta_inv).sqrt();

        // Set B via curl of A (simplified: B^1 ~ dA/dx2, B^2 ~ -dA/dx1)
        for i in 1..n1t - 1 {
            for j in 1..n2t - 1 {
                let da_dx2 = (a_phi[i * n2t + j + 1] - a_phi[i * n2t + j - 1]) / (2.0 * grid.dx2);
                let da_dx1 = (a_phi[(i + 1) * n2t + j] - a_phi[(i - 1) * n2t + j]) / (2.0 * grid.dx1);

                for k in 0..n3t {
                    let idx = grid.idx(i, j, k);
                    let p = prims.get_mut(idx);
                    p[B1] = b_norm * da_dx2;
                    p[B2] = -b_norm * da_dx1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fm_torus_creation() {
        let torus = FMTorus::schwarzschild(6.0, 12.0);
        assert!(torus.l > 0.0, "Angular momentum should be positive");
        // l = sqrt(r_max^2 / (r_max - 3)) = sqrt(144/9) = 4.0
        assert!((torus.l - 4.0).abs() < 0.01, "l = 4.0 for r_max=12, got {}", torus.l);
    }

    #[test]
    fn test_fm_torus_init() {
        let metric = KerrMetric::schwarzschild();
        let grid = Grid::new(32, 16, 1, 2.5, 40.0, metric);
        // r_in must be INSIDE the torus cusp (between ISCO=6 and r_max=12)
        let torus = FMTorus::schwarzschild(6.5, 12.0);
        let prims = torus.initialize(&grid);

        // Check that some cells have nonzero density (inside the torus)
        let mut max_rho = 0.0f64;
        for i in grid.ng..grid.ng + grid.n1 {
            for j in grid.ng..grid.ng + grid.n2 {
                let idx = grid.idx(i, j, grid.ng);
                let rho = prims.get(idx)[RHO];
                if rho > max_rho { max_rho = rho; }
            }
        }
        assert!(max_rho > 0.1, "Should have dense torus material, max_rho = {}", max_rho);
    }

    #[test]
    fn test_fm_torus_magnetic_field() {
        let metric = KerrMetric::schwarzschild();
        let grid = Grid::new(32, 16, 1, 2.5, 40.0, metric);
        let torus = FMTorus::schwarzschild(6.5, 12.0);
        let mut prims = torus.initialize(&grid);
        torus.add_magnetic_loop(&grid, &mut prims, 100.0); // beta = 100 (weak field)

        // Check that some cells have nonzero B-field
        let mut max_b = 0.0f64;
        for i in grid.ng..grid.ng + grid.n1 {
            for j in grid.ng..grid.ng + grid.n2 {
                let idx = grid.idx(i, j, grid.ng);
                let b1 = prims.get(idx)[B1];
                let b2 = prims.get(idx)[B2];
                let bmag = (b1 * b1 + b2 * b2).sqrt();
                if bmag > max_b { max_b = bmag; }
            }
        }
        assert!(max_b > 1e-10, "Should have nonzero B-field, max_b = {}", max_b);
    }
}
