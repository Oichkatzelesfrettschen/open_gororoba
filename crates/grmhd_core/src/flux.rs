//! Fused flux divergence computation for GRMHD.
//!
//! Translates three patterns into a single cell-update loop:
//! - steinmarder Instant-NGP: fuse reconstruct->riemann->update (no intermediate alloc)
//! - steinmarder D3Q19: SoA layout for cache-line-aligned access
//! - open_gororoba AssociatorWorkspace: per-thread reusable buffers via rayon map_init
//!
//! The main entry point `compute_rhs_1d` performs a 1D flux sweep along
//! a coordinate direction, computing the right-hand side dU/dt for each cell.

use crate::cons::{self, NCONS};
use crate::eos::GammaLaw;
use crate::grid::Grid;
use crate::metric::KerrMetric;
use crate::prims::{self, PrimGrid, Prim, NPRIM};
use crate::recon;
use crate::riemann;

/// Pre-computed metric quantities at each grid point.
/// Avoids repeated sqrt/trig during the flux loop.
/// Pattern from steinmarder: `__constant__` memory for LBM weights.
pub struct MetricCache {
    /// gcov[5] per cell: [g_tt, g_rr, g_thth, g_phph, g_tph]
    pub gcov: Vec<[f64; 5]>,
    /// gcon[5] per cell
    pub gcon: Vec<[f64; 5]>,
    /// sqrt(-g) * Jacobian per cell
    pub sqrt_neg_g: Vec<f64>,
    /// Lapse alpha per cell
    pub lapse: Vec<f64>,
}

impl MetricCache {
    /// Pre-compute all metric quantities on the grid.
    /// O(N) one-time cost, amortized over all timesteps.
    pub fn new(grid: &Grid) -> Self {
        let n = grid.n1_total() * grid.n2_total();
        let mut gcov = Vec::with_capacity(n);
        let mut gcon = Vec::with_capacity(n);
        let mut sqrt_neg_g = Vec::with_capacity(n);
        let mut lapse = Vec::with_capacity(n);

        for i in 0..grid.n1_total() {
            for j in 0..grid.n2_total() {
                let r = grid.r(i);
                let th = grid.theta(j);
                gcov.push(grid.metric.gcov(r, th));
                gcon.push(grid.metric.gcon(r, th));
                sqrt_neg_g.push(grid.sqrt_neg_g_at(i, j));
                lapse.push(grid.metric.lapse(r, th));
            }
        }

        Self { gcov, gcon, sqrt_neg_g, lapse }
    }

    /// Flat index into the metric cache for grid point (i, j).
    #[inline]
    fn idx(&self, i: usize, n2t: usize, j: usize) -> usize {
        i * n2t + j
    }
}

/// Per-thread workspace for the fused flux computation.
/// Pattern from cd_kernel AssociatorWorkspace: allocate once, reuse per face.
struct FluxWorkspace {
    prim_l: Prim,
    prim_r: Prim,
    cons_l: [f64; NCONS],
    cons_r: [f64; NCONS],
    flux_l: [f64; NCONS],
    flux_r: [f64; NCONS],
}

impl FluxWorkspace {
    fn new() -> Self {
        Self {
            prim_l: [0.0; NPRIM],
            prim_r: [0.0; NPRIM],
            cons_l: [0.0; NCONS],
            cons_r: [0.0; NCONS],
            flux_l: [0.0; NCONS],
            flux_r: [0.0; NCONS],
        }
    }
}

/// Compute the physical flux F^dir(P) for a given primitive state.
///
/// For direction dir=0 (radial): F^r_k = U_k * v^r + p_tot * delta^r_k - b^r * b_k
/// This is a simplified version that computes the flux from primitives directly.
fn compute_flux_from_prim(
    p: &Prim,
    _metric: &KerrMetric,
    _r: f64,
    _theta: f64,
    eos: &GammaLaw,
    _sqrt_neg_g: f64,
    dir: usize, // 0=r, 1=theta, 2=phi
) -> [f64; NCONS] {
    let rho = p[prims::RHO];
    let u = p[prims::UU];
    let v_dir = p[prims::V1 + dir]; // velocity in flux direction
    let pressure = eos.pressure(u);

    // Simplified flux (no magnetic terms for now -- pure hydro)
    let mut f = [0.0f64; NCONS];

    // Mass flux: F_D = D * v^dir
    f[0] = rho * v_dir;

    // Energy flux (simplified): F_E ~ (rho + u + p) * v^dir
    f[1] = (rho + u + pressure) * v_dir;

    // Momentum fluxes: F_{S_i} = S_i * v^dir + p * delta_{i,dir}
    for i in 0..3 {
        f[2 + i] = rho * v_dir * p[prims::V1 + i];
        if i == dir {
            f[2 + i] += pressure;
        }
    }

    // Magnetic flux: F_{B^i} (from induction equation, simplified)
    // dB^i/dt + d(v^dir B^i - v^i B^dir)/dx^dir = 0
    let b_dir = p[prims::B1 + dir];
    for i in 0..3 {
        f[5 + i] = v_dir * p[prims::B1 + i] - p[prims::V1 + i] * b_dir;
    }
    f[5 + dir] = 0.0; // div(B) = 0 constraint

    f
}

/// Compute the right-hand side dU/dt for a 1D sweep in direction `dir`.
///
/// Fused loop: for each face, load stencil -> reconstruct -> riemann -> accumulate.
/// Pattern from steinmarder fused collision+streaming kernel.
///
/// Returns the RHS vector (NCONS per cell, only interior cells filled).
pub fn compute_rhs_1d(
    prims: &PrimGrid,
    grid: &Grid,
    mc: &MetricCache,
    eos: &GammaLaw,
    dir: usize,
) -> Vec<[f64; NCONS]> {
    let ng = grid.ng;
    let n_active = match dir {
        0 => grid.n1,
        1 => grid.n2,
        2 => grid.n3,
        _ => panic!("Invalid direction {}", dir),
    };
    let n2t = grid.n2_total();
    let _n3t = grid.n3_total();

    // For simplicity, sweep radial direction (dir=0) at fixed j, k
    // Full 3D would loop over all (j, k) pairs
    let j_mid = ng + grid.n2 / 2; // equatorial plane
    let k_mid = ng; // phi = 0

    // Allocate RHS for active cells
    let mut rhs = vec![[0.0f64; NCONS]; n_active];

    // Fused face loop with workspace reuse
    let mut ws = FluxWorkspace::new();

    let dx = match dir {
        0 => grid.dx1,
        1 => grid.dx2,
        _ => grid.dx3,
    };

    // Sweep faces from ng to ng+n_active (inclusive for the last face)
    for face in ng..ng + n_active + 1 {
        let i_m2 = if face >= 2 { face - 2 } else { 0 };
        let i_m1 = if face >= 1 { face - 1 } else { 0 };
        let i_p0 = face;
        let i_p1 = (face + 1).min(grid.n1_total() - 1);

        // Load stencil primitives for reconstruction
        let idx_m2 = grid.idx(i_m2, j_mid, k_mid);
        let idx_m1 = grid.idx(i_m1, j_mid, k_mid);
        let idx_p0 = grid.idx(i_p0, j_mid, k_mid);
        let idx_p1 = grid.idx(i_p1, j_mid, k_mid);

        // Reconstruct each variable at the face (fused: no intermediate arrays)
        for var in 0..NPRIM {
            let qm2 = prims.get(idx_m2)[var];
            let qm1 = prims.get(idx_m1)[var];
            let qp0 = prims.get(idx_p0)[var];
            let qp1 = prims.get(idx_p1)[var];

            let (ql, qr) = recon::plm_lr(qm2, qm1, qp0, qp1);
            ws.prim_l[var] = ql;
            ws.prim_r[var] = qr;
        }

        // Compute fluxes from L/R states
        let r_face = grid.r(face);
        let th_face = grid.theta(j_mid);
        ws.flux_l = compute_flux_from_prim(
            &ws.prim_l, &grid.metric, r_face, th_face, eos, 1.0, dir,
        );
        ws.flux_r = compute_flux_from_prim(
            &ws.prim_r, &grid.metric, r_face, th_face, eos, 1.0, dir,
        );

        // Wave speed estimate
        let cs2_l = eos.cs2(ws.prim_l[prims::RHO], ws.prim_l[prims::UU]);
        let cs2_r = eos.cs2(ws.prim_r[prims::RHO], ws.prim_r[prims::UU]);
        let mc_idx = mc.idx(face, n2t, j_mid);
        let alpha = if mc_idx < mc.lapse.len() { mc.lapse[mc_idx] } else { 1.0 };

        let (sl, sr) = riemann::wave_speeds(
            ws.prim_l[prims::V1 + dir],
            ws.prim_r[prims::V1 + dir],
            cs2_l, cs2_r,
            0.0, 0.0, // va2 = 0 for pure hydro
            alpha,
        );

        // Compute conservative variables for HLL
        let mc_l = mc.idx(i_m1, n2t, j_mid);
        let mc_r = mc.idx(i_p0, n2t, j_mid);
        let sg_l = if mc_l < mc.sqrt_neg_g.len() { mc.sqrt_neg_g[mc_l] } else { 1.0 };
        let sg_r = if mc_r < mc.sqrt_neg_g.len() { mc.sqrt_neg_g[mc_r] } else { 1.0 };
        ws.cons_l = cons::prim2con(&ws.prim_l, &grid.metric, grid.r(i_m1), th_face, eos, sg_l);
        ws.cons_r = cons::prim2con(&ws.prim_r, &grid.metric, grid.r(i_p0), th_face, eos, sg_r);

        // HLL flux at this face
        let f_hll = riemann::hll_flux(&ws.flux_l, &ws.flux_r, &ws.cons_l, &ws.cons_r, sl, sr);

        // Accumulate divergence: rhs[i] -= (F_{i+1/2} - F_{i-1/2}) / dx
        // Left cell of this face: cell index = face - 1 - ng (in active coords)
        if face > ng {
            let cell_l = face - 1 - ng;
            if cell_l < n_active {
                for k in 0..NCONS {
                    rhs[cell_l][k] -= f_hll[k] / dx;
                }
            }
        }
        // Right cell: gets +F contribution (will be subtracted by the next face's -F)
        let cell_r = face - ng;
        if cell_r < n_active {
            for k in 0..NCONS {
                rhs[cell_r][k] += f_hll[k] / dx;
            }
        }
    }

    rhs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_cache_creation() {
        let metric = KerrMetric::schwarzschild();
        let grid = Grid::new(16, 8, 1, 2.5, 20.0, metric);
        let mc = MetricCache::new(&grid);
        assert_eq!(mc.gcov.len(), grid.n1_total() * grid.n2_total());
        assert!(mc.lapse[0] > 0.0);
    }

    #[test]
    fn test_compute_rhs_static_torus() {
        let metric = KerrMetric::schwarzschild();
        let grid = Grid::new(16, 8, 1, 2.5, 20.0, metric);
        let mc = MetricCache::new(&grid);
        let eos = GammaLaw::harm_default();

        let torus = crate::torus::FMTorus::schwarzschild(6.5, 12.0);
        let prims = torus.initialize(&grid);

        let rhs = compute_rhs_1d(&prims, &grid, &mc, &eos, 0);
        assert_eq!(rhs.len(), grid.n1);

        // RHS should be finite everywhere
        for (i, r) in rhs.iter().enumerate() {
            for (k, &v) in r.iter().enumerate() {
                assert!(v.is_finite(), "rhs[{}][{}] = {} is not finite", i, k, v);
            }
        }
    }

    #[test]
    fn test_flux_workspace_reuse() {
        // Verify FluxWorkspace can be reused without issues
        let mut ws = FluxWorkspace::new();
        let p1: Prim = [1.0, 0.01, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        let p2: Prim = [2.0, 0.02, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0];
        ws.prim_l = p1;
        ws.prim_r = p2;
        assert_eq!(ws.prim_l[prims::RHO], 1.0);
        ws.prim_l = p2;
        assert_eq!(ws.prim_l[prims::RHO], 2.0);
    }
}
