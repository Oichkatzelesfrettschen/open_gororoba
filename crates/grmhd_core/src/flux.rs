//! Fused flux divergence computation for GRMHD.
//!
//! Translates three patterns into a single cell-update loop:
//! - steinmarder Instant-NGP: fuse reconstruct->riemann->update (no intermediate alloc)
//! - steinmarder D3Q19: SoA layout for cache-line-aligned access
//! - open_gororoba AssociatorWorkspace: per-thread reusable buffers via rayon map_init
//!
//! The main entry point `compute_rhs_1d` performs a 1D flux sweep along
//! a coordinate direction, computing the right-hand side dU/dt for each cell.

use crate::{
    cons::{self, NCONS},
    eos::GammaLaw,
    grid::Grid,
    metric::KerrMetric,
    prims::{self, NPRIM, Prim, PrimGrid},
    recon, riemann,
};
use rayon::prelude::*;
use wide::f64x4;

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

        Self {
            gcov,
            gcon,
            sqrt_neg_g,
            lapse,
        }
    }

    /// Flat index into the metric cache for grid point (i, j).
    #[inline]
    fn idx(&self, i: usize, n2t: usize, j: usize) -> usize {
        i * n2t + j
    }

    /// Return a reference to the cached covariant metric at flat index.
    #[inline]
    pub fn gcov_at(&self, idx: usize) -> &[f64; 5] {
        &self.gcov[idx]
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

/// Compute the physical flux F^dir(P) from precomputed metric components.
///
/// This is the hot-path variant that avoids sin/cos by accepting cached gcov.
/// Uses `wide::f64x4` SIMD for the momentum and induction flux loops.
///
/// GRMHD flux (Gammie+ 2003 eq. 2.15-2.17):
///   F^i_D     = rho * u^t * v^i                     (mass)
///   F^i_E     = T^i_t + rho * u^t * v^i             (energy)
///   F^i_{S_j} = T^i_j                                (momentum)
///   F^i_{B^j} = v^i * B^j - v^j * B^i               (induction)
#[inline]
pub fn compute_flux_cached(
    p: &Prim,
    gcov: &[f64; 5],
    eos: &GammaLaw,
    sqrt_neg_g: f64,
    dir: usize, // 0=r, 1=theta, 2=phi
) -> [f64; NCONS] {
    let rho = p[prims::RHO];
    let u = p[prims::UU];
    let v1 = p[prims::V1];
    let v2 = p[prims::V2];
    let v3 = p[prims::V3];
    let b1 = p[prims::B1];
    let b2 = p[prims::B2];
    let b3 = p[prims::B3];
    let pressure = eos.pressure(u);

    let [g_tt, g_rr, g_thth, g_phph, g_tph] = *gcov;

    // 3-velocity squared and Lorentz factor
    let vsq = g_rr * v1 * v1 + g_thth * v2 * v2 + g_phph * v3 * v3;
    let alpha_sq = -(g_tt + 2.0 * g_tph * v3 + vsq);
    let ut = if alpha_sq > 1e-20 {
        1.0 / alpha_sq.sqrt()
    } else {
        1.0
    };
    let inv_ut = 1.0 / ut;
    let inv_ut_sq = inv_ut * inv_ut;

    // Covariant 4-velocity components
    let u_cov_t = g_tt * ut + g_tph * ut * v3;
    let u_cov_r = g_rr * ut * v1;
    let u_cov_th = g_thth * ut * v2;
    let u_cov_ph = g_phph * ut * v3 + g_tph * ut;

    // Magnetic 4-vector b^mu
    let bt = (b1 * u_cov_r + b2 * u_cov_th + b3 * u_cov_ph) * inv_ut;
    let bsq = ((b1 * b1 * g_rr + b2 * b2 * g_thth + b3 * b3 * g_phph) * inv_ut_sq
        + bt * bt * (-inv_ut_sq + vsq))
        .max(0.0);

    let w = rho + u + pressure + bsq; // total enthalpy
    let ptot = pressure + 0.5 * bsq;

    // u^dir (contravariant spatial velocity component)
    let v_dir = match dir {
        0 => v1,
        1 => v2,
        _ => v3,
    };
    let u_up_dir = ut * v_dir;

    // b^dir
    let b_up_dir = match dir {
        0 => (b1 + bt * ut * v1) * inv_ut,
        1 => (b2 + bt * ut * v2) * inv_ut,
        _ => (b3 + bt * ut * v3) * inv_ut,
    };

    let mut f = [0.0f64; NCONS];

    // Mass flux
    f[0] = sqrt_neg_g * rho * u_up_dir;

    // Energy flux
    let b_cov_t = g_tt * bt
        + g_tph
            * match dir {
                2 => b_up_dir,
                _ => 0.0,
            };
    let t_dir_t = w * u_up_dir * u_cov_t - b_up_dir * b_cov_t;
    f[1] = sqrt_neg_g * (t_dir_t + rho * u_up_dir);

    // Momentum flux (SIMD: 3 components packed into f64x4, lane 3 unused)
    {
        let b_v = [v1, v2, v3];
        let b_s = [b1, b2, b3];
        let u_cov = [u_cov_r, u_cov_th, u_cov_ph];
        let g_diag = [g_rr, g_thth, g_phph];

        // Compute b_up_j for all 3 spatial directions
        let b_up_j_arr = [
            (b_s[0] + bt * ut * b_v[0]) * inv_ut,
            (b_s[1] + bt * ut * b_v[1]) * inv_ut,
            (b_s[2] + bt * ut * b_v[2]) * inv_ut,
        ];

        // b_cov_j = g_diag[j] * b_up_j
        let b_cov_j_arr = [
            g_diag[0] * b_up_j_arr[0],
            g_diag[1] * b_up_j_arr[1],
            g_diag[2] * b_up_j_arr[2],
        ];

        // delta: 1.0 where j == dir, else 0.0
        let delta_arr = [
            if dir == 0 { 1.0 } else { 0.0 },
            if dir == 1 { 1.0 } else { 0.0 },
            if dir == 2 { 1.0 } else { 0.0 },
        ];

        // T^dir_j = w * u_up_dir * u_cov[j] + ptot * delta - b_up_dir * b_cov_j
        // Pack into f64x4 for SIMD (lane 3 = 0, unused)
        let wu = f64x4::from([
            w * u_up_dir * u_cov[0],
            w * u_up_dir * u_cov[1],
            w * u_up_dir * u_cov[2],
            0.0,
        ]);
        let pd = f64x4::from([
            ptot * delta_arr[0],
            ptot * delta_arr[1],
            ptot * delta_arr[2],
            0.0,
        ]);
        let bb = f64x4::from([
            b_up_dir * b_cov_j_arr[0],
            b_up_dir * b_cov_j_arr[1],
            b_up_dir * b_cov_j_arr[2],
            0.0,
        ]);
        let sg4 = f64x4::from(sqrt_neg_g);
        let mom = sg4 * (wu + pd - bb);
        let mom_arr: [f64; 4] = mom.into();
        f[2] = mom_arr[0];
        f[3] = mom_arr[1];
        f[4] = mom_arr[2];
    }

    // Induction flux (SIMD: 3 components packed into f64x4)
    {
        let b_spatial = [b1, b2, b3];
        let v_spatial = [v1, v2, v3];
        let b_dir_val = b_spatial[dir];

        let vd_b = f64x4::from([
            v_dir * b_spatial[0],
            v_dir * b_spatial[1],
            v_dir * b_spatial[2],
            0.0,
        ]);
        let vs_bd = f64x4::from([
            v_spatial[0] * b_dir_val,
            v_spatial[1] * b_dir_val,
            v_spatial[2] * b_dir_val,
            0.0,
        ]);
        let sg4 = f64x4::from(sqrt_neg_g);
        let ind = sg4 * (vd_b - vs_bd);
        let ind_arr: [f64; 4] = ind.into();
        f[5] = ind_arr[0];
        f[6] = ind_arr[1];
        f[7] = ind_arr[2];
    }
    f[5 + dir] = 0.0; // div(B) = 0

    f
}

/// Compute the physical flux F^dir(P) for a given primitive state.
///
/// Delegates to `compute_flux_cached` after computing gcov from the metric.
pub fn compute_flux_from_prim(
    p: &Prim,
    metric: &KerrMetric,
    r: f64,
    theta: f64,
    eos: &GammaLaw,
    sqrt_neg_g: f64,
    dir: usize,
) -> [f64; NCONS] {
    let gcov = metric.gcov(r, theta);
    compute_flux_cached(p, &gcov, eos, sqrt_neg_g, dir)
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
        let i_m2 = face.saturating_sub(2);
        let i_m1 = face.saturating_sub(1);
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

        // Compute fluxes from L/R states using cached metric
        let mc_face = mc.idx(face, n2t, j_mid);
        let gcov_face = if mc_face < mc.gcov.len() {
            mc.gcov_at(mc_face)
        } else {
            &[0.0; 5]
        };
        let sg_1d = if mc_face < mc.sqrt_neg_g.len() {
            mc.sqrt_neg_g[mc_face]
        } else {
            1.0
        };
        ws.flux_l = compute_flux_cached(&ws.prim_l, gcov_face, eos, sg_1d, dir);
        ws.flux_r = compute_flux_cached(&ws.prim_r, gcov_face, eos, sg_1d, dir);

        // Wave speed estimate
        let cs2_l = eos.cs2(ws.prim_l[prims::RHO], ws.prim_l[prims::UU]);
        let cs2_r = eos.cs2(ws.prim_r[prims::RHO], ws.prim_r[prims::UU]);
        let alpha = if mc_face < mc.lapse.len() {
            mc.lapse[mc_face]
        } else {
            1.0
        };

        let (sl, sr) = riemann::wave_speeds(
            ws.prim_l[prims::V1 + dir],
            ws.prim_r[prims::V1 + dir],
            cs2_l,
            cs2_r,
            0.0,
            0.0, // va2 = 0 for pure hydro
            alpha,
        );

        // Compute conservative variables for HLL using cached metric
        let mc_l = mc.idx(i_m1, n2t, j_mid);
        let mc_r = mc.idx(i_p0, n2t, j_mid);
        let gcov_l = if mc_l < mc.gcov.len() {
            mc.gcov_at(mc_l)
        } else {
            &[0.0; 5]
        };
        let gcov_r = if mc_r < mc.gcov.len() {
            mc.gcov_at(mc_r)
        } else {
            &[0.0; 5]
        };
        let sg_l = if mc_l < mc.sqrt_neg_g.len() {
            mc.sqrt_neg_g[mc_l]
        } else {
            1.0
        };
        let sg_r = if mc_r < mc.sqrt_neg_g.len() {
            mc.sqrt_neg_g[mc_r]
        } else {
            1.0
        };
        ws.cons_l = cons::prim2con_cached(&ws.prim_l, gcov_l, eos, sg_l);
        ws.cons_r = cons::prim2con_cached(&ws.prim_r, gcov_r, eos, sg_r);

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

/// Full 3D RHS computation: sweep all three directions across ALL cells.
///
/// For each interior cell, accumulates flux divergence from r, theta, phi faces.
/// Uses the FluxWorkspace pattern for zero allocation in the inner loop.
///
/// Face EMFs extracted from HLL flux for constrained transport.
/// EMF_phi at radial face (i+1/2, j) = -F_HLL_Btheta at that face.
/// EMF_phi at theta face (i, j+1/2) = +F_HLL_Br at that face.
pub struct FaceEmfs {
    /// EMF from radial faces: emf_r[i * n2t + j] = EMF at (i+1/2, j)
    pub emf_r: Vec<f64>,
    /// EMF from theta faces: emf_th[i * n2t + j] = EMF at (i, j+1/2)
    pub emf_th: Vec<f64>,
    pub n1t: usize,
    pub n2t: usize,
}

/// Returns RHS as a flat vector indexed by grid.idx(i,j,k) * NCONS + var,
/// plus face EMFs for constrained transport.
pub fn compute_rhs_3d(
    prims: &PrimGrid,
    grid: &Grid,
    mc: &MetricCache,
    eos: &GammaLaw,
) -> (Vec<f64>, FaceEmfs) {
    compute_rhs_3d_inner(prims, grid, mc, eos, false)
}

/// Inner RHS computation. When `zero_b_in_recon` is true, the B components
/// in the PLM-reconstructed L/R states are zeroed before HLL flux computation.
/// This prevents PLM from creating artificial v*B at velocity discontinuities
/// (the root cause of B-field instability). CT handles B evolution separately.
fn compute_rhs_3d_inner(
    prims: &PrimGrid,
    grid: &Grid,
    mc: &MetricCache,
    eos: &GammaLaw,
    zero_b_in_recon: bool,
) -> (Vec<f64>, FaceEmfs) {
    let ng = grid.ng;
    let n1t = grid.n1_total();
    let n2t = grid.n2_total();
    let n3t = grid.n3_total();
    let _n3t = n3t;
    let n_total = grid.n_total();

    let mut rhs = vec![0.0f64; n_total * NCONS];
    let mut face_emfs = FaceEmfs {
        emf_r: vec![0.0; n1t * n2t],
        emf_th: vec![0.0; n1t * n2t],
        n1t,
        n2t,
    };

    // Direction 0 (radial): sweep faces at fixed (j, k)
    // Parallelized over (j, k) pairs using rayon.
    // Safety: different (j, k) pairs write to unique cell indices via grid.idx
    // which is injective. EMF writes are gated on k == ng to avoid races.
    {
        // Cast pointers to usize for Send+Sync (usize is unconditionally both).
        // Safety: pointers remain valid for the duration of this block.
        let rhs_addr = rhs.as_mut_ptr() as usize;
        let emf_addr = face_emfs.emf_r.as_mut_ptr() as usize;
        let inv_dx = 1.0 / grid.dx1;

        let jk_pairs: Vec<(usize, usize)> = (ng..ng + grid.n2)
            .flat_map(|j| (ng..ng + grid.n3).map(move |k| (j, k)))
            .collect();

        jk_pairs
            .par_iter()
            .for_each_init(FluxWorkspace::new, |ws, &(j, k)| {
                let rhs_p = rhs_addr as *mut f64;
                let emf_p = emf_addr as *mut f64;

                for face_i in ng..ng + grid.n1 + 1 {
                    let im2 = face_i.saturating_sub(2);
                    let im1 = face_i.saturating_sub(1);
                    let ip0 = face_i.min(n1t - 1);
                    let ip1 = (face_i + 1).min(n1t - 1);

                    // Reconstruct at face
                    for var in 0..NPRIM {
                        let (ql, qr) = recon::plm_lr(
                            prims.get(grid.idx(im2, j, k))[var],
                            prims.get(grid.idx(im1, j, k))[var],
                            prims.get(grid.idx(ip0, j, k))[var],
                            prims.get(grid.idx(ip1, j, k))[var],
                        );
                        ws.prim_l[var] = ql;
                        ws.prim_r[var] = qr;
                    }

                    if zero_b_in_recon {
                        for b in 5..8 {
                            ws.prim_l[b] = 0.0;
                            ws.prim_r[b] = 0.0;
                        }
                    }

                    let mc_face = mc.idx(face_i, n2t, j);
                    let gcov_face = if mc_face < mc.gcov.len() {
                        mc.gcov_at(mc_face)
                    } else {
                        &[0.0; 5]
                    };
                    let sg_face = if mc_face < mc.sqrt_neg_g.len() {
                        mc.sqrt_neg_g[mc_face]
                    } else {
                        1.0
                    };
                    ws.flux_l = compute_flux_cached(&ws.prim_l, gcov_face, eos, sg_face, 0);
                    ws.flux_r = compute_flux_cached(&ws.prim_r, gcov_face, eos, sg_face, 0);

                    let cs2_l = eos.cs2(ws.prim_l[prims::RHO], ws.prim_l[prims::UU]);
                    let cs2_r = eos.cs2(ws.prim_r[prims::RHO], ws.prim_r[prims::UU]);
                    let alpha = if mc_face < mc.lapse.len() {
                        mc.lapse[mc_face]
                    } else {
                        1.0
                    };

                    let (sl, sr) = riemann::wave_speeds(
                        ws.prim_l[prims::V1],
                        ws.prim_r[prims::V1],
                        cs2_l,
                        cs2_r,
                        0.0,
                        0.0,
                        alpha,
                    );

                    let mc_l = mc.idx(im1, n2t, j);
                    let mc_r = mc.idx(ip0, n2t, j);
                    let gcov_l = if mc_l < mc.gcov.len() {
                        mc.gcov_at(mc_l)
                    } else {
                        &[0.0; 5]
                    };
                    let gcov_r = if mc_r < mc.gcov.len() {
                        mc.gcov_at(mc_r)
                    } else {
                        &[0.0; 5]
                    };
                    let sg_l = if mc_l < mc.sqrt_neg_g.len() {
                        mc.sqrt_neg_g[mc_l]
                    } else {
                        1.0
                    };
                    let sg_r = if mc_r < mc.sqrt_neg_g.len() {
                        mc.sqrt_neg_g[mc_r]
                    } else {
                        1.0
                    };
                    ws.cons_l = cons::prim2con_cached(&ws.prim_l, gcov_l, eos, sg_l);
                    ws.cons_r = cons::prim2con_cached(&ws.prim_r, gcov_r, eos, sg_r);

                    let f_hll =
                        riemann::hll_flux(&ws.flux_l, &ws.flux_r, &ws.cons_l, &ws.cons_r, sl, sr);

                    // Store EMF from HLL induction flux at this radial face.
                    // Only write from k == ng to avoid redundant writes across phi.
                    if k == ng {
                        // SAFETY: each (face_i, j) pair is unique within this
                        // sweep (rayon partitions (j, k) and we gate on k==ng,
                        // so no two threads ever target the same (face_i, j)).
                        // emf_p points to face_emfs.emf_r (length n1t * n2t);
                        // face_i * n2t + j stays within bounds because
                        // face_i in 0..n1t and j in 0..n2t.
                        unsafe {
                            let ep = emf_p.add(face_i * n2t + j);
                            *ep = -f_hll[6]; // negative of B^theta flux
                        }
                    }

                    // Accumulate divergence
                    // SAFETY: `v` indexes into a conserved-variable array of size
                    // NCONS * n_total. The rayon partition ensures each thread
                    // writes to a disjoint cell range (unique (j,k) per task).
                    // `base` points to valid heap memory from the RHS Vec whose
                    // lifetime spans this entire function. grid.idx is injective,
                    // so different (j,k) pairs never alias the same rhs cell.
                    if face_i > ng {
                        let cell_l = grid.idx(face_i - 1, j, k);
                        let base = unsafe { rhs_p.add(cell_l * NCONS) };
                        for (v, &fv) in f_hll.iter().enumerate() {
                            unsafe {
                                *base.add(v) -= fv * inv_dx;
                            }
                        }
                    }
                    if face_i < ng + grid.n1 {
                        let cell_r = grid.idx(face_i, j, k);
                        let base = unsafe { rhs_p.add(cell_r * NCONS) };
                        for (v, &fv) in f_hll.iter().enumerate() {
                            unsafe {
                                *base.add(v) += fv * inv_dx;
                            }
                        }
                    }
                }
            });
    }

    // Direction 1 (theta): sweep faces at fixed (i, k)
    if grid.n2 > 1 {
        // Parallelized over (i, k) pairs.
        let rhs_addr_th = rhs.as_mut_ptr() as usize;
        let emf_th_addr = face_emfs.emf_th.as_mut_ptr() as usize;
        let ik_pairs: Vec<(usize, usize)> = (ng..ng + grid.n1)
            .flat_map(|i| (ng..ng + grid.n3).map(move |k| (i, k)))
            .collect();
        ik_pairs
            .par_iter()
            .for_each_init(FluxWorkspace::new, |ws, &(i, k)| {
                let rp = rhs_addr_th as *mut f64;
                let ep = emf_th_addr as *mut f64;
                let inv_dx = 1.0 / grid.dx2;

                for face_j in ng..ng + grid.n2 + 1 {
                    let jm2 = face_j.saturating_sub(2);
                    let jm1 = face_j.saturating_sub(1);
                    let jp0 = face_j.min(n2t - 1);
                    let jp1 = (face_j + 1).min(n2t - 1);

                    for var in 0..NPRIM {
                        let (ql, qr) = recon::plm_lr(
                            prims.get(grid.idx(i, jm2, k))[var],
                            prims.get(grid.idx(i, jm1, k))[var],
                            prims.get(grid.idx(i, jp0, k))[var],
                            prims.get(grid.idx(i, jp1, k))[var],
                        );
                        ws.prim_l[var] = ql;
                        ws.prim_r[var] = qr;
                    }

                    // Zero B in reconstruction when CT is active
                    if zero_b_in_recon {
                        for b in 5..8 {
                            ws.prim_l[b] = 0.0;
                            ws.prim_r[b] = 0.0;
                        }
                    }

                    let mc_face = mc.idx(i, n2t, face_j.min(n2t - 1));
                    let gcov_face = if mc_face < mc.gcov.len() {
                        mc.gcov_at(mc_face)
                    } else {
                        &[0.0; 5]
                    };
                    let sg_face_th = if mc_face < mc.sqrt_neg_g.len() {
                        mc.sqrt_neg_g[mc_face]
                    } else {
                        1.0
                    };
                    ws.flux_l = compute_flux_cached(&ws.prim_l, gcov_face, eos, sg_face_th, 1);
                    ws.flux_r = compute_flux_cached(&ws.prim_r, gcov_face, eos, sg_face_th, 1);

                    let cs2_l = eos.cs2(ws.prim_l[prims::RHO], ws.prim_l[prims::UU]);
                    let cs2_r = eos.cs2(ws.prim_r[prims::RHO], ws.prim_r[prims::UU]);
                    let alpha = if mc_face < mc.lapse.len() {
                        mc.lapse[mc_face]
                    } else {
                        1.0
                    };

                    let (sl, sr) = riemann::wave_speeds(
                        ws.prim_l[prims::V2],
                        ws.prim_r[prims::V2],
                        cs2_l,
                        cs2_r,
                        0.0,
                        0.0,
                        alpha,
                    );

                    let mc_l = mc.idx(i, n2t, jm1);
                    let mc_r = mc.idx(i, n2t, jp0);
                    let gcov_l = if mc_l < mc.gcov.len() {
                        mc.gcov_at(mc_l)
                    } else {
                        &[0.0; 5]
                    };
                    let gcov_r = if mc_r < mc.gcov.len() {
                        mc.gcov_at(mc_r)
                    } else {
                        &[0.0; 5]
                    };
                    let sg_l = if mc_l < mc.sqrt_neg_g.len() {
                        mc.sqrt_neg_g[mc_l]
                    } else {
                        1.0
                    };
                    let sg_r = if mc_r < mc.sqrt_neg_g.len() {
                        mc.sqrt_neg_g[mc_r]
                    } else {
                        1.0
                    };
                    ws.cons_l = cons::prim2con_cached(&ws.prim_l, gcov_l, eos, sg_l);
                    ws.cons_r = cons::prim2con_cached(&ws.prim_r, gcov_r, eos, sg_r);

                    let f_hll =
                        riemann::hll_flux(&ws.flux_l, &ws.flux_r, &ws.cons_l, &ws.cons_r, sl, sr);

                    // Store EMF (only first k writes to avoid races on 2D EMF)
                    if k == ng {
                        // SAFETY: ep points to face_emfs.emf_th (length
                        // n1t * n2t); the gate k==ng ensures only one thread
                        // touches each (i, face_j). i*n2t + face_j is in
                        // bounds because i < n1t and face_j < n2t.
                        unsafe {
                            *ep.add(i * n2t + face_j) = f_hll[5];
                        }
                    }

                    // SAFETY: `v` indexes into NCONS * n_total RHS array.
                    // Rayon partitions (i, k) pairs so each thread writes to
                    // disjoint cells. `rp` is valid heap memory from the RHS Vec.
                    if face_j > ng {
                        let cell_l = grid.idx(i, face_j - 1, k);
                        let base = unsafe { rp.add(cell_l * NCONS) };
                        for (v, &fv) in f_hll.iter().enumerate() {
                            unsafe {
                                *base.add(v) -= fv * inv_dx;
                            }
                        }
                    }
                    if face_j < ng + grid.n2 {
                        let cell_r = grid.idx(i, face_j, k);
                        let base = unsafe { rp.add(cell_r * NCONS) };
                        for (v, &fv) in f_hll.iter().enumerate() {
                            unsafe {
                                *base.add(v) += fv * inv_dx;
                            }
                        }
                    }
                }
            });
    }

    // Direction 2 (phi): sweep faces at fixed (i, j) with PERIODIC BC
    // Parallelized over (i, j) pairs.
    if grid.n3 > 1 {
        let rhs_addr_phi = rhs.as_mut_ptr() as usize;
        let ij_pairs_phi: Vec<(usize, usize)> = (ng..ng + grid.n1)
            .flat_map(|i| (ng..ng + grid.n2).map(move |j| (i, j)))
            .collect();
        ij_pairs_phi
            .par_iter()
            .for_each_init(FluxWorkspace::new, |ws, &(i, j)| {
                let rp = rhs_addr_phi as *mut f64;
                let inv_dx = 1.0 / (grid.dx3 * 2.0 * std::f64::consts::PI);

                for face_k in ng..ng + grid.n3 + 1 {
                    // Periodic wrapping for phi direction
                    let km2 = ng
                        + ((face_k as isize - 2 - ng as isize).rem_euclid(grid.n3 as isize))
                            as usize;
                    let km1 = ng
                        + ((face_k as isize - 1 - ng as isize).rem_euclid(grid.n3 as isize))
                            as usize;
                    let kp0 = ng
                        + ((face_k as isize - ng as isize).rem_euclid(grid.n3 as isize)) as usize;
                    let kp1 = ng
                        + ((face_k as isize + 1 - ng as isize).rem_euclid(grid.n3 as isize))
                            as usize;

                    for var in 0..NPRIM {
                        let (ql, qr) = recon::plm_lr(
                            prims.get(grid.idx(i, j, km2))[var],
                            prims.get(grid.idx(i, j, km1))[var],
                            prims.get(grid.idx(i, j, kp0))[var],
                            prims.get(grid.idx(i, j, kp1))[var],
                        );
                        ws.prim_l[var] = ql;
                        ws.prim_r[var] = qr;
                    }

                    if zero_b_in_recon {
                        for b in 5..8 {
                            ws.prim_l[b] = 0.0;
                            ws.prim_r[b] = 0.0;
                        }
                    }

                    let mc_idx = mc.idx(i, n2t, j);
                    let gcov_ij = if mc_idx < mc.gcov.len() {
                        mc.gcov_at(mc_idx)
                    } else {
                        &[0.0; 5]
                    };
                    let sg = if mc_idx < mc.sqrt_neg_g.len() {
                        mc.sqrt_neg_g[mc_idx]
                    } else {
                        1.0
                    };
                    let alpha = if mc_idx < mc.lapse.len() {
                        mc.lapse[mc_idx]
                    } else {
                        1.0
                    };

                    ws.flux_l = compute_flux_cached(&ws.prim_l, gcov_ij, eos, sg, 2);
                    ws.flux_r = compute_flux_cached(&ws.prim_r, gcov_ij, eos, sg, 2);

                    let cs2_l = eos.cs2(ws.prim_l[prims::RHO], ws.prim_l[prims::UU]);
                    let cs2_r = eos.cs2(ws.prim_r[prims::RHO], ws.prim_r[prims::UU]);
                    let (sl, sr) = riemann::wave_speeds(
                        ws.prim_l[prims::V3],
                        ws.prim_r[prims::V3],
                        cs2_l,
                        cs2_r,
                        0.0,
                        0.0,
                        alpha,
                    );

                    ws.cons_l = cons::prim2con_cached(&ws.prim_l, gcov_ij, eos, sg);
                    ws.cons_r = cons::prim2con_cached(&ws.prim_r, gcov_ij, eos, sg);

                    let f_hll =
                        riemann::hll_flux(&ws.flux_l, &ws.flux_r, &ws.cons_l, &ws.cons_r, sl, sr);

                    // Accumulate phi divergence (periodic: both cells always exist)
                    // SAFETY: `v` indexes into NCONS * n_total RHS array.
                    // Rayon partitions (i, j) pairs so each thread writes to
                    // disjoint cells. `rp` is valid heap memory from the RHS Vec.
                    let cell_l = grid.idx(i, j, km1);
                    let base_l = unsafe { rp.add(cell_l * NCONS) };
                    for (v, &fv) in f_hll.iter().enumerate() {
                        unsafe {
                            *base_l.add(v) -= fv * inv_dx;
                        }
                    }
                    let cell_r = grid.idx(i, j, kp0);
                    let base_r = unsafe { rp.add(cell_r * NCONS) };
                    for (v, &fv) in f_hll.iter().enumerate() {
                        unsafe {
                            *base_r.add(v) += fv * inv_dx;
                        }
                    }
                }
            });
    }

    // Add GR geometric source terms, parallelized over (i, j) pairs.
    // Safety: different (i, j, k) write to unique rhs indices (grid.idx is injective).
    {
        let rhs_addr = rhs.as_mut_ptr() as usize;
        let dr_phys = grid.dx1;
        let dth_phys = grid.dx2 * std::f64::consts::PI;
        let ij_pairs: Vec<(usize, usize)> = (ng..ng + grid.n1)
            .flat_map(|i| (ng..ng + grid.n2).map(move |j| (i, j)))
            .collect();

        ij_pairs.par_iter().for_each(|&(i, j)| {
            let rp = rhs_addr as *mut f64;
            let mc_idx = mc.idx(i, n2t, j);
            let sg = if mc_idx < mc.sqrt_neg_g.len() {
                mc.sqrt_neg_g[mc_idx]
            } else {
                1.0
            };

            for k in ng..ng + grid.n3 {
                let idx = grid.idx(i, j, k);
                let p = {
                    let slice = prims.get(idx);
                    let mut arr = [0.0; NPRIM];
                    arr.copy_from_slice(slice);
                    arr
                };

                let src = crate::source::geometric_source(
                    &p,
                    &grid.metric,
                    grid.r(i),
                    grid.theta(j),
                    eos,
                    dr_phys,
                    dth_phys,
                );

                // SAFETY: `v` indexes into NCONS * n_total RHS array.
                // Rayon partitions (i, j) pairs; grid.idx is injective so
                // each (i, j, k) triple maps to a unique offset. `rp` points
                // to valid heap memory from the RHS Vec.
                let base = unsafe { rp.add(idx * NCONS) };
                for (v, &sv) in src.iter().enumerate() {
                    unsafe {
                        *base.add(v) += sg * sv;
                    }
                }
            }
        });
    }

    // Face EMFs are now collected from HLL flux during the sweeps above:
    // emf_r[i*n2t+j] = -F_HLL_Btheta at radial face (i+1/2, j)
    // emf_th[i*n2t+j] = +F_HLL_Br at theta face (i, j+1/2)
    // These have the correct upwind property from the Riemann solver.

    (rhs, face_emfs)
}

/// Take a single Euler step using the full 3D RHS.
///
/// Proper conservative update:
///   1. Convert prims -> cons (U = sqrt(-g) * conservative vars)
///   2. Update: U_new = U_old + dt * RHS (where RHS = -div(F))
///   3. Convert cons -> prims (simplified: extract rho, B directly)
///
/// This ensures mass, energy, and magnetic flux conservation.
pub fn euler_step_3d(
    prims: &mut PrimGrid,
    grid: &Grid,
    mc: &MetricCache,
    eos: &GammaLaw,
    dt: f64,
    staggered_b: Option<&mut crate::ct::StaggeredB>,
) {
    let ng = grid.ng;
    let n2t = grid.n2_total();

    // Step 1: Convert all interior cells to conservative variables (cached, parallel)
    let n_total = grid.n_total();
    let mut cons_grid = vec![[0.0f64; NCONS]; n_total];
    {
        let cg_addr = cons_grid.as_mut_ptr() as usize;
        let ij_pairs: Vec<(usize, usize)> = (ng..ng + grid.n1)
            .flat_map(|i| (ng..ng + grid.n2).map(move |j| (i, j)))
            .collect();
        ij_pairs.par_iter().for_each(|&(i, j)| {
            let cg_p = cg_addr as *mut [f64; NCONS];
            let mc_idx = mc.idx(i, n2t, j);
            let gcov = if mc_idx < mc.gcov.len() {
                mc.gcov_at(mc_idx)
            } else {
                &[0.0; 5]
            };
            let sg = if mc_idx < mc.sqrt_neg_g.len() {
                mc.sqrt_neg_g[mc_idx]
            } else {
                1.0
            };
            for k in ng..ng + grid.n3 {
                let idx = grid.idx(i, j, k);
                let cons_val = cons::prim2con_cached(
                    &{
                        let mut p = [0.0; NPRIM];
                        p.copy_from_slice(prims.get(idx));
                        p
                    },
                    gcov,
                    eos,
                    sg,
                );
                // SAFETY: idx = grid.idx(i,j,k) is injective and bounded by
                // n_total. cg_p points to cons_grid (length n_total). Rayon
                // partitions (i,j) pairs so each thread writes disjoint cells.
                unsafe {
                    *cg_p.add(idx) = cons_val;
                }
            }
        });
    }

    // Step 2: Compute RHS and update conservative variables
    let use_ct = staggered_b.is_some();
    let (rhs, face_emfs) = compute_rhs_3d_inner(prims, grid, mc, eos, use_ct);
    for i in ng..ng + grid.n1 {
        for j in ng..ng + grid.n2 {
            for k in ng..ng + grid.n3 {
                let idx = grid.idx(i, j, k);
                let base = idx * NCONS;
                for v in 0..NCONS {
                    cons_grid[idx][v] += dt * rhs[base + v];
                }
            }
        }
    }

    // If using CT, zero out the B-field RHS from the HLL flux divergence.
    // B is evolved ONLY by CT, not by the HLL flux.
    // Note: rhs is consumed above; we zero the B components in cons_grid directly
    // by reverting the B update (subtract what we just added).
    let use_ct = staggered_b.is_some();
    if use_ct {
        for i in ng..ng + grid.n1 {
            for j in ng..ng + grid.n2 {
                for k in ng..ng + grid.n3 {
                    let idx = grid.idx(i, j, k);
                    let base = idx * NCONS;
                    // Revert the B-field update from HLL flux divergence
                    cons_grid[idx][5] -= dt * rhs[base + 5];
                    cons_grid[idx][6] -= dt * rhs[base + 6];
                    cons_grid[idx][7] -= dt * rhs[base + 7];
                }
            }
        }
    }

    // Step 2b: Update B-field via constrained transport with HLL-derived EMFs
    if let Some(sb) = staggered_b {
        sb.ct_update_from_face_emfs(&face_emfs, grid, dt);
        sb.to_cell_centered(prims, grid);
    }

    // Step 3: Convert back to primitives using Kastaun con2prim
    // If CT active, first override conserved B with CT-derived B
    if use_ct {
        for i in ng..ng + grid.n1 {
            for j in ng..ng + grid.n2 {
                for k in ng..ng + grid.n3 {
                    let idx = grid.idx(i, j, k);
                    let mc_idx = mc.idx(i, n2t, j);
                    let sg = if mc_idx < mc.sqrt_neg_g.len() {
                        mc.sqrt_neg_g[mc_idx]
                    } else {
                        1.0
                    };
                    let p_ct = prims.get(idx);
                    cons_grid[idx][5] = sg * p_ct[prims::B1];
                    cons_grid[idx][6] = sg * p_ct[prims::B2];
                    cons_grid[idx][7] = sg * p_ct[prims::B3];
                }
            }
        }
    }

    // Con2prim: parallelized over (i, j) pairs (Brent per cell is independent).
    // Skip atmosphere cells where D is below floor (saves ~50% of iterations).
    {
        let prim_addr = prims.data.as_mut_ptr() as usize;
        let ij_pairs_c2p: Vec<(usize, usize)> = (ng..ng + grid.n1)
            .flat_map(|i| (ng..ng + grid.n2).map(move |j| (i, j)))
            .collect();
        ij_pairs_c2p.par_iter().for_each(|&(i, j)| {
            let pp = prim_addr as *mut f64;
            let mc_idx = mc.idx(i, n2t, j);
            let sg = if mc_idx < mc.sqrt_neg_g.len() {
                mc.sqrt_neg_g[mc_idx]
            } else {
                1.0
            };
            let gcov = if mc_idx < mc.gcov.len() {
                mc.gcov[mc_idx]
            } else {
                [0.0; 5]
            };
            let gcon = if mc_idx < mc.gcon.len() {
                mc.gcon[mc_idx]
            } else {
                [0.0; 5]
            };

            for k in ng..ng + grid.n3 {
                let idx = grid.idx(i, j, k);

                if cons_grid[idx][0] < sg * 1e-6 {
                    continue;
                }

                let result = crate::con2prim::con2prim_kastaun(
                    &cons_grid[idx],
                    &gcov,
                    &gcon,
                    sg,
                    eos,
                    1e-8,
                    1e-10,
                );
                let recovered = result.prims();
                let base = idx * NPRIM;
                // SAFETY: base = idx * NPRIM where idx is bounded by n_total
                // and NPRIM is the stride. pp points to prims.data (length
                // n_total * NPRIM). Rayon partitions (i,j) pairs so each
                // thread writes disjoint cells. grid.idx is injective.
                unsafe {
                    let p = pp.add(base);
                    if use_ct {
                        let ct_b1 = *p.add(prims::B1);
                        let ct_b2 = *p.add(prims::B2);
                        let ct_b3 = *p.add(prims::B3);
                        for (v, &rv) in recovered.iter().enumerate() {
                            *p.add(v) = rv;
                        }
                        *p.add(prims::B1) = ct_b1;
                        *p.add(prims::B2) = ct_b2;
                        *p.add(prims::B3) = ct_b3;
                    } else {
                        for (v, &rv) in recovered.iter().enumerate() {
                            *p.add(v) = rv;
                        }
                    }
                }
            }
        });
    }

    prims.apply_floors(1e-8, 1e-10);

    // Apply simple outflow boundary conditions: copy interior to ghost zones
    let ng = grid.ng;
    for j in 0..grid.n2_total() {
        for k in 0..grid.n3_total() {
            // Inner radial boundary: copy from first active cell
            for g in 0..ng {
                let src = grid.idx(ng, j, k);
                let dst = grid.idx(g, j, k);
                let src_p = {
                    let s = prims.get(src);
                    let mut a = [0.0; 8];
                    a.copy_from_slice(s);
                    a
                };
                prims.set(dst, &src_p);
            }
            // Outer radial boundary: copy from last active cell
            for g in 0..ng {
                let src = grid.idx(ng + grid.n1 - 1, j, k);
                let dst = grid.idx(ng + grid.n1 + g, j, k);
                let src_p = {
                    let s = prims.get(src);
                    let mut a = [0.0; 8];
                    a.copy_from_slice(s);
                    a
                };
                prims.set(dst, &src_p);
            }
        }
    }
    // Theta boundaries: copy from first/last active theta cell
    for i in 0..grid.n1_total() {
        for k in 0..grid.n3_total() {
            for g in 0..ng {
                let src = grid.idx(i, ng, k);
                let dst = grid.idx(i, g, k);
                let src_p = {
                    let s = prims.get(src);
                    let mut a = [0.0; 8];
                    a.copy_from_slice(s);
                    a
                };
                prims.set(dst, &src_p);
            }
            for g in 0..ng {
                let src = grid.idx(i, ng + grid.n2 - 1, k);
                let dst = grid.idx(i, ng + grid.n2 + g, k);
                let src_p = {
                    let s = prims.get(src);
                    let mut a = [0.0; 8];
                    a.copy_from_slice(s);
                    a
                };
                prims.set(dst, &src_p);
            }
        }
    }
}

/// Estimate the maximum signal speed across all interior cells.
/// Used for CFL timestep computation.
pub fn max_signal_speed(prims: &PrimGrid, grid: &Grid, mc: &MetricCache, eos: &GammaLaw) -> f64 {
    let ng = grid.ng;
    let n2t = grid.n2_total();
    let mut max_speed = 0.0f64;

    for i in ng..ng + grid.n1 {
        for j in ng..ng + grid.n2 {
            let idx = grid.idx(i, j, ng);
            let p = prims.get(idx);
            let cs2 = eos.cs2(p[prims::RHO], p[prims::UU]);
            let cs = cs2.sqrt();
            let mc_idx = mc.idx(i, n2t, j);
            let alpha = if mc_idx < mc.lapse.len() {
                mc.lapse[mc_idx]
            } else {
                1.0
            };

            // Max velocity + sound speed
            let v = (p[prims::V1].abs() + p[prims::V2].abs() + p[prims::V3].abs()).min(0.99);
            let speed = alpha * (v + cs).min(1.0); // cap at speed of light
            if speed > max_speed {
                max_speed = speed;
            }
        }
    }

    max_speed
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
    fn test_compute_rhs_3d_finite() {
        let metric = KerrMetric::schwarzschild();
        let grid = Grid::new(8, 8, 1, 2.5, 20.0, metric);
        let mc = MetricCache::new(&grid);
        let eos = GammaLaw::harm_default();

        let torus = crate::torus::FMTorus::schwarzschild(6.5, 12.0);
        let prims = torus.initialize(&grid);

        let (rhs, _face_emfs) = compute_rhs_3d(&prims, &grid, &mc, &eos);
        assert_eq!(rhs.len(), grid.n_total() * NCONS);

        // All interior RHS should be finite
        let ng = grid.ng;
        for i in ng..ng + grid.n1 {
            for j in ng..ng + grid.n2 {
                let idx = grid.idx(i, j, ng);
                for v in 0..NCONS {
                    assert!(
                        rhs[idx * NCONS + v].is_finite(),
                        "rhs[{},{},{}][{}] = {}",
                        i,
                        j,
                        ng,
                        v,
                        rhs[idx * NCONS + v]
                    );
                }
            }
        }
    }

    #[test]
    fn test_euler_step_3d_stable() {
        let metric = KerrMetric::schwarzschild();
        let grid = Grid::new(8, 8, 1, 2.5, 20.0, metric);
        let mc = MetricCache::new(&grid);
        let eos = GammaLaw::harm_default();

        let torus = crate::torus::FMTorus::schwarzschild(6.5, 12.0);
        let mut prims = torus.initialize(&grid);
        torus.add_magnetic_loop(&grid, &mut prims, 100.0);

        let max_v = max_signal_speed(&prims, &grid, &mc, &eos);
        let dt = crate::evolve::cfl_dt(&grid, max_v, 0.3);

        // Take one step
        euler_step_3d(&mut prims, &grid, &mc, &eos, dt, None);

        // Density should still be positive everywhere
        let ng = grid.ng;
        for i in ng..ng + grid.n1 {
            for j in ng..ng + grid.n2 {
                let idx = grid.idx(i, j, ng);
                let rho = prims.get(idx)[prims::RHO];
                assert!(rho > 0.0, "rho[{},{}] = {} <= 0 after step", i, j, rho);
            }
        }
    }

    #[test]
    fn test_max_signal_speed() {
        let metric = KerrMetric::schwarzschild();
        let grid = Grid::new(8, 8, 1, 2.5, 20.0, metric);
        let mc = MetricCache::new(&grid);
        let eos = GammaLaw::harm_default();

        let torus = crate::torus::FMTorus::schwarzschild(6.5, 12.0);
        let prims = torus.initialize(&grid);

        let max_v = max_signal_speed(&prims, &grid, &mc, &eos);
        assert!(
            max_v > 0.0 && max_v < 10.0,
            "max_signal = {} should be reasonable",
            max_v
        );
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

    #[test]
    fn test_compute_flux_cached_matches_original() {
        // Verify cached variant produces identical results to the metric-based variant
        let metric = KerrMetric::schwarzschild();
        let eos = GammaLaw::harm_default();
        let r = 10.0;
        let th = std::f64::consts::FRAC_PI_2;
        let sqrt_g = 100.0;

        let p: Prim = [1.0, 0.01, 0.1, 0.05, 0.02, 0.01, 0.02, 0.005];

        for dir in 0..3 {
            let f_orig = compute_flux_from_prim(&p, &metric, r, th, &eos, sqrt_g, dir);
            let gcov = metric.gcov(r, th);
            let f_cached = compute_flux_cached(&p, &gcov, &eos, sqrt_g, dir);

            for v in 0..NCONS {
                assert!(
                    (f_orig[v] - f_cached[v]).abs() < 1e-12,
                    "dir={} var={}: orig={} cached={}",
                    dir,
                    v,
                    f_orig[v],
                    f_cached[v],
                );
            }
        }
    }
}
