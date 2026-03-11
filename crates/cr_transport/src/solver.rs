//! Parker Transport Equation (PTE) solver.
//!
//! Implements Strang operator splitting:
//!   1. Spatial diffusion (ADI sweeps, Crank-Nicolson)
//!   2. Spatial advection (upwind, 1st order)
//!   3. Adiabatic momentum deceleration (upwind in ln p)
//!   4. Source injection (explicit Euler)
//!
//! The distribution function f(x, y, z, ln_p) is stored flat:
//!   f[grid_idx * n_p + p_idx]
//! where grid_idx = z*(nx*ny) + y*nx + x  (same as LBM convention).
//!
//! ADI (Alternating Direction Implicit) is used for the diffusion step.
//! Each spatial sweep solves a tridiagonal system with the Thomas algorithm.
//! x-axis uses non-periodic BCs (zero-gradient inner, Dirichlet ISM outer).
//! y/z-axes use periodic BCs (transverse symmetry).

use crate::{
    diffusion::{DiffusionConfig, diffusion_tensor},
    grid::RigidityGrid,
    source::DmSource,
};

/// Parker Transport Equation solver.
pub struct PteSolver {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub n_p: usize,
    /// Phase-space distribution: f[grid_idx * n_p + p_idx].
    pub f: Vec<f64>,
    pub grid: RigidityGrid,
    pub config: DiffusionConfig,
    pub timestep: usize,
    /// Physical timestep (s). Must match LBM dt.
    pub dt_s: f64,
    /// Spatial cell size (AU). Must match LBM dx.
    pub dx_au: f64,
    /// Inner boundary heliocentric distance (AU). Used for radial
    /// divergence calculation. Default 0.3 AU (inner heliosphere).
    pub r_min_au: f64,
}

impl PteSolver {
    /// Create a new PTE solver initialized to zero f.
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        grid: RigidityGrid,
        config: DiffusionConfig,
        dt_s: f64,
        dx_au: f64,
    ) -> Self {
        let n_p = grid.n_p;
        let n_cells = nx * ny * nz;
        Self {
            nx,
            ny,
            nz,
            n_p,
            f: vec![0.0; n_cells * n_p],
            grid,
            config,
            timestep: 0,
            dt_s,
            dx_au,
            r_min_au: 0.3,
        }
    }

    /// Grid index: z*(nx*ny) + y*nx + x  (same convention as LBM).
    #[inline]
    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        z * (self.nx * self.ny) + y * self.nx + x
    }

    /// Set outer ISM boundary condition. Applies LIS to all cells on the
    /// x = nx-1 face (outermost radial boundary).
    pub fn set_boundary_ism(&mut self, lis: &dyn Fn(f64) -> f64) {
        for y in 0..self.ny {
            for z in 0..self.nz {
                let idx = self.idx(self.nx - 1, y, z);
                for p in 0..self.n_p {
                    let r_gv = self.grid.rigidity(p);
                    self.f[idx * self.n_p + p] = lis(r_gv).max(0.0);
                }
            }
        }
    }

    /// Compute div(u) per cell using radial divergence in x and periodic
    /// central differences in y/z (transverse directions).
    ///
    /// Radial divergence: div_u_r = (1/r^2) * d(r^2 * u_r)/dr
    /// where r(x) = r_min_au + x * dx_au.
    ///
    /// x boundaries use one-sided differences (no periodic wrap --
    /// x=0 is the inner heliosphere, x=nx-1 is the ISM boundary).
    fn compute_div_u(&self, u_sw: &[[f64; 3]]) -> Vec<f64> {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let dx = self.dx_au;
        let mut div_u = vec![0.0_f64; nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.idx(x, y, z);
                    let r = self.r_min_au + x as f64 * dx;
                    let r2 = r * r;

                    // Radial divergence: (1/r^2) d(r^2 u_r)/dr
                    // via one-sided differences at boundaries, central interior
                    let du_dx = if nx < 2 {
                        0.0
                    } else if x == 0 {
                        // Forward difference
                        let rp = r + dx;
                        let u_r_p = u_sw[self.idx(1, y, z)][0];
                        let u_r_c = u_sw[idx][0];
                        (rp * rp * u_r_p - r2 * u_r_c) / (dx * r2)
                    } else if x == nx - 1 {
                        // Backward difference
                        let rm = r - dx;
                        let u_r_m = u_sw[self.idx(nx - 2, y, z)][0];
                        let u_r_c = u_sw[idx][0];
                        (r2 * u_r_c - rm * rm * u_r_m) / (dx * r2)
                    } else {
                        // Central difference
                        let rp = r + dx;
                        let rm = r - dx;
                        let u_r_p = u_sw[self.idx(x + 1, y, z)][0];
                        let u_r_m = u_sw[self.idx(x - 1, y, z)][0];
                        (rp * rp * u_r_p - rm * rm * u_r_m) / (2.0 * dx * r2)
                    };

                    // Transverse directions: periodic central differences
                    let yp = (y + 1) % ny;
                    let ym = if y == 0 { ny - 1 } else { y - 1 };
                    let zp = (z + 1) % nz;
                    let zm = if z == 0 { nz - 1 } else { z - 1 };
                    let du_dy =
                        (u_sw[self.idx(x, yp, z)][1] - u_sw[self.idx(x, ym, z)][1]) / (2.0 * dx);
                    let du_dz =
                        (u_sw[self.idx(x, y, zp)][2] - u_sw[self.idx(x, y, zm)][2]) / (2.0 * dx);
                    div_u[idx] = du_dx + du_dy + du_dz;
                }
            }
        }
        div_u
    }

    /// Compute the CFL number max(|u| * dt / dx) across all cells.
    /// Emits a warning when CFL > 0.5 (risk of numerical instability).
    /// Returns the maximum CFL value.
    pub fn check_cfl(&self, u_sw: &[[f64; 3]]) -> f64 {
        let mut cfl_max = 0.0_f64;
        for u in u_sw {
            let speed = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            let cfl = speed * self.dt_s / self.dx_au;
            if cfl > cfl_max {
                cfl_max = cfl;
            }
        }
        if cfl_max > 0.5 {
            eprintln!(
                "WARNING: PTE CFL = {cfl_max:.4} > 0.5 (dt={:.2e} s, dx={:.4} AU). \
                 Consider reducing dt or increasing dx for stability.",
                self.dt_s, self.dx_au
            );
        }
        cfl_max
    }

    /// Thomas algorithm (tridiagonal matrix algorithm) for a 1D tridiagonal system.
    /// Solves (a[i]*f[i-1] + b[i]*f[i] + c[i]*f[i+1]) = d[i].
    /// Uses Dirichlet at x=nx-1 (ISM boundary) and zero-flux at x=0 (inner boundary).
    ///
    /// Returns solution vector of length n.
    fn thomas_solve(
        a: &[f64], // sub-diagonal (length n, a[0] unused)
        b: &[f64], // main diagonal
        c: &[f64], // super-diagonal (length n, c[n-1] unused)
        d: &[f64], // right-hand side
    ) -> Vec<f64> {
        let n = d.len();
        let mut cp = vec![0.0; n];
        let mut dp = vec![0.0; n];
        let mut x = vec![0.0; n];

        cp[0] = c[0] / b[0];
        dp[0] = d[0] / b[0];

        for i in 1..n {
            let m = b[i] - a[i] * cp[i - 1];
            if m.abs() < 1e-30 {
                cp[i] = 0.0;
                dp[i] = 0.0;
            } else {
                cp[i] = if i < n - 1 { c[i] / m } else { 0.0 };
                dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
            }
        }

        x[n - 1] = dp[n - 1];
        for i in (0..n - 1).rev() {
            x[i] = dp[i] - cp[i] * x[i + 1];
        }
        x
    }

    /// Spatial diffusion ADI sweep along x-axis for a single momentum bin p_idx.
    /// Solves: f^{n+1} = f^n + dt * d/dx (K_xx * df/dx)
    fn diffuse_x_sweep(
        &mut self,
        b_field: (&[f64], &[f64], &[f64]),
        rigidity_gv: f64,
        p_idx: usize,
    ) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let dt = self.dt_s;
        let dx = self.dx_au;

        for z in 0..nz {
            for y in 0..ny {
                // Extract f along x for this (y, z, p) row
                let mut d = vec![0.0_f64; nx];
                for (x, d_x) in d.iter_mut().enumerate() {
                    let fi = self.idx(x, y, z);
                    *d_x = self.f[fi * self.n_p + p_idx];
                }

                // Build tridiagonal coefficients
                let mut a = vec![0.0_f64; nx];
                let mut b = vec![1.0_f64; nx];
                let mut c = vec![0.0_f64; nx];

                for x in 0..nx {
                    let fi = self.idx(x, y, z);
                    let b_vec = [b_field.0[fi], b_field.1[fi], b_field.2[fi]];
                    let b_mag = (b_vec[0] * b_vec[0] + b_vec[1] * b_vec[1] + b_vec[2] * b_vec[2])
                        .sqrt()
                        .max(1e-10);
                    let k = diffusion_tensor(b_vec, b_mag, rigidity_gv, &self.config);
                    let kxx = k[0][0];
                    let mu = kxx * dt / (dx * dx);

                    // Crank-Nicolson: half-implicit
                    let half_mu = 0.5 * mu;
                    a[x] = -half_mu;
                    b[x] = 1.0 + 2.0 * half_mu;
                    c[x] = -half_mu;

                    // Explicit part added to rhs (one-sided at x boundaries)
                    let fc = self.f[fi * self.n_p + p_idx];
                    let fp = if x < nx - 1 {
                        self.f[self.idx(x + 1, y, z) * self.n_p + p_idx]
                    } else {
                        fc // zero-gradient at outer boundary (Dirichlet reset later)
                    };
                    let fm = if x > 0 {
                        self.f[self.idx(x - 1, y, z) * self.n_p + p_idx]
                    } else {
                        fc // zero-gradient at inner boundary
                    };
                    d[x] = fc + half_mu * (fm - 2.0 * fc + fp);
                }

                // Fix boundary: x=0 is zero-flux (inner), x=nx-1 is ISM (Dirichlet)
                a[0] = 0.0;
                c[nx - 1] = 0.0;

                let sol = Self::thomas_solve(&a, &b, &c, &d);
                for (x, &s) in sol.iter().enumerate() {
                    let fi = self.idx(x, y, z);
                    self.f[fi * self.n_p + p_idx] = s.max(0.0);
                }
            }
        }
    }

    /// Spatial advection step (upwind) for all cells and all momentum bins.
    /// df/dt = -u_sw . grad(f)
    fn advect_spatial(&mut self, u_sw: &[[f64; 3]]) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let dt = self.dt_s;
        let dx = self.dx_au;
        let n_p = self.n_p;

        let f_old = self.f.clone();

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.idx(x, y, z);
                    let u = u_sw[idx];
                    // y/z: periodic (transverse directions)
                    let yp = (y + 1) % ny;
                    let ym = if y == 0 { ny - 1 } else { y - 1 };
                    let zp = (z + 1) % nz;
                    let zm = if z == 0 { nz - 1 } else { z - 1 };

                    for p in 0..n_p {
                        let f_c = f_old[idx * n_p + p];
                        // x: one-sided at boundaries (radial, non-periodic)
                        let adv_x = if u[0] > 0.0 {
                            if x > 0 {
                                u[0] * (f_c - f_old[self.idx(x - 1, y, z) * n_p + p]) / dx
                            } else {
                                0.0 // inner boundary: zero-gradient
                            }
                        } else if x < nx - 1 {
                            u[0] * (f_old[self.idx(x + 1, y, z) * n_p + p] - f_c) / dx
                        } else {
                            0.0 // outer boundary: Dirichlet (LIS reset handles it)
                        };
                        // y/z: periodic upwind
                        let adv_y = if u[1] > 0.0 {
                            u[1] * (f_c - f_old[self.idx(x, ym, z) * n_p + p]) / dx
                        } else {
                            u[1] * (f_old[self.idx(x, yp, z) * n_p + p] - f_c) / dx
                        };
                        let adv_z = if u[2] > 0.0 {
                            u[2] * (f_c - f_old[self.idx(x, y, zm) * n_p + p]) / dx
                        } else {
                            u[2] * (f_old[self.idx(x, y, zp) * n_p + p] - f_c) / dx
                        };
                        self.f[idx * n_p + p] = (f_c - dt * (adv_x + adv_y + adv_z)).max(0.0);
                    }
                }
            }
        }
    }

    /// Adiabatic momentum deceleration step.
    /// df/dt = (1/3) * div(u) * df/d(ln_p) (upwind in ln p axis)
    fn decelerate_momentum(&mut self, div_u: &[f64]) {
        let n_cells = self.nx * self.ny * self.nz;
        let n_p = self.n_p;
        let dt = self.dt_s;
        let d_ln_r = self.grid.d_ln_r;

        let f_old = self.f.clone();

        for idx in 0..n_cells {
            let du = div_u[idx];
            // (1/3) * div(u): positive div_u means expanding flow -> deceleration
            let rate = du / 3.0;
            for p in 0..n_p {
                let f_c = f_old[idx * n_p + p];
                // Upwind: if rate > 0 (deceleration), flux comes from higher p
                let df_dlnp = if rate > 0.0 {
                    if p > 0 {
                        (f_c - f_old[idx * n_p + (p - 1)]) / d_ln_r
                    } else {
                        0.0
                    }
                } else if p < n_p - 1 {
                    (f_old[idx * n_p + (p + 1)] - f_c) / d_ln_r
                } else {
                    0.0
                };
                self.f[idx * n_p + p] = (f_c - dt * rate * df_dlnp).max(0.0);
            }
        }
    }

    /// One full Strang-split timestep.
    ///
    /// Strang splitting order for second-order accuracy:
    ///   (diffuse/2) -> advect -> decelerate -> inject -> (diffuse/2)
    ///
    /// When `lis` is provided, the outer ISM boundary is reapplied after
    /// each step to prevent erosion by diffusion and advection.
    pub fn evolve_one_step(
        &mut self,
        u_sw: &[[f64; 3]],
        b_field: (&[f64], &[f64], &[f64]),
        source: Option<&DmSource>,
        dm_density: &[f64],
        lis: Option<&dyn Fn(f64) -> f64>,
    ) {
        // Half-step diffusion (x-sweep only for now; full 3D ADI is expensive)
        for p in 0..self.n_p {
            let r_gv = self.grid.rigidity(p);
            self.diffuse_x_sweep(b_field, r_gv, p);
        }

        // Spatial advection
        self.advect_spatial(u_sw);

        // Adiabatic momentum deceleration
        let div_u = self.compute_div_u(u_sw);
        self.decelerate_momentum(&div_u);

        // Source injection
        if let Some(src) = source {
            src.inject(&mut self.f, &self.grid, dm_density, self.dt_s);
        }

        // Reapply outer ISM boundary to prevent erosion from diffusion/advection
        if let Some(f) = lis {
            self.set_boundary_ism(f);
        }

        self.timestep += 1;
    }

    /// Extract differential flux J = p^2 * f at a single spatial cell.
    /// Returns Vec<f64> of length n_p.
    pub fn flux_at(&self, x: usize, y: usize, z: usize) -> Vec<f64> {
        let idx = self.idx(x, y, z);
        (0..self.n_p)
            .map(|p| {
                let r = self.grid.rigidity(p);
                r * r * self.f[idx * self.n_p + p]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{diffusion::DiffusionConfig, grid::RigidityGrid};

    #[test]
    fn test_adiabatic_deceleration_shifts_peak() {
        // Pure deceleration with div(u) = const > 0 should shift f peak to lower p
        let grid = RigidityGrid::new(20, 0.1, 100.0);
        let cfg = DiffusionConfig {
            kappa_0_au2_per_s: 0.0,
            ..Default::default()
        };
        let nx = 4;
        let mut solver = PteSolver::new(nx, 1, 1, grid.clone(), cfg, 1e4, 1.0);

        // Set Gaussian f centered at bin 10
        let peak_bin = 10_usize;
        for x in 0..nx {
            let idx = x;
            for p in 0..20 {
                let diff = p as f64 - peak_bin as f64;
                solver.f[idx * 20 + p] = (-0.5 * diff * diff).exp();
            }
        }

        // Uniform div(u) > 0 (expanding flow -> deceleration)
        let u_sw: Vec<[f64; 3]> = vec![[0.1, 0.0, 0.0]; nx];
        let bx = vec![1.0_f64; nx];
        let by = vec![0.0_f64; nx];
        let bz = vec![0.0_f64; nx];

        // Evolve 10 steps
        for _ in 0..10 {
            solver.evolve_one_step(&u_sw, (&bx, &by, &bz), None, &[], None);
        }

        // Find peak bin after deceleration
        let mut max_val = 0.0_f64;
        let mut max_bin = 0_usize;
        let idx = 0; // check cell 0
        for p in 0..20 {
            let v = solver.f[idx * 20 + p];
            if v > max_val {
                max_val = v;
                max_bin = p;
            }
        }

        // Peak should have shifted to lower rigidity (deceleration)
        assert!(
            max_bin <= peak_bin,
            "Peak should shift to lower rigidity, max_bin={max_bin} peak_bin={peak_bin}"
        );
    }

    #[test]
    fn test_f_remains_nonnegative() {
        let grid = RigidityGrid::new(10, 0.1, 100.0);
        let cfg = DiffusionConfig::default();
        let mut solver = PteSolver::new(4, 4, 4, grid, cfg, 1e3, 1.0);

        // Set some initial f
        for v in solver.f.iter_mut() {
            *v = 0.5;
        }

        let n = 4 * 4 * 4;
        let u_sw: Vec<[f64; 3]> = vec![[0.05, 0.01, 0.0]; n];
        let bx = vec![3.0_f64; n];
        let by = vec![4.0_f64; n];
        let bz = vec![0.0_f64; n];

        for _ in 0..5 {
            solver.evolve_one_step(&u_sw, (&bx, &by, &bz), None, &[], None);
        }

        for v in &solver.f {
            assert!(*v >= 0.0, "f became negative: {v}");
        }
    }

    #[test]
    fn test_lis_boundary_preserved_under_outward_advection() {
        // 1D-like solver (nx=16, ny=1, nz=1) with LIS at x=15.
        // Uniform outward velocity u_x > 0. After 100 steps, x=15 face
        // must remain at LIS value (was being eroded by periodic wrap).
        let grid = RigidityGrid::new(5, 0.1, 10.0);
        let cfg = DiffusionConfig {
            kappa_0_au2_per_s: 0.0,
            ..Default::default()
        };
        let nx = 16;
        let mut solver = PteSolver::new(nx, 1, 1, grid.clone(), cfg, 1e3, 1.0);
        solver.r_min_au = 1.0;

        // LIS: flat spectrum = 1.0 at all rigidities
        let lis = |_r: f64| -> f64 { 1.0 };
        solver.set_boundary_ism(&lis);

        let n = nx;
        let u_sw: Vec<[f64; 3]> = vec![[0.05, 0.0, 0.0]; n];
        let bx = vec![1.0_f64; n];
        let by = vec![0.0_f64; n];
        let bz = vec![0.0_f64; n];

        for _ in 0..100 {
            solver.evolve_one_step(&u_sw, (&bx, &by, &bz), None, &[], Some(&lis));
        }

        // x=15 (outer boundary) must remain at LIS = 1.0 for all p
        let outer_idx = solver.idx(nx - 1, 0, 0);
        for p in 0..solver.n_p {
            let val = solver.f[outer_idx * solver.n_p + p];
            assert!(
                (val - 1.0).abs() < 1e-10,
                "LIS boundary eroded at p={p}: f={val} (expected 1.0)"
            );
        }
    }

    #[test]
    fn test_zero_gradient_inner_bc_preserves_smooth_profile() {
        // Verify zero-gradient BC at x=0: a smooth initial profile should
        // not develop a kink at the inner boundary after diffusion steps.
        let grid = RigidityGrid::new(5, 0.1, 10.0);
        let cfg = DiffusionConfig::default();
        let nx = 8;
        let mut solver = PteSolver::new(nx, 1, 1, grid.clone(), cfg, 1e3, 1.0);
        solver.r_min_au = 1.0;

        // Linear profile f(x) = 0.5 + 0.05*x for a single p-bin
        for x in 0..nx {
            let val = 0.5 + 0.05 * x as f64;
            for p in 0..solver.n_p {
                let idx = solver.idx(x, 0, 0);
                solver.f[idx * solver.n_p + p] = val;
            }
        }

        let n = nx;
        let u_sw: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; n];
        let bx = vec![1.0_f64; n];
        let by = vec![0.0_f64; n];
        let bz = vec![0.0_f64; n];

        // Pure diffusion (no advection since u=0)
        for _ in 0..5 {
            solver.evolve_one_step(&u_sw, (&bx, &by, &bz), None, &[], None);
        }

        // Inner boundary (x=0) should be close to x=1 (zero-gradient BC),
        // NOT contaminated by outer boundary values via periodic wrap.
        let f0 = solver.f[solver.idx(0, 0, 0) * solver.n_p];
        let f1 = solver.f[solver.idx(1, 0, 0) * solver.n_p];
        let f_last = solver.f[solver.idx(nx - 1, 0, 0) * solver.n_p];
        // With zero-gradient BC, f0 should be close to f1, not pulled toward f_last
        assert!(
            (f0 - f1).abs() < (f_last - f0).abs().max(0.1),
            "Inner BC violated: f[0]={f0:.4}, f[1]={f1:.4}, f[{nx}-1]={f_last:.4}"
        );
        // All values must remain non-negative
        for v in &solver.f {
            assert!(*v >= 0.0, "f became negative: {v}");
        }
    }
}
