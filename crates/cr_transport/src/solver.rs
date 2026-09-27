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
//! y/z-axes use periodic BCs (transverse symmetry). Each implicit sweep uses
//! one diagonal tensor component; mixed and antisymmetric terms are omitted.

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
    /// Solves `(a[i]*f[i-1] + b[i]*f[i] + c[i]*f[i+1]) = d[i]`.
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

    fn cyclic_thomas_solve(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
        let n = d.len();
        if n <= 1 {
            return d.to_vec();
        }
        if n == 2 {
            let upper = c[0] + a[0];
            let lower = a[1] + c[1];
            let determinant = b[0] * b[1] - upper * lower;
            return vec![
                (d[0] * b[1] - upper * d[1]) / determinant,
                (b[0] * d[1] - lower * d[0]) / determinant,
            ];
        }

        // Row 0's lower entry couples to x[n-1] (top-right corner); row n-1's
        // upper entry couples to x[0] (bottom-left corner). Sherman-Morrison
        // writes the corners as u v^T with u = (gamma, 0, .., bottom_left) and
        // v = (1, 0, .., top_right / gamma).
        let top_right = a[0];
        let bottom_left = c[n - 1];
        let gamma = -b[0];
        let mut modified_b = b.to_vec();
        modified_b[0] = b[0] - gamma;
        modified_b[n - 1] = b[n - 1] - bottom_left * top_right / gamma;
        let mut modified_a = a.to_vec();
        let mut modified_c = c.to_vec();
        modified_a[0] = 0.0;
        modified_c[n - 1] = 0.0;

        let solution = Self::thomas_solve(&modified_a, &modified_b, &modified_c, d);
        let mut correction_rhs = vec![0.0; n];
        correction_rhs[0] = gamma;
        correction_rhs[n - 1] = bottom_left;
        let correction = Self::thomas_solve(&modified_a, &modified_b, &modified_c, &correction_rhs);
        let factor = (solution[0] + top_right * solution[n - 1] / gamma)
            / (1.0 + correction[0] + top_right * correction[n - 1] / gamma);
        solution
            .into_iter()
            .zip(correction)
            .map(|(value, correction)| value - factor * correction)
            .collect()
    }

    /// Advance one diagonal tensor component; mixed and antisymmetric terms are omitted.
    fn diffuse_axis_sweep(
        &mut self,
        axis: usize,
        periodic: bool,
        b_field: (&[f64], &[f64], &[f64]),
        rigidity_gv: f64,
        p_idx: usize,
        dt_s: f64,
    ) {
        let dimensions = [self.nx, self.ny, self.nz];
        let axis_length = dimensions[axis];
        let other_axis = (axis + 1) % 3;
        let remaining_axis = (axis + 2) % 3;
        let dx = self.dx_au;

        for fixed_remaining in 0..dimensions[remaining_axis] {
            for fixed_other in 0..dimensions[other_axis] {
                let mut coordinates = [0; 3];
                coordinates[other_axis] = fixed_other;
                coordinates[remaining_axis] = fixed_remaining;
                let mut rhs = vec![0.0_f64; axis_length];
                let mut lower = vec![0.0_f64; axis_length];
                let mut diagonal = vec![1.0_f64; axis_length];
                let mut upper = vec![0.0_f64; axis_length];

                for along_axis in 0..axis_length {
                    coordinates[axis] = along_axis;
                    let cell = self.idx(coordinates[0], coordinates[1], coordinates[2]);
                    let field = [b_field.0[cell], b_field.1[cell], b_field.2[cell]];
                    let field_magnitude =
                        (field[0] * field[0] + field[1] * field[1] + field[2] * field[2])
                            .sqrt()
                            .max(1e-10);
                    let tensor =
                        diffusion_tensor(field, field_magnitude, rigidity_gv, &self.config);
                    let mu = tensor[axis][axis] * dt_s / (dx * dx);
                    let half_mu = 0.5 * mu;
                    let center = self.f[cell * self.n_p + p_idx];
                    let previous = if along_axis > 0 {
                        coordinates[axis] = along_axis - 1;
                        let neighbor = self.idx(coordinates[0], coordinates[1], coordinates[2]);
                        self.f[neighbor * self.n_p + p_idx]
                    } else if periodic {
                        coordinates[axis] = axis_length - 1;
                        let neighbor = self.idx(coordinates[0], coordinates[1], coordinates[2]);
                        self.f[neighbor * self.n_p + p_idx]
                    } else {
                        center
                    };
                    let next = if along_axis + 1 < axis_length {
                        coordinates[axis] = along_axis + 1;
                        let neighbor = self.idx(coordinates[0], coordinates[1], coordinates[2]);
                        self.f[neighbor * self.n_p + p_idx]
                    } else if periodic {
                        coordinates[axis] = 0;
                        let neighbor = self.idx(coordinates[0], coordinates[1], coordinates[2]);
                        self.f[neighbor * self.n_p + p_idx]
                    } else {
                        center
                    };
                    coordinates[axis] = along_axis;

                    let has_previous = periodic || along_axis > 0;
                    let has_next = periodic || along_axis + 1 < axis_length;
                    lower[along_axis] = if has_previous { -half_mu } else { 0.0 };
                    upper[along_axis] = if has_next { -half_mu } else { 0.0 };
                    let neighbor_count =
                        if has_previous { 1.0 } else { 0.0 } + if has_next { 1.0 } else { 0.0 };
                    diagonal[along_axis] = 1.0 + half_mu * neighbor_count;
                    rhs[along_axis] = center + half_mu * (previous - 2.0 * center + next);
                }

                let solution = if periodic {
                    Self::cyclic_thomas_solve(&lower, &diagonal, &upper, &rhs)
                } else {
                    Self::thomas_solve(&lower, &diagonal, &upper, &rhs)
                };
                for (along_axis, value) in solution.into_iter().enumerate() {
                    coordinates[axis] = along_axis;
                    let cell = self.idx(coordinates[0], coordinates[1], coordinates[2]);
                    self.f[cell * self.n_p + p_idx] = value.max(0.0);
                }
            }
        }
    }

    fn diffuse_x_sweep(
        &mut self,
        b_field: (&[f64], &[f64], &[f64]),
        rigidity_gv: f64,
        p_idx: usize,
        dt_s: f64,
    ) {
        self.diffuse_axis_sweep(0, false, b_field, rigidity_gv, p_idx, dt_s);
    }

    fn diffuse_y_sweep(
        &mut self,
        b_field: (&[f64], &[f64], &[f64]),
        rigidity_gv: f64,
        p_idx: usize,
        dt_s: f64,
    ) {
        self.diffuse_axis_sweep(1, true, b_field, rigidity_gv, p_idx, dt_s);
    }

    fn diffuse_z_sweep(
        &mut self,
        b_field: (&[f64], &[f64], &[f64]),
        rigidity_gv: f64,
        p_idx: usize,
        dt_s: f64,
    ) {
        self.diffuse_axis_sweep(2, true, b_field, rigidity_gv, p_idx, dt_s);
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
    ///   x/2 -> y/2 -> z/2 -> advect -> decelerate -> inject -> z/2 -> y/2 -> x/2
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
        let diffusion_half_step = 0.5 * self.dt_s;
        for p in 0..self.n_p {
            let rigidity_gv = self.grid.rigidity(p);
            self.diffuse_x_sweep(b_field, rigidity_gv, p, diffusion_half_step);
            self.diffuse_y_sweep(b_field, rigidity_gv, p, diffusion_half_step);
            self.diffuse_z_sweep(b_field, rigidity_gv, p, diffusion_half_step);
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
        for p in 0..self.n_p {
            let rigidity_gv = self.grid.rigidity(p);
            self.diffuse_z_sweep(b_field, rigidity_gv, p, diffusion_half_step);
            self.diffuse_y_sweep(b_field, rigidity_gv, p, diffusion_half_step);
            self.diffuse_x_sweep(b_field, rigidity_gv, p, diffusion_half_step);
        }

        if let Some(f) = lis {
            self.set_boundary_ism(f);
        }

        self.timestep += 1;
    }

    /// Extract differential flux J = p^2 * f at a single spatial cell.
    /// Returns `Vec<f64>` of length n_p.
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
    use crate::{
        diffusion::{DiffusionConfig, diffusion_tensor},
        grid::RigidityGrid,
    };

    fn gaussian_mode_amplitude(solver: &PteSolver, axis: usize, momentum: usize) -> f64 {
        let dimensions = [solver.nx, solver.ny, solver.nz];
        let wave_number = 2.0 * std::f64::consts::PI / dimensions[axis] as f64;
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for z in 0..solver.nz {
            for y in 0..solver.ny {
                for x in 0..solver.nx {
                    let coordinates = [x, y, z];
                    let mode = (wave_number * coordinates[axis] as f64).cos();
                    let cell = solver.idx(x, y, z);
                    numerator += solver.f[cell * solver.n_p + momentum] * mode;
                    denominator += mode * mode;
                }
            }
        }
        numerator / denominator
    }

    fn gaussian_decay_for_axis(axis: usize, field: [f64; 3]) -> (f64, f64, f64) {
        let axis_length = 24;
        let grid = RigidityGrid::new(2, 1.0, 2.0);
        let config = DiffusionConfig {
            kappa_0_au2_per_s: 0.4,
            r_ref_gv: 1.0,
            alpha: 0.0,
            epsilon_perp: 0.1,
            solar_epoch_a: 0.0,
        };
        let mut solver =
            PteSolver::new(2, axis_length, axis_length, grid, config.clone(), 0.1, 1.0);
        let field_magnitude =
            (field[0] * field[0] + field[1] * field[1] + field[2] * field[2]).sqrt();
        let tensor = diffusion_tensor(field, field_magnitude, 1.0, &config);
        let diffusivity = tensor[axis][axis];
        let mut field_x = vec![field[0]; solver.nx * solver.ny * solver.nz];
        let mut field_y = vec![field[1]; field_x.len()];
        let mut field_z = vec![field[2]; field_x.len()];

        for z in 0..solver.nz {
            for y in 0..solver.ny {
                for x in 0..solver.nx {
                    let coordinates = [x, y, z];
                    let distance = coordinates[axis].min(axis_length - coordinates[axis]) as f64;
                    let value = (-0.5 * (distance / 4.0).powi(2)).exp();
                    let cell = solver.idx(x, y, z);
                    for momentum in 0..solver.n_p {
                        solver.f[cell * solver.n_p + momentum] = value;
                    }
                    field_x[cell] = field[0];
                    field_y[cell] = field[1];
                    field_z[cell] = field[2];
                }
            }
        }

        let initial_amplitude = gaussian_mode_amplitude(&solver, axis, 0);
        let zero_velocity = vec![[0.0; 3]; solver.nx * solver.ny * solver.nz];
        solver.evolve_one_step(
            &zero_velocity,
            (&field_x, &field_y, &field_z),
            None,
            &[],
            None,
        );
        let observed_amplification = gaussian_mode_amplitude(&solver, axis, 0) / initial_amplitude;
        let discrete_eigenvalue = 4.0 * (std::f64::consts::PI / axis_length as f64).sin().powi(2);
        let full_step_rate = diffusivity * discrete_eigenvalue * solver.dt_s;
        let half_step_amplification = (1.0 - full_step_rate / 4.0) / (1.0 + full_step_rate / 4.0);
        let expected_amplification = half_step_amplification * half_step_amplification;
        (observed_amplification, expected_amplification, diffusivity)
    }

    #[test]
    fn test_gaussian_decay_uses_kyy_and_kzz() {
        let field_along_y = [0.0, 5.0, 0.0];
        for axis in [1, 2] {
            let (observed, expected, diffusivity) = gaussian_decay_for_axis(axis, field_along_y);
            assert!(diffusivity > 0.0);
            assert!(
                (observed - expected).abs() < 1e-10,
                "axis {axis} amplification {observed} differs from K_ii prediction {expected}"
            );
        }
    }

    #[test]
    fn test_rotated_field_aligned_tensor_diffuses_in_y() {
        let rotated_field = [3.0, 4.0, 0.0];
        let (observed, expected, k_yy) = gaussian_decay_for_axis(1, rotated_field);
        let config = DiffusionConfig {
            kappa_0_au2_per_s: 0.4,
            r_ref_gv: 1.0,
            alpha: 0.0,
            epsilon_perp: 0.1,
            solar_epoch_a: 0.0,
        };
        let perpendicular =
            config.epsilon_perp * crate::diffusion::kappa_parallel(1.0, 5.0, &config);
        assert!(k_yy > perpendicular);
        assert!(observed < 1.0, "the y-profile did not diffuse: {observed}");
        assert!(
            (observed - expected).abs() < 1e-10,
            "rotated field amplification {observed} differs from K_yy prediction {expected}"
        );
    }

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

    #[test]
    fn test_cyclic_thomas_matches_dense_solve_with_unequal_corners() {
        let n = 6;
        let a: Vec<f64> = (0..n).map(|i| -0.3 - 0.07 * i as f64).collect();
        let c: Vec<f64> = (0..n).map(|i| -0.2 - 0.11 * i as f64).collect();
        let b: Vec<f64> = (0..n).map(|i| 2.0 + 0.13 * i as f64).collect();
        let d: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64).sin()).collect();
        let x = PteSolver::cyclic_thomas_solve(&a, &b, &c, &d);

        let mut dense = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            dense[i][i] = b[i];
            dense[i][(i + n - 1) % n] += a[i];
            dense[i][(i + 1) % n] += c[i];
        }
        for (row, expected) in dense.iter().zip(&d) {
            let product: f64 = row.iter().zip(&x).map(|(m, v)| m * v).sum();
            assert!(
                (product - expected).abs() < 1e-12,
                "cyclic solve residual {product} vs {expected}"
            );
        }
    }
}
