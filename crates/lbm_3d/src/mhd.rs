//! Magnetohydrodynamics extension for the D3Q19 LBM solver.
//!
//! Adds magnetic field evolution and Lorentz force coupling for
//! magnetized plasma simulations (e.g., solar wind Parker spiral).
//!
//! The B-field evolves via the induction equation:
//!   dB/dt = curl(v x B) - eta * curl(curl(B))
//!
//! where eta is magnetic resistivity. The Lorentz force J x B (with
//! J = curl(B)/mu_0) couples back to the LBM via the Guo forcing scheme.

use crate::boundary::GridIndex;

/// Configuration for MHD simulation parameters.
#[derive(Clone, Debug)]
pub struct MhdConfig {
    /// Reference magnetic field magnitude (nT). Parker spiral B_0 at r_0.
    pub b0_nt: f64,
    /// Solar rotation rate (rad/s). Carrington: 2.662e-6.
    pub omega: f64,
    /// Magnetic permeability of free space (H/m) in simulation units.
    /// For dimensionless LBM, set to 1.0.
    pub mu_0: f64,
    /// Magnetic resistivity (Ohmic dissipation). 0.0 for ideal MHD.
    pub eta: f64,
    /// MHD sub-timestep relative to LBM dt. Typically 1.0.
    pub dt_mhd: f64,
    /// Divergence cleaning rate (Dedner hyperbolic cleaning). 0.0 to disable.
    pub cleaning_rate: f64,
}

impl Default for MhdConfig {
    fn default() -> Self {
        Self {
            b0_nt: 5.0,
            omega: 2.662e-6,
            mu_0: 1.0,
            eta: 0.0,
            dt_mhd: 1.0,
            cleaning_rate: 0.0,
        }
    }
}

/// 3D magnetic field on the LBM grid.
pub struct MhdField {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// B-field x-component, flat array of length nx*ny*nz.
    pub bx: Vec<f64>,
    /// B-field y-component.
    pub by: Vec<f64>,
    /// B-field z-component.
    pub bz: Vec<f64>,
    /// Divergence cleaning potential (Dedner psi).
    pub psi: Vec<f64>,
    /// Configuration parameters.
    pub config: MhdConfig,
}

impl MhdField {
    /// Create a zero-initialized MHD field on the given grid.
    pub fn new(nx: usize, ny: usize, nz: usize, config: MhdConfig) -> Self {
        let n = nx * ny * nz;
        Self {
            nx,
            ny,
            nz,
            bx: vec![0.0; n],
            by: vec![0.0; n],
            bz: vec![0.0; n],
            psi: vec![0.0; n],
            config,
        }
    }

    /// Initialize with Parker spiral magnetic field.
    ///
    /// The Parker spiral in Cartesian coordinates centered on the Sun:
    ///   B_r = B_0 * (r_0/r)^2
    ///   B_phi = -B_0 * (r_0/r) * (Omega * r * sin(theta)) / v_sw
    ///
    /// We map the grid to a radial slab: x is radial (Sun -> outward),
    /// y and z are transverse. The grid center is at 1 AU equivalent.
    ///
    /// `v_sw` is solar wind speed in simulation units (e.g., 0.1 in LBM lattice units).
    pub fn parker_spiral_init(&mut self, v_sw: f64) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let b0 = self.config.b0_nt;
        let omega = self.config.omega;

        // Map grid: x in [0, nx) corresponds to radial distance.
        // r_0 is at grid center x = nx/2. Radial scaling: r/r_0 = x / (nx/2).
        let r0 = nx as f64 / 2.0;

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = GridIndex::new(x, y, z).linearize(nx, ny);

                    // Radial distance (avoid r=0 singularity)
                    let r = (x as f64).max(1.0);
                    let ratio = r0 / r;

                    // B_r along x-axis (radial)
                    let b_r = b0 * ratio * ratio;

                    // B_phi along y-axis (azimuthal)
                    // Parker spiral angle: tan(psi) = Omega * r / v_sw
                    let b_phi = if v_sw.abs() > 1e-15 {
                        -b0 * ratio * (omega * r) / v_sw
                    } else {
                        0.0
                    };

                    self.bx[idx] = b_r;
                    self.by[idx] = b_phi;
                    // Bz = 0 for equatorial Parker spiral
                    self.bz[idx] = 0.0;
                }
            }
        }
    }

    /// Evolve B-field by one MHD timestep using the induction equation.
    ///
    /// dB/dt = curl(v x B) - eta * curl(curl(B))
    ///
    /// Uses forward-time centered-space (FTCS) finite differences with
    /// periodic boundary conditions. For ideal MHD (eta=0), this reduces
    /// to dB/dt = curl(v x B).
    pub fn evolve_b_field(&mut self, u: &[[f64; 3]]) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let n = nx * ny * nz;
        let dt = self.config.dt_mhd;
        let eta = self.config.eta;

        debug_assert_eq!(u.len(), n, "velocity field length mismatch");

        // Compute v x B at each grid point
        let mut vxb_x = vec![0.0; n];
        let mut vxb_y = vec![0.0; n];
        let mut vxb_z = vec![0.0; n];

        for idx in 0..n {
            let [ux, uy, uz] = u[idx];
            let bx_i = self.bx[idx];
            let by_i = self.by[idx];
            let bz_i = self.bz[idx];
            // v x B = (uy*Bz - uz*By, uz*Bx - ux*Bz, ux*By - uy*Bx)
            vxb_x[idx] = uy * bz_i - uz * by_i;
            vxb_y[idx] = uz * bx_i - ux * bz_i;
            vxb_z[idx] = ux * by_i - uy * bx_i;
        }

        // Compute curl(v x B) via central differences (periodic)
        let mut dbx = vec![0.0; n];
        let mut dby = vec![0.0; n];
        let mut dbz = vec![0.0; n];

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = z * (nx * ny) + y * nx + x;

                    // Periodic neighbors
                    let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                    let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                    let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                    let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                    let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                    let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                    // curl(F)_x = dFz/dy - dFy/dz
                    // curl(F)_y = dFx/dz - dFz/dx
                    // curl(F)_z = dFy/dx - dFx/dy
                    let curl_vxb_x =
                        0.5 * (vxb_z[yp] - vxb_z[ym]) - 0.5 * (vxb_y[zp] - vxb_y[zm]);
                    let curl_vxb_y =
                        0.5 * (vxb_x[zp] - vxb_x[zm]) - 0.5 * (vxb_z[xp] - vxb_z[xm]);
                    let curl_vxb_z =
                        0.5 * (vxb_y[xp] - vxb_y[xm]) - 0.5 * (vxb_x[yp] - vxb_x[ym]);

                    dbx[idx] = curl_vxb_x;
                    dby[idx] = curl_vxb_y;
                    dbz[idx] = curl_vxb_z;

                    // Resistive term: -eta * curl(curl(B)) = eta * Laplacian(B)
                    // (using vector identity: curl(curl(B)) = grad(div B) - Lap(B),
                    //  and div B = 0 ideally)
                    if eta > 0.0 {
                        let lap_bx = self.bx[xp] + self.bx[xm] + self.bx[yp] + self.bx[ym]
                            + self.bx[zp] + self.bx[zm]
                            - 6.0 * self.bx[idx];
                        let lap_by = self.by[xp] + self.by[xm] + self.by[yp] + self.by[ym]
                            + self.by[zp] + self.by[zm]
                            - 6.0 * self.by[idx];
                        let lap_bz = self.bz[xp] + self.bz[xm] + self.bz[yp] + self.bz[ym]
                            + self.bz[zp] + self.bz[zm]
                            - 6.0 * self.bz[idx];
                        dbx[idx] += eta * lap_bx;
                        dby[idx] += eta * lap_by;
                        dbz[idx] += eta * lap_bz;
                    }
                }
            }
        }

        // Euler forward step
        for idx in 0..n {
            self.bx[idx] += dt * dbx[idx];
            self.by[idx] += dt * dby[idx];
            self.bz[idx] += dt * dbz[idx];
        }

        // Divergence cleaning (Dedner hyperbolic)
        if self.config.cleaning_rate > 0.0 {
            let ch = self.config.cleaning_rate;
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let idx = z * (nx * ny) + y * nx + x;
                        let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                        let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                        let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                        let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                        let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                        let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                        let div_b = 0.5 * (self.bx[xp] - self.bx[xm])
                            + 0.5 * (self.by[yp] - self.by[ym])
                            + 0.5 * (self.bz[zp] - self.bz[zm]);

                        self.psi[idx] = -ch * ch * div_b;
                    }
                }
            }
            // Apply correction: B -= dt * grad(psi)
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let idx = z * (nx * ny) + y * nx + x;
                        let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                        let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                        let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                        let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                        let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                        let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                        self.bx[idx] -= dt * 0.5 * (self.psi[xp] - self.psi[xm]);
                        self.by[idx] -= dt * 0.5 * (self.psi[yp] - self.psi[ym]);
                        self.bz[idx] -= dt * 0.5 * (self.psi[zp] - self.psi[zm]);
                    }
                }
            }
        }
    }

    /// Compute Lorentz force density: F = J x B / mu_0.
    ///
    /// J = curl(B) via central differences, then F = J x B.
    /// Returns force array compatible with `LbmSolver3D::set_force_field()`.
    pub fn lorentz_force(&self) -> Vec<[f64; 3]> {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let n = nx * ny * nz;
        let mu_0 = self.config.mu_0;

        let mut force = vec![[0.0; 3]; n];

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = z * (nx * ny) + y * nx + x;

                    let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                    let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                    let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                    let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                    let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                    let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                    // J = curl(B) / mu_0
                    let jx = (0.5 * (self.bz[yp] - self.bz[ym])
                        - 0.5 * (self.by[zp] - self.by[zm]))
                        / mu_0;
                    let jy = (0.5 * (self.bx[zp] - self.bx[zm])
                        - 0.5 * (self.bz[xp] - self.bz[xm]))
                        / mu_0;
                    let jz = (0.5 * (self.by[xp] - self.by[xm])
                        - 0.5 * (self.bx[yp] - self.bx[ym]))
                        / mu_0;

                    // F = J x B
                    let bx_i = self.bx[idx];
                    let by_i = self.by[idx];
                    let bz_i = self.bz[idx];
                    force[idx] = [
                        jy * bz_i - jz * by_i,
                        jz * bx_i - jx * bz_i,
                        jx * by_i - jy * bx_i,
                    ];
                }
            }
        }

        force
    }

    /// Compute the maximum divergence of B across the grid (should be ~0 for physical fields).
    pub fn max_div_b(&self) -> f64 {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let mut max_div = 0.0_f64;

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                    let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                    let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                    let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                    let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                    let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                    let div_b = 0.5 * (self.bx[xp] - self.bx[xm])
                        + 0.5 * (self.by[yp] - self.by[ym])
                        + 0.5 * (self.bz[zp] - self.bz[zm]);
                    max_div = max_div.max(div_b.abs());
                }
            }
        }
        max_div
    }

    /// Project B-field to its divergence-free component via Helmholtz decomposition.
    ///
    /// Solves the Poisson equation `Lap(phi) = div(B)` using Jacobi iteration
    /// with periodic boundary conditions, then sets `B <- B - grad(phi)`.
    /// The result satisfies `div(B) = 0` to within the solver tolerance.
    ///
    /// This should be called ONCE after initialization (e.g., `parker_spiral_init`)
    /// to remove the magnetic monopole artifacts from projecting a spherically
    /// symmetric field onto a Cartesian grid.
    ///
    /// Returns `(initial_max_div, final_max_div)` for diagnostics.
    pub fn project_divergence_free(&mut self, max_iters: usize, tol: f64) -> (f64, f64) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let n = nx * ny * nz;

        let initial_div = self.max_div_b();

        // Compute RHS: div(B) at each cell
        let mut rhs = vec![0.0; n];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = z * (nx * ny) + y * nx + x;
                    let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                    let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                    let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                    let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                    let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                    let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                    rhs[idx] = 0.5 * (self.bx[xp] - self.bx[xm])
                        + 0.5 * (self.by[yp] - self.by[ym])
                        + 0.5 * (self.bz[zp] - self.bz[zm]);
                }
            }
        }

        // Subtract mean from RHS (Poisson with periodic BC requires zero-mean RHS)
        let mean_rhs: f64 = rhs.iter().sum::<f64>() / n as f64;
        for v in &mut rhs {
            *v -= mean_rhs;
        }

        // Solve div_h(grad_h(phi)) = rhs via Jacobi iteration.
        //
        // Critical: the discrete Poisson operator MUST be consistent with the
        // central-difference div and grad operators used in max_div_b() and
        // the correction step. Central-difference div(grad(phi)) in 1D is:
        //   0.25*(phi[x+2] - 2*phi[x] + phi[x-2])
        // which is a WIDER stencil than the standard Laplacian. We solve
        // this directly to ensure div_h(B - grad_h(phi)) = 0 exactly.
        //
        // The wide Laplacian in 3D:
        //   L_wide(phi) = 0.25 * sum_dim (phi[+2] + phi[-2] - 2*phi[0])
        // Jacobi update: phi_new = (sum_wide_neighbors/4 - rhs) * 4/6
        //   where sum_wide = phi[x+2]+phi[x-2]+phi[y+2]+phi[y-2]+phi[z+2]+phi[z-2]

        // Helper: periodic index offset by +/-2
        let wrap = |v: usize, delta: isize, size: usize| -> usize {
            ((v as isize + delta).rem_euclid(size as isize)) as usize
        };

        let mut phi = vec![0.0; n];
        let mut phi_new = vec![0.0; n];

        for _iter in 0..max_iters {
            let mut max_residual = 0.0_f64;

            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let idx = z * (nx * ny) + y * nx + x;

                        // Wide stencil neighbors (offset +/-2)
                        let xp2 = z * (nx * ny) + y * nx + wrap(x, 2, nx);
                        let xm2 = z * (nx * ny) + y * nx + wrap(x, -2, nx);
                        let yp2 = z * (nx * ny) + wrap(y, 2, ny) * nx + x;
                        let ym2 = z * (nx * ny) + wrap(y, -2, ny) * nx + x;
                        let zp2 = wrap(z, 2, nz) * (nx * ny) + y * nx + x;
                        let zm2 = wrap(z, -2, nz) * (nx * ny) + y * nx + x;

                        let sum_wide =
                            phi[xp2] + phi[xm2] + phi[yp2] + phi[ym2] + phi[zp2] + phi[zm2];

                        // Wide Laplacian: L = 0.25*(sum_wide - 6*phi_center)
                        // Solving L(phi) = rhs:
                        //   0.25*(sum_wide - 6*phi) = rhs
                        //   phi = (sum_wide - 4*rhs) / 6
                        phi_new[idx] = (sum_wide - 4.0 * rhs[idx]) / 6.0;

                        // Residual check
                        let lap_wide =
                            0.25 * (sum_wide - 6.0 * phi[idx]);
                        let residual = (lap_wide - rhs[idx]).abs();
                        max_residual = max_residual.max(residual);
                    }
                }
            }

            std::mem::swap(&mut phi, &mut phi_new);

            if max_residual < tol {
                break;
            }
        }

        // Apply correction: B <- B - grad_h(phi)
        // grad_h uses the same central-difference stencil as div_h
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = z * (nx * ny) + y * nx + x;
                    let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                    let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                    let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                    let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                    let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                    let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                    self.bx[idx] -= 0.5 * (phi[xp] - phi[xm]);
                    self.by[idx] -= 0.5 * (phi[yp] - phi[ym]);
                    self.bz[idx] -= 0.5 * (phi[zp] - phi[zm]);
                }
            }
        }

        let final_div = self.max_div_b();
        (initial_div, final_div)
    }

    /// Total magnetic energy: integral of B^2 / (2 * mu_0) over the grid.
    pub fn magnetic_energy(&self) -> f64 {
        let n = self.nx * self.ny * self.nz;
        let mut energy = 0.0;
        for idx in 0..n {
            let b_sq = self.bx[idx] * self.bx[idx]
                + self.by[idx] * self.by[idx]
                + self.bz[idx] * self.bz[idx];
            energy += b_sq;
        }
        energy / (2.0 * self.config.mu_0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mhd_field_creation() {
        let field = MhdField::new(8, 8, 8, MhdConfig::default());
        assert_eq!(field.bx.len(), 512);
        assert_eq!(field.by.len(), 512);
        assert_eq!(field.bz.len(), 512);
        assert!(field.bx.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_parker_spiral_init() {
        let mut field = MhdField::new(16, 8, 8, MhdConfig {
            b0_nt: 5.0,
            omega: 2.662e-6,
            ..MhdConfig::default()
        });
        field.parker_spiral_init(0.1);

        // B_r should be positive everywhere (radial outward)
        let n = 16 * 8 * 8;
        assert!(field.bx.iter().take(n).all(|&v| v > 0.0));

        // B_r should decrease with radial distance (x)
        // Compare x=2 vs x=14 at y=4, z=4
        let idx_near = GridIndex::new(2, 4, 4).linearize(16, 8);
        let idx_far = GridIndex::new(14, 4, 4).linearize(16, 8);
        assert!(field.bx[idx_near] > field.bx[idx_far]);
    }

    #[test]
    fn test_uniform_b_zero_lorentz() {
        // Uniform B-field should produce zero current (curl(B)=0) and thus zero Lorentz force
        let mut field = MhdField::new(8, 8, 8, MhdConfig::default());
        let n = 8 * 8 * 8;
        for idx in 0..n {
            field.bx[idx] = 1.0;
            field.by[idx] = 0.0;
            field.bz[idx] = 0.0;
        }

        let force = field.lorentz_force();
        for f in &force {
            assert!(f[0].abs() < 1e-14);
            assert!(f[1].abs() < 1e-14);
            assert!(f[2].abs() < 1e-14);
        }
    }

    #[test]
    fn test_divergence_uniform_field() {
        let mut field = MhdField::new(8, 8, 8, MhdConfig::default());
        let n = 8 * 8 * 8;
        for idx in 0..n {
            field.bx[idx] = 3.0;
            field.by[idx] = -1.0;
            field.bz[idx] = 2.0;
        }
        // Uniform field has zero divergence
        assert!(field.max_div_b() < 1e-14);
    }

    #[test]
    fn test_magnetic_energy() {
        let mut field = MhdField::new(4, 4, 4, MhdConfig {
            mu_0: 1.0,
            ..MhdConfig::default()
        });
        let n = 4 * 4 * 4;
        for idx in 0..n {
            field.bx[idx] = 1.0;
        }
        // Energy = sum(B^2) / (2 * mu_0) = 64 * 1.0 / 2.0 = 32.0
        assert!((field.magnetic_energy() - 32.0).abs() < 1e-12);
    }

    #[test]
    fn test_evolve_frozen_in() {
        // Ideal MHD (eta=0) with zero velocity: B should not change
        let mut field = MhdField::new(8, 8, 8, MhdConfig {
            eta: 0.0,
            ..MhdConfig::default()
        });
        let n = 8 * 8 * 8;
        // Set uniform B
        for idx in 0..n {
            field.bx[idx] = 1.0;
        }
        let u = vec![[0.0, 0.0, 0.0]; n];
        let energy_before = field.magnetic_energy();

        field.evolve_b_field(&u);

        let energy_after = field.magnetic_energy();
        assert!(
            (energy_before - energy_after).abs() < 1e-12,
            "energy_before={energy_before}, energy_after={energy_after}"
        );
    }

    #[test]
    fn test_projection_uniform_is_identity() {
        // A uniform B-field already has div(B) = 0.
        // Projection should leave it unchanged.
        let mut field = MhdField::new(8, 8, 8, MhdConfig::default());
        let n = 8 * 8 * 8;
        for idx in 0..n {
            field.bx[idx] = 3.0;
            field.by[idx] = -1.0;
            field.bz[idx] = 2.0;
        }
        let energy_before = field.magnetic_energy();
        let (init_div, final_div) = field.project_divergence_free(1000, 1e-12);

        assert!(init_div < 1e-14, "uniform field should have zero div: {init_div}");
        assert!(final_div < 1e-14, "projection should preserve zero div: {final_div}");

        let energy_after = field.magnetic_energy();
        let rel_change = (energy_after - energy_before).abs() / energy_before;
        assert!(
            rel_change < 1e-10,
            "projection of divergence-free field should preserve energy: rel_change={rel_change:.3e}"
        );
    }

    #[test]
    fn test_projection_parker_reduces_div() {
        // Parker spiral on Cartesian grid has large div(B).
        // Projection should reduce it by several orders of magnitude.
        let mut field = MhdField::new(16, 8, 8, MhdConfig {
            b0_nt: 5.0,
            omega: 2.662e-6,
            ..MhdConfig::default()
        });
        field.parker_spiral_init(0.1);

        let (init_div, final_div) = field.project_divergence_free(5000, 1e-10);

        assert!(
            init_div > 1.0,
            "Parker spiral should have significant div(B): {init_div:.3e}"
        );
        assert!(
            final_div < init_div * 1e-3,
            "projection should reduce div(B) by >3 orders: {init_div:.3e} -> {final_div:.3e}"
        );
    }

    #[test]
    fn test_projection_preserves_energy_order() {
        // Projection removes the irrotational (gradient) component.
        // For a Parker spiral, the solenoidal component carries most of the energy.
        // Energy should remain within the same order of magnitude.
        let mut field = MhdField::new(16, 8, 8, MhdConfig {
            b0_nt: 5.0,
            omega: 2.662e-6,
            ..MhdConfig::default()
        });
        field.parker_spiral_init(0.1);
        let energy_before = field.magnetic_energy();

        field.project_divergence_free(5000, 1e-10);

        let energy_after = field.magnetic_energy();
        // Energy may decrease (we're removing the gradient component) but should
        // stay within a factor of 10
        assert!(
            energy_after > energy_before * 0.1,
            "energy should remain same order: before={energy_before:.3e}, after={energy_after:.3e}"
        );
        assert!(
            energy_after <= energy_before * 1.01,
            "energy should not increase: before={energy_before:.3e}, after={energy_after:.3e}"
        );
    }

    #[test]
    fn test_projection_preserves_br_monotonicity() {
        // After projection, B_r should still decrease with x (radial distance)
        // along the midline, though values will be modified.
        let mut field = MhdField::new(16, 8, 8, MhdConfig {
            b0_nt: 5.0,
            omega: 2.662e-6,
            ..MhdConfig::default()
        });
        field.parker_spiral_init(0.1);
        field.project_divergence_free(5000, 1e-10);

        // Check B_r decreases from x=2 to x=14 at midline
        let idx_near = GridIndex::new(2, 4, 4).linearize(16, 8);
        let idx_far = GridIndex::new(14, 4, 4).linearize(16, 8);
        assert!(
            field.bx[idx_near] > field.bx[idx_far],
            "B_r should still decrease with radius after projection: near={:.3e}, far={:.3e}",
            field.bx[idx_near],
            field.bx[idx_far]
        );
    }
}
