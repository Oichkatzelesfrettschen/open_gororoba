use crate::simulation::state_3d::{FrustrationField3D, LbmBackend3D};
use algebra_core::lie::e7_geometry::generate_e7_roots;
#[cfg(feature = "gpu")]
use anyhow::Context;
use anyhow::Result;
use ndarray::{Array3, Zip};
use spectral_core::ndfft::{fft_3d, ifft_3d, real_to_complex_3d};

/// Frustration field derived from E7 Lie algebra roots.
pub struct E7SpectralFilter {
    pub mask: Array3<f32>,
    #[cfg(feature = "gpu")]
    pub d_mask: Option<cudarc::driver::CudaSlice<f32>>,
    pub base_damping: f32,
    pub enstrophy_crit: f32,
    pub current_damping: f32,
    pub step_count: usize,
}

impl E7SpectralFilter {
    pub fn new(nx: usize, ny: usize, nz: usize, base_damping: f32) -> Self {
        let mut mask = Array3::zeros((nx, ny, nz));
        let roots = generate_e7_roots();
        let sigma: f32 = 1.5;
        let sigma_sq_inv = 1.0 / (2.0 * sigma * sigma);

        for root in roots {
            let rx = root.root.coords[2] as f32;
            let ry = root.root.coords[3] as f32;
            let rz = root.root.coords[4] as f32;
            let scale = 8.0f32;
            let center_x = rx * scale + nx as f32 / 2.0;
            let center_y = ry * scale + ny as f32 / 2.0;
            let center_z = rz * scale + nz as f32 / 2.0;
            let radius = (sigma * 4.0).ceil() as isize;

            for dz in -radius..=radius {
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let ix = center_x.round() as isize + dx;
                        let iy = center_y.round() as isize + dy;
                        let iz = center_z.round() as isize + dz;
                        let gx = ix.rem_euclid(nx as isize) as usize;
                        let gy = iy.rem_euclid(ny as isize) as usize;
                        let gz = iz.rem_euclid(nz as isize) as usize;
                        let dist_sq = (dx * dx + dy * dy + dz * dz) as f32;
                        mask[[gx, gy, gz]] += (-dist_sq * sigma_sq_inv).exp();
                    }
                }
            }
        }

        let max_val = mask.iter().cloned().fold(f32::NAN, f32::max);
        mask.mapv_inplace(|v| v / max_val);
        mask[[0, 0, 0]] = 1.0;

        Self {
            mask,
            #[cfg(feature = "gpu")]
            d_mask: None,
            base_damping,
            enstrophy_crit: 1e-4,
            current_damping: 1.0,
            step_count: 0,
        }
    }
}

impl FrustrationField3D for E7SpectralFilter {
    fn apply(&mut self, fluid: &mut LbmBackend3D, nx: usize, ny: usize, nz: usize) -> Result<()> {
        self.step_count += 1;
        if !self.step_count.is_multiple_of(10) {
            return Ok(());
        }

        let enstrophy = match fluid {
            LbmBackend3D::Cpu(solver) => self.calculate_enstrophy_cpu(solver, nx, ny, nz) as f32,
            #[cfg(feature = "gpu")]
            LbmBackend3D::Gpu(solver) => solver.calculate_enstrophy()?,
            #[cfg(not(feature = "gpu"))]
            LbmBackend3D::Gpu(_) => panic!("GPU backend not enabled"),
        };

        let alpha = (enstrophy / self.enstrophy_crit).tanh();
        self.current_damping = 1.0f32 - (1.0f32 - self.base_damping) * alpha;

        match fluid {
            LbmBackend3D::Cpu(solver) => {
                let mut ux = Array3::zeros((nx, ny, nz));
                let mut uy = Array3::zeros((nx, ny, nz));
                let mut uz = Array3::zeros((nx, ny, nz));
                for (idx, vel) in solver.u.iter().enumerate() {
                    let z = idx / (nx * ny);
                    let y = (idx % (nx * ny)) / nx;
                    let x = idx % nx;
                    ux[[x, y, z]] = vel[0] as f32;
                    uy[[x, y, z]] = vel[1] as f32;
                    uz[[x, y, z]] = vel[2] as f32;
                }
                self.filter_component_cpu(&mut ux, self.current_damping);
                self.filter_component_cpu(&mut uy, self.current_damping);
                self.filter_component_cpu(&mut uz, self.current_damping);
                for (idx, vel) in solver.u.iter_mut().enumerate() {
                    let z = idx / (nx * ny);
                    let y = (idx % (nx * ny)) / nx;
                    let x = idx % nx;
                    vel[0] = ux[[x, y, z]] as f64;
                    vel[1] = uy[[x, y, z]] as f64;
                    vel[2] = uz[[x, y, z]] as f64;
                }
            }
            #[cfg(feature = "gpu")]
            LbmBackend3D::Gpu(solver) => {
                if self.d_mask.is_none() {
                    let mask_flat: Vec<f32> = self.mask.iter().cloned().collect();
                    self.d_mask = Some(solver.stream().clone_htod(&mask_flat)?);
                }

                let d_mask = self.d_mask.as_ref().unwrap();

                #[cfg(feature = "cufft")]
                {
                    for comp in 0..3 {
                        // Borrow solver buffers without any device-to-device clones.
                        //
                        // Pipeline per component:
                        // 1) Real (u) -> complex (u_hat)
                        // 2) FFT (forward) u_hat -> u_hat_out
                        // 3) Mask in frequency domain (in-place on u_hat_out)
                        // 4) IFFT (inverse) u_hat_out -> u_hat
                        // 5) Complex -> real, with 1/N normalization (cuFFT is unnormalized)
                        let mut d_u_hat = solver
                            .d_u_hat
                            .take()
                            .context("GPU spectral filter missing solver.d_u_hat buffer")?;
                        let mut d_u_hat_out = solver
                            .d_u_hat_out
                            .take()
                            .context("GPU spectral filter missing solver.d_u_hat_out buffer")?;

                        let res = (|| -> Result<()> {
                            solver.convert_real_to_complex(&mut d_u_hat, comp)?;
                            solver.fft_3d_c2c_into(&d_u_hat, &mut d_u_hat_out, -1)?;
                            solver.apply_spectral_mask(
                                &mut d_u_hat_out,
                                d_mask,
                                self.current_damping,
                            )?;
                            solver.fft_3d_c2c_into(&d_u_hat_out, &mut d_u_hat, 1)?;

                            let scale = 1.0f32 / (nx * ny * nz) as f32;
                            solver.convert_complex_to_real(&d_u_hat, comp, scale)?;
                            Ok(())
                        })();

                        solver.d_u_hat = Some(d_u_hat);
                        solver.d_u_hat_out = Some(d_u_hat_out);

                        res?;
                    }
                }
            }
            #[cfg(not(feature = "gpu"))]
            LbmBackend3D::Gpu(_) => panic!("GPU backend not enabled"),
        }
        Ok(())
    }

    fn trace_algebra_norm(&self) -> f64 {
        // Export damping-induced coupling activity as a non-placeholder scalar.
        // 0 => no extra damping, higher => stronger active spectral suppression.
        (1.0f32 - self.current_damping).max(0.0) as f64
    }
}

impl E7SpectralFilter {
    fn filter_component_cpu(&self, field: &mut Array3<f32>, damping: f32) {
        let field_f64 = field.mapv(|x| x as f64);
        let complex_field = real_to_complex_3d(&field_f64);
        let mut field_hat = fft_3d(&complex_field);
        Zip::from(&mut field_hat)
            .and(&self.mask)
            .for_each(|val, &m| {
                if m < 0.5 {
                    *val *= damping as f64;
                }
            });
        let field_filtered = ifft_3d(&field_hat);
        Zip::from(field).and(&field_filtered).for_each(|out, inp| {
            *out = inp.re as f32;
        });
    }

    fn calculate_enstrophy_cpu(
        &self,
        solver: &lbm_3d::solver::LbmSolver3D,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> f64 {
        let mut enstrophy = 0.0;
        let u = &solver.u;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = |xi: usize, yi: usize, zi: usize| xi + nx * (yi + ny * zi);
                    let xp = (x + 1) % nx;
                    let xm = (x + nx - 1) % nx;
                    let yp = (y + 1) % ny;
                    let ym = (y + ny - 1) % ny;
                    let zp = (z + 1) % nz;
                    let zm = (z + nz - 1) % nz;
                    let duz_dy = (u[idx(x, yp, z)][2] - u[idx(x, ym, z)][2]) * 0.5;
                    let duy_dz = (u[idx(x, y, zp)][1] - u[idx(x, y, zm)][1]) * 0.5;
                    let dux_dz = (u[idx(x, y, zp)][0] - u[idx(x, y, zm)][0]) * 0.5;
                    let duz_dx = (u[idx(xp, y, z)][2] - u[idx(xm, y, z)][2]) * 0.5;
                    let duy_dx = (u[idx(xp, y, z)][1] - u[idx(xm, y, z)][1]) * 0.5;
                    let dux_dy = (u[idx(x, yp, z)][0] - u[idx(x, ym, z)][0]) * 0.5;
                    let wx = duz_dy - duy_dz;
                    let wy = dux_dz - duz_dx;
                    let wz = duy_dx - dux_dy;
                    enstrophy += wx * wx + wy * wy + wz * wz;
                }
            }
        }
        enstrophy / (nx * ny * nz) as f64
    }
}
