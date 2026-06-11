//! GPU-accelerated Algebraic Lensing using CUDA.
//!
//! Bridges the GRIN RK4 solver to custom CUDA kernels for massive throughput.
//! Optimized for Ada Lovelace (SM89) hardware.

#[cfg(any(feature = "gpu", feature = "vulkan", feature = "cubecl"))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[cfg(feature = "gpu")]
use cudarc::driver::{CudaStream, DeviceRepr, PushKernelArg, ValidAsZeroBits};
#[cfg(feature = "gpu")]
use gororoba_gpu_cuda::{
    Buffer, CompileOptions, Context, KernelHandle, LaunchConfig, ModuleRegistry,
};
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
const KERNEL_SRC: &str = include_str!("algebraic_lensing_grin.cu");

#[cfg(feature = "cubecl")]
mod cubecl_backend {
    use super::GpuVec3;
    use cubecl::prelude::*;
    use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

    const WORKGROUP_SIZE: u32 = 256;

    #[derive(Clone, Copy, Debug)]
    pub struct AlgebraicLensingCubeclConfig {
        pub nx: usize,
        pub ny: usize,
        pub nz: usize,
        pub alpha: f32,
        pub dt: f32,
        pub max_steps: usize,
        pub eps_grad: f32,
    }

    impl AlgebraicLensingCubeclConfig {
        pub fn new(
            nx: usize,
            ny: usize,
            nz: usize,
            alpha: f32,
            dt: f32,
            max_steps: usize,
        ) -> Result<Self, String> {
            let config = Self {
                nx,
                ny,
                nz,
                alpha,
                dt,
                max_steps,
                eps_grad: 0.1,
            };
            config.validate()?;
            Ok(config)
        }

        pub fn n_cells(&self) -> usize {
            self.nx * self.ny * self.nz
        }

        fn validate(&self) -> Result<(), String> {
            if self.nx == 0 || self.ny == 0 || self.nz == 0 {
                return Err(format!(
                    "grid dimensions must be positive, got {}x{}x{}",
                    self.nx, self.ny, self.nz
                ));
            }
            let n_cells = (self.nx as u64)
                .checked_mul(self.ny as u64)
                .and_then(|value| value.checked_mul(self.nz as u64))
                .ok_or_else(|| "density grid cell count overflows u64".to_string())?;
            if n_cells > u32::MAX as u64 {
                return Err(format!("density grid exceeds u32 indexing: {n_cells}"));
            }
            if !(self.alpha.is_finite() && self.dt.is_finite() && self.dt > 0.0) {
                return Err(format!(
                    "alpha must be finite and dt must be finite positive, got alpha={}, dt={}",
                    self.alpha, self.dt
                ));
            }
            if !(self.eps_grad.is_finite() && self.eps_grad > 0.0) {
                return Err(format!(
                    "eps_grad must be finite and positive, got {}",
                    self.eps_grad
                ));
            }
            if self.max_steps == 0 {
                return Err("max_steps must be at least 1".to_string());
            }
            if self.max_steps > u32::MAX as usize {
                return Err(format!("max_steps exceeds u32: {}", self.max_steps));
            }
            Ok(())
        }
    }

    pub struct AlgebraicLensingCubecl;

    impl AlgebraicLensingCubecl {
        pub fn is_available() -> bool {
            gororoba_gpu_cubecl::Runtime::probe()
        }

        pub fn trace_rays(
            config: AlgebraicLensingCubeclConfig,
            density_field: &[f32],
            initial_pos: &[GpuVec3],
            initial_dir: &[GpuVec3],
        ) -> Result<(Vec<GpuVec3>, Vec<GpuVec3>), String> {
            validate_trace_inputs(config, density_field, initial_pos, initial_dir)?;
            if !Self::is_available() {
                return Err("algebraic lensing cubecl adapter unavailable".to_string());
            }

            let n_rays = initial_pos.len();
            let device = WgpuDevice::default();
            let client = WgpuRuntime::client(&device);

            let density_handle = client.create_from_slice(&encode_f32_slice(density_field)?);
            let pos_handle =
                client.create_from_slice(&encode_f32_slice(&flatten_vec3(initial_pos))?);
            let dir_handle =
                client.create_from_slice(&encode_f32_slice(&flatten_vec3(initial_dir))?);
            let final_pos_handle = client.empty(3 * n_rays * std::mem::size_of::<f32>());
            let final_dir_handle = client.empty(3 * n_rays * std::mem::size_of::<f32>());
            let final_pos_readback = final_pos_handle.clone();
            let final_dir_readback = final_dir_handle.clone();

            let cube_dim = CubeDim::new_1d(WORKGROUP_SIZE);
            let cube_count = CubeCount::new_1d(dispatch_groups(n_rays)?);

            // SAFETY: launch_unchecked requires exact buffer proof. The density
            // buffer has nx * ny * nz f32 values, ray buffers have 3 * n_rays
            // f32 values, and the kernel terminates threads past n_rays.
            unsafe {
                algebraic_lensing_cubecl_kernel::launch_unchecked::<WgpuRuntime>(
                    &client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts(density_handle, density_field.len()),
                    ArrayArg::from_raw_parts(pos_handle, 3 * n_rays),
                    ArrayArg::from_raw_parts(dir_handle, 3 * n_rays),
                    ArrayArg::from_raw_parts(final_pos_handle, 3 * n_rays),
                    ArrayArg::from_raw_parts(final_dir_handle, 3 * n_rays),
                    config.nx as u32,
                    config.ny as u32,
                    config.nz as u32,
                    n_rays as u32,
                    config.max_steps as u32,
                    config.alpha.to_bits(),
                    config.dt.to_bits(),
                    config.eps_grad.to_bits(),
                );
            }

            let final_pos = decode_f32_output(
                &client.read_one_unchecked(final_pos_readback),
                3 * n_rays,
                "final_pos",
            )?;
            let final_dir = decode_f32_output(
                &client.read_one_unchecked(final_dir_readback),
                3 * n_rays,
                "final_dir",
            )?;
            Ok((inflate_vec3(&final_pos), inflate_vec3(&final_dir)))
        }
    }

    #[cube(launch_unchecked)]
    #[allow(clippy::too_many_arguments)] // cubecl launch ABI: one kernel arg per buffer/scalar.
    #[allow(clippy::manual_range_contains)] // CubeCL device code keeps scalar comparisons in kernel IR.
    pub fn algebraic_lensing_cubecl_kernel(
        density_field: &Array<f32>,
        initial_pos: &Array<f32>,
        initial_dir: &Array<f32>,
        final_pos: &mut Array<f32>,
        final_dir: &mut Array<f32>,
        #[comptime] nx: u32,
        #[comptime] ny: u32,
        #[comptime] nz: u32,
        #[comptime] n_rays: u32,
        #[comptime] max_steps: u32,
        #[comptime] alpha_bits: u32,
        #[comptime] dt_bits: u32,
        #[comptime] eps_grad_bits: u32,
    ) {
        let ray = ABSOLUTE_POS;
        if ray >= n_rays as usize {
            terminate!();
        }

        let base = ray * 3usize;
        let mut p_x = initial_pos[base];
        let mut p_y = initial_pos[base + 1usize];
        let mut p_z = initial_pos[base + 2usize];
        let mut d_x = initial_dir[base];
        let mut d_y = initial_dir[base + 1usize];
        let mut d_z = initial_dir[base + 2usize];
        let dt = f32::reinterpret(dt_bits);

        let mut step: u32 = 0u32;
        while step < max_steps {
            let k1_p_x = d_x;
            let k1_p_y = d_y;
            let k1_p_z = d_z;
            let k1_d_x = curvature_component(
                density_field,
                p_x,
                p_y,
                p_z,
                d_x,
                d_y,
                d_z,
                0u32,
                nx,
                ny,
                nz,
                alpha_bits,
                eps_grad_bits,
            );
            let k1_d_y = curvature_component(
                density_field,
                p_x,
                p_y,
                p_z,
                d_x,
                d_y,
                d_z,
                1u32,
                nx,
                ny,
                nz,
                alpha_bits,
                eps_grad_bits,
            );
            let k1_d_z = curvature_component(
                density_field,
                p_x,
                p_y,
                p_z,
                d_x,
                d_y,
                d_z,
                2u32,
                nx,
                ny,
                nz,
                alpha_bits,
                eps_grad_bits,
            );

            let p2_x = p_x + k1_p_x * (dt * 0.5);
            let p2_y = p_y + k1_p_y * (dt * 0.5);
            let p2_z = p_z + k1_p_z * (dt * 0.5);
            let d2_x_raw = d_x + k1_d_x * (dt * 0.5);
            let d2_y_raw = d_y + k1_d_y * (dt * 0.5);
            let d2_z_raw = d_z + k1_d_z * (dt * 0.5);
            let d2_inv_len = inv_len_or_one(d2_x_raw, d2_y_raw, d2_z_raw);
            let d2_x = d2_x_raw * d2_inv_len;
            let d2_y = d2_y_raw * d2_inv_len;
            let d2_z = d2_z_raw * d2_inv_len;

            let k2_p_x = d2_x;
            let k2_p_y = d2_y;
            let k2_p_z = d2_z;
            let k2_d_x = curvature_component(
                density_field,
                p2_x,
                p2_y,
                p2_z,
                d2_x,
                d2_y,
                d2_z,
                0u32,
                nx,
                ny,
                nz,
                alpha_bits,
                eps_grad_bits,
            );
            let k2_d_y = curvature_component(
                density_field,
                p2_x,
                p2_y,
                p2_z,
                d2_x,
                d2_y,
                d2_z,
                1u32,
                nx,
                ny,
                nz,
                alpha_bits,
                eps_grad_bits,
            );
            let k2_d_z = curvature_component(
                density_field,
                p2_x,
                p2_y,
                p2_z,
                d2_x,
                d2_y,
                d2_z,
                2u32,
                nx,
                ny,
                nz,
                alpha_bits,
                eps_grad_bits,
            );

            let p3_x = p_x + k2_p_x * (dt * 0.5);
            let p3_y = p_y + k2_p_y * (dt * 0.5);
            let p3_z = p_z + k2_p_z * (dt * 0.5);
            let d3_x_raw = d_x + k2_d_x * (dt * 0.5);
            let d3_y_raw = d_y + k2_d_y * (dt * 0.5);
            let d3_z_raw = d_z + k2_d_z * (dt * 0.5);
            let d3_inv_len = inv_len_or_one(d3_x_raw, d3_y_raw, d3_z_raw);
            let d3_x = d3_x_raw * d3_inv_len;
            let d3_y = d3_y_raw * d3_inv_len;
            let d3_z = d3_z_raw * d3_inv_len;

            let k3_p_x = d3_x;
            let k3_p_y = d3_y;
            let k3_p_z = d3_z;
            let k3_d_x = curvature_component(
                density_field,
                p3_x,
                p3_y,
                p3_z,
                d3_x,
                d3_y,
                d3_z,
                0u32,
                nx,
                ny,
                nz,
                alpha_bits,
                eps_grad_bits,
            );
            let k3_d_y = curvature_component(
                density_field,
                p3_x,
                p3_y,
                p3_z,
                d3_x,
                d3_y,
                d3_z,
                1u32,
                nx,
                ny,
                nz,
                alpha_bits,
                eps_grad_bits,
            );
            let k3_d_z = curvature_component(
                density_field,
                p3_x,
                p3_y,
                p3_z,
                d3_x,
                d3_y,
                d3_z,
                2u32,
                nx,
                ny,
                nz,
                alpha_bits,
                eps_grad_bits,
            );

            let p4_x = p_x + k3_p_x * dt;
            let p4_y = p_y + k3_p_y * dt;
            let p4_z = p_z + k3_p_z * dt;
            let d4_x_raw = d_x + k3_d_x * dt;
            let d4_y_raw = d_y + k3_d_y * dt;
            let d4_z_raw = d_z + k3_d_z * dt;
            let d4_inv_len = inv_len_or_one(d4_x_raw, d4_y_raw, d4_z_raw);
            let d4_x = d4_x_raw * d4_inv_len;
            let d4_y = d4_y_raw * d4_inv_len;
            let d4_z = d4_z_raw * d4_inv_len;

            let k4_p_x = d4_x;
            let k4_p_y = d4_y;
            let k4_p_z = d4_z;
            let k4_d_x = curvature_component(
                density_field,
                p4_x,
                p4_y,
                p4_z,
                d4_x,
                d4_y,
                d4_z,
                0u32,
                nx,
                ny,
                nz,
                alpha_bits,
                eps_grad_bits,
            );
            let k4_d_y = curvature_component(
                density_field,
                p4_x,
                p4_y,
                p4_z,
                d4_x,
                d4_y,
                d4_z,
                1u32,
                nx,
                ny,
                nz,
                alpha_bits,
                eps_grad_bits,
            );
            let k4_d_z = curvature_component(
                density_field,
                p4_x,
                p4_y,
                p4_z,
                d4_x,
                d4_y,
                d4_z,
                2u32,
                nx,
                ny,
                nz,
                alpha_bits,
                eps_grad_bits,
            );

            p_x += (k1_p_x + k2_p_x * 2.0 + k3_p_x * 2.0 + k4_p_x) * (dt / 6.0);
            p_y += (k1_p_y + k2_p_y * 2.0 + k3_p_y * 2.0 + k4_p_y) * (dt / 6.0);
            p_z += (k1_p_z + k2_p_z * 2.0 + k3_p_z * 2.0 + k4_p_z) * (dt / 6.0);

            let d_next_x = d_x + (k1_d_x + k2_d_x * 2.0 + k3_d_x * 2.0 + k4_d_x) * (dt / 6.0);
            let d_next_y = d_y + (k1_d_y + k2_d_y * 2.0 + k3_d_y * 2.0 + k4_d_y) * (dt / 6.0);
            let d_next_z = d_z + (k1_d_z + k2_d_z * 2.0 + k3_d_z * 2.0 + k4_d_z) * (dt / 6.0);
            let d_next_inv_len = inv_len_or_one(d_next_x, d_next_y, d_next_z);
            d_x = d_next_x * d_next_inv_len;
            d_y = d_next_y * d_next_inv_len;
            d_z = d_next_z * d_next_inv_len;

            if p_x < -100.0
                || p_x > 200.0
                || p_y < -100.0
                || p_y > 200.0
                || p_z < -100.0
                || p_z > 200.0
            {
                step = max_steps;
            } else {
                step += 1u32;
            }
        }

        final_pos[base] = p_x;
        final_pos[base + 1usize] = p_y;
        final_pos[base + 2usize] = p_z;
        final_dir[base] = d_x;
        final_dir[base + 1usize] = d_y;
        final_dir[base + 2usize] = d_z;
    }

    #[cube]
    #[allow(clippy::too_many_arguments)] // device helper mirrors the scalar ray equation.
    fn curvature_component(
        density_field: &Array<f32>,
        p_x: f32,
        p_y: f32,
        p_z: f32,
        d_x: f32,
        d_y: f32,
        d_z: f32,
        component: u32,
        nx: u32,
        ny: u32,
        nz: u32,
        alpha_bits: u32,
        eps_grad_bits: u32,
    ) -> f32 {
        let grad_x = gradient_component(
            density_field,
            p_x,
            p_y,
            p_z,
            0u32,
            nx,
            ny,
            nz,
            alpha_bits,
            eps_grad_bits,
        );
        let grad_y = gradient_component(
            density_field,
            p_x,
            p_y,
            p_z,
            1u32,
            nx,
            ny,
            nz,
            alpha_bits,
            eps_grad_bits,
        );
        let grad_z = gradient_component(
            density_field,
            p_x,
            p_y,
            p_z,
            2u32,
            nx,
            ny,
            nz,
            alpha_bits,
            eps_grad_bits,
        );
        let n = sample_n(density_field, p_x, p_y, p_z, nx, ny, nz, alpha_bits);
        let d_dot_grad = d_x * grad_x + d_y * grad_y + d_z * grad_z;
        let mut result = (grad_z - d_z * d_dot_grad) * (1.0 / n);
        if component == 0u32 {
            result = (grad_x - d_x * d_dot_grad) * (1.0 / n);
        }
        if component == 1u32 {
            result = (grad_y - d_y * d_dot_grad) * (1.0 / n);
        }
        result
    }

    #[cube]
    #[allow(clippy::too_many_arguments)] // device helper mirrors central differences.
    fn gradient_component(
        density_field: &Array<f32>,
        p_x: f32,
        p_y: f32,
        p_z: f32,
        component: u32,
        nx: u32,
        ny: u32,
        nz: u32,
        alpha_bits: u32,
        eps_grad_bits: u32,
    ) -> f32 {
        let eps = f32::reinterpret(eps_grad_bits);
        let mut result = (sample_n(density_field, p_x, p_y, p_z + eps, nx, ny, nz, alpha_bits)
            - sample_n(density_field, p_x, p_y, p_z - eps, nx, ny, nz, alpha_bits))
            / (2.0 * eps);
        if component == 0u32 {
            result = (sample_n(density_field, p_x + eps, p_y, p_z, nx, ny, nz, alpha_bits)
                - sample_n(density_field, p_x - eps, p_y, p_z, nx, ny, nz, alpha_bits))
                / (2.0 * eps);
        }
        if component == 1u32 {
            result = (sample_n(density_field, p_x, p_y + eps, p_z, nx, ny, nz, alpha_bits)
                - sample_n(density_field, p_x, p_y - eps, p_z, nx, ny, nz, alpha_bits))
                / (2.0 * eps);
        }
        result
    }

    #[cube]
    fn sample_n(
        density_field: &Array<f32>,
        p_x: f32,
        p_y: f32,
        p_z: f32,
        nx: u32,
        ny: u32,
        nz: u32,
        alpha_bits: u32,
    ) -> f32 {
        1.0 + f32::reinterpret(alpha_bits)
            * sample_density(density_field, p_x, p_y, p_z, nx, ny, nz)
    }

    #[cube]
    fn sample_density(
        density_field: &Array<f32>,
        p_x: f32,
        p_y: f32,
        p_z: f32,
        nx: u32,
        ny: u32,
        nz: u32,
    ) -> f32 {
        let x = wrap_coord(p_x, nx);
        let y = wrap_coord(p_y, ny);
        let z = wrap_coord(p_z, nz);
        let x0 = x as u32;
        let y0 = y as u32;
        let z0 = z as u32;
        let x1 = (x0 + 1u32) % nx;
        let y1 = (y0 + 1u32) % ny;
        let z1 = (z0 + 1u32) % nz;
        let u = x - x0 as f32;
        let v = y - y0 as f32;
        let w = z - z0 as f32;

        let c000 = density_field[density_index(x0, y0, z0, nx, ny)];
        let c100 = density_field[density_index(x1, y0, z0, nx, ny)];
        let c010 = density_field[density_index(x0, y1, z0, nx, ny)];
        let c110 = density_field[density_index(x1, y1, z0, nx, ny)];
        let c001 = density_field[density_index(x0, y0, z1, nx, ny)];
        let c101 = density_field[density_index(x1, y0, z1, nx, ny)];
        let c011 = density_field[density_index(x0, y1, z1, nx, ny)];
        let c111 = density_field[density_index(x1, y1, z1, nx, ny)];

        let i00 = c000 * (1.0 - u) + c100 * u;
        let i10 = c010 * (1.0 - u) + c110 * u;
        let i01 = c001 * (1.0 - u) + c101 * u;
        let i11 = c011 * (1.0 - u) + c111 * u;
        let i0 = i00 * (1.0 - v) + i10 * v;
        let i1 = i01 * (1.0 - v) + i11 * v;
        i0 * (1.0 - w) + i1 * w
    }

    #[cube]
    fn wrap_coord(value: f32, limit: u32) -> f32 {
        let limit_f = limit as f32;
        let mut wrapped = value;
        while wrapped < 0.0 {
            wrapped += limit_f;
        }
        while wrapped >= limit_f {
            wrapped -= limit_f;
        }
        wrapped
    }

    #[cube]
    fn density_index(ix: u32, iy: u32, iz: u32, nx: u32, ny: u32) -> usize {
        (iz * (nx * ny) + iy * nx + ix) as usize
    }

    #[cube]
    fn inv_len_or_one(x: f32, y: f32, z: f32) -> f32 {
        let len_sq = x * x + y * y + z * z;
        let mut inv_len = 1.0 / f32::sqrt(len_sq);
        if len_sq < 1.0e-30 {
            inv_len = 1.0;
        }
        inv_len
    }

    pub fn trace_rays_cpu_reference_cubecl(
        config: AlgebraicLensingCubeclConfig,
        density_field: &[f32],
        initial_pos: &[GpuVec3],
        initial_dir: &[GpuVec3],
    ) -> Result<(Vec<GpuVec3>, Vec<GpuVec3>), String> {
        validate_trace_inputs(config, density_field, initial_pos, initial_dir)?;
        let mut final_pos = Vec::with_capacity(initial_pos.len());
        let mut final_dir = Vec::with_capacity(initial_pos.len());
        for (&pos, &dir) in initial_pos.iter().zip(initial_dir.iter()) {
            let (p, d) = trace_one_cpu(config, density_field, pos, dir);
            final_pos.push(p);
            final_dir.push(d);
        }
        Ok((final_pos, final_dir))
    }

    fn trace_one_cpu(
        config: AlgebraicLensingCubeclConfig,
        density_field: &[f32],
        initial_pos: GpuVec3,
        initial_dir: GpuVec3,
    ) -> (GpuVec3, GpuVec3) {
        let mut p = initial_pos;
        let mut d = initial_dir;
        for _ in 0..config.max_steps {
            let (k1_p, k1_d) = get_derivatives_cpu(config, density_field, p, d);
            let p2 = add_vec3(p, scale_vec3(k1_p, config.dt * 0.5));
            let d2 = normalize_cpu(add_vec3(d, scale_vec3(k1_d, config.dt * 0.5)));
            let (k2_p, k2_d) = get_derivatives_cpu(config, density_field, p2, d2);

            let p3 = add_vec3(p, scale_vec3(k2_p, config.dt * 0.5));
            let d3 = normalize_cpu(add_vec3(d, scale_vec3(k2_d, config.dt * 0.5)));
            let (k3_p, k3_d) = get_derivatives_cpu(config, density_field, p3, d3);

            let p4 = add_vec3(p, scale_vec3(k3_p, config.dt));
            let d4 = normalize_cpu(add_vec3(d, scale_vec3(k3_d, config.dt)));
            let (k4_p, k4_d) = get_derivatives_cpu(config, density_field, p4, d4);

            p = add_vec3(
                p,
                scale_vec3(
                    add_vec3(
                        add_vec3(k1_p, scale_vec3(k2_p, 2.0)),
                        add_vec3(scale_vec3(k3_p, 2.0), k4_p),
                    ),
                    config.dt / 6.0,
                ),
            );
            d = normalize_cpu(add_vec3(
                d,
                scale_vec3(
                    add_vec3(
                        add_vec3(k1_d, scale_vec3(k2_d, 2.0)),
                        add_vec3(scale_vec3(k3_d, 2.0), k4_d),
                    ),
                    config.dt / 6.0,
                ),
            ));

            if p.x < -100.0
                || p.x > 200.0
                || p.y < -100.0
                || p.y > 200.0
                || p.z < -100.0
                || p.z > 200.0
            {
                break;
            }
        }
        (p, d)
    }

    fn get_derivatives_cpu(
        config: AlgebraicLensingCubeclConfig,
        density_field: &[f32],
        p: GpuVec3,
        d: GpuVec3,
    ) -> (GpuVec3, GpuVec3) {
        let (grad_n, n) = get_gradient_n_cpu(config, density_field, p);
        let d_dot_grad = dot_vec3(d, grad_n);
        let dp = d;
        let dd = scale_vec3(sub_vec3(grad_n, scale_vec3(d, d_dot_grad)), 1.0 / n);
        (dp, dd)
    }

    fn get_gradient_n_cpu(
        config: AlgebraicLensingCubeclConfig,
        density_field: &[f32],
        p: GpuVec3,
    ) -> (GpuVec3, f32) {
        let eps = config.eps_grad;
        let grad_x = (sample_n_cpu(config, density_field, GpuVec3 { x: p.x + eps, ..p })
            - sample_n_cpu(config, density_field, GpuVec3 { x: p.x - eps, ..p }))
            / (2.0 * eps);
        let grad_y = (sample_n_cpu(config, density_field, GpuVec3 { y: p.y + eps, ..p })
            - sample_n_cpu(config, density_field, GpuVec3 { y: p.y - eps, ..p }))
            / (2.0 * eps);
        let grad_z = (sample_n_cpu(config, density_field, GpuVec3 { z: p.z + eps, ..p })
            - sample_n_cpu(config, density_field, GpuVec3 { z: p.z - eps, ..p }))
            / (2.0 * eps);
        (
            GpuVec3 {
                x: grad_x,
                y: grad_y,
                z: grad_z,
            },
            sample_n_cpu(config, density_field, p),
        )
    }

    fn sample_n_cpu(
        config: AlgebraicLensingCubeclConfig,
        density_field: &[f32],
        p: GpuVec3,
    ) -> f32 {
        1.0 + config.alpha * sample_density_cpu(config, density_field, p)
    }

    fn sample_density_cpu(
        config: AlgebraicLensingCubeclConfig,
        density_field: &[f32],
        p: GpuVec3,
    ) -> f32 {
        let x = wrap_coord_cpu(p.x, config.nx);
        let y = wrap_coord_cpu(p.y, config.ny);
        let z = wrap_coord_cpu(p.z, config.nz);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let z0 = z.floor() as usize;
        let x1 = (x0 + 1) % config.nx;
        let y1 = (y0 + 1) % config.ny;
        let z1 = (z0 + 1) % config.nz;
        let u = x - x.floor();
        let v = y - y.floor();
        let w = z - z.floor();

        let get = |ix: usize, iy: usize, iz: usize| -> f32 {
            density_field[iz * (config.nx * config.ny) + iy * config.nx + ix]
        };
        let c000 = get(x0, y0, z0);
        let c100 = get(x1, y0, z0);
        let c010 = get(x0, y1, z0);
        let c110 = get(x1, y1, z0);
        let c001 = get(x0, y0, z1);
        let c101 = get(x1, y0, z1);
        let c011 = get(x0, y1, z1);
        let c111 = get(x1, y1, z1);
        let i00 = c000 * (1.0 - u) + c100 * u;
        let i10 = c010 * (1.0 - u) + c110 * u;
        let i01 = c001 * (1.0 - u) + c101 * u;
        let i11 = c011 * (1.0 - u) + c111 * u;
        let i0 = i00 * (1.0 - v) + i10 * v;
        let i1 = i01 * (1.0 - v) + i11 * v;
        i0 * (1.0 - w) + i1 * w
    }

    fn validate_trace_inputs(
        config: AlgebraicLensingCubeclConfig,
        density_field: &[f32],
        initial_pos: &[GpuVec3],
        initial_dir: &[GpuVec3],
    ) -> Result<(), String> {
        config.validate()?;
        if density_field.len() != config.n_cells() {
            return Err(format!(
                "density length mismatch: got {}, expected {}",
                density_field.len(),
                config.n_cells()
            ));
        }
        if initial_pos.is_empty() {
            return Err("at least one ray is required".to_string());
        }
        if initial_pos.len() != initial_dir.len() {
            return Err(format!(
                "ray position/direction length mismatch: {} vs {}",
                initial_pos.len(),
                initial_dir.len()
            ));
        }
        if initial_pos.len() > u32::MAX as usize {
            return Err(format!("ray count exceeds u32: {}", initial_pos.len()));
        }
        if density_field.iter().any(|value| !value.is_finite()) {
            return Err("density field contains a non-finite value".to_string());
        }
        Ok(())
    }

    fn dispatch_groups(n_rays: usize) -> Result<u32, String> {
        Ok(u32::try_from(n_rays)
            .map_err(|_| format!("ray count exceeds u32: {n_rays}"))?
            .div_ceil(WORKGROUP_SIZE))
    }

    fn flatten_vec3(values: &[GpuVec3]) -> Vec<f32> {
        let mut out = Vec::with_capacity(values.len() * 3);
        for value in values {
            out.push(value.x);
            out.push(value.y);
            out.push(value.z);
        }
        out
    }

    fn inflate_vec3(values: &[f32]) -> Vec<GpuVec3> {
        values
            .chunks_exact(3)
            .map(|chunk| GpuVec3 {
                x: chunk[0],
                y: chunk[1],
                z: chunk[2],
            })
            .collect()
    }

    fn wrap_coord_cpu(value: f32, limit: usize) -> f32 {
        let limit_f = limit as f32;
        let mut wrapped = value - (value / limit_f).floor() * limit_f;
        if wrapped < 0.0 {
            wrapped += limit_f;
        }
        if wrapped >= limit_f {
            wrapped = 0.0;
        }
        wrapped
    }

    fn add_vec3(a: GpuVec3, b: GpuVec3) -> GpuVec3 {
        GpuVec3 {
            x: a.x + b.x,
            y: a.y + b.y,
            z: a.z + b.z,
        }
    }

    fn sub_vec3(a: GpuVec3, b: GpuVec3) -> GpuVec3 {
        GpuVec3 {
            x: a.x - b.x,
            y: a.y - b.y,
            z: a.z - b.z,
        }
    }

    fn scale_vec3(v: GpuVec3, scale: f32) -> GpuVec3 {
        GpuVec3 {
            x: v.x * scale,
            y: v.y * scale,
            z: v.z * scale,
        }
    }

    fn dot_vec3(a: GpuVec3, b: GpuVec3) -> f32 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    fn normalize_cpu(v: GpuVec3) -> GpuVec3 {
        let len_sq = dot_vec3(v, v);
        if len_sq < 1.0e-30 {
            v
        } else {
            scale_vec3(v, len_sq.sqrt().recip())
        }
    }

    fn encode_f32_slice(values: &[f32]) -> Result<Vec<u8>, String> {
        let byte_len = values
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "algebraic lensing cubecl buffer size overflows".to_string())?;
        let mut bytes = Vec::with_capacity(byte_len);
        for &value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        Ok(bytes)
    }

    fn decode_f32_output(bytes: &[u8], output_len: usize, label: &str) -> Result<Vec<f32>, String> {
        let expected_bytes = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("algebraic lensing cubecl {label} length overflows bytes"))?;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "algebraic lensing cubecl {label} readback returned {} bytes, expected {expected_bytes}",
                bytes.len()
            ));
        }

        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    pub use AlgebraicLensingCubecl as PublicAlgebraicLensingCubecl;
    pub use AlgebraicLensingCubeclConfig as PublicAlgebraicLensingCubeclConfig;
    pub use trace_rays_cpu_reference_cubecl as public_trace_rays_cpu_reference_cubecl;

    #[cfg(test)]
    mod tests {
        use super::*;

        fn fixture_config() -> AlgebraicLensingCubeclConfig {
            AlgebraicLensingCubeclConfig::new(8, 8, 8, 1.0, 0.1, 10).unwrap()
        }

        fn gradient_density(config: AlgebraicLensingCubeclConfig) -> Vec<f32> {
            let mut density = vec![0.0f32; config.n_cells()];
            for z in 0..config.nz {
                for y in 0..config.ny {
                    for x in 0..config.nx {
                        let idx = z * (config.nx * config.ny) + y * config.nx + x;
                        density[idx] = 0.2 + 0.01 * x as f32 + 0.005 * y as f32;
                    }
                }
            }
            density
        }

        fn fixture_rays() -> (Vec<GpuVec3>, Vec<GpuVec3>) {
            (
                vec![
                    GpuVec3 {
                        x: 4.0,
                        y: 4.0,
                        z: 0.0,
                    },
                    GpuVec3 {
                        x: 2.5,
                        y: 3.5,
                        z: 1.0,
                    },
                ],
                vec![
                    GpuVec3 {
                        x: 0.0,
                        y: 0.0,
                        z: 1.0,
                    },
                    normalize_cpu(GpuVec3 {
                        x: 0.1,
                        y: 0.0,
                        z: 1.0,
                    }),
                ],
            )
        }

        #[test]
        fn algebraic_lensing_cubecl_available_does_not_panic() {
            let _ = AlgebraicLensingCubecl::is_available();
        }

        #[test]
        fn cubecl_cpu_reference_uniform_density_advances_rays() {
            let config = fixture_config();
            let density = vec![0.375f32; config.n_cells()];
            let pos = vec![GpuVec3 {
                x: 4.0,
                y: 4.0,
                z: 0.0,
            }];
            let dir = vec![GpuVec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            }];
            let (final_pos, final_dir) =
                trace_rays_cpu_reference_cubecl(config, &density, &pos, &dir).unwrap();
            assert!((final_pos[0].z - 1.0).abs() < 1.0e-5);
            assert!((final_dir[0].z - 1.0).abs() < 1.0e-5);
        }

        #[test]
        fn cubecl_invalid_trace_inputs_are_rejected() {
            let config = fixture_config();
            let density = vec![0.0f32; config.n_cells()];
            let (pos, dir) = fixture_rays();
            assert!(
                trace_rays_cpu_reference_cubecl(config, &density[..density.len() - 1], &pos, &dir)
                    .is_err()
            );
            assert!(trace_rays_cpu_reference_cubecl(config, &density, &[], &[]).is_err());
            assert!(trace_rays_cpu_reference_cubecl(config, &density, &pos, &dir[..1]).is_err());
        }

        #[test]
        fn cubecl_f32_roundtrip_decodes_outputs() {
            let values = [0.0f32, 1.0, -2.5, 4.25];
            let bytes = encode_f32_slice(&values).unwrap();
            assert_eq!(
                decode_f32_output(&bytes, values.len(), "test").unwrap(),
                values
            );
        }

        #[test]
        fn algebraic_lensing_cubecl_matches_cpu_when_adapter_available() {
            if !AlgebraicLensingCubecl::is_available() {
                return;
            }

            let config = fixture_config();
            let density = gradient_density(config);
            let (pos, dir) = fixture_rays();
            let cpu = trace_rays_cpu_reference_cubecl(config, &density, &pos, &dir).unwrap();
            let cubecl = AlgebraicLensingCubecl::trace_rays(config, &density, &pos, &dir).unwrap();
            assert_vec3_close(&cpu.0, &cubecl.0);
            assert_vec3_close(&cpu.1, &cubecl.1);
        }

        fn assert_vec3_close(expected: &[GpuVec3], observed: &[GpuVec3]) {
            assert_eq!(expected.len(), observed.len());
            for (idx, (lhs, rhs)) in expected.iter().zip(observed.iter()).enumerate() {
                for (component, lhs_value, rhs_value) in [
                    ("x", lhs.x, rhs.x),
                    ("y", lhs.y, rhs.y),
                    ("z", lhs.z, rhs.z),
                ] {
                    let scale = lhs_value.abs().max(1.0);
                    let rel = (lhs_value - rhs_value).abs() / scale;
                    assert!(
                        rel < 2.0e-4,
                        "ray {idx} component {component}: cpu={lhs_value}, cubecl={rhs_value}, rel={rel}"
                    );
                }
            }
        }
    }
}

#[cfg(feature = "cubecl")]
pub use cubecl_backend::{
    PublicAlgebraicLensingCubecl as AlgebraicLensingCubecl,
    PublicAlgebraicLensingCubeclConfig as AlgebraicLensingCubeclConfig,
    public_trace_rays_cpu_reference_cubecl as trace_rays_cpu_reference_cubecl,
};

#[cfg(feature = "gpu")]
unsafe impl DeviceRepr for GpuVec3 {}
#[cfg(feature = "gpu")]
unsafe impl ValidAsZeroBits for GpuVec3 {}

#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug)]
pub struct AlgebraicLensingGpuConfig {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub alpha: f32,
    pub dt: f32,
    pub max_steps: usize,
    pub eps_grad: f32,
}

#[cfg(feature = "gpu")]
impl AlgebraicLensingGpuConfig {
    pub fn new(nx: usize, ny: usize, nz: usize, alpha: f32, dt: f32, max_steps: usize) -> Self {
        Self {
            nx,
            ny,
            nz,
            alpha,
            dt,
            max_steps,
            eps_grad: 0.1,
        }
    }
}

/// GPU-accelerated Algebraic Lensing simulation.
#[cfg(feature = "gpu")]
pub struct AlgebraicLensingGpu {
    _ctx: Context,
    stream: Arc<CudaStream>,
    trace_kernel: KernelHandle,
}

#[cfg(feature = "gpu")]
impl AlgebraicLensingGpu {
    /// Initialize CUDA context and compile kernels.
    pub fn new() -> anyhow::Result<Self> {
        let ctx = Context::with_default_device()?;
        let stream = ctx.default_stream();

        let opts = CompileOptions::for_arch(8, 9).include_path("/opt/cuda/include");
        let ptx = CompileOptions::compile_ptx(KERNEL_SRC, &opts)?;
        let registry = ModuleRegistry::load(ctx.raw(), ptx, &["trace_rays_kernel"])?;
        let trace_kernel = registry.get("trace_rays_kernel")?;

        Ok(Self {
            _ctx: ctx,
            stream,
            trace_kernel,
        })
    }

    /// Trace a batch of rays through the algebraic vacuum.
    ///
    /// # Arguments
    /// * `density_field` - Precomputed imbalance density (nx * ny * nz)
    /// * `config` - Grid dimensions and RK4 parameters
    /// * `initial_pos` - Starting positions
    /// * `initial_dir` - Starting unit directions
    pub fn trace_rays(
        &self,
        config: AlgebraicLensingGpuConfig,
        density_field: &[f32],
        initial_pos: &[GpuVec3],
        initial_dir: &[GpuVec3],
    ) -> anyhow::Result<(Vec<GpuVec3>, Vec<GpuVec3>)> {
        let n_rays = initial_pos.len();

        let d_density = Buffer::htod(&self.stream, density_field)?;
        let d_initial_pos = Buffer::htod(&self.stream, initial_pos)?;
        let d_initial_dir = Buffer::htod(&self.stream, initial_dir)?;
        let mut d_final_pos = Buffer::alloc_zeros(&self.stream, n_rays)?;
        let mut d_final_dir = Buffer::alloc_zeros(&self.stream, n_rays)?;

        let launch_config = LaunchConfig::launch_1d(n_rays as u32);

        let nx_i = config.nx as i32;
        let ny_i = config.ny as i32;
        let nz_i = config.nz as i32;
        let n_rays_i = n_rays as i32;
        let max_steps_i = config.max_steps as i32;

        let mut builder = self.stream.launch_builder(&self.trace_kernel);
        builder
            .arg(d_density.raw())
            .arg(&nx_i)
            .arg(&ny_i)
            .arg(&nz_i)
            .arg(d_initial_pos.raw())
            .arg(d_initial_dir.raw())
            .arg(d_final_pos.raw_mut())
            .arg(d_final_dir.raw_mut())
            .arg(&n_rays_i)
            .arg(&config.alpha)
            .arg(&config.dt)
            .arg(&max_steps_i)
            .arg(&config.eps_grad);

        unsafe { builder.launch(launch_config) }?;

        Ok((d_final_pos.dtoh_vec()?, d_final_dir.dtoh_vec()?))
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::GpuVec3;
    use gororoba_gpu_vulkan::{
        Adapter, ComputePipeline, ComputePipelineBuilder, DescriptorPool, DescriptorSet,
        DescriptorSetLayout, DescriptorSetLayoutSpec, Device, DeviceBuilder, DispatchScope,
        HostVisibleBuffer, Instance, InstanceBuilder, QueueFamilyRequirement, ShaderModule,
        ValidationPolicy,
    };

    const WORKGROUP_SIZE: u32 = 256;
    const DISPATCH_TIMEOUT_NS: u64 = 30_000_000_000;
    const ENTRY_POINT: &str = "trace_rays_kernel";

    pub const ALGEBRAIC_LENSING_VULKAN_WGSL: &str = r#"
struct F32Buffer {
    values: array<f32>,
};

struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    n_rays: u32,
    max_steps: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    alpha: f32,
    dt: f32,
    eps_grad: f32,
    pad3: f32,
};

@group(0) @binding(0)
var<storage, read> density_field: F32Buffer;
@group(0) @binding(1)
var<storage, read> initial_pos: F32Buffer;
@group(0) @binding(2)
var<storage, read> initial_dir: F32Buffer;
@group(0) @binding(3)
var<storage, read_write> final_pos: F32Buffer;
@group(0) @binding(4)
var<storage, read_write> final_dir: F32Buffer;
@group(0) @binding(5)
var<uniform> params: Params;

fn ray_index(ray: u32, component: u32) -> u32 {
    return ray * 3u + component;
}

fn load_initial_pos(ray: u32) -> vec3<f32> {
    return vec3<f32>(
        initial_pos.values[ray_index(ray, 0u)],
        initial_pos.values[ray_index(ray, 1u)],
        initial_pos.values[ray_index(ray, 2u)]
    );
}

fn load_initial_dir(ray: u32) -> vec3<f32> {
    return vec3<f32>(
        initial_dir.values[ray_index(ray, 0u)],
        initial_dir.values[ray_index(ray, 1u)],
        initial_dir.values[ray_index(ray, 2u)]
    );
}

fn normalize_safe(v: vec3<f32>) -> vec3<f32> {
    let len_sq: f32 = dot(v, v);
    if (len_sq < 1.0e-30) {
        return v;
    }
    return v * inverseSqrt(len_sq);
}

fn wrap_coord(value: f32, limit: u32) -> f32 {
    let limit_f: f32 = f32(limit);
    var wrapped: f32 = value - floor(value / limit_f) * limit_f;
    if (wrapped < 0.0) {
        wrapped = wrapped + limit_f;
    }
    if (wrapped >= limit_f) {
        wrapped = 0.0;
    }
    return wrapped;
}

fn density_index(ix: u32, iy: u32, iz: u32) -> u32 {
    return iz * (params.nx * params.ny) + iy * params.nx + ix;
}

fn sample_density(p: vec3<f32>) -> f32 {
    let x: f32 = wrap_coord(p.x, params.nx);
    let y: f32 = wrap_coord(p.y, params.ny);
    let z: f32 = wrap_coord(p.z, params.nz);

    let x0: u32 = u32(floor(x));
    let y0: u32 = u32(floor(y));
    let z0: u32 = u32(floor(z));
    let x1: u32 = (x0 + 1u) % params.nx;
    let y1: u32 = (y0 + 1u) % params.ny;
    let z1: u32 = (z0 + 1u) % params.nz;

    let u: f32 = x - floor(x);
    let v: f32 = y - floor(y);
    let w: f32 = z - floor(z);

    let c000: f32 = density_field.values[density_index(x0, y0, z0)];
    let c100: f32 = density_field.values[density_index(x1, y0, z0)];
    let c010: f32 = density_field.values[density_index(x0, y1, z0)];
    let c110: f32 = density_field.values[density_index(x1, y1, z0)];
    let c001: f32 = density_field.values[density_index(x0, y0, z1)];
    let c101: f32 = density_field.values[density_index(x1, y0, z1)];
    let c011: f32 = density_field.values[density_index(x0, y1, z1)];
    let c111: f32 = density_field.values[density_index(x1, y1, z1)];

    let i00: f32 = c000 * (1.0 - u) + c100 * u;
    let i10: f32 = c010 * (1.0 - u) + c110 * u;
    let i01: f32 = c001 * (1.0 - u) + c101 * u;
    let i11: f32 = c011 * (1.0 - u) + c111 * u;
    let i0: f32 = i00 * (1.0 - v) + i10 * v;
    let i1: f32 = i01 * (1.0 - v) + i11 * v;
    return i0 * (1.0 - w) + i1 * w;
}

fn sample_n(p: vec3<f32>) -> f32 {
    return 1.0 + params.alpha * sample_density(p);
}

fn gradient_and_n(p: vec3<f32>) -> vec4<f32> {
    let eps: f32 = params.eps_grad;
    let grad_x: f32 = (
        sample_n(vec3<f32>(p.x + eps, p.y, p.z)) -
        sample_n(vec3<f32>(p.x - eps, p.y, p.z))
    ) / (2.0 * eps);
    let grad_y: f32 = (
        sample_n(vec3<f32>(p.x, p.y + eps, p.z)) -
        sample_n(vec3<f32>(p.x, p.y - eps, p.z))
    ) / (2.0 * eps);
    let grad_z: f32 = (
        sample_n(vec3<f32>(p.x, p.y, p.z + eps)) -
        sample_n(vec3<f32>(p.x, p.y, p.z - eps))
    ) / (2.0 * eps);
    return vec4<f32>(grad_x, grad_y, grad_z, sample_n(p));
}

fn curvature(p: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    let sample: vec4<f32> = gradient_and_n(p);
    let grad: vec3<f32> = sample.xyz;
    let d_dot_grad: f32 = dot(d, grad);
    return (grad - d * d_dot_grad) * (1.0 / sample.w);
}

@compute @workgroup_size(256)
fn trace_rays_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;
    if (idx >= params.n_rays) {
        return;
    }

    var p: vec3<f32> = load_initial_pos(idx);
    var d: vec3<f32> = load_initial_dir(idx);

    for (var step: u32 = 0u; step < params.max_steps; step = step + 1u) {
        let k1_p: vec3<f32> = d;
        let k1_d: vec3<f32> = curvature(p, d);
        let p2: vec3<f32> = p + k1_p * (params.dt * 0.5);
        let d2: vec3<f32> = normalize_safe(d + k1_d * (params.dt * 0.5));
        let k2_p: vec3<f32> = d2;
        let k2_d: vec3<f32> = curvature(p2, d2);

        let p3: vec3<f32> = p + k2_p * (params.dt * 0.5);
        let d3: vec3<f32> = normalize_safe(d + k2_d * (params.dt * 0.5));
        let k3_p: vec3<f32> = d3;
        let k3_d: vec3<f32> = curvature(p3, d3);

        let p4: vec3<f32> = p + k3_p * params.dt;
        let d4: vec3<f32> = normalize_safe(d + k3_d * params.dt);
        let k4_p: vec3<f32> = d4;
        let k4_d: vec3<f32> = curvature(p4, d4);

        p = p + (k1_p + k2_p * 2.0 + k3_p * 2.0 + k4_p) * (params.dt / 6.0);
        d = normalize_safe(d + (k1_d + k2_d * 2.0 + k3_d * 2.0 + k4_d) * (params.dt / 6.0));

        if (p.x < -100.0 || p.x > 200.0 || p.y < -100.0 || p.y > 200.0 || p.z < -100.0 || p.z > 200.0) {
            break;
        }
    }

    let base: u32 = idx * 3u;
    final_pos.values[base + 0u] = p.x;
    final_pos.values[base + 1u] = p.y;
    final_pos.values[base + 2u] = p.z;
    final_dir.values[base + 0u] = d.x;
    final_dir.values[base + 1u] = d.y;
    final_dir.values[base + 2u] = d.z;
}
"#;

    #[derive(Clone, Copy, Debug)]
    pub struct AlgebraicLensingVulkanConfig {
        pub nx: usize,
        pub ny: usize,
        pub nz: usize,
        pub alpha: f32,
        pub dt: f32,
        pub max_steps: usize,
        pub eps_grad: f32,
    }

    impl AlgebraicLensingVulkanConfig {
        pub fn new(
            nx: usize,
            ny: usize,
            nz: usize,
            alpha: f32,
            dt: f32,
            max_steps: usize,
        ) -> Result<Self, String> {
            let config = Self {
                nx,
                ny,
                nz,
                alpha,
                dt,
                max_steps,
                eps_grad: 0.1,
            };
            config.validate()?;
            Ok(config)
        }

        pub fn n_cells(&self) -> usize {
            self.nx * self.ny * self.nz
        }

        fn validate(&self) -> Result<(), String> {
            if self.nx == 0 || self.ny == 0 || self.nz == 0 {
                return Err(format!(
                    "grid dimensions must be positive, got {}x{}x{}",
                    self.nx, self.ny, self.nz
                ));
            }
            let n_cells = (self.nx as u64)
                .checked_mul(self.ny as u64)
                .and_then(|value| value.checked_mul(self.nz as u64))
                .ok_or_else(|| "density grid cell count overflows u64".to_string())?;
            if n_cells > u32::MAX as u64 {
                return Err(format!("density grid exceeds u32 indexing: {n_cells}"));
            }
            if !(self.alpha.is_finite() && self.dt.is_finite() && self.dt > 0.0) {
                return Err(format!(
                    "alpha must be finite and dt must be finite positive, got alpha={}, dt={}",
                    self.alpha, self.dt
                ));
            }
            if !(self.eps_grad.is_finite() && self.eps_grad > 0.0) {
                return Err(format!(
                    "eps_grad must be finite and positive, got {}",
                    self.eps_grad
                ));
            }
            if self.max_steps == 0 {
                return Err("max_steps must be at least 1".to_string());
            }
            if self.max_steps > u32::MAX as usize {
                return Err(format!("max_steps exceeds u32: {}", self.max_steps));
            }
            Ok(())
        }
    }

    pub struct AlgebraicLensingVulkan {
        pipeline: ComputePipeline,
        descriptor_layout: DescriptorSetLayout,
        device: Device,
        adapter: Adapter,
        #[allow(dead_code)]
        instance: Instance,
    }

    impl AlgebraicLensingVulkan {
        pub fn new() -> Result<Self, String> {
            let (instance, adapter, device) = Self::build_context()?;
            let (pipeline, descriptor_layout) = Self::build_pipeline(&device)?;

            Ok(Self {
                pipeline,
                descriptor_layout,
                device,
                adapter,
                instance,
            })
        }

        fn build_context() -> Result<(Instance, Adapter, Device), String> {
            let instance = InstanceBuilder::new("optics_core_algebraic_lensing_vulkan")
                .validation(ValidationPolicy::Enable)
                .build()
                .map_err(|e| format!("algebraic lensing Vulkan instance creation failed: {e}"))?;
            let adapter = Adapter::pick(&instance, QueueFamilyRequirement::Compute)
                .map_err(|e| format!("algebraic lensing Vulkan adapter pick failed: {e}"))?;
            let device = DeviceBuilder::new(adapter.clone())
                .build(&instance)
                .map_err(|e| format!("algebraic lensing Vulkan device creation failed: {e}"))?;
            Ok((instance, adapter, device))
        }

        fn build_pipeline(
            device: &Device,
        ) -> Result<(ComputePipeline, DescriptorSetLayout), String> {
            let shader = Self::build_shader(device)?;
            let descriptor_layout = DescriptorSetLayoutSpec::new()
                .storage_buffer(0)
                .storage_buffer(1)
                .storage_buffer(2)
                .storage_buffer(3)
                .storage_buffer(4)
                .uniform_buffer(5)
                .build(device)
                .map_err(|e| format!("algebraic lensing descriptor layout failed: {e}"))?;
            let pipeline = ComputePipelineBuilder::new(device, &shader)
                .descriptor_layout(&descriptor_layout)
                .build()
                .map_err(|e| format!("algebraic lensing compute pipeline build failed: {e}"))?;

            Ok((pipeline, descriptor_layout))
        }

        fn build_shader(device: &Device) -> Result<ShaderModule, String> {
            ShaderModule::from_wgsl(device, ALGEBRAIC_LENSING_VULKAN_WGSL, ENTRY_POINT)
                .map_err(|e| format!("algebraic lensing WGSL compile failed: {e}"))
        }

        pub fn wgsl_source() -> &'static str {
            ALGEBRAIC_LENSING_VULKAN_WGSL
        }

        pub fn trace_rays(
            &self,
            config: AlgebraicLensingVulkanConfig,
            density_field: &[f32],
            initial_pos: &[GpuVec3],
            initial_dir: &[GpuVec3],
        ) -> Result<(Vec<GpuVec3>, Vec<GpuVec3>), String> {
            validate_trace_inputs(config, density_field, initial_pos, initial_dir)?;
            let buffers = VulkanTraceBuffers::new(
                &self.device,
                &self.adapter,
                config,
                density_field,
                initial_pos,
                initial_dir,
            )?;

            let descriptor_pool =
                DescriptorPool::for_layout(&self.device, &self.descriptor_layout, 1)
                    .map_err(|e| format!("descriptor pool allocation failed: {e}"))?;
            let descriptor_set = descriptor_pool
                .allocate_set(&self.descriptor_layout)
                .map_err(|e| format!("descriptor set allocation failed: {e}"))?;
            buffers.write_descriptor_set(&descriptor_set);

            let dispatch = DispatchScope::new(&self.device)
                .map_err(|e| format!("dispatch scope creation failed: {e}"))?;
            dispatch
                .dispatch(
                    &self.pipeline,
                    descriptor_set.raw(),
                    dispatch_groups(buffers.n_rays)?,
                    1,
                    1,
                    DISPATCH_TIMEOUT_NS,
                )
                .map_err(|e| format!("algebraic lensing Vulkan dispatch failed: {e}"))?;

            buffers.read_output()
        }
    }

    struct VulkanTraceBuffers {
        density: HostVisibleBuffer,
        pos_in: HostVisibleBuffer,
        dir_in: HostVisibleBuffer,
        pos_out: HostVisibleBuffer,
        dir_out: HostVisibleBuffer,
        params: HostVisibleBuffer,
        n_rays: usize,
    }

    impl VulkanTraceBuffers {
        fn new(
            device: &Device,
            adapter: &Adapter,
            config: AlgebraicLensingVulkanConfig,
            density_field: &[f32],
            initial_pos: &[GpuVec3],
            initial_dir: &[GpuVec3],
        ) -> Result<Self, String> {
            let n_rays = initial_pos.len();
            let density = allocate_storage_f32(device, adapter, density_field.len(), "density")?;
            let pos_in = allocate_storage_f32(device, adapter, 3 * n_rays, "initial_pos")?;
            let dir_in = allocate_storage_f32(device, adapter, 3 * n_rays, "initial_dir")?;
            let pos_out = allocate_storage_f32(device, adapter, 3 * n_rays, "final_pos")?;
            let dir_out = allocate_storage_f32(device, adapter, 3 * n_rays, "final_dir")?;
            let params = HostVisibleBuffer::uniform(device, adapter, 48)
                .map_err(|e| format!("algebraic lensing params allocation failed: {e}"))?;

            density
                .write_f32_slice(density_field)
                .map_err(|e| format!("density upload failed: {e}"))?;
            pos_in
                .write_f32_slice(&flatten_vec3(initial_pos))
                .map_err(|e| format!("initial position upload failed: {e}"))?;
            dir_in
                .write_f32_slice(&flatten_vec3(initial_dir))
                .map_err(|e| format!("initial direction upload failed: {e}"))?;
            let zeroed = vec![0.0f32; 3 * n_rays];
            pos_out
                .write_f32_slice(&zeroed)
                .map_err(|e| format!("final position initialization failed: {e}"))?;
            dir_out
                .write_f32_slice(&zeroed)
                .map_err(|e| format!("final direction initialization failed: {e}"))?;
            params
                .write_bytes(&encode_params(config, n_rays)?)
                .map_err(|e| format!("params upload failed: {e}"))?;

            Ok(Self {
                density,
                pos_in,
                dir_in,
                pos_out,
                dir_out,
                params,
                n_rays,
            })
        }

        fn write_descriptor_set(&self, descriptor_set: &DescriptorSet) {
            descriptor_set.write_storage_buffer(0, &self.density);
            descriptor_set.write_storage_buffer(1, &self.pos_in);
            descriptor_set.write_storage_buffer(2, &self.dir_in);
            descriptor_set.write_storage_buffer(3, &self.pos_out);
            descriptor_set.write_storage_buffer(4, &self.dir_out);
            descriptor_set.write_uniform_buffer(5, &self.params);
        }

        fn read_output(&self) -> Result<(Vec<GpuVec3>, Vec<GpuVec3>), String> {
            let len = 3 * self.n_rays;
            let final_pos = inflate_vec3(
                &self
                    .pos_out
                    .read_f32_slice(len)
                    .map_err(|e| format!("final position readback failed: {e}"))?,
            );
            let final_dir = inflate_vec3(
                &self
                    .dir_out
                    .read_f32_slice(len)
                    .map_err(|e| format!("final direction readback failed: {e}"))?,
            );
            Ok((final_pos, final_dir))
        }
    }

    pub fn trace_rays_cpu_reference(
        config: AlgebraicLensingVulkanConfig,
        density_field: &[f32],
        initial_pos: &[GpuVec3],
        initial_dir: &[GpuVec3],
    ) -> Result<(Vec<GpuVec3>, Vec<GpuVec3>), String> {
        validate_trace_inputs(config, density_field, initial_pos, initial_dir)?;
        let mut final_pos = Vec::with_capacity(initial_pos.len());
        let mut final_dir = Vec::with_capacity(initial_pos.len());
        for (&pos, &dir) in initial_pos.iter().zip(initial_dir.iter()) {
            let (p, d) = trace_one_cpu(config, density_field, pos, dir);
            final_pos.push(p);
            final_dir.push(d);
        }
        Ok((final_pos, final_dir))
    }

    fn trace_one_cpu(
        config: AlgebraicLensingVulkanConfig,
        density_field: &[f32],
        initial_pos: GpuVec3,
        initial_dir: GpuVec3,
    ) -> (GpuVec3, GpuVec3) {
        let mut p = initial_pos;
        let mut d = initial_dir;
        for _ in 0..config.max_steps {
            let (k1_p, k1_d) = get_derivatives_cpu(config, density_field, p, d);
            let p2 = add_vec3(p, scale_vec3(k1_p, config.dt * 0.5));
            let d2 = normalize_cpu(add_vec3(d, scale_vec3(k1_d, config.dt * 0.5)));
            let (k2_p, k2_d) = get_derivatives_cpu(config, density_field, p2, d2);

            let p3 = add_vec3(p, scale_vec3(k2_p, config.dt * 0.5));
            let d3 = normalize_cpu(add_vec3(d, scale_vec3(k2_d, config.dt * 0.5)));
            let (k3_p, k3_d) = get_derivatives_cpu(config, density_field, p3, d3);

            let p4 = add_vec3(p, scale_vec3(k3_p, config.dt));
            let d4 = normalize_cpu(add_vec3(d, scale_vec3(k3_d, config.dt)));
            let (k4_p, k4_d) = get_derivatives_cpu(config, density_field, p4, d4);

            p = add_vec3(
                p,
                scale_vec3(
                    add_vec3(
                        add_vec3(k1_p, scale_vec3(k2_p, 2.0)),
                        add_vec3(scale_vec3(k3_p, 2.0), k4_p),
                    ),
                    config.dt / 6.0,
                ),
            );
            d = normalize_cpu(add_vec3(
                d,
                scale_vec3(
                    add_vec3(
                        add_vec3(k1_d, scale_vec3(k2_d, 2.0)),
                        add_vec3(scale_vec3(k3_d, 2.0), k4_d),
                    ),
                    config.dt / 6.0,
                ),
            ));

            if p.x < -100.0
                || p.x > 200.0
                || p.y < -100.0
                || p.y > 200.0
                || p.z < -100.0
                || p.z > 200.0
            {
                break;
            }
        }
        (p, d)
    }

    fn get_derivatives_cpu(
        config: AlgebraicLensingVulkanConfig,
        density_field: &[f32],
        p: GpuVec3,
        d: GpuVec3,
    ) -> (GpuVec3, GpuVec3) {
        let (grad_n, n) = get_gradient_n_cpu(config, density_field, p);
        let d_dot_grad = dot_vec3(d, grad_n);
        let dp = d;
        let dd = scale_vec3(sub_vec3(grad_n, scale_vec3(d, d_dot_grad)), 1.0 / n);
        (dp, dd)
    }

    fn get_gradient_n_cpu(
        config: AlgebraicLensingVulkanConfig,
        density_field: &[f32],
        p: GpuVec3,
    ) -> (GpuVec3, f32) {
        let eps = config.eps_grad;
        let grad_x = (sample_n_cpu(config, density_field, GpuVec3 { x: p.x + eps, ..p })
            - sample_n_cpu(config, density_field, GpuVec3 { x: p.x - eps, ..p }))
            / (2.0 * eps);
        let grad_y = (sample_n_cpu(config, density_field, GpuVec3 { y: p.y + eps, ..p })
            - sample_n_cpu(config, density_field, GpuVec3 { y: p.y - eps, ..p }))
            / (2.0 * eps);
        let grad_z = (sample_n_cpu(config, density_field, GpuVec3 { z: p.z + eps, ..p })
            - sample_n_cpu(config, density_field, GpuVec3 { z: p.z - eps, ..p }))
            / (2.0 * eps);
        (
            GpuVec3 {
                x: grad_x,
                y: grad_y,
                z: grad_z,
            },
            sample_n_cpu(config, density_field, p),
        )
    }

    fn sample_n_cpu(
        config: AlgebraicLensingVulkanConfig,
        density_field: &[f32],
        p: GpuVec3,
    ) -> f32 {
        1.0 + config.alpha * sample_density_cpu(config, density_field, p)
    }

    fn sample_density_cpu(
        config: AlgebraicLensingVulkanConfig,
        density_field: &[f32],
        p: GpuVec3,
    ) -> f32 {
        let x = wrap_coord_cpu(p.x, config.nx);
        let y = wrap_coord_cpu(p.y, config.ny);
        let z = wrap_coord_cpu(p.z, config.nz);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let z0 = z.floor() as usize;
        let x1 = (x0 + 1) % config.nx;
        let y1 = (y0 + 1) % config.ny;
        let z1 = (z0 + 1) % config.nz;
        let u = x - x.floor();
        let v = y - y.floor();
        let w = z - z.floor();

        let get = |ix: usize, iy: usize, iz: usize| -> f32 {
            density_field[iz * (config.nx * config.ny) + iy * config.nx + ix]
        };
        let c000 = get(x0, y0, z0);
        let c100 = get(x1, y0, z0);
        let c010 = get(x0, y1, z0);
        let c110 = get(x1, y1, z0);
        let c001 = get(x0, y0, z1);
        let c101 = get(x1, y0, z1);
        let c011 = get(x0, y1, z1);
        let c111 = get(x1, y1, z1);
        let i00 = c000 * (1.0 - u) + c100 * u;
        let i10 = c010 * (1.0 - u) + c110 * u;
        let i01 = c001 * (1.0 - u) + c101 * u;
        let i11 = c011 * (1.0 - u) + c111 * u;
        let i0 = i00 * (1.0 - v) + i10 * v;
        let i1 = i01 * (1.0 - v) + i11 * v;
        i0 * (1.0 - w) + i1 * w
    }

    fn validate_trace_inputs(
        config: AlgebraicLensingVulkanConfig,
        density_field: &[f32],
        initial_pos: &[GpuVec3],
        initial_dir: &[GpuVec3],
    ) -> Result<(), String> {
        config.validate()?;
        if density_field.len() != config.n_cells() {
            return Err(format!(
                "density length mismatch: got {}, expected {}",
                density_field.len(),
                config.n_cells()
            ));
        }
        if initial_pos.is_empty() {
            return Err("at least one ray is required".to_string());
        }
        if initial_pos.len() != initial_dir.len() {
            return Err(format!(
                "ray position/direction length mismatch: {} vs {}",
                initial_pos.len(),
                initial_dir.len()
            ));
        }
        if initial_pos.len() > u32::MAX as usize {
            return Err(format!("ray count exceeds u32: {}", initial_pos.len()));
        }
        if density_field.iter().any(|value| !value.is_finite()) {
            return Err("density field contains a non-finite value".to_string());
        }
        Ok(())
    }

    fn encode_params(
        config: AlgebraicLensingVulkanConfig,
        n_rays: usize,
    ) -> Result<[u8; 48], String> {
        let words = [
            u32::try_from(config.nx)
                .map_err(|_| "nx does not fit u32".to_string())?
                .to_le_bytes(),
            u32::try_from(config.ny)
                .map_err(|_| "ny does not fit u32".to_string())?
                .to_le_bytes(),
            u32::try_from(config.nz)
                .map_err(|_| "nz does not fit u32".to_string())?
                .to_le_bytes(),
            u32::try_from(n_rays)
                .map_err(|_| "ray count does not fit u32".to_string())?
                .to_le_bytes(),
            u32::try_from(config.max_steps)
                .map_err(|_| "max_steps does not fit u32".to_string())?
                .to_le_bytes(),
            0u32.to_le_bytes(),
            0u32.to_le_bytes(),
            0u32.to_le_bytes(),
            config.alpha.to_le_bytes(),
            config.dt.to_le_bytes(),
            config.eps_grad.to_le_bytes(),
            0f32.to_le_bytes(),
        ];
        let mut bytes = [0u8; 48];
        for (idx, word) in words.iter().enumerate() {
            let start = idx * 4;
            bytes[start..start + 4].copy_from_slice(word);
        }
        Ok(bytes)
    }

    fn allocate_storage_f32(
        device: &Device,
        adapter: &Adapter,
        len: usize,
        label: &str,
    ) -> Result<HostVisibleBuffer, String> {
        let bytes =
            len.checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| format!("{label} byte length overflows usize"))? as u64;
        HostVisibleBuffer::storage(device, adapter, bytes)
            .map_err(|e| format!("{label} buffer allocation failed: {e}"))
    }

    fn dispatch_groups(n_rays: usize) -> Result<u32, String> {
        Ok(u32::try_from(n_rays)
            .map_err(|_| format!("ray count exceeds u32: {n_rays}"))?
            .div_ceil(WORKGROUP_SIZE))
    }

    fn flatten_vec3(values: &[GpuVec3]) -> Vec<f32> {
        let mut out = Vec::with_capacity(values.len() * 3);
        for value in values {
            out.push(value.x);
            out.push(value.y);
            out.push(value.z);
        }
        out
    }

    fn inflate_vec3(values: &[f32]) -> Vec<GpuVec3> {
        values
            .chunks_exact(3)
            .map(|chunk| GpuVec3 {
                x: chunk[0],
                y: chunk[1],
                z: chunk[2],
            })
            .collect()
    }

    fn wrap_coord_cpu(value: f32, limit: usize) -> f32 {
        let limit_f = limit as f32;
        let mut wrapped = value - (value / limit_f).floor() * limit_f;
        if wrapped < 0.0 {
            wrapped += limit_f;
        }
        if wrapped >= limit_f {
            wrapped = 0.0;
        }
        wrapped
    }

    fn add_vec3(a: GpuVec3, b: GpuVec3) -> GpuVec3 {
        GpuVec3 {
            x: a.x + b.x,
            y: a.y + b.y,
            z: a.z + b.z,
        }
    }

    fn sub_vec3(a: GpuVec3, b: GpuVec3) -> GpuVec3 {
        GpuVec3 {
            x: a.x - b.x,
            y: a.y - b.y,
            z: a.z - b.z,
        }
    }

    fn scale_vec3(v: GpuVec3, scale: f32) -> GpuVec3 {
        GpuVec3 {
            x: v.x * scale,
            y: v.y * scale,
            z: v.z * scale,
        }
    }

    fn dot_vec3(a: GpuVec3, b: GpuVec3) -> f32 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    fn normalize_cpu(v: GpuVec3) -> GpuVec3 {
        let len_sq = dot_vec3(v, v);
        if len_sq < 1.0e-30 {
            v
        } else {
            scale_vec3(v, len_sq.sqrt().recip())
        }
    }

    pub use AlgebraicLensingVulkan as PublicAlgebraicLensingVulkan;
    pub use AlgebraicLensingVulkanConfig as PublicAlgebraicLensingVulkanConfig;
    pub use trace_rays_cpu_reference as public_trace_rays_cpu_reference;

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::collections::BTreeMap;

        fn fixture_config() -> AlgebraicLensingVulkanConfig {
            AlgebraicLensingVulkanConfig::new(8, 8, 8, 1.0, 0.1, 10).unwrap()
        }

        fn gradient_density(config: AlgebraicLensingVulkanConfig) -> Vec<f32> {
            let mut density = vec![0.0f32; config.n_cells()];
            for z in 0..config.nz {
                for y in 0..config.ny {
                    for x in 0..config.nx {
                        let idx = z * (config.nx * config.ny) + y * config.nx + x;
                        density[idx] = 0.2 + 0.01 * x as f32 + 0.005 * y as f32;
                    }
                }
            }
            density
        }

        fn fixture_rays() -> (Vec<GpuVec3>, Vec<GpuVec3>) {
            (
                vec![
                    GpuVec3 {
                        x: 4.0,
                        y: 4.0,
                        z: 0.0,
                    },
                    GpuVec3 {
                        x: 2.5,
                        y: 3.5,
                        z: 1.0,
                    },
                ],
                vec![
                    GpuVec3 {
                        x: 0.0,
                        y: 0.0,
                        z: 1.0,
                    },
                    normalize_cpu(GpuVec3 {
                        x: 0.1,
                        y: 0.0,
                        z: 1.0,
                    }),
                ],
            )
        }

        #[test]
        fn algebraic_lensing_vulkan_wgsl_parses_and_emits_compute_spirv() {
            let module =
                naga::front::wgsl::parse_str(AlgebraicLensingVulkan::wgsl_source()).unwrap();
            let override_ids: BTreeMap<&str, u32> = module
                .overrides
                .iter()
                .filter_map(|(_, override_constant)| {
                    Some((
                        override_constant.name.as_deref()?,
                        u32::from(override_constant.id?),
                    ))
                })
                .collect();
            assert!(override_ids.is_empty());
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            let pipeline_options = naga::back::spv::PipelineOptions {
                shader_stage: naga::ShaderStage::Compute,
                entry_point: ENTRY_POINT.to_string(),
            };
            let spirv = naga::back::spv::write_vec(
                &module,
                &info,
                &naga::back::spv::Options::default(),
                Some(&pipeline_options),
            )
            .unwrap();
            assert!(!spirv.is_empty());
        }

        #[test]
        fn cpu_reference_uniform_density_advances_rays() {
            let config = fixture_config();
            let density = vec![0.375f32; config.n_cells()];
            let pos = vec![GpuVec3 {
                x: 4.0,
                y: 4.0,
                z: 0.0,
            }];
            let dir = vec![GpuVec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            }];
            let (final_pos, final_dir) =
                trace_rays_cpu_reference(config, &density, &pos, &dir).unwrap();
            assert!((final_pos[0].z - 1.0).abs() < 1.0e-5);
            assert!((final_dir[0].z - 1.0).abs() < 1.0e-5);
        }

        #[test]
        fn invalid_trace_inputs_are_rejected() {
            let config = fixture_config();
            let density = vec![0.0f32; config.n_cells()];
            let (pos, dir) = fixture_rays();
            assert!(
                trace_rays_cpu_reference(config, &density[..density.len() - 1], &pos, &dir)
                    .is_err()
            );
            assert!(trace_rays_cpu_reference(config, &density, &[], &[]).is_err());
            assert!(trace_rays_cpu_reference(config, &density, &pos, &dir[..1]).is_err());
        }

        #[test]
        #[ignore = "requires local Vulkan compute device"]
        fn algebraic_lensing_vulkan_context_smoke() {
            let (_instance, _adapter, _device) = AlgebraicLensingVulkan::build_context().unwrap();
        }

        #[test]
        #[ignore = "requires local Vulkan compute device"]
        fn algebraic_lensing_vulkan_shader_smoke() {
            let (_instance, _adapter, device) = AlgebraicLensingVulkan::build_context().unwrap();
            let _shader = AlgebraicLensingVulkan::build_shader(&device).unwrap();
        }

        #[test]
        #[ignore = "requires local Vulkan compute device"]
        fn algebraic_lensing_vulkan_pipeline_smoke() {
            let _tracer = AlgebraicLensingVulkan::new().unwrap();
        }

        #[test]
        #[ignore = "requires local Vulkan compute device"]
        fn algebraic_lensing_vulkan_uniform_density_smoke() {
            let config = fixture_config();
            let density = vec![0.375f32; config.n_cells()];
            let pos = vec![GpuVec3 {
                x: 4.0,
                y: 4.0,
                z: 0.0,
            }];
            let dir = vec![GpuVec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            }];
            let (final_pos, final_dir) = AlgebraicLensingVulkan::new()
                .unwrap()
                .trace_rays(config, &density, &pos, &dir)
                .unwrap();
            assert!((final_pos[0].z - 1.0).abs() < 1.0e-4);
            assert!((final_dir[0].z - 1.0).abs() < 1.0e-4);
        }

        #[test]
        #[ignore = "requires local Vulkan compute device"]
        fn algebraic_lensing_vulkan_matches_cpu_reference() {
            let config = fixture_config();
            let density = gradient_density(config);
            let (pos, dir) = fixture_rays();
            let cpu = trace_rays_cpu_reference(config, &density, &pos, &dir).unwrap();
            let vulkan = AlgebraicLensingVulkan::new()
                .unwrap()
                .trace_rays(config, &density, &pos, &dir)
                .unwrap();
            assert_vec3_close(&cpu.0, &vulkan.0);
            assert_vec3_close(&cpu.1, &vulkan.1);
        }

        fn assert_vec3_close(expected: &[GpuVec3], observed: &[GpuVec3]) {
            assert_eq!(expected.len(), observed.len());
            for (idx, (lhs, rhs)) in expected.iter().zip(observed.iter()).enumerate() {
                for (component, lhs_value, rhs_value) in [
                    ("x", lhs.x, rhs.x),
                    ("y", lhs.y, rhs.y),
                    ("z", lhs.z, rhs.z),
                ] {
                    let scale = lhs_value.abs().max(1.0);
                    let rel = (lhs_value - rhs_value).abs() / scale;
                    assert!(
                        rel < 1.0e-4,
                        "ray {idx} component {component}: cpu={lhs_value}, vulkan={rhs_value}, rel={rel}"
                    );
                }
            }
        }
    }
}

#[cfg(feature = "vulkan")]
pub use vulkan::{
    PublicAlgebraicLensingVulkan as AlgebraicLensingVulkan,
    PublicAlgebraicLensingVulkanConfig as AlgebraicLensingVulkanConfig,
    public_trace_rays_cpu_reference as trace_rays_cpu_reference,
};

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_initialization() {
        if !Context::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }
        let tracer = AlgebraicLensingGpu::new();
        assert!(tracer.is_ok());
    }

    #[test]
    fn test_gpu_trace_smoke() {
        if !Context::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }
        let tracer = AlgebraicLensingGpu::new().unwrap();
        let density = vec![0.375f32; 8 * 8 * 8];
        let pos = vec![GpuVec3 {
            x: 4.0,
            y: 4.0,
            z: 0.0,
        }];
        let dir = vec![GpuVec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }];

        let config = AlgebraicLensingGpuConfig::new(8, 8, 8, 1.0, 0.1, 10);
        let result = tracer.trace_rays(config, &density, &pos, &dir);
        assert!(result.is_ok());
        let (f_pos, _) = result.unwrap();
        assert_eq!(f_pos.len(), 1);
        // Position should have advanced in Z
        assert!(f_pos[0].z > 0.0);
    }
}
