// GPU-accelerated Lattice Boltzmann Method (D3Q19) with CUDA
// Runtime kernel compilation via cudarc NVRTC

use anyhow::{ensure, Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

/// Bit-compatible wrapper for Complex32 to satisfy CUDA traits.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct ComplexDevice {
    pub re: f32,
    pub im: f32,
}

unsafe impl cudarc::driver::DeviceRepr for ComplexDevice {}
unsafe impl cudarc::driver::ValidAsZeroBits for ComplexDevice {}

#[cfg(feature = "cufft")]
#[allow(unused_imports)]
use cudarc::cufft::result as cufft;

const KERNEL_SRC: &str = include_str!("kernels.cu");
const KERNEL_BF16_SRC: &str = include_str!("kernels_bf16.cu");

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Precision {
    FP32,
    BF16,
}

/// GPU-accelerated D3Q19 LBM solver with mixed-precision support (FP32/BF16)
pub struct LbmSolver3DCuda {
    nx: usize,
    ny: usize,
    nz: usize,
    n_cells: usize,
    pub precision: Precision,
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    d_f: CudaSlice<u8>,
    d_f_tmp: CudaSlice<u8>,
    d_rho: CudaSlice<u8>,
    d_u: CudaSlice<u8>,
    d_tau: CudaSlice<u8>,
    d_force: CudaSlice<u8>,
    initialize_uniform_kernel: CudaFunction,
    initialize_custom_kernel: CudaFunction,
    lbm_step_fused_kernel: CudaFunction,
    lbm_step_fused_4d_kernel: Option<CudaFunction>,
    enstrophy_cell_kernel: CudaFunction,
    reduce_sum_rho_kernel: CudaFunction,
    reduce_sum_f32_kernel: CudaFunction,
    zero_kernel: CudaFunction,
    apply_mask_kernel: CudaFunction,
    convert_real_to_complex_kernel: CudaFunction,
    convert_complex_to_real_kernel: CudaFunction,
    lbm_block_dim: (u32, u32, u32),
    #[cfg(feature = "cufft")]
    fft_plan: Option<cudarc::cufft::sys::cufftHandle>,
    d_reduction_out: CudaSlice<f32>,
    d_reduction_buffer_f32: Option<CudaSlice<f32>>,
    pub d_u_hat: Option<CudaSlice<ComplexDevice>>,
    pub d_u_hat_out: Option<CudaSlice<ComplexDevice>>,
    pub rho: Vec<f32>,
    pub u: Vec<[f32; 3]>,
}

impl LbmSolver3DCuda {
    fn parse_block_dim_env() -> Option<(u32, u32, u32)> {
        let raw = std::env::var("GOROROBA_LBM_BLOCK_DIM").ok()?;
        let cleaned = raw
            .trim()
            .replace(',', "x")
            .replace(' ', "")
            .to_ascii_lowercase();
        let parts: Vec<&str> = cleaned.split('x').filter(|p| !p.is_empty()).collect();
        if parts.len() != 3 {
            return None;
        }
        let bx: u32 = parts[0].parse().ok()?;
        let by: u32 = parts[1].parse().ok()?;
        let bz: u32 = parts[2].parse().ok()?;
        if bx == 0 || by == 0 || bz == 0 {
            return None;
        }
        if (bx as u64) * (by as u64) * (bz as u64) > 1024 {
            return None;
        }
        Some((bx, by, bz))
    }

    pub fn new(nx: usize, ny: usize, nz: usize, tau: f64, precision: Precision) -> Result<Self> {
        let n_cells = nx * ny * nz;
        let ctx = CudaContext::new(0).context("CUDA Init Failed")?;
        let stream = ctx.default_stream();
        let src = match precision {
            Precision::FP32 => KERNEL_SRC,
            Precision::BF16 => KERNEL_BF16_SRC,
        };

        use cudarc::nvrtc::CompileOptions;
        let opts = if precision == Precision::BF16 {
            CompileOptions {
                include_paths: vec!["/opt/cuda/include".to_string()],
                arch: Some("sm_89"),
                ..Default::default()
            }
        } else {
            CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            }
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(src, opts)?;
        let module = ctx.load_module(ptx)?;

        let lbm_step_fused_kernel = module.load_function(if precision == Precision::BF16 {
            "lbm_step_fused_bf16_kernel"
        } else {
            "lbm_step_fused_kernel"
        })?;
        let lbm_step_fused_4d_kernel = if precision == Precision::BF16 {
            Some(module.load_function("lbm_step_fused_bf16_4d_batch_kernel")?)
        } else {
            None
        };
        let initialize_uniform_kernel = module.load_function(if precision == Precision::BF16 {
            "initialize_uniform_bf16_kernel"
        } else {
            "initialize_uniform_kernel"
        })?;
        let initialize_custom_kernel = module.load_function(if precision == Precision::BF16 {
            "initialize_custom_bf16_kernel"
        } else {
            "initialize_custom_kernel"
        })?;
        let enstrophy_cell_kernel = module.load_function("compute_enstrophy_cell_kernel")?;
        let reduce_sum_rho_kernel = module.load_function(if precision == Precision::BF16 {
            "reduce_sum_bf16_to_f32_kernel"
        } else {
            "reduce_sum_kernel"
        })?;
        let reduce_sum_f32_kernel = module.load_function("reduce_sum_kernel")?;
        let zero_kernel = module.load_function(if precision == Precision::BF16 {
            "zero_f32_kernel"
        } else {
            "zero_kernel"
        })?;
        let apply_mask_kernel = module.load_function("apply_spectral_mask_kernel")?;
        let convert_real_to_complex_kernel =
            module.load_function(if precision == Precision::BF16 {
                "convert_real_bf16_to_complex_f32_kernel"
            } else {
                "convert_real_to_complex_kernel"
            })?;
        let convert_complex_to_real_kernel =
            module.load_function(if precision == Precision::BF16 {
                "convert_complex_f32_to_real_bf16_kernel"
            } else {
                "convert_complex_to_real_kernel"
            })?;

        let es = if precision == Precision::FP32 { 4 } else { 2 };
        let mut d_f = stream.alloc_zeros::<u8>(19 * n_cells * es)?;
        let d_f_tmp = stream.alloc_zeros::<u8>(19 * n_cells * es)?;
        let mut d_rho = stream.alloc_zeros::<u8>(n_cells * es)?;
        let mut d_u = stream.alloc_zeros::<u8>(3 * n_cells * es)?;
        let d_force = stream.alloc_zeros::<u8>(3 * n_cells * es)?;
        let d_tau = if precision == Precision::FP32 {
            let v = vec![tau as f32; n_cells];
            let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            stream.clone_htod(&b)?
        } else {
            let v = vec![half::bf16::from_f32(tau as f32).to_bits(); n_cells];
            let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            stream.clone_htod(&b)?
        };

        let d_reduction_out = stream.alloc_zeros::<f32>(1)?;
        let d_reduction_buffer_f32 = Some(stream.alloc_zeros::<f32>(n_cells)?);
        let d_u_hat = Some(stream.alloc_zeros::<ComplexDevice>(n_cells)?);
        let d_u_hat_out = Some(stream.alloc_zeros::<ComplexDevice>(n_cells)?);

        let (nx_i, ny_i, nz_i) = (nx as i32, ny as i32, nz as i32);
        let rho_init = 1.0f32;
        let u_init = 0.0f32;
        let mut init = stream.launch_builder(&initialize_uniform_kernel);
        init.arg(&mut d_f)
            .arg(&mut d_rho)
            .arg(&mut d_u)
            .arg(&rho_init)
            .arg(&u_init)
            .arg(&u_init)
            .arg(&u_init)
            .arg(&nx_i)
            .arg(&ny_i)
            .arg(&nz_i);
        unsafe { init.launch(LaunchConfig::for_num_elems(n_cells as u32)) }?;

        // Default kernel geometry. Override with:
        //   GOROROBA_LBM_BLOCK_DIM=8x4x4 (or 8,4,4)
        let default_block_dim = (4u32, 4u32, 4u32);
        let lbm_block_dim = Self::parse_block_dim_env().unwrap_or(default_block_dim);

        Ok(Self {
            nx,
            ny,
            nz,
            n_cells,
            precision,
            _ctx: ctx,
            stream,
            d_f,
            d_f_tmp,
            d_rho,
            d_u,
            d_tau,
            d_force,
            initialize_uniform_kernel,
            initialize_custom_kernel,
            lbm_step_fused_kernel,
            lbm_step_fused_4d_kernel,
            enstrophy_cell_kernel,
            reduce_sum_rho_kernel,
            reduce_sum_f32_kernel,
            zero_kernel,
            apply_mask_kernel,
            convert_real_to_complex_kernel,
            convert_complex_to_real_kernel,
            lbm_block_dim,
            #[cfg(feature = "cufft")]
            fft_plan: None,
            d_reduction_out,
            d_reduction_buffer_f32,
            d_u_hat,
            d_u_hat_out,
            rho: vec![1.0; n_cells],
            u: vec![[0.0; 3]; n_cells],
        })
    }

    fn encode_f32_to_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    }

    fn encode_bf16_to_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_bits())
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<u8>>()
    }

    pub fn step_4d(&mut self, nw: usize) -> Result<()> {
        let (nx, ny, nz_total) = (self.nx as i32, self.ny as i32, self.nz as i32);
        let nw_i = nw as i32;
        ensure!(nw > 0, "nw must be > 0");
        let nz_sub = nz_total / nw_i;
        ensure!(
            nz_total % nw_i == 0,
            "Total Z dimension {} must be divisible by nw {}",
            nz_total,
            nw
        );

        let kernel = self
            .lbm_step_fused_4d_kernel
            .as_ref()
            .context("4D kernel not available (requires BF16)")?;

        let (bx, by, bz) = self.lbm_block_dim;
        let config = LaunchConfig {
            grid_dim: (
                self.nx.div_ceil(bx as usize) as u32,
                self.ny.div_ceil(by as usize) as u32,
                self.nz.div_ceil(bz as usize) as u32,
            ),
            block_dim: (bx, by, bz),
            shared_mem_bytes: 0,
        };

        let mut b = self.stream.launch_builder(kernel);
        b.arg(&self.d_f)
            .arg(&mut self.d_f_tmp)
            .arg(&mut self.d_rho)
            .arg(&mut self.d_u)
            .arg(&self.d_force)
            .arg(&self.d_tau)
            .arg(&nx)
            .arg(&ny)
            .arg(&nz_sub)
            .arg(&nw_i);
        unsafe { b.launch(config) }?;
        std::mem::swap(&mut self.d_f, &mut self.d_f_tmp);
        Ok(())
    }

    pub fn initialize_uniform(&mut self, rho: f32, u: [f32; 3]) -> Result<()> {
        let (nx_i, ny_i, nz_i) = (self.nx as i32, self.ny as i32, self.nz as i32);
        let rho_init = rho;
        let (ux, uy, uz) = (u[0], u[1], u[2]);
        let mut init = self.stream.launch_builder(&self.initialize_uniform_kernel);
        init.arg(&mut self.d_f)
            .arg(&mut self.d_rho)
            .arg(&mut self.d_u)
            .arg(&rho_init)
            .arg(&ux)
            .arg(&uy)
            .arg(&uz)
            .arg(&nx_i)
            .arg(&ny_i)
            .arg(&nz_i);
        unsafe { init.launch(LaunchConfig::for_num_elems(self.n_cells as u32)) }?;
        Ok(())
    }

    pub fn initialize_custom(&mut self, rho: &[f64], u: &[[f64; 3]]) -> Result<()> {
        ensure!(
            rho.len() == self.n_cells,
            "rho length mismatch: got {}, expected {}",
            rho.len(),
            self.n_cells
        );
        ensure!(
            u.len() == self.n_cells,
            "u length mismatch: got {}, expected {}",
            u.len(),
            self.n_cells
        );

        let mut rho_flat = Vec::with_capacity(self.n_cells);
        for &v in rho {
            rho_flat.push(v as f32);
        }
        let mut u_flat = Vec::with_capacity(self.n_cells * 3);
        for v in u {
            u_flat.push(v[0] as f32);
            u_flat.push(v[1] as f32);
            u_flat.push(v[2] as f32);
        }

        let rho_bytes = match self.precision {
            Precision::FP32 => Self::encode_f32_to_bytes(&rho_flat),
            Precision::BF16 => Self::encode_bf16_to_bytes(&rho_flat),
        };
        let u_bytes = match self.precision {
            Precision::FP32 => Self::encode_f32_to_bytes(&u_flat),
            Precision::BF16 => Self::encode_bf16_to_bytes(&u_flat),
        };

        let d_rho_in = self.stream.clone_htod(&rho_bytes)?;
        let d_u_in = self.stream.clone_htod(&u_bytes)?;

        let (nx_i, ny_i, nz_i) = (self.nx as i32, self.ny as i32, self.nz as i32);
        let mut init = self.stream.launch_builder(&self.initialize_custom_kernel);
        init.arg(&mut self.d_f)
            .arg(&mut self.d_rho)
            .arg(&mut self.d_u)
            .arg(&d_rho_in)
            .arg(&d_u_in)
            .arg(&nx_i)
            .arg(&ny_i)
            .arg(&nz_i);
        unsafe { init.launch(LaunchConfig::for_num_elems(self.n_cells as u32)) }?;
        Ok(())
    }

    pub fn set_force_field(&mut self, force_field: &[[f64; 3]]) -> Result<()> {
        ensure!(
            force_field.len() == self.n_cells,
            "force field length mismatch: got {}, expected {}",
            force_field.len(),
            self.n_cells
        );

        let mut force_flat = Vec::with_capacity(self.n_cells * 3);
        for v in force_field {
            force_flat.push(v[0] as f32);
            force_flat.push(v[1] as f32);
            force_flat.push(v[2] as f32);
        }

        let bytes = match self.precision {
            Precision::FP32 => Self::encode_f32_to_bytes(&force_flat),
            Precision::BF16 => Self::encode_bf16_to_bytes(&force_flat),
        };
        self.d_force = self.stream.clone_htod(&bytes)?;
        Ok(())
    }

    pub fn set_viscosity_field(&mut self, viscosity_field: &[f64]) -> Result<()> {
        ensure!(
            viscosity_field.len() == self.n_cells,
            "viscosity field length mismatch: got {}, expected {}",
            viscosity_field.len(),
            self.n_cells
        );

        // LBM lattice units (D3Q19 BGK): nu = (tau - 0.5) / 3  =>  tau = 0.5 + 3*nu
        let mut tau_flat = Vec::with_capacity(self.n_cells);
        for &nu in viscosity_field {
            tau_flat.push((0.5 + 3.0 * nu) as f32);
        }

        let bytes = match self.precision {
            Precision::FP32 => Self::encode_f32_to_bytes(&tau_flat),
            Precision::BF16 => Self::encode_bf16_to_bytes(&tau_flat),
        };
        self.d_tau = self.stream.clone_htod(&bytes)?;
        Ok(())
    }

    pub fn evolve(&mut self, steps: usize) -> Result<()> {
        for _ in 0..steps {
            self.step()?;
        }
        self.sync_to_host()?;
        Ok(())
    }

    fn decode_f32_from_bytes(bytes: &[u8], out: &mut [f32]) -> Result<()> {
        ensure!(
            bytes.len() == out.len() * 4,
            "unexpected f32 byte length: got {}, expected {}",
            bytes.len(),
            out.len() * 4
        );
        for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
            *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(())
    }

    fn decode_bf16_from_bytes(bytes: &[u8], out: &mut [f32]) -> Result<()> {
        ensure!(
            bytes.len() == out.len() * 2,
            "unexpected bf16 byte length: got {}, expected {}",
            bytes.len(),
            out.len() * 2
        );
        for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            *dst = half::bf16::from_bits(bits).to_f32();
        }
        Ok(())
    }

    pub fn step(&mut self) -> Result<()> {
        let (nx, ny, nz) = (self.nx as i32, self.ny as i32, self.nz as i32);
        let (bx, by, bz) = self.lbm_block_dim;
        let config = LaunchConfig {
            grid_dim: (
                self.nx.div_ceil(bx as usize) as u32,
                self.ny.div_ceil(by as usize) as u32,
                self.nz.div_ceil(bz as usize) as u32,
            ),
            block_dim: (bx, by, bz),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.lbm_step_fused_kernel);
        b.arg(&self.d_f)
            .arg(&mut self.d_f_tmp)
            .arg(&mut self.d_rho)
            .arg(&mut self.d_u)
            .arg(&self.d_force)
            .arg(&self.d_tau)
            .arg(&nx)
            .arg(&ny)
            .arg(&nz);
        unsafe { b.launch(config) }?;
        std::mem::swap(&mut self.d_f, &mut self.d_f_tmp);
        Ok(())
    }

    pub fn calculate_enstrophy(&mut self) -> Result<f32> {
        let (nx, ny, nz, n) = (
            self.nx as i32,
            self.ny as i32,
            self.nz as i32,
            self.n_cells as i32,
        );

        let (out_ptr, _) = self.d_reduction_out.device_ptr(&self.stream);
        let mut bz = self.stream.launch_builder(&self.zero_kernel);
        bz.arg(&out_ptr);
        unsafe { bz.launch(LaunchConfig::for_num_elems(1)) }?;

        let mut b = self.stream.launch_builder(&self.enstrophy_cell_kernel);
        let d_u_ptr = self.d_u.device_ptr(&self.stream).0;
        let d_enstrophy_buffer_ptr = self
            .d_reduction_buffer_f32
            .as_ref()
            .unwrap()
            .device_ptr(&self.stream)
            .0;

        b.arg(&d_u_ptr)
            .arg(&d_enstrophy_buffer_ptr)
            .arg(&nx)
            .arg(&ny)
            .arg(&nz);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (
                    self.nx.div_ceil(2) as u32,
                    self.ny.div_ceil(2) as u32,
                    self.nz.div_ceil(2) as u32,
                ),
                block_dim: (2, 2, 2),
                shared_mem_bytes: 0,
            })
        }?;

        let grid_size = self.n_cells.div_ceil(256) as u32;
        let mut b_reduce = self.stream.launch_builder(&self.reduce_sum_f32_kernel);
        b_reduce
            .arg(self.d_reduction_buffer_f32.as_ref().unwrap())
            .arg(&mut self.d_reduction_out)
            .arg(&n);
        unsafe {
            b_reduce.launch(LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;

        let res = self.stream.clone_dtoh(&self.d_reduction_out)?;
        Ok(res[0] / self.n_cells as f32)
    }

    pub fn calculate_mean_density(&mut self) -> Result<f32> {
        let n = self.n_cells as i32;
        let (out_ptr, _) = self.d_reduction_out.device_ptr(&self.stream);
        let mut bz = self.stream.launch_builder(&self.zero_kernel);
        bz.arg(&out_ptr);
        unsafe { bz.launch(LaunchConfig::for_num_elems(1)) }?;
        let mut b = self.stream.launch_builder(&self.reduce_sum_rho_kernel);

        // Correctly passing &self.d_rho instead of &self.d_rho.device_ptr
        b.arg(&self.d_rho).arg(&mut self.d_reduction_out).arg(&n);

        let grid_size = self.n_cells.div_ceil(256) as u32;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        let res = self.stream.clone_dtoh(&self.d_reduction_out)?;
        Ok(res[0] / self.n_cells as f32)
    }

    pub fn sync_to_host(&mut self) -> Result<()> {
        let rho_bytes = self.stream.clone_dtoh(&self.d_rho)?;
        let u_bytes = self.stream.clone_dtoh(&self.d_u)?;

        match self.precision {
            Precision::FP32 => {
                Self::decode_f32_from_bytes(&rho_bytes, &mut self.rho)?;
                let mut u_flat = vec![0.0f32; self.n_cells * 3];
                Self::decode_f32_from_bytes(&u_bytes, &mut u_flat)?;
                for idx in 0..self.n_cells {
                    let base = idx * 3;
                    self.u[idx] = [u_flat[base], u_flat[base + 1], u_flat[base + 2]];
                }
            }
            Precision::BF16 => {
                Self::decode_bf16_from_bytes(&rho_bytes, &mut self.rho)?;
                let mut u_flat = vec![0.0f32; self.n_cells * 3];
                Self::decode_bf16_from_bytes(&u_bytes, &mut u_flat)?;
                for idx in 0..self.n_cells {
                    let base = idx * 3;
                    self.u[idx] = [u_flat[base], u_flat[base + 1], u_flat[base + 2]];
                }
            }
        }

        Ok(())
    }

    #[cfg(feature = "cufft")]
    fn ensure_fft_plan(&mut self) -> Result<cudarc::cufft::sys::cufftHandle> {
        if let Some(handle) = self.fft_plan {
            return Ok(handle);
        }
        let handle = cufft::plan_3d(
            self.nx as i32,
            self.ny as i32,
            self.nz as i32,
            cudarc::cufft::sys::cufftType::CUFFT_C2C,
        )
        .context("cufftPlan3d failed")?;
        unsafe {
            cufft::set_stream(handle, self.stream.cu_stream() as _)
                .context("cufftSetStream failed")?;
        }
        self.fft_plan = Some(handle);
        Ok(handle)
    }

    /// Out-of-place complex-to-complex FFT using cuFFT.
    ///
    /// `direction` is `-1` for forward and `1` for inverse.
    #[cfg(feature = "cufft")]
    pub fn fft_3d_c2c_into(
        &mut self,
        input: &CudaSlice<ComplexDevice>,
        output: &mut CudaSlice<ComplexDevice>,
        direction: i32,
    ) -> Result<()> {
        let handle = self.ensure_fft_plan()?;
        let (i_ptr, _) = input.device_ptr(&self.stream);
        let (o_ptr, _) = output.device_ptr(&self.stream);
        unsafe {
            cufft::exec_c2c(handle, i_ptr as *mut _, o_ptr as *mut _, direction)
                .context("cufftExecC2C failed")?;
        }
        Ok(())
    }

    pub fn apply_spectral_mask(
        &self,
        u_hat: &mut CudaSlice<ComplexDevice>,
        mask: &CudaSlice<f32>,
        damping: f32,
    ) -> Result<()> {
        let n = self.n_cells as i32;
        let mut b = self.stream.launch_builder(&self.apply_mask_kernel);
        b.arg(u_hat).arg(mask).arg(&damping).arg(&n);
        unsafe { b.launch(LaunchConfig::for_num_elems(self.n_cells as u32)) }?;
        Ok(())
    }

    pub fn convert_real_to_complex(
        &self,
        d_u_hat: &mut CudaSlice<ComplexDevice>,
        component: usize,
    ) -> Result<()> {
        let (c, n) = (component as i32, self.n_cells as i32);
        let mut b = self
            .stream
            .launch_builder(&self.convert_real_to_complex_kernel);
        b.arg(&self.d_u).arg(d_u_hat).arg(&c).arg(&n);
        unsafe { b.launch(LaunchConfig::for_num_elems(self.n_cells as u32)) }?;
        Ok(())
    }

    pub fn convert_complex_to_real(
        &mut self,
        d_u_hat: &CudaSlice<ComplexDevice>,
        component: usize,
        scale: f32,
    ) -> Result<()> {
        let (c, n) = (component as i32, self.n_cells as i32);
        let mut b = self
            .stream
            .launch_builder(&self.convert_complex_to_real_kernel);
        b.arg(d_u_hat)
            .arg(&mut self.d_u)
            .arg(&c)
            .arg(&scale)
            .arg(&n);
        unsafe { b.launch(LaunchConfig::for_num_elems(self.n_cells as u32)) }?;
        Ok(())
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl Drop for LbmSolver3DCuda {
    fn drop(&mut self) {
        #[cfg(feature = "cufft")]
        if let Some(handle) = self.fft_plan.take() {
            // Best-effort cleanup. Avoid panicking in Drop.
            let _ = unsafe { cufft::destroy(handle) };
        }
    }
}

#[cfg(test)]
impl LbmSolver3DCuda {
    fn set_all_distributions_for_test(&mut self, value: f32) -> Result<()> {
        let count = 19 * self.n_cells;
        let bytes: Vec<u8> = match self.precision {
            Precision::FP32 => {
                let values = vec![value; count];
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>()
            }
            Precision::BF16 => {
                let bf = half::bf16::from_f32(value).to_bits();
                let values = vec![bf; count];
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>()
            }
        };
        self.d_f = self.stream.clone_htod(&bytes)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn gpu_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    fn maybe_solver(precision: Precision) -> Option<LbmSolver3DCuda> {
        if !gpu_available() {
            eprintln!("Skipping GPU test: CUDA device unavailable");
            return None;
        }
        match LbmSolver3DCuda::new(4, 4, 4, 0.6, precision) {
            Ok(solver) => Some(solver),
            Err(err) => {
                eprintln!("Skipping GPU test: failed to initialize solver ({err})");
                None
            }
        }
    }

    #[test]
    fn init_and_first_step_are_finite_fp32() {
        let Some(mut solver) = maybe_solver(Precision::FP32) else {
            return;
        };
        let mean0 = solver
            .calculate_mean_density()
            .expect("mean density should compute");
        assert!(mean0.is_finite());
        assert_abs_diff_eq!(mean0, 1.0, epsilon = 1.0e-3);

        solver.step().expect("first step should succeed");
        let mean1 = solver
            .calculate_mean_density()
            .expect("mean density after first step");
        assert!(mean1.is_finite());
        assert_abs_diff_eq!(mean1, 1.0, epsilon = 1.0e-2);
    }

    #[test]
    fn mean_density_is_mean_not_sum() {
        let Some(mut solver) = maybe_solver(Precision::FP32) else {
            return;
        };
        // With uniform rho=1 init, true mean must be ~1 regardless of cell count.
        let mean = solver
            .calculate_mean_density()
            .expect("mean density should compute");
        assert!(mean.is_finite());
        assert!(mean < 2.0, "expected mean close to 1, got {mean}");
    }

    #[test]
    fn sync_to_host_populates_fresh_buffers_bf16() {
        let Some(mut solver) = maybe_solver(Precision::BF16) else {
            return;
        };
        solver.step().expect("step should succeed");
        solver.sync_to_host().expect("sync_to_host should succeed");

        assert_eq!(solver.rho.len(), solver.n_cells);
        assert_eq!(solver.u.len(), solver.n_cells);
        assert!(solver.rho.iter().all(|v| v.is_finite()));
        assert!(solver
            .u
            .iter()
            .all(|v| v[0].is_finite() && v[1].is_finite() && v[2].is_finite()));
    }

    #[test]
    fn zero_density_guard_prevents_nan_propagation_fp32() {
        let Some(mut solver) = maybe_solver(Precision::FP32) else {
            return;
        };
        solver
            .set_all_distributions_for_test(0.0)
            .expect("test setup should succeed");
        solver
            .step()
            .expect("step should succeed with zeroed distributions");

        let mean = solver
            .calculate_mean_density()
            .expect("mean density should compute");
        assert!(mean.is_finite());
        assert!(mean > 0.0);

        solver.sync_to_host().expect("sync_to_host should succeed");
        assert!(solver.rho.iter().all(|v| v.is_finite()));
    }
}
