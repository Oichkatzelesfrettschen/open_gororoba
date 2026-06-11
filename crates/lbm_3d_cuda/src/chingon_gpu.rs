// GPU-accelerated Chingon AVT tensor contraction via cudarc NVRTC.
//
// Performs the bilinear tensor contraction on GPU (FP32) while the
// RK4 integration loop and orbital mechanics stay on CPU (f64).
//
// Dimension-parametric: works for 64D, 128D, 256D, etc.
//
// Data flow per RK4 sub-step:
//   CPU -> GPU: orbital parameters (~100 bytes as kernel args)
//   GPU:        build_state (O(dim)) -> contraction (O(violations)) -> project (O(block3))
//   GPU -> CPU: 3D force vector (12 bytes)

use anyhow::{Context, Result};
use cudarc::driver::{CudaFunction, CudaStream, PushKernelArg};
use gororoba_algebra::construction::chingon::PackedAvt;
use gororoba_gpu_cuda::{
    Buffer, CompileOptions, Context as CudaContextHelper, LaunchConfig, ModuleRegistry,
};
use gr_core::forces::chingon_bivector_drag::block_layout;
use std::sync::Arc;

const KERNEL_SRC: &str = include_str!("kernels_chingon.cu");

/// GPU pipeline for Chingon AVT tensor contraction.
///
/// Holds compiled kernels, device buffers, and the packed AVT data.
/// Created once at startup; reused for every RK4 sub-step of every flyby.
///
/// Dimension-parametric: the `dim` field determines buffer sizes and
/// block layout. All kernels receive block boundaries as arguments.
pub struct ChingonGpuPipeline {
    _ctx: CudaContextHelper,
    stream: Arc<CudaStream>,
    _module_registry: ModuleRegistry,
    // Kernels
    build_state_kernel: CudaFunction,
    contraction_kernel: CudaFunction,
    project_kernel: CudaFunction,
    zero_kernel: CudaFunction,
    // Device buffers (persistent across calls)
    d_packed_avt: Buffer<u32>,
    d_v_nd: Buffer<f32>,
    d_force_nd: Buffer<f32>,
    d_force_3d: Buffer<f32>,
    // Metadata
    dim: u32,
    index_bits: u32,
    n_violations: u32,
    inv_n_viol: f32,
    // Block layout (precomputed from dim)
    block_size: u32,
    block3_size: u32,
    b1_start: u32,
    b2_start: u32,
    b3_start: u32,
    n_phases: u32,
    n_phases_b3: u32,
}

impl ChingonGpuPipeline {
    /// Compile kernels and upload packed AVT to GPU.
    ///
    /// Call this once at program startup. The packed AVT data is uploaded
    /// to device memory and persists for the lifetime of the pipeline.
    pub fn new(packed: &PackedAvt) -> Result<Self> {
        let ctx =
            CudaContextHelper::with_default_device().context("CUDA init for Chingon pipeline")?;
        let stream = ctx.default_stream();

        let opts = CompileOptions::with_arch(crate::preferred_cuda_arch());
        let ptx = CompileOptions::compile_ptx(KERNEL_SRC, &opts)
            .context("NVRTC compile kernels_chingon.cu")?;
        let module_registry = ModuleRegistry::load(
            ctx.raw(),
            ptx,
            &[
                "chingon_build_state_3body",
                "chingon_avt_contraction",
                "chingon_project_3body",
                "chingon_zero_buffer",
            ],
        )
        .context("Load Chingon CUDA module")?;

        let build_state_kernel = module_registry.get("chingon_build_state_3body")?;
        let contraction_kernel = module_registry.get("chingon_avt_contraction")?;
        let project_kernel = module_registry.get("chingon_project_3body")?;
        let zero_kernel = module_registry.get("chingon_zero_buffer")?;

        let dim = packed.dim as usize;

        // Compute block layout
        let (bs, np, b1s, b2s, b3s, b3sz) = block_layout(dim);
        let n_phases_b3 = b3sz.div_ceil(3);

        // Upload packed AVT (read-only for entire simulation)
        let d_packed_avt =
            Buffer::htod(&stream, &packed.data).context("Upload packed AVT to GPU")?;

        // Allocate working buffers
        let d_v_nd = Buffer::alloc_zeros(&stream, dim).context("Alloc v_Nd buffer")?;
        let d_force_nd = Buffer::alloc_zeros(&stream, dim).context("Alloc force_Nd buffer")?;
        let d_force_3d = Buffer::alloc_zeros(&stream, 3).context("Alloc force_3d buffer")?;

        let n_violations = packed.violation_count;
        let inv_n_viol = 1.0f32 / (n_violations.max(1) as f32);

        Ok(Self {
            _ctx: ctx,
            stream,
            _module_registry: module_registry,
            build_state_kernel,
            contraction_kernel,
            project_kernel,
            zero_kernel,
            d_packed_avt,
            d_v_nd,
            d_force_nd,
            d_force_3d,
            dim: packed.dim,
            index_bits: packed.index_bits,
            n_violations,
            inv_n_viol,
            block_size: bs as u32,
            block3_size: b3sz as u32,
            b1_start: b1s as u32,
            b2_start: b2s as u32,
            b3_start: b3s as u32,
            n_phases: np as u32,
            n_phases_b3: n_phases_b3 as u32,
        })
    }

    /// Execute the full tensor contraction pipeline for one RK4 sub-step.
    ///
    /// All orbital parameters are passed as f64 from the CPU integrator
    /// inside the shared `ThreeBodyOrbitalParams` bundle (computed by
    /// `gr_core::forces::ThreeBodyOrbitalParams::compute`), cast to f32
    /// for the GPU kernel. The 3D force is returned as f64 for
    /// integration back into the f64 RK4 loop.
    pub fn compute_force_3body(
        &mut self,
        params: &gr_core::forces::chingon_bivector_drag::ThreeBodyOrbitalParams,
        alpha: f64,
    ) -> Result<[f64; 3]> {
        // Step 1: Zero the force buffers
        zero_device_buffer(
            &self.stream,
            &self.zero_kernel,
            &mut self.d_force_nd,
            self.dim as usize,
        )?;
        zero_device_buffer(&self.stream, &self.zero_kernel, &mut self.d_force_3d, 3)?;

        // Step 2: Build the N-D state vector
        self.build_state(params)?;

        // Step 3: AVT tensor contraction
        self.run_contraction(alpha as f32)?;

        // Step 4: Project N-D force to 3D
        self.run_projection(
            &params.triad_solar,
            &params.h_triad_solar,
            &params.vhat_triad_solar,
            params.h_earth_norm,
            params.cross_sign,
        )?;

        // Step 5: Read back 3D force
        let force: Vec<f32> = self.d_force_3d.dtoh_vec().context("Read back force_3d")?;

        Ok([force[0] as f64, force[1] as f64, force[2] as f64])
    }

    fn build_state(
        &mut self,
        params: &gr_core::forces::chingon_bivector_drag::ThreeBodyOrbitalParams,
    ) -> Result<()> {
        // Launch enough threads to cover all axes
        let threads = 256u32.min(self.dim);
        let blocks = self.dim.div_ceil(threads);
        let cfg = LaunchConfig::launch_blocks_1d(blocks, threads);

        let triad_earth = &params.triad_earth;
        let triad_lunar = &params.triad_lunar;
        let triad_solar = &params.triad_solar;

        // Cast all f64 -> f32 for kernel args
        let e_v_earth_x = triad_earth[0][0] as f32;
        let e_v_earth_y = triad_earth[0][1] as f32;
        let e_v_earth_z = triad_earth[0][2] as f32;
        let e_h_earth_x = triad_earth[1][0] as f32;
        let e_h_earth_y = triad_earth[1][1] as f32;
        let e_h_earth_z = triad_earth[1][2] as f32;
        let e_n_earth_x = triad_earth[2][0] as f32;
        let e_n_earth_y = triad_earth[2][1] as f32;
        let e_n_earth_z = triad_earth[2][2] as f32;

        let e_v_lunar_x = triad_lunar[0][0] as f32;
        let e_v_lunar_y = triad_lunar[0][1] as f32;
        let e_v_lunar_z = triad_lunar[0][2] as f32;
        let e_h_lunar_x = triad_lunar[1][0] as f32;
        let e_h_lunar_y = triad_lunar[1][1] as f32;
        let e_h_lunar_z = triad_lunar[1][2] as f32;
        let e_n_lunar_x = triad_lunar[2][0] as f32;
        let e_n_lunar_y = triad_lunar[2][1] as f32;
        let e_n_lunar_z = triad_lunar[2][2] as f32;

        let e_v_solar_x = triad_solar[0][0] as f32;
        let e_v_solar_y = triad_solar[0][1] as f32;
        let e_v_solar_z = triad_solar[0][2] as f32;
        let e_h_solar_x = triad_solar[1][0] as f32;
        let e_h_solar_y = triad_solar[1][1] as f32;
        let e_h_solar_z = triad_solar[1][2] as f32;
        let e_n_solar_x = triad_solar[2][0] as f32;
        let e_n_solar_y = triad_solar[2][1] as f32;
        let e_n_solar_z = triad_solar[2][2] as f32;

        let ht_e0 = params.h_triad_earth[0] as f32;
        let ht_e1 = params.h_triad_earth[1] as f32;
        let ht_e2 = params.h_triad_earth[2] as f32;
        let vr_l0 = params.vrel_triad_lunar[0] as f32;
        let vr_l1 = params.vrel_triad_lunar[1] as f32;
        let vr_l2 = params.vrel_triad_lunar[2] as f32;
        let ht_s0 = params.h_triad_solar[0] as f32;
        let ht_s1 = params.h_triad_solar[1] as f32;
        let ht_s2 = params.h_triad_solar[2] as f32;
        let vh_s0 = params.vhat_triad_solar[0] as f32;
        let vh_s1 = params.vhat_triad_solar[1] as f32;
        let vh_s2 = params.vhat_triad_solar[2] as f32;
        let h_norm = params.h_earth_norm as f32;
        let v_norm = params.v_rel_norm as f32;
        let cs = params.cross_sign as f32;

        let mut builder = self.stream.launch_builder(&self.build_state_kernel);
        builder.arg(self.d_v_nd.raw_mut());
        // Block layout parameters
        builder.arg(&self.dim);
        builder.arg(&self.block_size);
        builder.arg(&self.block3_size);
        builder.arg(&self.b1_start);
        builder.arg(&self.b2_start);
        builder.arg(&self.b3_start);
        builder.arg(&self.n_phases);
        builder.arg(&self.n_phases_b3);
        // Earth triad
        builder.arg(&e_v_earth_x);
        builder.arg(&e_v_earth_y);
        builder.arg(&e_v_earth_z);
        builder.arg(&e_h_earth_x);
        builder.arg(&e_h_earth_y);
        builder.arg(&e_h_earth_z);
        builder.arg(&e_n_earth_x);
        builder.arg(&e_n_earth_y);
        builder.arg(&e_n_earth_z);
        // Lunar triad
        builder.arg(&e_v_lunar_x);
        builder.arg(&e_v_lunar_y);
        builder.arg(&e_v_lunar_z);
        builder.arg(&e_h_lunar_x);
        builder.arg(&e_h_lunar_y);
        builder.arg(&e_h_lunar_z);
        builder.arg(&e_n_lunar_x);
        builder.arg(&e_n_lunar_y);
        builder.arg(&e_n_lunar_z);
        // Solar triad
        builder.arg(&e_v_solar_x);
        builder.arg(&e_v_solar_y);
        builder.arg(&e_v_solar_z);
        builder.arg(&e_h_solar_x);
        builder.arg(&e_h_solar_y);
        builder.arg(&e_h_solar_z);
        builder.arg(&e_n_solar_x);
        builder.arg(&e_n_solar_y);
        builder.arg(&e_n_solar_z);
        // Triad projections
        builder.arg(&ht_e0);
        builder.arg(&ht_e1);
        builder.arg(&ht_e2);
        builder.arg(&vr_l0);
        builder.arg(&vr_l1);
        builder.arg(&vr_l2);
        builder.arg(&ht_s0);
        builder.arg(&ht_s1);
        builder.arg(&ht_s2);
        builder.arg(&vh_s0);
        builder.arg(&vh_s1);
        builder.arg(&vh_s2);
        // Scalars
        builder.arg(&h_norm);
        builder.arg(&v_norm);
        builder.arg(&cs);

        unsafe { builder.launch(cfg).context("Launch build_state_3body")? };
        Ok(())
    }

    fn run_contraction(&mut self, alpha: f32) -> Result<()> {
        let block_size = 256u32;
        // At 256D with 3.97M violations, 256 blocks gives only 65K threads each
        // processing ~60 violations. Allow up to 4096 blocks (1M threads) to
        // saturate the RTX 4070 Ti's 60 SMs with multiple warps per SM.
        let grid_size = (self.n_violations.div_ceil(block_size)).clamp(1, 4096);
        let shared_bytes = self.dim * 4; // sizeof(float) * dim

        let mut cfg = LaunchConfig::launch_blocks_1d(grid_size, block_size);
        cfg.shared_mem_bytes = shared_bytes;

        let mut builder = self.stream.launch_builder(&self.contraction_kernel);
        builder.arg(self.d_packed_avt.raw());
        builder.arg(self.d_v_nd.raw());
        builder.arg(self.d_force_nd.raw_mut());
        builder.arg(&self.n_violations);
        builder.arg(&self.dim);
        builder.arg(&self.index_bits);
        builder.arg(&alpha);
        builder.arg(&self.inv_n_viol);

        unsafe { builder.launch(cfg).context("Launch avt_contraction")? };
        Ok(())
    }

    fn run_projection(
        &mut self,
        triad_solar: &[[f64; 3]; 3],
        h_triad_solar: &[f64; 3],
        vhat_triad_solar: &[f64; 3],
        h_earth_norm: f64,
        cross_sign: f64,
    ) -> Result<()> {
        let cfg = LaunchConfig::launch_blocks_1d(1, 1);

        let e_v_solar_x = triad_solar[0][0] as f32;
        let e_v_solar_y = triad_solar[0][1] as f32;
        let e_v_solar_z = triad_solar[0][2] as f32;
        let e_h_solar_x = triad_solar[1][0] as f32;
        let e_h_solar_y = triad_solar[1][1] as f32;
        let e_h_solar_z = triad_solar[1][2] as f32;
        let e_n_solar_x = triad_solar[2][0] as f32;
        let e_n_solar_y = triad_solar[2][1] as f32;
        let e_n_solar_z = triad_solar[2][2] as f32;
        let ht_s0 = h_triad_solar[0] as f32;
        let ht_s1 = h_triad_solar[1] as f32;
        let ht_s2 = h_triad_solar[2] as f32;
        let vh_s0 = vhat_triad_solar[0] as f32;
        let vh_s1 = vhat_triad_solar[1] as f32;
        let vh_s2 = vhat_triad_solar[2] as f32;
        let h_norm = h_earth_norm as f32;
        let cs = cross_sign as f32;

        let mut builder = self.stream.launch_builder(&self.project_kernel);
        builder.arg(self.d_force_nd.raw());
        builder.arg(self.d_force_3d.raw_mut());
        // Block layout
        builder.arg(&self.dim);
        builder.arg(&self.b3_start);
        builder.arg(&self.block3_size);
        builder.arg(&self.n_phases_b3);
        // Solar triad
        builder.arg(&e_v_solar_x);
        builder.arg(&e_v_solar_y);
        builder.arg(&e_v_solar_z);
        builder.arg(&e_h_solar_x);
        builder.arg(&e_h_solar_y);
        builder.arg(&e_h_solar_z);
        builder.arg(&e_n_solar_x);
        builder.arg(&e_n_solar_y);
        builder.arg(&e_n_solar_z);
        builder.arg(&ht_s0);
        builder.arg(&ht_s1);
        builder.arg(&ht_s2);
        builder.arg(&vh_s0);
        builder.arg(&vh_s1);
        builder.arg(&vh_s2);
        builder.arg(&h_norm);
        builder.arg(&cs);

        unsafe { builder.launch(cfg).context("Launch project_3body")? };
        Ok(())
    }

    /// Read the raw N-D force vector from GPU (for debugging/validation).
    pub fn read_force_nd(&self) -> Result<Vec<f32>> {
        self.d_force_nd.dtoh_vec().context("Read force_Nd")
    }

    /// Read the N-D state vector from GPU (for debugging/validation).
    pub fn read_state_nd(&self) -> Result<Vec<f32>> {
        self.d_v_nd.dtoh_vec().context("Read v_Nd")
    }
}

/// Zero a GPU buffer by launching the zero kernel.
///
/// Extracted as a free function to avoid borrow conflicts when zeroing
/// `self.d_force_nd` / `self.d_force_3d` inside `compute_force_3body`.
fn zero_device_buffer(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    buf: &mut Buffer<f32>,
    n: usize,
) -> Result<()> {
    let n_u32 = n as u32;
    let cfg = LaunchConfig::launch_1d(n_u32);
    let mut builder = stream.launch_builder(kernel);
    builder.arg(buf.raw_mut());
    builder.arg(&n_u32);
    unsafe { builder.launch(cfg).context("Launch zero_buffer")? };
    Ok(())
}
