// Copyright (c) 2026 Terascale Functionalists
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Dark-halo detector: three-pipeline Vulkan compute backend.
//
// Pipeline 1 (dark_halo_viscosity.wgsl): writes a per-cell tau field
// from a deterministic Murmur-inspired spatial hash of (x, y, z, seed).
//
// Pipeline 2 (dark_halo_lbm_step.wgsl): N D3Q19 BGK PUSH timesteps,
// reading per-cell tau from the buffer above rather than a uniform.
// Ping-pong between two f buffers avoids read-write conflicts.
//
// Pipeline 3 (dark_halo_detector.wgsl): classifies each cell as a
// halo candidate when all three criteria hold:
//   ZD proxy (tau - tau_base)/tau_amp > zd_threshold,
//   |u| < velocity_epsilon, and rho > density_factor * rho_mean.
//
// Buffer layout (SoA):
//   f:    f[i * n_cells + cell],  i = 0..18
//   u:    ux[0..N] | uy[N..2N] | uz[2N..3N],  N = n_cells
//   tau, rho: flat f32 arrays of length n_cells
//   halo_mask: flat u32 array of length n_cells (0=background, 1=halo)
//
// Descriptor set bindings match the three WGSL shader headers.

use ash::vk;
use gororoba_gpu_vulkan::{
    Adapter, ComputePipeline, ComputePipelineBuilder, DescriptorSetLayout, DescriptorSetLayoutSpec,
    Device, DeviceBuilder, DispatchScope, Instance, InstanceBuilder, QueueFamilyRequirement,
    ShaderModule, ValidationPolicy, VulkanError,
};

const VISC_WGSL: &str = include_str!("../shaders/dark_halo_viscosity.wgsl");
const LBM_STEP_WGSL: &str = include_str!("../shaders/dark_halo_lbm_step.wgsl");
const DETECTOR_WGSL: &str = include_str!("../shaders/dark_halo_detector.wgsl");
const VISC_ENTRY: &str = "main";
const LBM_STEP_ENTRY: &str = "cs_step";
const DETECTOR_ENTRY: &str = "main";
const D3Q19_CHANNELS: usize = 19;
const WORKGROUP_1D: u32 = 64;
const WORKGROUP_DET: u32 = 8;
const STEP_TIMEOUT_NS: u64 = 30_000_000_000;

/// Errors from the dark-halo Vulkan pipeline.
#[derive(Debug, thiserror::Error)]
pub enum DarkHaloError {
    #[error("vulkan helper error: {0}")]
    Vulkan(#[from] VulkanError),
    #[error("vulkan API error: {0:?}")]
    Vk(vk::Result),
    #[error("grid dimensions must all be positive (got nx={nx}, ny={ny}, nz={nz})")]
    EmptyGrid { nx: usize, ny: usize, nz: usize },
    #[error("grid too large for u32 cell indexing: nx*ny*nz = {0}")]
    GridTooLarge(u64),
    #[error("tau_base must satisfy tau_base > 0.5 for BGK stability (got {0})")]
    UnstableTauBase(f32),
    #[error("initial f slice length {got} does not match nx*ny*nz*19 = {expected}")]
    UploadLengthMismatch { got: usize, expected: usize },
}

impl From<vk::Result> for DarkHaloError {
    fn from(r: vk::Result) -> Self {
        Self::Vk(r)
    }
}

/// Configuration for one dark-halo detector run.
#[derive(Clone, Debug)]
pub struct DarkHaloConfig {
    /// Cayley-Dickson dimension; drives lambda = ln(k_dim as f32) in the
    /// viscosity hash formula.
    pub k_dim: u32,
    /// Number of D3Q19 BGK timesteps to evolve before classification.
    pub steps: usize,
    /// Spatial hash seed for the ZD viscosity field.
    pub seed: u32,
    /// Baseline relaxation time; must be > 0.5 for BGK stability.
    pub tau_base: f32,
    /// Amplitude of ZD viscosity modulation around tau_base.
    pub tau_amp: f32,
    /// ZD proxy threshold: (tau - tau_base) / tau_amp > zd_threshold.
    pub zd_threshold: f32,
    /// Speed threshold: |u| < velocity_epsilon triggers the low-velocity
    /// criterion.
    pub velocity_epsilon: f32,
    /// Density factor: rho > density_factor * rho_mean triggers the
    /// high-density criterion.
    pub density_factor: f32,
}

/// Result of one dark-halo detector run.
#[derive(Clone, Debug)]
pub struct DarkHaloVulkanResult {
    /// Number of cells satisfying all three classifier criteria.
    pub halo_count: u32,
    /// Total number of cells in the grid.
    pub n_cells: usize,
    /// halo_count / n_cells.
    pub halo_fraction: f32,
}

/// Uniform-buffer payload mirroring `ViscParams` in dark_halo_viscosity.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ViscParamsUbo {
    nx: u32,
    ny: u32,
    nz: u32,
    seed: u32,
    k_dim: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    tau_base: f32,
    tau_amp: f32,
    lambda: f32,
    _pad3: f32,
}
// SAFETY: repr(C) with only u32/f32 fields; no padding; all bit patterns valid.
unsafe impl bytemuck::Pod for ViscParamsUbo {}
unsafe impl bytemuck::Zeroable for ViscParamsUbo {}

/// Uniform-buffer payload mirroring `Params` in dark_halo_lbm_step.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct LbmParamsUbo {
    nx: u32,
    ny: u32,
    nz: u32,
    pad: u32,
}
// SAFETY: repr(C) with only u32 fields; no padding; all bit patterns valid.
unsafe impl bytemuck::Pod for LbmParamsUbo {}
unsafe impl bytemuck::Zeroable for LbmParamsUbo {}

/// Uniform-buffer payload mirroring `HaloConstants` in dark_halo_detector.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct HaloConstantsUbo {
    nx: u32,
    ny: u32,
    nz: u32,
    tau_base: f32,
    tau_amp: f32,
    zd_threshold: f32,
    velocity_epsilon: f32,
    density_factor: f32,
    rho_mean: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
// SAFETY: repr(C) with only u32/f32 fields; no padding; all bit patterns valid.
unsafe impl bytemuck::Pod for HaloConstantsUbo {}
unsafe impl bytemuck::Zeroable for HaloConstantsUbo {}

/// One Vulkan-owned buffer plus its backing device memory.
struct DeviceBuffer {
    device: Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: handles created by allocate_buffer on self.device;
        // Device is Arc-backed so the device outlives this Drop.
        unsafe {
            self.device.raw().destroy_buffer(self.buffer, None);
            self.device.raw().free_memory(self.memory, None);
        }
    }
}

/// Vulkan-backed dark-halo detector.
///
/// Owns the instance, device, three pipelines, nine device buffers, and
/// the four pre-baked descriptor sets.  Call [`DarkHaloVulkan::run`] to
/// execute the full viscosity + LBM + classify pipeline.
pub struct DarkHaloVulkan {
    nx: usize,
    ny: usize,
    nz: usize,
    n_cells: usize,
    // current_in tracks which f buffer holds the latest state:
    // 0 -> f_bufs[0] is current input, 1 -> f_bufs[1] is current input.
    current_in: usize,
    f_bufs: [DeviceBuffer; 2],
    // tau_buf is written by the viscosity GPU dispatch; Rust never reads the field
    // directly after construction, but the DeviceBuffer must live as long as the
    // descriptor sets that bind its vk::Buffer handle.
    #[allow(dead_code)]
    tau_buf: DeviceBuffer,
    rho_buf: DeviceBuffer,
    u_buf: DeviceBuffer,
    halo_mask_buf: DeviceBuffer,
    // Uniform buffers; contents updated before each pipeline dispatch.
    #[allow(dead_code)]
    visc_ubo: DeviceBuffer,
    #[allow(dead_code)]
    lbm_ubo: DeviceBuffer,
    #[allow(dead_code)]
    halo_ubo: DeviceBuffer,
    // Descriptor set layouts (must outlive their pipelines).
    #[allow(dead_code)]
    visc_dsl: DescriptorSetLayout,
    #[allow(dead_code)]
    lbm_dsl: DescriptorSetLayout,
    #[allow(dead_code)]
    detector_dsl: DescriptorSetLayout,
    visc_pipeline: ComputePipeline,
    lbm_pipeline: ComputePipeline,
    detector_pipeline: ComputePipeline,
    dispatch: DispatchScope,
    descriptor_pool: vk::DescriptorPool,
    // visc_set: visc UBO + tau_buf
    // lbm_sets[0]: f_bufs[0]->f_bufs[1] + tau_buf + lbm UBO
    // lbm_sets[1]: f_bufs[1]->f_bufs[0] + tau_buf + lbm UBO
    // detector_set: tau_buf + rho_buf + u_buf + halo_mask_buf + halo UBO
    visc_set: vk::DescriptorSet,
    lbm_sets: [vk::DescriptorSet; 2],
    detector_set: vk::DescriptorSet,
    #[allow(dead_code)]
    adapter: Adapter,
    device: Device,
    #[allow(dead_code)]
    instance: Instance,
}

impl DarkHaloVulkan {
    /// Allocate all Vulkan resources for an nx*ny*nz periodic D3Q19 grid.
    ///
    /// The initial f field is the equilibrium rest state (`f[i,cell] = w_i`).
    /// Inject a custom state via the `f_init` argument to `run`.
    pub fn new(nx: usize, ny: usize, nz: usize) -> Result<Self, DarkHaloError> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(DarkHaloError::EmptyGrid { nx, ny, nz });
        }
        let n_cells_u64 = (nx as u64) * (ny as u64) * (nz as u64);
        if n_cells_u64 > u32::MAX as u64 {
            return Err(DarkHaloError::GridTooLarge(n_cells_u64));
        }
        let n_cells = nx * ny * nz;

        let instance = InstanceBuilder::new("dark_halo_vulkan")
            .api_version(vk::API_VERSION_1_2)
            .validation(ValidationPolicy::default_for_profile())
            .build()?;
        let adapter = Adapter::pick(&instance, QueueFamilyRequirement::Compute)?;
        let device = DeviceBuilder::new(adapter.clone()).build(&instance)?;

        // Three shaders, three pipelines with distinct descriptor set layouts.
        let visc_shader = ShaderModule::from_wgsl(&device, VISC_WGSL, VISC_ENTRY)?;
        let lbm_shader = ShaderModule::from_wgsl(&device, LBM_STEP_WGSL, LBM_STEP_ENTRY)?;
        let detector_shader = ShaderModule::from_wgsl(&device, DETECTOR_WGSL, DETECTOR_ENTRY)?;

        // Viscosity: binding 0 = tau_out (storage), binding 1 = ViscParams (uniform).
        let visc_dsl = DescriptorSetLayoutSpec::new()
            .storage_buffer(0)
            .uniform_buffer(1)
            .build(&device)?;
        // LBM step: binding 0 = f_in, 1 = f_out, 2 = tau_buf (all storage),
        //           binding 3 = Params (uniform).
        let lbm_dsl = DescriptorSetLayoutSpec::new()
            .storage_buffer(0)
            .storage_buffer(1)
            .storage_buffer(2)
            .uniform_buffer(3)
            .build(&device)?;
        // Detector: binding 0 = tau, 1 = rho, 2 = u, 3 = halo_mask (all storage),
        //           binding 4 = HaloConstants (uniform).
        let detector_dsl = DescriptorSetLayoutSpec::new()
            .storage_buffer(0)
            .storage_buffer(1)
            .storage_buffer(2)
            .storage_buffer(3)
            .uniform_buffer(4)
            .build(&device)?;

        let visc_pipeline = ComputePipelineBuilder::new(&device, &visc_shader)
            .descriptor_layout(&visc_dsl)
            .build()?;
        let lbm_pipeline = ComputePipelineBuilder::new(&device, &lbm_shader)
            .descriptor_layout(&lbm_dsl)
            .build()?;
        let detector_pipeline = ComputePipelineBuilder::new(&device, &detector_shader)
            .descriptor_layout(&detector_dsl)
            .build()?;

        // Allocate device buffers.
        let f_bytes = (n_cells * D3Q19_CHANNELS * std::mem::size_of::<f32>()) as vk::DeviceSize;
        let cell_f32_bytes = (n_cells * std::mem::size_of::<f32>()) as vk::DeviceSize;
        let cell_u32_bytes = (n_cells * std::mem::size_of::<u32>()) as vk::DeviceSize;
        let u_bytes = (3 * n_cells * std::mem::size_of::<f32>()) as vk::DeviceSize;

        let f_buf0 = allocate_storage_buffer(&device, &adapter, &instance, f_bytes)?;
        let f_buf1 = allocate_storage_buffer(&device, &adapter, &instance, f_bytes)?;
        let tau_buf = allocate_storage_buffer(&device, &adapter, &instance, cell_f32_bytes)?;
        let rho_buf = allocate_storage_buffer(&device, &adapter, &instance, cell_f32_bytes)?;
        let u_buf = allocate_storage_buffer(&device, &adapter, &instance, u_bytes)?;
        let halo_mask_buf = allocate_storage_buffer(&device, &adapter, &instance, cell_u32_bytes)?;

        let visc_ubo = allocate_uniform_buffer(
            &device,
            &adapter,
            &instance,
            std::mem::size_of::<ViscParamsUbo>() as vk::DeviceSize,
        )?;
        let lbm_ubo = allocate_uniform_buffer(
            &device,
            &adapter,
            &instance,
            std::mem::size_of::<LbmParamsUbo>() as vk::DeviceSize,
        )?;
        let halo_ubo = allocate_uniform_buffer(
            &device,
            &adapter,
            &instance,
            std::mem::size_of::<HaloConstantsUbo>() as vk::DeviceSize,
        )?;

        // Initialise both f buffers to the D3Q19 equilibrium rest state.
        let weights = d3q19_weights_f32();
        let f_init: Vec<f32> = (0..n_cells * D3Q19_CHANNELS)
            .map(|k| weights[k / n_cells])
            .collect();
        upload_f32_slice(&device, &f_buf0, &f_init)?;
        upload_f32_slice(&device, &f_buf1, &f_init)?;

        // Descriptor pool: 4 sets, 11 storage + 4 uniform descriptors.
        //   visc_set:       1 storage + 1 uniform
        //   lbm_sets[0]:    3 storage + 1 uniform
        //   lbm_sets[1]:    3 storage + 1 uniform
        //   detector_set:   4 storage + 1 uniform
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 11,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 4,
            },
        ];
        // SAFETY: pool_sizes outlives the call.
        let descriptor_pool = unsafe {
            device.raw().create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(4)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }?;

        let visc_raw = visc_dsl.raw();
        let lbm_raw = lbm_dsl.raw();
        let det_raw = detector_dsl.raw();
        let set_layouts = [visc_raw, lbm_raw, lbm_raw, det_raw];
        // SAFETY: descriptor_pool just created; set_layouts outlives the call.
        let sets_vec = unsafe {
            device.raw().allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&set_layouts),
            )
        }?;
        let visc_set = sets_vec[0];
        let lbm_sets = [sets_vec[1], sets_vec[2]];
        let detector_set = sets_vec[3];

        // Pre-bake all descriptor sets; uniform buffer contents are written
        // before each dispatch in `run` but the buffer handles never change.
        bind_visc_set(&device, visc_set, &tau_buf, &visc_ubo);
        bind_lbm_step_set(&device, lbm_sets[0], &f_buf0, &f_buf1, &tau_buf, &lbm_ubo);
        bind_lbm_step_set(&device, lbm_sets[1], &f_buf1, &f_buf0, &tau_buf, &lbm_ubo);
        bind_detector_set(
            &device,
            detector_set,
            &tau_buf,
            &rho_buf,
            &u_buf,
            &halo_mask_buf,
            &halo_ubo,
        );

        let dispatch = DispatchScope::new(&device)?;

        Ok(Self {
            nx,
            ny,
            nz,
            n_cells,
            current_in: 0,
            f_bufs: [f_buf0, f_buf1],
            tau_buf,
            rho_buf,
            u_buf,
            halo_mask_buf,
            visc_ubo,
            lbm_ubo,
            halo_ubo,
            visc_dsl,
            lbm_dsl,
            detector_dsl,
            visc_pipeline,
            lbm_pipeline,
            detector_pipeline,
            dispatch,
            descriptor_pool,
            visc_set,
            lbm_sets,
            detector_set,
            adapter,
            device,
            instance,
        })
    }

    /// Execute the full dark-halo detection pipeline.
    ///
    /// `f_init` must be SoA-laid-out: `f[i * n_cells + cell]` for i=0..18.
    /// Pass `None` to use the D3Q19 equilibrium rest state set at construction.
    pub fn run(
        &mut self,
        f_init: Option<&[f32]>,
        config: &DarkHaloConfig,
    ) -> Result<DarkHaloVulkanResult, DarkHaloError> {
        if config.tau_base.is_nan() || config.tau_base <= 0.5 {
            return Err(DarkHaloError::UnstableTauBase(config.tau_base));
        }
        let n = self.n_cells;
        let group_1d = (n as u32).div_ceil(WORKGROUP_1D);
        let gx_det = (self.nx as u32).div_ceil(WORKGROUP_DET);
        let gy_det = (self.ny as u32).div_ceil(WORKGROUP_DET);
        let gz_det = (self.nz as u32).div_ceil(WORKGROUP_DET);

        // Upload initial f state if provided; otherwise the equilibrium
        // rest state written at construction is already in f_bufs[0].
        self.current_in = 0;
        if let Some(f_host) = f_init {
            if f_host.len() != n * D3Q19_CHANNELS {
                return Err(DarkHaloError::UploadLengthMismatch {
                    got: f_host.len(),
                    expected: n * D3Q19_CHANNELS,
                });
            }
            upload_f32_slice(&self.device, &self.f_bufs[0], f_host)?;
        }

        // Step 1: dispatch viscosity to fill tau_buf.
        let lambda = (config.k_dim as f32).ln();
        let visc_params = ViscParamsUbo {
            nx: self.nx as u32,
            ny: self.ny as u32,
            nz: self.nz as u32,
            seed: config.seed,
            k_dim: config.k_dim,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            tau_base: config.tau_base,
            tau_amp: config.tau_amp,
            lambda,
            _pad3: 0.0,
        };
        upload_pod(&self.device, &self.visc_ubo, &visc_params)?;
        self.dispatch.dispatch(
            &self.visc_pipeline,
            self.visc_set,
            group_1d,
            1,
            1,
            STEP_TIMEOUT_NS,
        )?;

        // Step 2: LBM step params (constant across all steps).
        let lbm_params = LbmParamsUbo {
            nx: self.nx as u32,
            ny: self.ny as u32,
            nz: self.nz as u32,
            pad: 0,
        };
        upload_pod(&self.device, &self.lbm_ubo, &lbm_params)?;

        // Step 3: evolve config.steps BGK timesteps with ping-pong.
        for _ in 0..config.steps {
            let lbm_set = self.lbm_sets[self.current_in];
            self.dispatch
                .dispatch(&self.lbm_pipeline, lbm_set, group_1d, 1, 1, STEP_TIMEOUT_NS)?;
            self.current_in ^= 1;
        }

        // Step 4: download current f field, compute macroscopic fields on CPU.
        let f_current = download_f32_slice(&self.device, &self.f_bufs[self.current_in])?;
        let (rho, u_flat, rho_mean) = compute_macroscopic_cpu(&f_current, n);

        // Step 5: upload rho and u to device for the detector shader.
        upload_f32_slice(&self.device, &self.rho_buf, &rho)?;
        upload_f32_slice(&self.device, &self.u_buf, &u_flat)?;

        // Step 6: dispatch detector.
        let halo_pc = HaloConstantsUbo {
            nx: self.nx as u32,
            ny: self.ny as u32,
            nz: self.nz as u32,
            tau_base: config.tau_base,
            tau_amp: config.tau_amp,
            zd_threshold: config.zd_threshold,
            velocity_epsilon: config.velocity_epsilon,
            density_factor: config.density_factor,
            rho_mean,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        upload_pod(&self.device, &self.halo_ubo, &halo_pc)?;
        self.dispatch.dispatch(
            &self.detector_pipeline,
            self.detector_set,
            gx_det,
            gy_det,
            gz_det,
            STEP_TIMEOUT_NS,
        )?;

        // Step 7: read halo mask and sum.
        let halo_mask = download_u32_slice(&self.device, &self.halo_mask_buf)?;
        let halo_count: u32 = halo_mask.iter().sum();

        Ok(DarkHaloVulkanResult {
            halo_count,
            n_cells: n,
            halo_fraction: halo_count as f32 / n as f32,
        })
    }

    pub fn grid(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }
}

impl Drop for DarkHaloVulkan {
    fn drop(&mut self) {
        // SAFETY: descriptor_pool was created on self.device and is destroyed
        // exactly once here. Descriptor sets allocated from it are reclaimed
        // implicitly by destroy_descriptor_pool.
        unsafe {
            self.device
                .raw()
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

// ---- CPU helpers ----

fn compute_macroscopic_cpu(f: &[f32], n: usize) -> (Vec<f32>, Vec<f32>, f32) {
    let mut rho = vec![0.0_f32; n];
    let mut ux = vec![0.0_f32; n];
    let mut uy = vec![0.0_f32; n];
    let mut uz = vec![0.0_f32; n];
    for cell in 0..n {
        let mut r = 0.0_f32;
        for i in 0..D3Q19_CHANNELS {
            r += f[i * n + cell];
        }
        // Guard against rho=0 in degenerate initial states.
        let inv_r = if r > 1e-30 { 1.0 / r } else { 0.0 };
        let g = |i: usize| f[i * n + cell];
        let mx = g(1) - g(2) + g(7) - g(8) + g(9) - g(10) + g(11) - g(12) + g(13) - g(14);
        let my = g(3) - g(4) + g(7) - g(8) - g(9) + g(10) + g(15) - g(16) + g(17) - g(18);
        let mz = g(5) - g(6) + g(11) - g(12) - g(13) + g(14) + g(15) - g(16) - g(17) + g(18);
        rho[cell] = r;
        ux[cell] = mx * inv_r;
        uy[cell] = my * inv_r;
        uz[cell] = mz * inv_r;
    }
    let rho_mean = rho.iter().sum::<f32>() / n as f32;
    // Pack u as SoA matching dark_halo_detector.wgsl: ux[0..N] | uy[N..2N] | uz[2N..3N].
    let mut u_flat = vec![0.0_f32; 3 * n];
    u_flat[..n].copy_from_slice(&ux);
    u_flat[n..2 * n].copy_from_slice(&uy);
    u_flat[2 * n..3 * n].copy_from_slice(&uz);
    (rho, u_flat, rho_mean)
}

fn d3q19_weights_f32() -> [f32; 19] {
    const W0: f32 = 1.0 / 3.0;
    const W1: f32 = 1.0 / 18.0;
    const W2: f32 = 1.0 / 36.0;
    [
        W0, W1, W1, W1, W1, W1, W1, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2,
    ]
}

// ---- Vulkan memory helpers (identical contract to lbm_d3q19_vulkan.rs) ----

fn allocate_storage_buffer(
    device: &Device,
    adapter: &Adapter,
    instance: &Instance,
    size: vk::DeviceSize,
) -> Result<DeviceBuffer, DarkHaloError> {
    allocate_buffer(
        device,
        adapter,
        instance,
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )
}

fn allocate_uniform_buffer(
    device: &Device,
    adapter: &Adapter,
    instance: &Instance,
    size: vk::DeviceSize,
) -> Result<DeviceBuffer, DarkHaloError> {
    allocate_buffer(
        device,
        adapter,
        instance,
        size,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
    )
}

fn allocate_buffer(
    device: &Device,
    adapter: &Adapter,
    _instance: &Instance,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Result<DeviceBuffer, DarkHaloError> {
    // SAFETY: device + adapter share the same instance; size is bounded by
    // n_cells and the calling allocation functions.
    let buffer = unsafe {
        device.raw().create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }?;
    let req = unsafe { device.raw().get_buffer_memory_requirements(buffer) };
    let mem_props = adapter.memory_properties;
    let mut mem_type_index = u32::MAX;
    for i in 0..mem_props.memory_type_count {
        let supported = (req.memory_type_bits & (1 << i)) != 0;
        let flags = mem_props.memory_types[i as usize].property_flags;
        if supported
            && flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            && flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            mem_type_index = i;
            break;
        }
    }
    if mem_type_index == u32::MAX {
        // SAFETY: buffer created above; cleaned up before propagating the error.
        unsafe {
            device.raw().destroy_buffer(buffer, None);
        }
        return Err(DarkHaloError::Vk(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
    }
    // SAFETY: req.size + mem_type_index are Vulkan-produced; size is bounded.
    let memory = unsafe {
        device.raw().allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type_index),
            None,
        )
    }?;
    // SAFETY: buffer + memory created on the same device.
    unsafe {
        device.raw().bind_buffer_memory(buffer, memory, 0)?;
    }
    Ok(DeviceBuffer {
        device: device.clone(),
        buffer,
        memory,
        size: req.size,
    })
}

fn upload_f32_slice(
    device: &Device,
    buf: &DeviceBuffer,
    data: &[f32],
) -> Result<(), DarkHaloError> {
    let bytes = bytemuck::cast_slice::<f32, u8>(data);
    // SAFETY: HOST_VISIBLE + HOST_COHERENT memory; bytes.len() <= buf.size.
    unsafe {
        let ptr = device
            .raw()
            .map_memory(buf.memory, 0, buf.size, vk::MemoryMapFlags::empty())?
            as *mut u8;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
        device.raw().unmap_memory(buf.memory);
    }
    Ok(())
}

fn upload_pod<T: bytemuck::Pod>(
    device: &Device,
    buf: &DeviceBuffer,
    value: &T,
) -> Result<(), DarkHaloError> {
    let bytes = bytemuck::bytes_of(value);
    // SAFETY: HOST_VISIBLE + HOST_COHERENT; bytes.len() <= buf.size.
    unsafe {
        let ptr = device
            .raw()
            .map_memory(buf.memory, 0, buf.size, vk::MemoryMapFlags::empty())?
            as *mut u8;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
        device.raw().unmap_memory(buf.memory);
    }
    Ok(())
}

fn download_f32_slice(device: &Device, buf: &DeviceBuffer) -> Result<Vec<f32>, DarkHaloError> {
    let mut out = vec![0.0_f32; buf.size as usize / std::mem::size_of::<f32>()];
    // SAFETY: HOST_VISIBLE + HOST_COHERENT; size matches buf.size.
    unsafe {
        let ptr = device
            .raw()
            .map_memory(buf.memory, 0, buf.size, vk::MemoryMapFlags::empty())?
            as *const u8;
        let out_bytes = bytemuck::cast_slice_mut::<f32, u8>(&mut out);
        std::ptr::copy_nonoverlapping(ptr, out_bytes.as_mut_ptr(), out_bytes.len());
        device.raw().unmap_memory(buf.memory);
    }
    Ok(out)
}

fn download_u32_slice(device: &Device, buf: &DeviceBuffer) -> Result<Vec<u32>, DarkHaloError> {
    let mut out = vec![0u32; buf.size as usize / std::mem::size_of::<u32>()];
    // SAFETY: HOST_VISIBLE + HOST_COHERENT; size matches buf.size.
    unsafe {
        let ptr = device
            .raw()
            .map_memory(buf.memory, 0, buf.size, vk::MemoryMapFlags::empty())?
            as *const u8;
        let out_bytes = bytemuck::cast_slice_mut::<u32, u8>(&mut out);
        std::ptr::copy_nonoverlapping(ptr, out_bytes.as_mut_ptr(), out_bytes.len());
        device.raw().unmap_memory(buf.memory);
    }
    Ok(out)
}

// ---- Descriptor-set bind helpers ----

fn bind_visc_set(
    device: &Device,
    set: vk::DescriptorSet,
    tau_out: &DeviceBuffer,
    ubo: &DeviceBuffer,
) {
    let info_tau = [vk::DescriptorBufferInfo::default()
        .buffer(tau_out.buffer)
        .offset(0)
        .range(tau_out.size)];
    let info_ubo = [vk::DescriptorBufferInfo::default()
        .buffer(ubo.buffer)
        .offset(0)
        .range(ubo.size)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info_tau),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&info_ubo),
    ];
    // SAFETY: info arrays outlive the call; set was allocated from the pool.
    unsafe {
        device.raw().update_descriptor_sets(&writes, &[]);
    }
}

fn bind_lbm_step_set(
    device: &Device,
    set: vk::DescriptorSet,
    f_in: &DeviceBuffer,
    f_out: &DeviceBuffer,
    tau_buf: &DeviceBuffer,
    ubo: &DeviceBuffer,
) {
    let info_fin = [vk::DescriptorBufferInfo::default()
        .buffer(f_in.buffer)
        .offset(0)
        .range(f_in.size)];
    let info_fout = [vk::DescriptorBufferInfo::default()
        .buffer(f_out.buffer)
        .offset(0)
        .range(f_out.size)];
    let info_tau = [vk::DescriptorBufferInfo::default()
        .buffer(tau_buf.buffer)
        .offset(0)
        .range(tau_buf.size)];
    let info_ubo = [vk::DescriptorBufferInfo::default()
        .buffer(ubo.buffer)
        .offset(0)
        .range(ubo.size)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info_fin),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info_fout),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info_tau),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&info_ubo),
    ];
    // SAFETY: info arrays outlive the call; set was allocated from the pool.
    unsafe {
        device.raw().update_descriptor_sets(&writes, &[]);
    }
}

fn bind_detector_set(
    device: &Device,
    set: vk::DescriptorSet,
    tau_in: &DeviceBuffer,
    rho_in: &DeviceBuffer,
    u_in: &DeviceBuffer,
    halo_mask: &DeviceBuffer,
    ubo: &DeviceBuffer,
) {
    let info_tau = [vk::DescriptorBufferInfo::default()
        .buffer(tau_in.buffer)
        .offset(0)
        .range(tau_in.size)];
    let info_rho = [vk::DescriptorBufferInfo::default()
        .buffer(rho_in.buffer)
        .offset(0)
        .range(rho_in.size)];
    let info_u = [vk::DescriptorBufferInfo::default()
        .buffer(u_in.buffer)
        .offset(0)
        .range(u_in.size)];
    let info_halo = [vk::DescriptorBufferInfo::default()
        .buffer(halo_mask.buffer)
        .offset(0)
        .range(halo_mask.size)];
    let info_ubo = [vk::DescriptorBufferInfo::default()
        .buffer(ubo.buffer)
        .offset(0)
        .range(ubo.size)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info_tau),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info_rho),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info_u),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info_halo),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&info_ubo),
    ];
    // SAFETY: info arrays outlive the call; set was allocated from the pool.
    unsafe {
        device.raw().update_descriptor_sets(&writes, &[]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d3q19_weights_sum_to_one() {
        let sum: f32 = d3q19_weights_f32().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
    }

    #[test]
    fn compute_macroscopic_cpu_rest_state() {
        let n = 8usize;
        let weights = d3q19_weights_f32();
        let f: Vec<f32> = (0..n * D3Q19_CHANNELS).map(|k| weights[k / n]).collect();
        let (rho, u_flat, rho_mean) = compute_macroscopic_cpu(&f, n);
        for cell in 0..n {
            assert!(
                (rho[cell] - 1.0).abs() < 1e-6,
                "rho[{cell}] = {}",
                rho[cell]
            );
            assert!(u_flat[cell].abs() < 1e-6, "ux[{cell}] non-zero");
            assert!(u_flat[n + cell].abs() < 1e-6, "uy[{cell}] non-zero");
            assert!(u_flat[2 * n + cell].abs() < 1e-6, "uz[{cell}] non-zero");
        }
        assert!((rho_mean - 1.0).abs() < 1e-6, "rho_mean = {rho_mean}");
    }
}
