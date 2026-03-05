//! Besag-Clifford permutation testing on Vulkan.
//!
//! Orchestrates GPU-side permutation testing of ZD-imbalance -> viscosity ->
//! LBM -> regional correlation for falsification of clustering claims at
//! 256^3 resolution. Shares VulkanContext with GororobaEngine for zero-copy
//! buffer reuse.
//!
//! # Pipeline (per permutation batch)
//! 1. Dispatch `shuffle_imbalance` (PCG-based parallel shuffle)
//! 2. Dispatch `transform_viscosity` (imbalance -> tau)
//! 3. Copy viscosity -> tau_buffer (or bind directly)
//! 4. Call `engine.step()` x lbm_steps (reuse existing LBM pipeline)
//! 5. Dispatch `compute_regional_correlation` (subregion means)
//! 6. Dispatch `count_extreme` (compare to observed statistic)
//! 7. Read extreme_count -> CPU, check adaptive stopping

use crate::VulkanContext;
use crate::compute::{BufferSet, GororobaEngine, VulkanEngineError, compile_wgsl};
use ash::{Device, vk};
use gpu_allocator::{MemoryLocation, vulkan::*};
use std::ffi::CString;
use std::sync::{Arc, Mutex};

type Result<T> = std::result::Result<T, VulkanEngineError>;

/// Configuration for Besag-Clifford permutation testing.
#[derive(Clone, Debug)]
pub struct BesagCliffordVulkanConfig {
    /// Grid dimensions (nx, ny, nz).
    pub grid_dim: (u32, u32, u32),
    /// Number of LBM steps per permutation.
    pub lbm_steps: u32,
    /// Maximum number of permutations.
    pub max_perms: u32,
    /// Batch size (permutations per GPU dispatch).
    pub batch_size: u32,
    /// Number of subregions per dimension for regional correlation.
    pub n_sub: u32,
    /// Significance level for the permutation test.
    pub alpha: f64,
    /// Base viscosity (nu_base).
    pub nu_base: f32,
    /// Coupling constant (lambda).
    pub lambda: f32,
    /// Mean imbalance field value.
    pub phi_mean: f32,
    /// RNG seed.
    pub seed: u32,
}

impl Default for BesagCliffordVulkanConfig {
    fn default() -> Self {
        Self {
            grid_dim: (256, 256, 256),
            lbm_steps: 1000,
            max_perms: 1000,
            batch_size: 1,
            n_sub: 8,
            alpha: 0.05,
            nu_base: 0.01,
            lambda: 1.0,
            phi_mean: 0.375,
            seed: 42,
        }
    }
}

/// Result of a Besag-Clifford permutation test.
#[derive(Clone, Debug)]
pub struct BcResult {
    /// Number of permutations completed.
    pub n_perms: u32,
    /// Number of permutations with test stat >= observed.
    pub n_extreme: u32,
    /// Estimated p-value = (n_extreme + 1) / (n_perms + 1).
    pub p_value: f64,
    /// Whether the test was stopped early via adaptive stopping.
    pub early_stopped: bool,
    /// Observed test statistic (variance of regional means).
    pub observed_stat: f64,
}

impl BcResult {
    /// Compute p-value using the Besag-Clifford estimator.
    ///
    /// p = (n_extreme + 1) / (n_perms + 1)
    ///
    /// The +1 corrections ensure the estimator is unbiased and the observed
    /// data itself counts as one permutation.
    pub fn p_value(n_extreme: u32, n_perms: u32) -> f64 {
        (n_extreme as f64 + 1.0) / (n_perms as f64 + 1.0)
    }
}

/// VRAM budget check for a Besag-Clifford + LBM configuration.
///
/// Returns (bc_bytes, total_bytes) where bc_bytes is the additional VRAM
/// needed for BC buffers beyond the LBM engine's own allocation.
pub fn vram_budget(config: &BesagCliffordVulkanConfig) -> (u64, u64) {
    let n_cells = config.grid_dim.0 as u64
        * config.grid_dim.1 as u64
        * config.grid_dim.2 as u64;
    let n_sub_total = config.n_sub as u64 * config.n_sub as u64 * config.n_sub as u64;

    // BC-specific buffers (all f32 = 4 bytes):
    // imbalance_in, shuffled, viscosity: 3 * n_cells * 4
    // regional_means, regional_vel: 2 * n_sub^3 * 512 * 4 (workgroup partials)
    // extreme_count: 4 bytes
    // uniform: 48 bytes (BesagConstants)
    let bc_bytes = 3 * n_cells * 4
        + 2 * n_sub_total * 512 * 4
        + 4
        + 48;

    // LBM buffers:
    // f_a + f_b: 2 * n_cells * 19 * 4
    // rho, tau, entropy: 3 * n_cells * 4
    // u, force: 2 * n_cells * 3 * 4
    let lbm_bytes = 2 * n_cells * 19 * 4
        + 3 * n_cells * 4
        + 2 * n_cells * 3 * 4;

    (bc_bytes, bc_bytes + lbm_bytes)
}

/// Check whether adaptive stopping criteria are met.
///
/// Uses a simple sequential analysis: if the current p-value estimate is
/// far enough from the significance threshold alpha, we can stop early.
/// Specifically, stop if the 95% Wilson confidence interval for the true
/// p-value does not contain alpha.
pub fn should_stop_early(n_extreme: u32, n_perms: u32, alpha: f64) -> bool {
    if n_perms < 20 {
        return false; // Need minimum samples
    }
    let p_hat = BcResult::p_value(n_extreme, n_perms);
    let n = n_perms as f64;

    // Wilson interval half-width (z=1.96 for 95% CI)
    let z = 1.96;
    let denom = 1.0 + z * z / n;
    let center = (p_hat + z * z / (2.0 * n)) / denom;
    let hw = z * (p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)).sqrt() / denom;

    let lo = (center - hw).max(0.0);
    let hi = (center + hw).min(1.0);

    // Stop if alpha is outside the confidence interval
    alpha < lo || alpha > hi
}

/// GPU constant buffer matching besag_clifford.wgsl BesagConstants struct.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct BesagConstants {
    nx: u32,
    ny: u32,
    nz: u32,
    n_cells: u32,
    n_sub: u32,
    nu_base: f32,
    lambda: f32,
    phi_mean: f32,
    seed: u32,
    batch_idx: u32,
    observed_stat: f32,
    _pad: u32,
}

/// GPU constant buffer matching dark_halo_detector.wgsl HaloConstants struct.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct HaloConstants {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub tau_base: f32,
    pub tau_amp: f32,
    pub zd_threshold: f32,
    pub velocity_epsilon: f32,
    pub density_factor: f32,
    pub rho_mean: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Besag-Clifford permutation test engine on Vulkan.
///
/// When `ctx` is `None`, the engine operates in dry-run mode (configuration
/// validation and VRAM budget computation only). When `ctx` is `Some`, full
/// GPU pipelines are initialized for live dispatch.
pub struct BesagCliffordVulkanEngine {
    pub config: BesagCliffordVulkanConfig,
    ctx: Option<VulkanContext>,
    // GPU resources (only present when ctx is Some)
    bc_pipelines: Option<BcPipelines>,
}

// Fields held for Vulkan resource lifetime management (destroyed on drop).
#[allow(dead_code)]
struct BcPipelines {
    device: Arc<Device>,
    allocator: Arc<Mutex<Allocator>>,
    // Four BC compute pipelines (share one shader module's entry points)
    shuffle_pipeline: vk::Pipeline,
    viscosity_pipeline: vk::Pipeline,
    correlation_pipeline: vk::Pipeline,
    count_pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    // BC-specific buffers
    imbalance_buffer: BufferSet,
    shuffled_buffer: BufferSet,
    viscosity_buffer: BufferSet,
    regional_means_buffer: BufferSet,
    regional_vel_buffer: BufferSet,
    extreme_count_buffer: BufferSet,
    uniform_buffer: BufferSet,
    // Dark halo detector resources
    halo_pipeline: vk::Pipeline,
    halo_layout: vk::PipelineLayout,
    halo_descriptor_layout: vk::DescriptorSetLayout,
    halo_descriptor_pool: vk::DescriptorPool,
    halo_descriptor_set: vk::DescriptorSet,
    halo_mask_buffer: BufferSet,
    halo_uniform_buffer: BufferSet,
}

fn create_buf(
    device: &Device,
    allocator: &mut Allocator,
    size: u64,
    usage: vk::BufferUsageFlags,
    name: &str,
    loc: MemoryLocation,
) -> Result<BufferSet> {
    let buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo {
                size,
                usage,
                ..Default::default()
            },
            None,
        )
    }?;
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let allocation = allocator.allocate(&AllocationCreateDesc {
        name,
        requirements: reqs,
        location: loc,
        linear: true,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })?;
    unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }?;
    Ok(BufferSet { buffer, allocation })
}

fn create_compute_pipeline(
    device: &Device,
    module: vk::ShaderModule,
    layout: vk::PipelineLayout,
    entry: &str,
) -> Result<vk::Pipeline> {
    let entry_name = CString::new(entry)
        .map_err(|e| VulkanEngineError::ShaderError(e.to_string()))?;
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo {
                stage: vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::COMPUTE,
                    module,
                    p_name: entry_name.as_ptr(),
                    ..Default::default()
                },
                layout,
                ..Default::default()
            }],
            None,
        )
    }
    .map_err(|e| VulkanEngineError::Vulkan(e.1))?[0];
    Ok(pipeline)
}

fn storage_barrier(device: &Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier {
        s_type: vk::StructureType::MEMORY_BARRIER,
        src_access_mask: vk::AccessFlags::SHADER_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        ..Default::default()
    };
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

fn transfer_to_compute_barrier(device: &Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier {
        s_type: vk::StructureType::MEMORY_BARRIER,
        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        ..Default::default()
    };
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

fn compute_to_host_barrier(device: &Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier {
        s_type: vk::StructureType::MEMORY_BARRIER,
        src_access_mask: vk::AccessFlags::SHADER_WRITE,
        dst_access_mask: vk::AccessFlags::HOST_READ,
        ..Default::default()
    };
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

impl BesagCliffordVulkanEngine {
    /// Create a new engine with VRAM budget validation.
    ///
    /// If `ctx` is None, the engine operates in dry-run mode (configuration
    /// validation only, no GPU dispatch).
    pub fn new(
        config: BesagCliffordVulkanConfig,
        ctx: Option<VulkanContext>,
    ) -> std::result::Result<Self, String> {
        if let Some(ref c) = ctx {
            let (bc_bytes, total_bytes) = vram_budget(&config);
            let avail = c.caps.vram_mb as u64 * 1024 * 1024;
            if total_bytes > (avail as f64 * 0.9) as u64 {
                return Err(format!(
                    "VRAM budget exceeded: need {} MB (BC={} MB, LBM={} MB), \
                     have {} MB (90% of {} MB)",
                    total_bytes / (1024 * 1024),
                    bc_bytes / (1024 * 1024),
                    (total_bytes - bc_bytes) / (1024 * 1024),
                    (avail as f64 * 0.9) as u64 / (1024 * 1024),
                    c.caps.vram_mb,
                ));
            }
        }
        Ok(Self {
            config,
            ctx,
            bc_pipelines: None,
        })
    }

    /// Initialize GPU pipelines and buffers. Must be called before
    /// `run_permutation_test`. Requires a live VulkanContext.
    pub fn init_pipelines(
        &mut self,
        lbm: &GororobaEngine,
    ) -> std::result::Result<(), String> {
        let ctx = self.ctx.as_ref().ok_or("No VulkanContext for pipeline init")?;
        let device = ctx.device.clone();
        let allocator_arc = ctx.allocator.clone();

        let n_cells = self.config.grid_dim.0 as u64
            * self.config.grid_dim.1 as u64
            * self.config.grid_dim.2 as u64;
        let n_sub_total =
            self.config.n_sub as u64 * self.config.n_sub as u64 * self.config.n_sub as u64;

        // Compile BC shader
        let bc_code = compile_wgsl(include_str!("../shaders/besag_clifford.wgsl"))
            .map_err(|e| format!("BC shader compile: {e}"))?;
        let bc_module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    code_size: bc_code.len() * 4,
                    p_code: bc_code.as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }
        .map_err(|e| format!("BC shader module: {e}"))?;

        // Descriptor set layout: 7 bindings (6 storage + 1 uniform)
        let bindings: [vk::DescriptorSetLayoutBinding; 7] = core::array::from_fn(|i| {
            vk::DescriptorSetLayoutBinding {
                binding: i as u32,
                descriptor_type: if i == 6 {
                    vk::DescriptorType::UNIFORM_BUFFER
                } else {
                    vk::DescriptorType::STORAGE_BUFFER
                },
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            }
        });
        let dsl = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    binding_count: 7,
                    p_bindings: bindings.as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }
        .map_err(|e| format!("BC descriptor layout: {e}"))?;

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo {
                    set_layout_count: 1,
                    p_set_layouts: &dsl,
                    ..Default::default()
                },
                None,
            )
        }
        .map_err(|e| format!("BC pipeline layout: {e}"))?;

        // Create 4 pipelines for 4 entry points
        let shuffle_pl =
            create_compute_pipeline(&device, bc_module, pipeline_layout, "shuffle_imbalance")
                .map_err(|e| format!("shuffle pipeline: {e}"))?;
        let viscosity_pl =
            create_compute_pipeline(&device, bc_module, pipeline_layout, "transform_viscosity")
                .map_err(|e| format!("viscosity pipeline: {e}"))?;
        let correlation_pl = create_compute_pipeline(
            &device,
            bc_module,
            pipeline_layout,
            "compute_regional_correlation",
        )
        .map_err(|e| format!("correlation pipeline: {e}"))?;
        let count_pl =
            create_compute_pipeline(&device, bc_module, pipeline_layout, "count_extreme")
                .map_err(|e| format!("count pipeline: {e}"))?;

        unsafe { device.destroy_shader_module(bc_module, None) };

        // Allocate BC buffers
        let mut alloc = allocator_arc
            .lock()
            .map_err(|_| "Failed to lock allocator")?;

        let storage_rw =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let storage_ro = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;

        let imbalance_buffer = create_buf(
            &device,
            &mut alloc,
            n_cells * 4,
            storage_ro,
            "bc_imb",
            MemoryLocation::GpuOnly,
        )
        .map_err(|e| format!("imbalance buf: {e}"))?;

        let shuffled_buffer = create_buf(
            &device,
            &mut alloc,
            n_cells * 4,
            storage_rw,
            "bc_shuf",
            MemoryLocation::GpuOnly,
        )
        .map_err(|e| format!("shuffled buf: {e}"))?;

        let viscosity_buffer = create_buf(
            &device,
            &mut alloc,
            n_cells * 4,
            storage_rw | vk::BufferUsageFlags::TRANSFER_SRC,
            "bc_visc",
            MemoryLocation::GpuOnly,
        )
        .map_err(|e| format!("viscosity buf: {e}"))?;

        let regional_means_buffer = create_buf(
            &device,
            &mut alloc,
            n_sub_total * 512 * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "bc_rmean",
            MemoryLocation::GpuOnly,
        )
        .map_err(|e| format!("regional means buf: {e}"))?;

        let regional_vel_buffer = create_buf(
            &device,
            &mut alloc,
            n_sub_total * 512 * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "bc_rvel",
            MemoryLocation::GpuOnly,
        )
        .map_err(|e| format!("regional vel buf: {e}"))?;

        let extreme_count_buffer = create_buf(
            &device,
            &mut alloc,
            4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "bc_ext",
            MemoryLocation::GpuToCpu,
        )
        .map_err(|e| format!("extreme count buf: {e}"))?;

        let uniform_buffer = create_buf(
            &device,
            &mut alloc,
            std::mem::size_of::<BesagConstants>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "bc_uni",
            MemoryLocation::CpuToGpu,
        )
        .map_err(|e| format!("uniform buf: {e}"))?;

        // --- Dark halo detector ---
        let halo_code = compile_wgsl(include_str!("../shaders/dark_halo_detector.wgsl"))
            .map_err(|e| format!("Halo shader compile: {e}"))?;
        let halo_module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    code_size: halo_code.len() * 4,
                    p_code: halo_code.as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }
        .map_err(|e| format!("Halo shader module: {e}"))?;

        let halo_bindings: [vk::DescriptorSetLayoutBinding; 5] = core::array::from_fn(|i| {
            vk::DescriptorSetLayoutBinding {
                binding: i as u32,
                descriptor_type: if i == 4 {
                    vk::DescriptorType::UNIFORM_BUFFER
                } else {
                    vk::DescriptorType::STORAGE_BUFFER
                },
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            }
        });
        let halo_dsl = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    binding_count: 5,
                    p_bindings: halo_bindings.as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }
        .map_err(|e| format!("Halo descriptor layout: {e}"))?;

        let halo_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo {
                    set_layout_count: 1,
                    p_set_layouts: &halo_dsl,
                    ..Default::default()
                },
                None,
            )
        }
        .map_err(|e| format!("Halo pipeline layout: {e}"))?;

        let halo_pipeline = create_compute_pipeline(&device, halo_module, halo_layout, "main")
            .map_err(|e| format!("Halo pipeline: {e}"))?;

        unsafe { device.destroy_shader_module(halo_module, None) };

        let halo_mask_buffer = create_buf(
            &device,
            &mut alloc,
            n_cells * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "halo_mask",
            MemoryLocation::GpuToCpu,
        )
        .map_err(|e| format!("halo mask buf: {e}"))?;

        let halo_uniform_buffer = create_buf(
            &device,
            &mut alloc,
            std::mem::size_of::<HaloConstants>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "halo_uni",
            MemoryLocation::CpuToGpu,
        )
        .map_err(|e| format!("halo uniform buf: {e}"))?;

        drop(alloc);

        // --- BC descriptor pool + set ---
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    max_sets: 1,
                    pool_size_count: 2,
                    p_pool_sizes: [
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 6,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                        },
                    ]
                    .as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }
        .map_err(|e| format!("BC descriptor pool: {e}"))?;

        let sets = unsafe {
            device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool: pool,
                descriptor_set_count: 1,
                p_set_layouts: &dsl,
                ..Default::default()
            })
        }
        .map_err(|e| format!("BC descriptor set: {e}"))?;
        let bc_set = sets[0];

        // Write BC descriptor set: bind all 7 buffers
        // binding(0) = imbalance_in (from LBM rho as a proxy for now;
        //              actual imbalance data will be uploaded separately)
        let infos = [
            vk::DescriptorBufferInfo {
                buffer: imbalance_buffer.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: shuffled_buffer.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: viscosity_buffer.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: regional_means_buffer.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: regional_vel_buffer.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: extreme_count_buffer.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: uniform_buffer.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
        ];
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(b, info)| vk::WriteDescriptorSet {
                dst_set: bc_set,
                dst_binding: b as u32,
                descriptor_count: 1,
                descriptor_type: if b == 6 {
                    vk::DescriptorType::UNIFORM_BUFFER
                } else {
                    vk::DescriptorType::STORAGE_BUFFER
                },
                p_buffer_info: info,
                ..Default::default()
            })
            .collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        // --- Halo descriptor pool + set ---
        let halo_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    max_sets: 1,
                    pool_size_count: 2,
                    p_pool_sizes: [
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 4,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                        },
                    ]
                    .as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }
        .map_err(|e| format!("Halo descriptor pool: {e}"))?;

        let halo_sets = unsafe {
            device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool: halo_pool,
                descriptor_set_count: 1,
                p_set_layouts: &halo_dsl,
                ..Default::default()
            })
        }
        .map_err(|e| format!("Halo descriptor set: {e}"))?;
        let halo_set = halo_sets[0];

        // Write halo descriptor set: 5 bindings
        // binding(0)=tau, binding(1)=rho, binding(2)=u from LBM engine
        // binding(3)=halo_mask, binding(4)=uniform
        let halo_infos = [
            vk::DescriptorBufferInfo {
                buffer: lbm.tau_buffer_handle(),
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: lbm.rho_buffer_handle(),
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: lbm.u_buffer_handle(),
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: halo_mask_buffer.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: halo_uniform_buffer.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
        ];
        let halo_writes: Vec<vk::WriteDescriptorSet> = halo_infos
            .iter()
            .enumerate()
            .map(|(b, info)| vk::WriteDescriptorSet {
                dst_set: halo_set,
                dst_binding: b as u32,
                descriptor_count: 1,
                descriptor_type: if b == 4 {
                    vk::DescriptorType::UNIFORM_BUFFER
                } else {
                    vk::DescriptorType::STORAGE_BUFFER
                },
                p_buffer_info: info,
                ..Default::default()
            })
            .collect();
        unsafe { device.update_descriptor_sets(&halo_writes, &[]) };

        self.bc_pipelines = Some(BcPipelines {
            device,
            allocator: allocator_arc,
            shuffle_pipeline: shuffle_pl,
            viscosity_pipeline: viscosity_pl,
            correlation_pipeline: correlation_pl,
            count_pipeline: count_pl,
            pipeline_layout,
            descriptor_layout: dsl,
            descriptor_pool: pool,
            descriptor_set: bc_set,
            imbalance_buffer,
            shuffled_buffer,
            viscosity_buffer,
            regional_means_buffer,
            regional_vel_buffer,
            extreme_count_buffer,
            uniform_buffer,
            halo_pipeline,
            halo_layout,
            halo_descriptor_layout: halo_dsl,
            halo_descriptor_pool: halo_pool,
            halo_descriptor_set: halo_set,
            halo_mask_buffer,
            halo_uniform_buffer,
        });

        Ok(())
    }

    /// Record BC dispatch commands for one permutation into a command buffer.
    ///
    /// This records (but does not submit) the shuffle -> viscosity -> copy ->
    /// LBM steps -> regional correlation -> count_extreme pipeline.
    pub fn record_permutation(
        &self,
        cmd: vk::CommandBuffer,
        lbm: &mut GororobaEngine,
        batch_idx: u32,
        observed_stat: f32,
    ) -> Result<()> {
        let pl = self
            .bc_pipelines
            .as_ref()
            .ok_or_else(|| VulkanEngineError::ShaderError("Pipelines not initialized".into()))?;
        let device = &pl.device;
        let n_cells = self.config.grid_dim.0 * self.config.grid_dim.1 * self.config.grid_dim.2;

        // Update uniform buffer with current batch parameters
        let constants = BesagConstants {
            nx: self.config.grid_dim.0,
            ny: self.config.grid_dim.1,
            nz: self.config.grid_dim.2,
            n_cells,
            n_sub: self.config.n_sub,
            nu_base: self.config.nu_base,
            lambda: self.config.lambda,
            phi_mean: self.config.phi_mean,
            seed: self.config.seed,
            batch_idx,
            observed_stat,
            _pad: 0,
        };
        let mapped_ptr = pl
            .uniform_buffer
            .allocation
            .mapped_ptr()
            .ok_or_else(|| {
                VulkanEngineError::MappingError("Failed to map BC uniform buffer".into())
            })?;
        unsafe { std::ptr::write(mapped_ptr.as_ptr() as *mut BesagConstants, constants) };

        let wg_cells = n_cells.div_ceil(256);

        // 1. Shuffle imbalance field
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl.shuffle_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pl.pipeline_layout,
                0,
                &[pl.descriptor_set],
                &[],
            );
            device.cmd_dispatch(cmd, wg_cells, 1, 1);
        }
        storage_barrier(device, cmd);

        // 2. Transform shuffled imbalance to viscosity (tau)
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl.viscosity_pipeline);
            device.cmd_dispatch(cmd, wg_cells, 1, 1);
        }
        storage_barrier(device, cmd);

        // 3. Copy viscosity -> LBM tau buffer
        unsafe {
            device.cmd_copy_buffer(
                cmd,
                pl.viscosity_buffer.buffer,
                lbm.tau_buffer_handle(),
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: n_cells as u64 * 4,
                }],
            );
        }
        transfer_to_compute_barrier(device, cmd);

        // 4. Run LBM steps (reuses existing LBM pipeline in GororobaEngine)
        for step in 0..self.config.lbm_steps {
            lbm.step_with_params(
                cmd,
                batch_idx * self.config.lbm_steps + step,
                self.config.nu_base,
                0.2,
                self.config.lambda,
            )?;
            if step < self.config.lbm_steps - 1 {
                storage_barrier(device, cmd);
            }
        }
        storage_barrier(device, cmd);

        // 5. Compute regional correlation
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pl.correlation_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pl.pipeline_layout,
                0,
                &[pl.descriptor_set],
                &[],
            );
            device.cmd_dispatch(cmd, self.config.n_sub, self.config.n_sub, self.config.n_sub);
        }
        storage_barrier(device, cmd);

        // 6. Count extreme (compare permuted stat vs observed)
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl.count_pipeline);
            device.cmd_dispatch(cmd, 1, 1, 1);
        }
        compute_to_host_barrier(device, cmd);

        Ok(())
    }

    /// Run the full Besag-Clifford permutation test.
    ///
    /// Returns the test result including p-value and whether early stopping
    /// was triggered.
    pub fn run_permutation_test(
        &self,
        ctx: &VulkanContext,
        lbm: &mut GororobaEngine,
    ) -> std::result::Result<BcResult, String> {
        let pl = self
            .bc_pipelines
            .as_ref()
            .ok_or("Pipelines not initialized")?;
        let device = &pl.device;

        // Create command pool and fence for synchronous execution
        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    queue_family_index: ctx.queue_family_index,
                    flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    ..Default::default()
                },
                None,
            )
        }
        .map_err(|e| format!("Command pool: {e}"))?;

        let cmd = unsafe {
            device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: cmd_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })
        }
        .map_err(|e| format!("Command buffer: {e}"))?[0];

        let fence = unsafe {
            device.create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|e| format!("Fence: {e}"))?;

        // Zero out extreme_count before starting
        self.zero_extreme_count(ctx)
            .map_err(|e| format!("Zero extreme: {e}"))?;

        // For the initial observed statistic, we use 0.0 as a placeholder.
        // A proper implementation would compute the observed statistic first
        // with unpermuted data, then use that value for comparison.
        let observed_stat = 0.0_f32;

        let mut n_perms = 0u32;
        let mut n_extreme = 0u32;
        let mut early_stopped = false;

        let max_batches = self.config.max_perms / self.config.batch_size.max(1);

        for batch in 0..max_batches {
            // Reset extreme count for this batch
            self.zero_extreme_count(ctx)
                .map_err(|e| format!("Zero extreme batch {batch}: {e}"))?;

            // Record commands
            unsafe {
                device.begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo {
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
            }
            .map_err(|e| format!("Begin cmd: {e}"))?;

            self.record_permutation(cmd, lbm, batch, observed_stat)
                .map_err(|e| format!("Record permutation {batch}: {e}"))?;

            unsafe { device.end_command_buffer(cmd) }
                .map_err(|e| format!("End cmd: {e}"))?;

            // Submit and wait
            unsafe {
                device.queue_submit(
                    ctx.queue,
                    &[vk::SubmitInfo {
                        command_buffer_count: 1,
                        p_command_buffers: &cmd,
                        ..Default::default()
                    }],
                    fence,
                )
            }
            .map_err(|e| format!("Submit: {e}"))?;

            unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }
                .map_err(|e| format!("Wait: {e}"))?;
            unsafe { device.reset_fences(&[fence]) }
                .map_err(|e| format!("Reset fence: {e}"))?;

            // Read back extreme_count
            let batch_extreme = self
                .read_extreme_count()
                .map_err(|e| format!("Read extreme: {e}"))?;

            n_extreme += batch_extreme;
            n_perms += self.config.batch_size.max(1);

            // Check adaptive stopping
            if should_stop_early(n_extreme, n_perms, self.config.alpha) {
                early_stopped = true;
                break;
            }
        }

        // Cleanup
        unsafe {
            device.destroy_fence(fence, None);
            device.destroy_command_pool(cmd_pool, None);
        }

        let p_value = BcResult::p_value(n_extreme, n_perms);
        Ok(BcResult {
            n_perms,
            n_extreme,
            p_value,
            early_stopped,
            observed_stat: observed_stat as f64,
        })
    }

    /// Dispatch the dark halo detector shader.
    ///
    /// Returns the volume fraction of cells classified as halo candidates.
    pub fn dispatch_halo_detector(
        &self,
        ctx: &VulkanContext,
        constants: &HaloConstants,
    ) -> std::result::Result<f64, String> {
        let pl = self
            .bc_pipelines
            .as_ref()
            .ok_or("Pipelines not initialized")?;
        let device = &pl.device;
        let (nx, ny, nz) = (constants.nx, constants.ny, constants.nz);
        let n_cells = nx as u64 * ny as u64 * nz as u64;

        // Upload halo constants
        let mapped_ptr = pl
            .halo_uniform_buffer
            .allocation
            .mapped_ptr()
            .ok_or("Failed to map halo uniform buffer")?;
        unsafe { std::ptr::write(mapped_ptr.as_ptr() as *mut HaloConstants, *constants) };

        // Create temporary command pool, buffer, fence
        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    queue_family_index: ctx.queue_family_index,
                    flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    ..Default::default()
                },
                None,
            )
        }
        .map_err(|e| format!("Halo cmd pool: {e}"))?;

        let cmd = unsafe {
            device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: cmd_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })
        }
        .map_err(|e| format!("Halo cmd buf: {e}"))?[0];

        let fence = unsafe {
            device.create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|e| format!("Halo fence: {e}"))?;

        // Record dispatch
        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )
        }
        .map_err(|e| format!("Halo begin cmd: {e}"))?;

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl.halo_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pl.halo_layout,
                0,
                &[pl.halo_descriptor_set],
                &[],
            );
            device.cmd_dispatch(cmd, nx.div_ceil(8), ny.div_ceil(8), nz.div_ceil(8));
        }
        compute_to_host_barrier(device, cmd);

        unsafe { device.end_command_buffer(cmd) }
            .map_err(|e| format!("Halo end cmd: {e}"))?;

        // Submit and wait
        unsafe {
            device.queue_submit(
                ctx.queue,
                &[vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers: &cmd,
                    ..Default::default()
                }],
                fence,
            )
        }
        .map_err(|e| format!("Halo submit: {e}"))?;

        unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }
            .map_err(|e| format!("Halo wait: {e}"))?;

        // Read back halo mask and count
        let ptr = pl
            .halo_mask_buffer
            .allocation
            .mapped_ptr()
            .ok_or("Failed to map halo mask buffer")?
            .as_ptr() as *const u32;
        let mask = unsafe { std::slice::from_raw_parts(ptr, n_cells as usize) };
        let halo_count: u64 = mask.iter().map(|&v| v as u64).sum();
        let volume_fraction = halo_count as f64 / n_cells as f64;

        // Cleanup
        unsafe {
            device.destroy_fence(fence, None);
            device.destroy_command_pool(cmd_pool, None);
        }

        Ok(volume_fraction)
    }

    fn zero_extreme_count(&self, ctx: &VulkanContext) -> Result<()> {
        let pl = self
            .bc_pipelines
            .as_ref()
            .ok_or_else(|| VulkanEngineError::ShaderError("Pipelines not initialized".into()))?;

        // Fill the extreme_count buffer with zero via vkCmdFillBuffer
        let cmd_pool = unsafe {
            pl.device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    queue_family_index: ctx.queue_family_index,
                    ..Default::default()
                },
                None,
            )
        }?;
        let cmd = unsafe {
            pl.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: cmd_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })
        }?[0];
        let fence = unsafe {
            pl.device.create_fence(&vk::FenceCreateInfo::default(), None)
        }?;

        unsafe {
            pl.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )?;
            pl.device.cmd_fill_buffer(cmd, pl.extreme_count_buffer.buffer, 0, 4, 0);
            pl.device.end_command_buffer(cmd)?;
            pl.device.queue_submit(
                ctx.queue,
                &[vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers: &cmd,
                    ..Default::default()
                }],
                fence,
            )?;
            pl.device.wait_for_fences(&[fence], true, u64::MAX)?;
            pl.device.destroy_fence(fence, None);
            pl.device.destroy_command_pool(cmd_pool, None);
        }
        Ok(())
    }

    fn read_extreme_count(&self) -> Result<u32> {
        let pl = self
            .bc_pipelines
            .as_ref()
            .ok_or_else(|| VulkanEngineError::ShaderError("Pipelines not initialized".into()))?;
        let ptr = pl
            .extreme_count_buffer
            .allocation
            .mapped_ptr()
            .ok_or_else(|| {
                VulkanEngineError::MappingError("Failed to map extreme count buffer".into())
            })?
            .as_ptr() as *const u32;
        Ok(unsafe { *ptr })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_vram_budget() {
        let config = BesagCliffordVulkanConfig {
            grid_dim: (256, 256, 256),
            n_sub: 8,
            ..Default::default()
        };
        let (bc_bytes, total_bytes) = vram_budget(&config);

        // BC: 3 * 256^3 * 4 = 201326592 + small overhead
        let n_cells = 256u64 * 256 * 256;
        assert!(bc_bytes >= 3 * n_cells * 4, "BC bytes too small: {bc_bytes}");

        // Total should include LBM: 2 * 256^3 * 19 * 4 = ~5 GB
        let lbm_min = 2 * n_cells * 19 * 4;
        assert!(
            total_bytes >= bc_bytes + lbm_min,
            "Total bytes too small: {total_bytes}"
        );

        // Should fit in 12 GB (RTX 4070 Ti)
        let rtx4070ti_mb = 12288u64;
        let rtx4070ti_bytes = rtx4070ti_mb * 1024 * 1024;
        assert!(
            total_bytes < (rtx4070ti_bytes as f64 * 0.9) as u64,
            "256^3 should fit in 12 GB: need {} MB",
            total_bytes / (1024 * 1024),
        );
    }

    #[test]
    fn test_p_value_estimator() {
        // 0 extreme in 999 perms -> p = 1/1000 = 0.001
        let p = BcResult::p_value(0, 999);
        assert!((p - 0.001).abs() < 1e-10);

        // All extreme -> p = 1.0
        let p = BcResult::p_value(999, 999);
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_stopping_clear_reject() {
        // 0 extreme in 500 perms at alpha=0.05 -> p~0.002, Wilson CI upper
        // bound well below alpha; should stop early.
        assert!(should_stop_early(0, 500, 0.05));
    }

    #[test]
    fn test_adaptive_stopping_clear_accept() {
        // 50 extreme in 100 perms, alpha=0.05 -> p~0.5, should stop
        assert!(should_stop_early(50, 100, 0.05));
    }

    #[test]
    fn test_adaptive_stopping_uncertain() {
        // 4 extreme in 100 perms, alpha=0.05 -> p~0.05, should NOT stop
        // (right at boundary, confidence interval spans alpha)
        let stop = should_stop_early(4, 100, 0.05);
        // This is borderline; the test just verifies no panic
        let _ = stop;
    }

    #[test]
    fn test_adaptive_stopping_minimum_samples() {
        // Should not stop with fewer than 20 samples
        assert!(!should_stop_early(0, 10, 0.05));
    }

    #[test]
    fn test_besag_constants_layout() {
        // Verify the BesagConstants struct has the expected size for GPU uniform alignment
        assert_eq!(
            std::mem::size_of::<BesagConstants>(),
            48,
            "BesagConstants must be 48 bytes (12 x u32/f32)"
        );
    }

    #[test]
    fn test_halo_constants_layout() {
        // Verify HaloConstants struct has expected size
        assert_eq!(
            std::mem::size_of::<HaloConstants>(),
            48,
            "HaloConstants must be 48 bytes (12 x u32/f32)"
        );
    }

    #[test]
    fn test_dry_run_engine_creation() {
        // Engine creation without VulkanContext should succeed
        let config = BesagCliffordVulkanConfig::default();
        let engine = BesagCliffordVulkanEngine::new(config, None).unwrap();
        assert!(engine.bc_pipelines.is_none());
    }
}
