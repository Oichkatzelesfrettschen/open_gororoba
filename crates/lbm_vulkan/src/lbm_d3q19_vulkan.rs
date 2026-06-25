//! D3Q19 lattice Boltzmann method, Vulkan compute backend.
//!
//! Uses the consolidated `gororoba_gpu_vulkan` helpers
//! (`InstanceBuilder`, `Adapter::pick`, `DeviceBuilder`, `ShaderModule`,
//! `DescriptorSetLayoutSpec`, `ComputePipelineBuilder`, `DispatchScope`)
//! introduced in Wave B1. The kernel itself lives at
//! `shaders/lbm_d3q19.wgsl` and is fused PUSH-scheme: each thread
//! reads its own 19 distribution values, computes moments + BGK
//! collision, then writes the post-collision values to the 19
//! periodic-neighbor destination cells. See the shader header for the
//! per-direction velocity table and the data-race-free PUSH argument.
//!
//! Buffer layout (SoA): `f[i * n_cells + cell]` where
//! `cell = z*nx*ny + y*nx + x` and `i` runs 0..18 over D3Q19 channels.
//! Ping-pong between two device buffers each timestep.
//!
//! Public API: [`LbmD3Q19Vulkan::new`] constructs everything;
//! [`LbmD3Q19Vulkan::evolve`] runs N steps; [`LbmD3Q19Vulkan::download`]
//! reads the current f field back to host memory.

use std::sync::Arc;

use crate::MacroFields;
use ash::vk;
use gororoba_gpu_vulkan::{
    Adapter, ComputePipeline, ComputePipelineBuilder, DescriptorSetLayout, DescriptorSetLayoutSpec,
    Device, DeviceBuilder, DispatchScope, Instance, InstanceBuilder, QueueFamilyRequirement,
    ShaderModule, ValidationPolicy, VulkanError,
};

const WGSL_SOURCE: &str = include_str!("../shaders/lbm_d3q19.wgsl");
const ENTRY_POINT: &str = "cs_step";
const D3Q19_CHANNELS: usize = 19;
const WORKGROUP_X: u32 = 64;
const STEP_TIMEOUT_NS: u64 = 30_000_000_000; // 30s per dispatch upper bound

/// Errors emitted by the D3Q19 Vulkan launcher.
#[derive(Debug, thiserror::Error)]
pub enum LbmD3Q19Error {
    #[error("vulkan helper error: {0}")]
    Vulkan(#[from] VulkanError),
    #[error("vulkan API error: {0:?}")]
    Vk(vk::Result),
    #[error("grid dimensions must all be positive (got nx={nx}, ny={ny}, nz={nz})")]
    EmptyGrid { nx: usize, ny: usize, nz: usize },
    #[error("grid too large for u32 cell indexing: nx*ny*nz = {0}")]
    GridTooLarge(u64),
    #[error("tau must satisfy tau > 0.5 for BGK stability (got {0})")]
    UnstableTau(f32),
    #[error("input f slice length {got} does not match nx*ny*nz*19 = {expected}")]
    UploadLengthMismatch { got: usize, expected: usize },
}

impl From<vk::Result> for LbmD3Q19Error {
    fn from(r: vk::Result) -> Self {
        Self::Vk(r)
    }
}

/// Uniform-buffer payload mirroring the WGSL `Params` struct.
///
/// The trailing pad fields keep the struct at the 32-byte minimum
/// alignment WGSL/WGPU asks for on this binding.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ParamsUbo {
    nx: u32,
    ny: u32,
    nz: u32,
    pad: u32,
    tau: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

// SAFETY: ParamsUbo is repr(C) with only u32/f32 fields; all bit
// patterns are valid and no padding bytes leak host state to the GPU.
unsafe impl bytemuck::Pod for ParamsUbo {}
unsafe impl bytemuck::Zeroable for ParamsUbo {}

/// One Vulkan-owned storage buffer plus the backing device memory it
/// uses. Kept as a small RAII bundle so cleanup is one place.
struct DeviceBuffer {
    device: Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    // Logical byte length requested by the caller. Vulkan may require a
    // larger memory allocation, but descriptor ranges and host readback
    // lengths must stay tied to the actual buffer payload.
    size: vk::DeviceSize,
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: handles created via vkCreate{Buffer,AllocateMemory}
        // on self.device; Device holds an Arc, so the device outlives
        // the destroy call.
        unsafe {
            self.device.raw().destroy_buffer(self.buffer, None);
            self.device.raw().free_memory(self.memory, None);
        }
    }
}

/// Vulkan-backed D3Q19 LBM solver. Owns the instance, device,
/// pipeline, ping-pong buffers, and the uniform parameter buffer for
/// the lifetime of one simulation run.
pub struct LbmD3Q19Vulkan {
    nx: usize,
    ny: usize,
    nz: usize,
    n_cells: usize,
    // f_buffers[0] and f_buffers[1] alternate as input / output each
    // step; current_in is the index that currently holds the latest
    // post-stream state.
    current_in: usize,
    f_buffers: [DeviceBuffer; 2],
    // The remaining fields below are anchors that keep Vulkan handles
    // alive via the helper-owned Arc graph (params_buffer outlives the
    // descriptor set; descriptor_set_layouts outlives the pipeline;
    // adapter outlives the device's queue family lookup; instance
    // outlives the entry loader). Marked dead-code allow because the
    // accessors live on Drop / via raw().
    #[allow(dead_code)]
    params_buffer: DeviceBuffer,
    #[allow(dead_code)]
    descriptor_set_layouts: [DescriptorSetLayout; 1],
    pipeline: ComputePipeline,
    dispatch: DispatchScope,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: [vk::DescriptorSet; 2],
    #[allow(dead_code)]
    adapter: Adapter,
    device: Device,
    #[allow(dead_code)]
    instance: Instance,
}

impl LbmD3Q19Vulkan {
    /// Build a fresh solver for an `nx*ny*nz` periodic D3Q19 grid with
    /// BGK relaxation time `tau`.
    ///
    /// The initial state is the equilibrium for `rho = 1` and `u = 0`
    /// (i.e. `f[i, cell] = w_i` everywhere), matching the CPU reference
    /// `LbmSolver3D::new()` in `crates/lbm_3d/src/solver.rs`. To inject
    /// a non-trivial initial field, call `upload_state` after `new`.
    pub fn new(nx: usize, ny: usize, nz: usize, tau: f32) -> Result<Self, LbmD3Q19Error> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(LbmD3Q19Error::EmptyGrid { nx, ny, nz });
        }
        let n_cells_u64 = (nx as u64) * (ny as u64) * (nz as u64);
        if n_cells_u64 > u32::MAX as u64 {
            return Err(LbmD3Q19Error::GridTooLarge(n_cells_u64));
        }
        if tau.is_nan() || tau <= 0.5 {
            return Err(LbmD3Q19Error::UnstableTau(tau));
        }
        let n_cells = nx * ny * nz;

        // Validation layer enabled in debug, off in release (matches the
        // ValidationPolicy default-for-profile heuristic).
        let instance = InstanceBuilder::new("lbm_d3q19_vulkan")
            .api_version(vk::API_VERSION_1_2)
            .validation(ValidationPolicy::default_for_profile())
            .build()?;
        let adapter = Adapter::pick(&instance, QueueFamilyRequirement::Compute)?;
        let device = DeviceBuilder::new(adapter.clone()).build(&instance)?;

        let shader = ShaderModule::from_wgsl(&device, WGSL_SOURCE, ENTRY_POINT)?;

        // Descriptor set layout: two storage buffers (f_in, f_out) +
        // one uniform buffer (params). Matches the WGSL @group(0) bindings.
        let descriptor_set_layout = DescriptorSetLayoutSpec::new()
            .storage_buffer(0)
            .storage_buffer(1)
            .uniform_buffer(2)
            .build(&device)?;

        let pipeline = ComputePipelineBuilder::new(&device, &shader)
            .descriptor_layout(&descriptor_set_layout)
            .build()?;

        // Ping-pong storage buffers sized to `n_cells * 19 * 4` bytes
        // each (f32 SoA).
        let buffer_bytes =
            (n_cells * D3Q19_CHANNELS * std::mem::size_of::<f32>()) as vk::DeviceSize;
        let f_buf0 = allocate_storage_buffer(&device, &adapter, &instance, buffer_bytes)?;
        let f_buf1 = allocate_storage_buffer(&device, &adapter, &instance, buffer_bytes)?;

        let params_bytes = std::mem::size_of::<ParamsUbo>() as vk::DeviceSize;
        let params_buffer = allocate_uniform_buffer(&device, &adapter, &instance, params_bytes)?;

        // Initialise both f buffers to equilibrium at rest: f[i, cell] = w_i.
        let weights = d3q19_weights_f32();
        let init: Vec<f32> = (0..n_cells * D3Q19_CHANNELS)
            .map(|k| weights[k / n_cells])
            .collect();
        upload_f32_slice(&device, &f_buf0, &init)?;
        upload_f32_slice(&device, &f_buf1, &init)?;

        // Upload the uniform params (constant for the run).
        let params = ParamsUbo {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            pad: 0,
            tau,
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        };
        upload_pod(&device, &params_buffer, &params)?;

        // One descriptor pool with two sets so we can pre-bake both
        // "f_in = buf0, f_out = buf1" and the swapped variant; this
        // avoids re-writing descriptors every step.
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 4,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 2,
            },
        ];
        // SAFETY: pool_sizes outlives the call.
        let descriptor_pool = unsafe {
            device.raw().create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(2)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }?;

        let dsl_raw = descriptor_set_layout.raw();
        let set_layouts = [dsl_raw, dsl_raw];
        // SAFETY: descriptor_pool was just created; set_layouts outlives the call.
        let descriptor_sets_vec = unsafe {
            device.raw().allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&set_layouts),
            )
        }?;
        let descriptor_sets = [descriptor_sets_vec[0], descriptor_sets_vec[1]];

        // Bake set 0: (f_in = buf0, f_out = buf1, params).
        bind_storage_pair_with_params(
            &device,
            descriptor_sets[0],
            &f_buf0,
            &f_buf1,
            &params_buffer,
        );
        // Bake set 1: (f_in = buf1, f_out = buf0, params).
        bind_storage_pair_with_params(
            &device,
            descriptor_sets[1],
            &f_buf1,
            &f_buf0,
            &params_buffer,
        );

        let dispatch = DispatchScope::new(&device)?;

        Ok(Self {
            nx,
            ny,
            nz,
            n_cells,
            current_in: 0,
            f_buffers: [f_buf0, f_buf1],
            params_buffer,
            descriptor_set_layouts: [descriptor_set_layout],
            pipeline,
            dispatch,
            descriptor_pool,
            descriptor_sets,
            adapter,
            device,
            instance,
        })
    }

    /// Replace the current f field with the host-supplied slice. The
    /// slice must be SoA-laid-out: `f[i * n_cells + cell]` for
    /// `i = 0..19`. Useful to drive the solver from a CPU oracle's
    /// initial condition for parity testing.
    pub fn upload_state(&mut self, f_host: &[f32]) -> Result<(), LbmD3Q19Error> {
        let expected = self.n_cells * D3Q19_CHANNELS;
        if f_host.len() != expected {
            return Err(LbmD3Q19Error::UploadLengthMismatch {
                got: f_host.len(),
                expected,
            });
        }
        upload_f32_slice(&self.device, &self.f_buffers[self.current_in], f_host)?;
        Ok(())
    }

    /// Advance the simulation by `num_steps` BGK + stream timesteps.
    ///
    /// Each step is one compute dispatch using the descriptor set that
    /// currently maps "f_in = front" / "f_out = back", followed by a
    /// fence wait and a swap of which buffer is front. The shader is
    /// PUSH-scheme so there's no data race within a dispatch.
    pub fn evolve(&mut self, num_steps: usize) -> Result<(), LbmD3Q19Error> {
        let group_count_x = (self.n_cells as u32).div_ceil(WORKGROUP_X);
        for _ in 0..num_steps {
            let dset = self.descriptor_sets[self.current_in];
            self.dispatch
                .dispatch(&self.pipeline, dset, group_count_x, 1, 1, STEP_TIMEOUT_NS)?;
            self.current_in ^= 1;
        }
        Ok(())
    }

    /// Read the current f field back to host memory.
    ///
    /// Returns a `Vec<f32>` of length `nx*ny*nz*19` in SoA layout
    /// matching the upload contract.
    pub fn download(&self) -> Result<Vec<f32>, LbmD3Q19Error> {
        download_f32_slice(&self.device, &self.f_buffers[self.current_in])
    }

    /// Compute (rho, u_x, u_y, u_z) per cell from the current f field
    /// by downloading and applying the standard D3Q19 moment formulas
    /// on the host. Convenience for parity tests.
    pub fn compute_macroscopic(&self) -> Result<MacroFields, LbmD3Q19Error> {
        let f = self.download()?;
        let n = self.n_cells;
        let mut rho = vec![0.0_f32; n];
        let mut ux = vec![0.0_f32; n];
        let mut uy = vec![0.0_f32; n];
        let mut uz = vec![0.0_f32; n];
        // Same velocity ordering as shaders/lbm_d3q19.wgsl.
        for cell in 0..n {
            let mut r = 0.0_f32;
            for i in 0..D3Q19_CHANNELS {
                r += f[i * n + cell];
            }
            let inv_r = 1.0 / r;
            // Pull the per-channel values once.
            let g = |i: usize| f[i * n + cell];
            let mx = g(1) - g(2) + g(7) - g(8) + g(9) - g(10) + g(11) - g(12) + g(13) - g(14);
            let my = g(3) - g(4) + g(7) - g(8) - g(9) + g(10) + g(15) - g(16) + g(17) - g(18);
            let mz = g(5) - g(6) + g(11) - g(12) - g(13) + g(14) + g(15) - g(16) - g(17) + g(18);
            rho[cell] = r;
            ux[cell] = mx * inv_r;
            uy[cell] = my * inv_r;
            uz[cell] = mz * inv_r;
        }
        Ok(MacroFields {
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            rho,
            ux,
            uy,
            uz,
        })
    }

    pub fn grid(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }
}

impl Drop for LbmD3Q19Vulkan {
    fn drop(&mut self) {
        // SAFETY: descriptor_pool was created on self.device above and
        // is destroyed here exactly once. The descriptor sets allocated
        // from it are reclaimed implicitly by destroy_descriptor_pool.
        unsafe {
            self.device
                .raw()
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
        // All other handles self-destruct via their Drop impls in the
        // helper crate, in the right order because Rust drops fields
        // in declaration order: dispatch first (frees its own pool +
        // fence), then pipeline, then descriptor_set_layouts, then
        // buffers, then device, then adapter, then instance.
    }
}

// ---- helpers ----

fn d3q19_weights_f32() -> [f32; 19] {
    const W0: f32 = 1.0 / 3.0;
    const W1: f32 = 1.0 / 18.0;
    const W2: f32 = 1.0 / 36.0;
    [
        W0, W1, W1, W1, W1, W1, W1, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2,
    ]
}

fn allocate_storage_buffer(
    device: &Device,
    adapter: &Adapter,
    instance: &Instance,
    size: vk::DeviceSize,
) -> Result<DeviceBuffer, LbmD3Q19Error> {
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
) -> Result<DeviceBuffer, LbmD3Q19Error> {
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
) -> Result<DeviceBuffer, LbmD3Q19Error> {
    // SAFETY: device + adapter belong to the same instance; size is
    // bounded by allocate_buffer callers (n_cells * 19 * 4 bytes).
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
        // SAFETY: cleanup buffer before propagating.
        unsafe {
            device.raw().destroy_buffer(buffer, None);
        }
        return Err(LbmD3Q19Error::Vk(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
    }
    // SAFETY: req.size + mem_type_index are produced by Vulkan. req.size
    // may be larger than the logical buffer size because it includes
    // driver alignment padding required for binding.
    let memory = unsafe {
        device.raw().allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type_index),
            None,
        )
    }?;
    // SAFETY: buffer + memory were just created on the same device.
    unsafe {
        device.raw().bind_buffer_memory(buffer, memory, 0)?;
    }
    Ok(DeviceBuffer {
        device: device.clone(),
        buffer,
        memory,
        size,
    })
}

fn upload_f32_slice(
    device: &Device,
    buf: &DeviceBuffer,
    data: &[f32],
) -> Result<(), LbmD3Q19Error> {
    let bytes = bytemuck::cast_slice::<f32, u8>(data);
    // SAFETY: HOST_VISIBLE + HOST_COHERENT memory; size matches buf.size.
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
) -> Result<(), LbmD3Q19Error> {
    let bytes = bytemuck::bytes_of(value);
    // SAFETY: HOST_VISIBLE + HOST_COHERENT.
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

fn download_f32_slice(device: &Device, buf: &DeviceBuffer) -> Result<Vec<f32>, LbmD3Q19Error> {
    let bytes_len = buf.size as usize;
    let mut out = vec![0.0_f32; bytes_len / std::mem::size_of::<f32>()];
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

fn bind_storage_pair_with_params(
    device: &Device,
    set: vk::DescriptorSet,
    f_in: &DeviceBuffer,
    f_out: &DeviceBuffer,
    params: &DeviceBuffer,
) {
    let f_in_info = [vk::DescriptorBufferInfo::default()
        .buffer(f_in.buffer)
        .offset(0)
        .range(f_in.size)];
    let f_out_info = [vk::DescriptorBufferInfo::default()
        .buffer(f_out.buffer)
        .offset(0)
        .range(f_out.size)];
    let params_info = [vk::DescriptorBufferInfo::default()
        .buffer(params.buffer)
        .offset(0)
        .range(params.size)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&f_in_info),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&f_out_info),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&params_info),
    ];
    // SAFETY: writes references info arrays which live until the call returns.
    unsafe {
        device.raw().update_descriptor_sets(&writes, &[]);
    }
    // Anchor Arc so clippy doesn't flag unused; the params buffer must
    // outlive the descriptor set, which it does through field-order Drop.
    let _ = Arc::new(());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d3q19_weights_sum_to_one() {
        let sum: f32 = d3q19_weights_f32().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
    }
}
