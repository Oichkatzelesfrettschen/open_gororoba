//! VulkanQuantizer: dispatches the quantize.comp SPIR-V shader for
//! TurboQuant boundary-search quantization on the GPU.
//!
//! This module is feature-gated on `vulkan` (which enables the `ash` dep
//! plus `gororoba_gpu_vulkan`'s helpers). Initialization boilerplate
//! (Vulkan loader, instance, physical-device selection, queue, shader
//! module, descriptor set layout, compute pipeline) is delegated to the
//! shared helpers in `gororoba_gpu_vulkan`; the bespoke per-call
//! quantize() pipeline (host-visible memory allocation, push constants,
//! descriptor pool/set, command buffer recording, fence wait) stays in
//! this file because no other consumer needs that exact shape.
//!
//! # Universal SAFETY argument
//!
//! Every `unsafe { device.<vk_fn>(...) }` call follows the same pattern
//! used by lbm_vulkan/compute.rs and documented at the top of that file:
//! 1. Active instance/device borrowed from VulkanQuantizer fields
//! 2. Handle lifetimes are paired with Drop destruction (this file +
//!    the gpu_vulkan helpers' Drop impls)
//! 3. Synchronization via vk::Fence
//! 4. Descriptor-set bindings match the pipeline layout's signature
//! 5. Memory alignment is satisfied by selecting HOST_VISIBLE +
//!    HOST_COHERENT memory types whose alignment >= the
//!    vk::MemoryRequirements reported by the device.

#![cfg(feature = "vulkan")]

use ash::vk;
use gororoba_gpu_vulkan::{
    Adapter, ComputePipeline, ComputePipelineBuilder, DescriptorSetLayout,
    DescriptorSetLayoutSpec, Device, DeviceBuilder, Instance, InstanceBuilder, QueueFamilyRequirement,
    ShaderModule, ValidationPolicy, VulkanError,
};

use super::shaders::quantize_spv;

/// Errors emitted by the VulkanQuantizer constructor or quantize().
#[derive(Debug)]
pub enum VulkanQuantizerError {
    /// glslc was missing at build time so no SPIR-V is embedded.
    NoShaderBytes,
    /// Vulkan loader failed to initialize.
    VulkanLoad(String),
    /// Vulkan API returned a non-success result.
    Vk(vk::Result),
    /// No Vulkan device with compute queue support found.
    NoComputeDevice,
}

impl std::fmt::Display for VulkanQuantizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulkanQuantizerError::NoShaderBytes => write!(
                f,
                "VulkanQuantizer: SPIR-V bytes are empty (glslc missing at build)"
            ),
            VulkanQuantizerError::VulkanLoad(s) => write!(f, "Vulkan loader: {}", s),
            VulkanQuantizerError::Vk(r) => write!(f, "Vulkan: {:?}", r),
            VulkanQuantizerError::NoComputeDevice => {
                write!(f, "VulkanQuantizer: no Vulkan device with compute queue")
            }
        }
    }
}

impl std::error::Error for VulkanQuantizerError {}

impl From<vk::Result> for VulkanQuantizerError {
    fn from(r: vk::Result) -> Self {
        VulkanQuantizerError::Vk(r)
    }
}

impl From<VulkanError> for VulkanQuantizerError {
    fn from(err: VulkanError) -> Self {
        match err {
            VulkanError::LoaderLoad(e) => VulkanQuantizerError::VulkanLoad(e.to_string()),
            VulkanError::Vk(r) => VulkanQuantizerError::Vk(r),
            VulkanError::NoMatchingPhysicalDevice(_) | VulkanError::NoMatchingQueueFamily(_) => {
                VulkanQuantizerError::NoComputeDevice
            }
            other => VulkanQuantizerError::VulkanLoad(other.to_string()),
        }
    }
}

/// Owns a Vulkan instance + compute device + the quantize compute
/// pipeline. Construct once per process; reuse across many quantize()
/// calls.
///
/// Field-order rationale: Rust drops fields in declaration order, and
/// the gpu_vulkan helpers each hold an `Arc<Device>` (or `Arc<Instance>`)
/// internally, so their Drop is deferred until the last reference goes
/// away. The order below mirrors creation order so the raw `cmd_pool`
/// (the only handle this file destroys directly) is released before the
/// `Device` reference count reaches zero.
pub struct VulkanQuantizer {
    pipeline: ComputePipeline,
    descriptor_set_layout: DescriptorSetLayout,
    // shader_module is intentionally retained for symmetry and possible
    // future cache reuse even though Vulkan permits destroying it after
    // pipeline creation. The Drop chain keeps the rest of the helpers
    // alive long enough that an unused field here costs nothing.
    #[allow(dead_code)]
    shader_module: ShaderModule,
    cmd_pool: vk::CommandPool,
    device: Device,
    adapter: Adapter,
    // instance is held to anchor the Arc<InstanceInner> reference graph
    // so the loader stays loaded for the lifetime of this quantizer.
    #[allow(dead_code)]
    instance: Instance,
}

impl VulkanQuantizer {
    /// Construct a new quantizer. Picks the first Vulkan physical device
    /// with a compute queue (via `gororoba_gpu_vulkan::Adapter::pick`),
    /// loads the embedded quantize.spv into a shader module, and builds
    /// the descriptor-set / compute-pipeline objects through the
    /// consolidated helper builders. Future calls to quantize() reuse
    /// this state.
    pub fn new() -> Result<Self, VulkanQuantizerError> {
        let spv_bytes = quantize_spv().ok_or(VulkanQuantizerError::NoShaderBytes)?;

        // Instance: API 1.2 + no validation layer (matches the prior
        // hand-rolled invocation; lbm_vulkan picks 1.3 + opt-in
        // validation). The helper sets the engine name/version itself.
        let instance = InstanceBuilder::new("turboquant_vulkan_quantizer")
            .api_version(vk::API_VERSION_1_2)
            .validation(ValidationPolicy::Disable)
            .build()?;

        // Physical device: first match with COMPUTE-capable queue. Pre-
        // queries memory properties so quantize() does not need to re-
        // query per call.
        let adapter = Adapter::pick(&instance, QueueFamilyRequirement::Compute)?;

        // Logical device: no extra features beyond core. The bespoke
        // quantize() path reaches into raw handles via Device::raw().
        let device = DeviceBuilder::new(adapter.clone()).build(&instance)?;

        // Shader module: from pre-compiled SPIR-V bytes. The helper
        // copies into a u32 buffer internally so alignment is satisfied.
        let shader_module = ShaderModule::from_spv_bytes(&device, spv_bytes, "main")?;

        // Descriptor set layout: 3 storage buffers (values, boundaries,
        // indices). All at COMPUTE stage by default.
        let descriptor_set_layout = DescriptorSetLayoutSpec::new()
            .storage_buffer(0)
            .storage_buffer(1)
            .storage_buffer(2)
            .build(&device)?;

        // Compute pipeline: 16-byte push constant range (n_values,
        // n_boundaries, pad0, pad1 -- four i32 fields).
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16);
        let pipeline = ComputePipelineBuilder::new(&device, &shader_module)
            .descriptor_layout(&descriptor_set_layout)
            .push_constant_range(push_constant_range)
            .build()?;

        // Persistent command pool for per-call command-buffer alloc.
        // The pool flag matches the prior site so quantize() can free a
        // single command buffer without resetting the whole pool.
        let cmd_pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.queue_family_index())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: device was just created above and outlives the pool
        // via the field order in VulkanQuantizer (cmd_pool is destroyed
        // in Drop before the Device reference count hits zero).
        let cmd_pool = unsafe { device.raw().create_command_pool(&cmd_pool_ci, None) }?;

        Ok(VulkanQuantizer {
            pipeline,
            descriptor_set_layout,
            shader_module,
            cmd_pool,
            device,
            adapter,
            instance,
        })
    }

    /// Convenience predicate: probe the loader without committing to a
    /// device. Used by Backend::Vulkan dispatch to decide whether to
    /// fall back to CPU at runtime.
    pub fn is_available() -> bool {
        if quantize_spv().is_none() {
            return false;
        }
        // Match the prior probe: only check that the loader can load
        // libvulkan; do not pay the cost of enumerating physical devices
        // or building an instance.
        // SAFETY: ash::Entry::load opens libvulkan via dlopen and
        // returns Ok only if the load succeeds; no further state is
        // mutated.
        unsafe { ash::Entry::load() }.is_ok()
    }

    /// Run the boundary-search quantization on the GPU. `values` are
    /// the f32 inputs; `boundaries` are the sorted f32 thresholds
    /// (length = 2^bits - 1); `out` receives the u8 indices (one per
    /// value).
    ///
    /// Allocates HOST_VISIBLE | HOST_COHERENT buffers (no staging),
    /// uploads, dispatches the compute shader with workgroup_size=256,
    /// waits on a fence, and reads back the packed u32 indices into
    /// `out` as unpacked u8 bytes.
    pub fn quantize(
        &self,
        values: &[f32],
        boundaries: &[f32],
        out: &mut [u8],
    ) -> Result<(), VulkanQuantizerError> {
        assert_eq!(values.len(), out.len(), "values.len() must equal out.len()");
        let n_values = values.len();
        if n_values == 0 {
            return Ok(());
        }
        let n_boundaries = boundaries.len();
        // packed_indices: 4 u8 indices per u32 word, ceil(n_values / 4).
        let packed_words = n_values.div_ceil(4);

        let values_size = (n_values * 4) as vk::DeviceSize;
        let boundaries_size = (n_boundaries * 4).max(4) as vk::DeviceSize;
        let indices_size = (packed_words * 4) as vk::DeviceSize;

        let device = self.device.raw();

        // ---- Allocate the 3 buffers + their backing memory blocks. ----
        let values_buf = self.create_storage_buffer(values_size)?;
        let boundaries_buf = self.create_storage_buffer(boundaries_size)?;
        let indices_buf = self.create_storage_buffer(indices_size)?;
        let values_mem = self.alloc_host_visible(&values_buf)?;
        let boundaries_mem = self.alloc_host_visible(&boundaries_buf)?;
        let indices_mem = self.alloc_host_visible(&indices_buf)?;
        unsafe {
            // SAFETY: each (buf, mem) pair was created in this device;
            // mem alignment satisfies vk::MemoryRequirements.
            device.bind_buffer_memory(values_buf, values_mem, 0)?;
            device.bind_buffer_memory(boundaries_buf, boundaries_mem, 0)?;
            device.bind_buffer_memory(indices_buf, indices_mem, 0)?;
        }

        // ---- Upload values + boundaries; zero-fill indices. ----
        unsafe {
            self.upload_f32(values_mem, values)?;
            self.upload_f32(boundaries_mem, boundaries)?;
            // Zero-fill the indices buffer (the shader uses atomicOr).
            let zeros = vec![0u32; packed_words];
            self.upload_u32(indices_mem, &zeros)?;
        }

        // ---- Allocate descriptor pool + descriptor set; bind buffers. ----
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 3,
        }];
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    max_sets: 1,
                    pool_size_count: pool_sizes.len() as u32,
                    p_pool_sizes: pool_sizes.as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }?;
        let dsl_raw = self.descriptor_set_layout.raw();
        let dset = unsafe {
            device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool: pool,
                descriptor_set_count: 1,
                p_set_layouts: &dsl_raw,
                ..Default::default()
            })?
        }[0];
        let buf_infos = [
            vk::DescriptorBufferInfo {
                buffer: values_buf,
                offset: 0,
                range: values_size,
            },
            vk::DescriptorBufferInfo {
                buffer: boundaries_buf,
                offset: 0,
                range: boundaries_size,
            },
            vk::DescriptorBufferInfo {
                buffer: indices_buf,
                offset: 0,
                range: indices_size,
            },
        ];
        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, info)| vk::WriteDescriptorSet {
                dst_set: dset,
                dst_binding: i as u32,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: info,
                ..Default::default()
            })
            .collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        // ---- Allocate + record + submit command buffer. ----
        let cmd_buf = unsafe {
            device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: self.cmd_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })
        }?[0];
        let push_constants: [i32; 4] = [n_values as i32, n_boundaries as i32, 0, 0];
        let workgroup_count = n_values.div_ceil(256) as u32;
        unsafe {
            device.begin_command_buffer(
                cmd_buf,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )?;
            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.raw(),
            );
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.layout(),
                0,
                &[dset],
                &[],
            );
            // Push constants: reinterpret the [i32; 4] as &[u8; 16] via
            // bytemuck so clippy's transmute lint does not fire.
            let pc_bytes: &[u8] = bytemuck::cast_slice(&push_constants);
            device.cmd_push_constants(
                cmd_buf,
                self.pipeline.layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );
            device.cmd_dispatch(cmd_buf, workgroup_count, 1, 1);
            device.end_command_buffer(cmd_buf)?;
        }

        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        unsafe {
            device.queue_submit(
                self.device.queue(),
                &[vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers: &cmd_buf,
                    ..Default::default()
                }],
                fence,
            )?;
            // Wait up to 5 seconds for the GPU to finish.
            device.wait_for_fences(&[fence], true, 5_000_000_000)?;
        }

        // ---- Read back packed indices and unpack to u8. ----
        let mut packed = vec![0u32; packed_words];
        unsafe {
            self.download_u32(indices_mem, &mut packed)?;
        }
        for (i, slot) in out.iter_mut().enumerate() {
            let word = packed[i / 4];
            let shift = (i % 4) * 8;
            *slot = ((word >> shift) & 0xFF) as u8;
        }

        // ---- Cleanup: per-call resources. ----
        unsafe {
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.cmd_pool, &[cmd_buf]);
            device.destroy_descriptor_pool(pool, None);
            device.free_memory(indices_mem, None);
            device.free_memory(boundaries_mem, None);
            device.free_memory(values_mem, None);
            device.destroy_buffer(indices_buf, None);
            device.destroy_buffer(boundaries_buf, None);
            device.destroy_buffer(values_buf, None);
        }
        Ok(())
    }

    fn create_storage_buffer(
        &self,
        size: vk::DeviceSize,
    ) -> Result<vk::Buffer, VulkanQuantizerError> {
        let buf = unsafe {
            self.device.raw().create_buffer(
                &vk::BufferCreateInfo {
                    size,
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )
        }?;
        Ok(buf)
    }

    fn alloc_host_visible(
        &self,
        buf: &vk::Buffer,
    ) -> Result<vk::DeviceMemory, VulkanQuantizerError> {
        let req = unsafe { self.device.raw().get_buffer_memory_requirements(*buf) };
        // `adapter.memory_properties` is pre-queried at pick time -- no
        // need to round-trip through the instance again.
        let mem_props = self.adapter.memory_properties;
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
            return Err(VulkanQuantizerError::Vk(
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            ));
        }
        let mem = unsafe {
            self.device.raw().allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: req.size,
                    memory_type_index: mem_type_index,
                    ..Default::default()
                },
                None,
            )
        }?;
        Ok(mem)
    }

    /// Map memory, copy &[f32] into it, unmap. SAFETY: caller must
    /// ensure mem was allocated with HOST_VISIBLE | HOST_COHERENT and
    /// is at least values.len()*4 bytes.
    unsafe fn upload_f32(
        &self,
        mem: vk::DeviceMemory,
        values: &[f32],
    ) -> Result<(), VulkanQuantizerError> {
        unsafe {
            let size = (values.len() * 4) as vk::DeviceSize;
            let ptr = self
                .device
                .raw()
                .map_memory(mem, 0, size, vk::MemoryMapFlags::empty())?
                as *mut f32;
            std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
            self.device.raw().unmap_memory(mem);
            Ok(())
        }
    }

    /// Same as upload_f32 but for u32 (used to zero-init the indices).
    unsafe fn upload_u32(
        &self,
        mem: vk::DeviceMemory,
        words: &[u32],
    ) -> Result<(), VulkanQuantizerError> {
        unsafe {
            let size = (words.len() * 4) as vk::DeviceSize;
            let ptr = self
                .device
                .raw()
                .map_memory(mem, 0, size, vk::MemoryMapFlags::empty())?
                as *mut u32;
            std::ptr::copy_nonoverlapping(words.as_ptr(), ptr, words.len());
            self.device.raw().unmap_memory(mem);
            Ok(())
        }
    }

    /// Map memory, copy out into &mut [u32], unmap.
    unsafe fn download_u32(
        &self,
        mem: vk::DeviceMemory,
        words: &mut [u32],
    ) -> Result<(), VulkanQuantizerError> {
        unsafe {
            let size = (words.len() * 4) as vk::DeviceSize;
            let ptr = self
                .device
                .raw()
                .map_memory(mem, 0, size, vk::MemoryMapFlags::empty())?
                as *const u32;
            std::ptr::copy_nonoverlapping(ptr, words.as_mut_ptr(), words.len());
            self.device.raw().unmap_memory(mem);
            Ok(())
        }
    }
}

impl Drop for VulkanQuantizer {
    fn drop(&mut self) {
        // SAFETY: the command pool was created by Self::new on
        // self.device.raw(); the Device handle inside VulkanQuantizer is
        // dropped only after this Drop runs (field-order semantics), so
        // the destroy call is legal.
        //
        // The gpu_vulkan helpers (pipeline, descriptor_set_layout,
        // shader_module, device, instance) destroy their own resources
        // in their Drop impls, each holding an Arc to the parent so the
        // tear-down order is reverse of construction order regardless
        // of struct field order.
        unsafe {
            self.device
                .raw()
                .destroy_command_pool(self.cmd_pool, None);
        }
    }
}
