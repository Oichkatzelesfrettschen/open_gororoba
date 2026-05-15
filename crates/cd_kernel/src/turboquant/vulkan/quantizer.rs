//! VulkanQuantizer: dispatches the quantize.comp SPIR-V shader for
//! TurboQuant boundary-search quantization on the GPU.
//!
//! This module is feature-gated on `vulkan` (which enables the `ash` dep).
//! It owns its own Vulkan instance/device for simplicity; future work
//! may share a VulkanContext with lbm_vulkan to avoid double-init.
//!
//! # Universal SAFETY argument
//!
//! Every `unsafe { device.<vk_fn>(...) }` call follows the same pattern
//! used by lbm_vulkan/compute.rs and documented at the top of that file:
//! 1. Active instance/device borrowed from VulkanQuantizer fields
//! 2. Handle lifetimes are paired with Drop destruction (this file)
//! 3. Synchronization via vk::Fence
//! 4. Descriptor-set bindings match the pipeline layout's signature
//! 5. Memory alignment is satisfied by selecting HOST_VISIBLE +
//!    HOST_COHERENT memory types whose alignment >= the
//!    vk::MemoryRequirements reported by the device.

#![cfg(feature = "vulkan")]

use ash::{Device, Entry, Instance, vk};
use std::ffi::CString;

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

/// Owns a Vulkan instance + compute device + the quantize compute
/// pipeline. Construct once per process; reuse across many quantize()
/// calls.
pub struct VulkanQuantizer {
    // Drop order: pipeline -> pipeline_layout -> dsl -> shader_module
    //          -> cmd_pool -> device -> instance -> entry. Rust drops
    // fields in declaration order; arrange accordingly.
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    shader_module: vk::ShaderModule,
    cmd_pool: vk::CommandPool,
    #[allow(dead_code)] // queue used by quantize() in Phase C
    queue: vk::Queue,
    #[allow(dead_code)] // queue_family_index used by quantize() in Phase C
    queue_family_index: u32,
    device: Device,
    #[allow(dead_code)] // physical_device used by quantize() in Phase C for memory-type queries
    physical_device: vk::PhysicalDevice,
    instance: Instance,
    _entry: Entry,
}

impl VulkanQuantizer {
    /// Construct a new quantizer. Picks the first Vulkan physical device
    /// with a compute queue, loads the embedded quantize.spv into a
    /// shader module, and constructs the descriptor-set / pipeline
    /// objects. Future calls to quantize() reuse this state.
    pub fn new() -> Result<Self, VulkanQuantizerError> {
        let spv = quantize_spv().ok_or(VulkanQuantizerError::NoShaderBytes)?;

        let entry = unsafe { Entry::load() }
            .map_err(|e| VulkanQuantizerError::VulkanLoad(e.to_string()))?;
        let app_name = CString::new("turboquant_vulkan_quantizer").unwrap();
        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            api_version: vk::API_VERSION_1_2,
            ..Default::default()
        };
        let instance_ci = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            ..Default::default()
        };
        let instance = unsafe { entry.create_instance(&instance_ci, None) }?;

        let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
        let (physical_device, queue_family_index) = physical_devices
            .into_iter()
            .find_map(|pd| {
                let qfp = unsafe { instance.get_physical_device_queue_family_properties(pd) };
                qfp.iter()
                    .enumerate()
                    .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                    .map(|(i, _)| (pd, i as u32))
            })
            .ok_or(VulkanQuantizerError::NoComputeDevice)?;

        let priorities = [1.0_f32];
        let queue_ci = vk::DeviceQueueCreateInfo {
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: priorities.as_ptr(),
            ..Default::default()
        };
        let device_ci = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_ci,
            ..Default::default()
        };
        let device = unsafe { instance.create_device(physical_device, &device_ci, None) }?;
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // SPIR-V is embedded as &[u8]; ash wants &[u32] aligned. Reinterpret.
        // SAFETY: SPIR-V is 4-byte-aligned by spec; the build.rs glslc output
        // satisfies this. The lifetime of the slice is bounded by spv which
        // lives for the duration of this function.
        let spv_u32: Vec<u32> = spv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shader_module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    code_size: spv_u32.len() * 4,
                    p_code: spv_u32.as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }?;

        // Descriptor set layout: 3 storage buffers (values, boundaries, indices).
        let bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];
        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    binding_count: bindings.len() as u32,
                    p_bindings: bindings.as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }?;

        // Push constants: 4 i32 fields = 16 bytes (n_values, n_boundaries, pad0, pad1).
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 16,
        };
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo {
                    set_layout_count: 1,
                    p_set_layouts: &descriptor_set_layout,
                    push_constant_range_count: 1,
                    p_push_constant_ranges: &push_constant_range,
                    ..Default::default()
                },
                None,
            )
        }?;

        let entry_main = CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: entry_main.as_ptr(),
            ..Default::default()
        };
        let pipeline_ci = vk::ComputePipelineCreateInfo {
            stage,
            layout: pipeline_layout,
            ..Default::default()
        };
        let pipelines = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
        }
        .map_err(|(_, r)| VulkanQuantizerError::Vk(r))?;
        let pipeline = pipelines[0];

        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    queue_family_index,
                    flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    ..Default::default()
                },
                None,
            )
        }?;

        Ok(VulkanQuantizer {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            shader_module,
            cmd_pool,
            queue,
            queue_family_index,
            device,
            physical_device,
            instance,
            _entry: entry,
        })
    }

    /// Convenience predicate: probe the loader without committing to a
    /// device. Used by Backend::Vulkan dispatch to decide whether to
    /// fall back to CPU at runtime.
    pub fn is_available() -> bool {
        if quantize_spv().is_none() {
            return false;
        }
        unsafe { Entry::load() }.is_ok()
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
        let packed_words = (n_values + 3) / 4;

        let values_size = (n_values * 4) as vk::DeviceSize;
        let boundaries_size = (n_boundaries * 4).max(4) as vk::DeviceSize;
        let indices_size = (packed_words * 4) as vk::DeviceSize;

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
            self.device.bind_buffer_memory(values_buf, values_mem, 0)?;
            self.device
                .bind_buffer_memory(boundaries_buf, boundaries_mem, 0)?;
            self.device
                .bind_buffer_memory(indices_buf, indices_mem, 0)?;
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
            self.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    max_sets: 1,
                    pool_size_count: pool_sizes.len() as u32,
                    p_pool_sizes: pool_sizes.as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }?;
        let dset = unsafe {
            self.device
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                    descriptor_pool: pool,
                    descriptor_set_count: 1,
                    p_set_layouts: &self.descriptor_set_layout,
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
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        // ---- Allocate + record + submit command buffer. ----
        let cmd_buf = unsafe {
            self.device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                    command_pool: self.cmd_pool,
                    level: vk::CommandBufferLevel::PRIMARY,
                    command_buffer_count: 1,
                    ..Default::default()
                })
        }?[0];
        let push_constants: [i32; 4] = [n_values as i32, n_boundaries as i32, 0, 0];
        let workgroup_count = ((n_values + 255) / 256) as u32;
        unsafe {
            self.device.begin_command_buffer(
                cmd_buf,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )?;
            self.device
                .cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[dset],
                &[],
            );
            // Push constants: cast the [i32; 4] to &[u8; 16].
            let pc_bytes: [u8; 16] = std::mem::transmute(push_constants);
            self.device.cmd_push_constants(
                cmd_buf,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &pc_bytes,
            );
            self.device.cmd_dispatch(cmd_buf, workgroup_count, 1, 1);
            self.device.end_command_buffer(cmd_buf)?;
        }

        let fence = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }?;
        unsafe {
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers: &cmd_buf,
                    ..Default::default()
                }],
                fence,
            )?;
            // Wait up to 5 seconds for the GPU to finish.
            self.device.wait_for_fences(&[fence], true, 5_000_000_000)?;
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
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.cmd_pool, &[cmd_buf]);
            self.device.destroy_descriptor_pool(pool, None);
            self.device.free_memory(indices_mem, None);
            self.device.free_memory(boundaries_mem, None);
            self.device.free_memory(values_mem, None);
            self.device.destroy_buffer(indices_buf, None);
            self.device.destroy_buffer(boundaries_buf, None);
            self.device.destroy_buffer(values_buf, None);
        }
        Ok(())
    }

    fn create_storage_buffer(
        &self,
        size: vk::DeviceSize,
    ) -> Result<vk::Buffer, VulkanQuantizerError> {
        let buf = unsafe {
            self.device.create_buffer(
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
        let req = unsafe { self.device.get_buffer_memory_requirements(*buf) };
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
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
            self.device.allocate_memory(
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
                .map_memory(mem, 0, size, vk::MemoryMapFlags::empty())?
                as *mut f32;
            std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
            self.device.unmap_memory(mem);
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
                .map_memory(mem, 0, size, vk::MemoryMapFlags::empty())?
                as *mut u32;
            std::ptr::copy_nonoverlapping(words.as_ptr(), ptr, words.len());
            self.device.unmap_memory(mem);
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
                .map_memory(mem, 0, size, vk::MemoryMapFlags::empty())?
                as *const u32;
            std::ptr::copy_nonoverlapping(ptr, words.as_mut_ptr(), words.len());
            self.device.unmap_memory(mem);
            Ok(())
        }
    }
}

impl Drop for VulkanQuantizer {
    fn drop(&mut self) {
        // SAFETY: the device is alive (we own it). Each handle was
        // created by self.device and is destroyed in reverse order of
        // creation per Vulkan spec.
        unsafe {
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_shader_module(self.shader_module, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
