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

use ash::{Entry, Instance, Device, vk};
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
            self.device
                .destroy_shader_module(self.shader_module, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
