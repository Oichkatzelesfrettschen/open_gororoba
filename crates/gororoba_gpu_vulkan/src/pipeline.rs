//! Compute pipeline builder.
//!
//! Consolidates the 9 sites in lbm_vulkan (compute.rs:508/700/873/1071,
//! besag_clifford_vulkan.rs:279/468/471/473/481, alignment_vulkan.rs:133)
//! and cd_kernel turboquant (quantizer.rs:216-225) that build identical
//! vk::ComputePipelineCreateInfo from a ShaderModule + PipelineLayout.

use std::{ffi::CString, sync::Arc};

use ash::vk;

use crate::{
    descriptor::DescriptorSetLayout,
    device::Device,
    error::{Result, VulkanError},
    shader::ShaderModule,
};

/// Builder for a compute pipeline.
pub struct ComputePipelineBuilder<'a> {
    device: &'a Device,
    shader: &'a ShaderModule,
    descriptor_layouts: Vec<&'a DescriptorSetLayout>,
    push_constant_ranges: Vec<vk::PushConstantRange>,
}

impl<'a> ComputePipelineBuilder<'a> {
    pub fn new(device: &'a Device, shader: &'a ShaderModule) -> Self {
        Self {
            device,
            shader,
            descriptor_layouts: Vec::new(),
            push_constant_ranges: Vec::new(),
        }
    }

    pub fn descriptor_layout(mut self, layout: &'a DescriptorSetLayout) -> Self {
        self.descriptor_layouts.push(layout);
        self
    }

    pub fn push_constant_range(mut self, range: vk::PushConstantRange) -> Self {
        self.push_constant_ranges.push(range);
        self
    }

    /// Build the pipeline + return it bundled with its pipeline layout.
    pub fn build(self) -> Result<ComputePipeline> {
        let set_layouts: Vec<vk::DescriptorSetLayout> =
            self.descriptor_layouts.iter().map(|l| l.raw()).collect();
        let pl_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&self.push_constant_ranges);
        // SAFETY: set_layouts + push_constant_ranges outlive this call;
        // ash copies them into a driver-owned object.
        let pipeline_layout = unsafe { self.device.raw().create_pipeline_layout(&pl_ci, None) }?;

        let entry_cstring =
            CString::new(self.shader.entry_point()).expect("entry_point free of NUL");
        let stage_ci = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(self.shader.raw())
            .name(entry_cstring.as_c_str());

        let pipeline_ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage_ci)
            .layout(pipeline_layout);

        // SAFETY: pipeline_ci references stage_ci which holds entry_cstring
        // via its borrow; entry_cstring lives to the end of this fn.
        let pipelines = unsafe {
            self.device.raw().create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_ci),
                None,
            )
        }
        .map_err(|(_, err)| VulkanError::Vk(err))?;

        let pipeline = pipelines
            .into_iter()
            .next()
            .ok_or(VulkanError::UnsupportedFeature(
                "create_compute_pipelines returned empty vec",
            ))?;

        Ok(ComputePipeline {
            device: Arc::new(self.device.clone()),
            raw: pipeline,
            layout: pipeline_layout,
        })
    }
}

/// Owned compute pipeline + its layout, with deterministic Drop.
pub struct ComputePipeline {
    device: Arc<Device>,
    raw: vk::Pipeline,
    layout: vk::PipelineLayout,
}

impl ComputePipeline {
    pub fn raw(&self) -> vk::Pipeline {
        self.raw
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        // SAFETY: Both handles were created by build() above; Arc<Device>
        // keeps the device alive past this Drop.
        unsafe {
            self.device.raw().destroy_pipeline(self.raw, None);
            self.device.raw().destroy_pipeline_layout(self.layout, None);
        }
    }
}
