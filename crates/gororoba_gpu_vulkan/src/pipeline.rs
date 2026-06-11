//! Compute pipeline builder.
//!
//! Consolidates the 9 sites in lbm_vulkan (compute.rs:508/700/873/1071,
//! besag_clifford_vulkan.rs:279/468/471/473/481, alignment_vulkan.rs:133)
//! and cd_kernel turboquant (quantizer.rs:216-225) that build identical
//! vk::ComputePipelineCreateInfo from a ShaderModule + PipelineLayout.

use std::{collections::BTreeMap, ffi::CString, sync::Arc};

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
    specialization_u32: BTreeMap<u32, u32>,
}

impl<'a> ComputePipelineBuilder<'a> {
    pub fn new(device: &'a Device, shader: &'a ShaderModule) -> Self {
        Self {
            device,
            shader,
            descriptor_layouts: Vec::new(),
            push_constant_ranges: Vec::new(),
            specialization_u32: BTreeMap::new(),
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

    /// Set a u32 Vulkan specialization constant by WGSL override name.
    pub fn with_override(mut self, name: &str, value: u32) -> Result<Self> {
        let constant_id = self.shader.override_constant_id(name).ok_or_else(|| {
            VulkanError::PipelineOverride(format!(
                "WGSL override `{name}` is missing or has no explicit @id"
            ))
        })?;
        self.specialization_u32.insert(constant_id, value);
        Ok(self)
    }

    /// Set a u32 Vulkan specialization constant by numeric constant ID.
    pub fn with_override_id(mut self, constant_id: u32, value: u32) -> Self {
        self.specialization_u32.insert(constant_id, value);
        self
    }

    /// Build with a batch of named u32 WGSL overrides.
    pub fn build_specialised(mut self, overrides: &[(String, u32)]) -> Result<ComputePipeline> {
        for (name, value) in overrides {
            self = self.with_override(name, *value)?;
        }
        self.build()
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
        let specialization_entries: Vec<vk::SpecializationMapEntry> = self
            .specialization_u32
            .keys()
            .enumerate()
            .map(|(index, constant_id)| {
                vk::SpecializationMapEntry::default()
                    .constant_id(*constant_id)
                    .offset((index * size_of::<u32>()) as u32)
                    .size(size_of::<u32>())
            })
            .collect();
        let specialization_data: Vec<u8> = self
            .specialization_u32
            .values()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let specialization_info = if specialization_entries.is_empty() {
            None
        } else {
            Some(
                vk::SpecializationInfo::default()
                    .map_entries(&specialization_entries)
                    .data(&specialization_data),
            )
        };

        let mut stage_ci = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(self.shader.raw())
            .name(entry_cstring.as_c_str());
        if let Some(ref info) = specialization_info {
            stage_ci = stage_ci.specialization_info(info);
        }

        let pipeline_ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage_ci)
            .layout(pipeline_layout);

        // SAFETY: pipeline_ci references stage_ci which holds entry_cstring
        // via its borrow; entry_cstring lives to the end of this fn.
        let pipelines_result = unsafe {
            self.device.raw().create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_ci),
                None,
            )
        };

        let pipelines = match pipelines_result {
            Ok(pipelines) => pipelines,
            Err((_, err)) => {
                // pipeline_layout was created above but create_compute_pipelines
                // failed; clean it up before propagating so repeated build
                // failures do not leak driver-side layout objects.
                // SAFETY: pipeline_layout was just created on self.device.
                unsafe {
                    self.device
                        .raw()
                        .destroy_pipeline_layout(pipeline_layout, None);
                }
                return Err(VulkanError::Vk(err));
            }
        };

        let pipeline = match pipelines.into_iter().next() {
            Some(p) => p,
            None => {
                // Same cleanup obligation as the error arm above.
                // SAFETY: pipeline_layout still owned by us.
                unsafe {
                    self.device
                        .raw()
                        .destroy_pipeline_layout(pipeline_layout, None);
                }
                return Err(VulkanError::UnsupportedFeature(
                    "create_compute_pipelines returned empty vec",
                ));
            }
        };

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
