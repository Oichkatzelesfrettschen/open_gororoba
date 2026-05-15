//! Descriptor set layout builder.
//!
//! Consolidates the 7 ad-hoc DescriptorSetLayoutBinding sites in
//! lbm_vulkan (compute.rs:430-545, alignment_vulkan.rs:61-160,
//! besag_clifford_vulkan.rs:443-780) and cd_kernel turboquant
//! (quantizer.rs:156-188). All sites mechanically build the same
//! pattern, varying only binding count and descriptor types.

use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::error::Result;

/// Descriptor type vocabulary aligned with the workspace's prior usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorType {
    /// vk::DescriptorType::STORAGE_BUFFER (compute kernels' read/write
    /// f-distribution + state buffers).
    StorageBuffer,
    /// vk::DescriptorType::UNIFORM_BUFFER (constants + dispatch parameters).
    UniformBuffer,
    /// vk::DescriptorType::STORAGE_IMAGE (compute kernels writing to images).
    StorageImage,
}

impl DescriptorType {
    pub fn to_vk(self) -> vk::DescriptorType {
        match self {
            Self::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
            Self::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            Self::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
        }
    }
}

/// Spec for a single descriptor binding.
#[derive(Debug, Clone, Copy)]
pub struct BindingSpec {
    pub binding: u32,
    pub descriptor_type: DescriptorType,
    pub count: u32,
}

impl BindingSpec {
    pub fn new(binding: u32, descriptor_type: DescriptorType) -> Self {
        Self {
            binding,
            descriptor_type,
            count: 1,
        }
    }
}

/// Builds a descriptor set layout for a compute shader.
///
/// All bindings are exposed at `ShaderStageFlags::COMPUTE` since the
/// workspace's prior call sites all dispatch compute kernels. Future
/// expansion (graphics pipelines) would add a stage-flags parameter.
#[derive(Default)]
pub struct DescriptorSetLayoutSpec {
    bindings: Vec<BindingSpec>,
}

impl DescriptorSetLayoutSpec {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a binding. Convenience for builder-style construction.
    pub fn binding(mut self, spec: BindingSpec) -> Self {
        self.bindings.push(spec);
        self
    }

    /// Append a storage-buffer binding at the given index.
    pub fn storage_buffer(self, binding: u32) -> Self {
        self.binding(BindingSpec::new(binding, DescriptorType::StorageBuffer))
    }

    /// Append a uniform-buffer binding at the given index.
    pub fn uniform_buffer(self, binding: u32) -> Self {
        self.binding(BindingSpec::new(binding, DescriptorType::UniformBuffer))
    }

    /// Append a storage-image binding at the given index.
    pub fn storage_image(self, binding: u32) -> Self {
        self.binding(BindingSpec::new(binding, DescriptorType::StorageImage))
    }

    /// Build the layout on `device`.
    pub fn build(self, device: &Device) -> Result<DescriptorSetLayout> {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = self
            .bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(b.descriptor_type.to_vk())
                    .descriptor_count(b.count)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: bindings outlives this call; ash copies the layout into
        // a driver-owned object.
        let raw = unsafe { device.raw().create_descriptor_set_layout(&ci, None) }?;
        Ok(DescriptorSetLayout {
            device: Arc::new(device.clone()),
            raw,
            bindings: self.bindings,
        })
    }
}

/// Owned VkDescriptorSetLayout with deterministic Drop.
pub struct DescriptorSetLayout {
    device: Arc<Device>,
    raw: vk::DescriptorSetLayout,
    bindings: Vec<BindingSpec>,
}

impl DescriptorSetLayout {
    pub fn raw(&self) -> vk::DescriptorSetLayout {
        self.raw
    }

    pub fn bindings(&self) -> &[BindingSpec] {
        &self.bindings
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        // SAFETY: self.raw was created by build() above; the Arc<Device>
        // keeps the device alive past this Drop.
        unsafe {
            self.device
                .raw()
                .destroy_descriptor_set_layout(self.raw, None);
        }
    }
}
