//! Descriptor set layout builder.
//!
//! Consolidates the 7 ad-hoc DescriptorSetLayoutBinding sites in
//! lbm_vulkan (compute.rs:430-545, alignment_vulkan.rs:61-160,
//! besag_clifford_vulkan.rs:443-780) and cd_kernel turboquant
//! (quantizer.rs:156-188). All sites mechanically build the same
//! pattern, varying only binding count and descriptor types.

use std::sync::Arc;

use ash::vk;

use crate::{buffer::HostVisibleBuffer, device::Device, error::Result};

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

/// Owned descriptor pool sized for one or more sets matching a layout.
pub struct DescriptorPool {
    device: Arc<Device>,
    raw: vk::DescriptorPool,
}

impl DescriptorPool {
    pub fn for_layout(
        device: &Device,
        layout: &DescriptorSetLayout,
        max_sets: u32,
    ) -> Result<Self> {
        let mut storage_buffers = 0u32;
        let mut uniform_buffers = 0u32;
        let mut storage_images = 0u32;
        for binding in layout.bindings() {
            let count = binding.count.saturating_mul(max_sets);
            match binding.descriptor_type {
                DescriptorType::StorageBuffer => storage_buffers += count,
                DescriptorType::UniformBuffer => uniform_buffers += count,
                DescriptorType::StorageImage => storage_images += count,
            }
        }
        let mut pool_sizes = Vec::new();
        if storage_buffers != 0 {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: storage_buffers,
            });
        }
        if uniform_buffers != 0 {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: uniform_buffers,
            });
        }
        if storage_images != 0 {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: storage_images,
            });
        }
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);
        // SAFETY: pool_sizes outlives this call; ash copies it into a
        // driver-owned pool object.
        let raw = unsafe { device.raw().create_descriptor_pool(&pool_ci, None) }?;
        Ok(Self {
            device: Arc::new(device.clone()),
            raw,
        })
    }

    pub fn allocate_set(&self, layout: &DescriptorSetLayout) -> Result<DescriptorSet> {
        let layouts = [layout.raw()];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.raw)
            .set_layouts(&layouts);
        // SAFETY: pool and layout are live and created on the same device.
        let sets = unsafe { self.device.raw().allocate_descriptor_sets(&alloc_info) }?;
        let raw = sets
            .first()
            .copied()
            .ok_or(crate::error::VulkanError::UnsupportedFeature(
                "allocate_descriptor_sets returned empty vec",
            ))?;
        Ok(DescriptorSet {
            device: Arc::clone(&self.device),
            raw,
        })
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        // SAFETY: self.raw was created by for_layout above; Arc<Device>
        // keeps the device alive past this Drop.
        unsafe {
            self.device.raw().destroy_descriptor_pool(self.raw, None);
        }
    }
}

/// Descriptor set handle owned by a DescriptorPool.
pub struct DescriptorSet {
    device: Arc<Device>,
    raw: vk::DescriptorSet,
}

impl DescriptorSet {
    pub fn raw(&self) -> vk::DescriptorSet {
        self.raw
    }

    pub fn write_storage_buffer(&self, binding: u32, buffer: &HostVisibleBuffer) {
        self.write_buffer(binding, buffer, vk::DescriptorType::STORAGE_BUFFER);
    }

    pub fn write_uniform_buffer(&self, binding: u32, buffer: &HostVisibleBuffer) {
        self.write_buffer(binding, buffer, vk::DescriptorType::UNIFORM_BUFFER);
    }

    fn write_buffer(
        &self,
        binding: u32,
        buffer: &HostVisibleBuffer,
        descriptor_type: vk::DescriptorType,
    ) {
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer.raw())
            .offset(0)
            .range(buffer.size())];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(self.raw)
            .dst_binding(binding)
            .dst_array_element(0)
            .descriptor_type(descriptor_type)
            .buffer_info(&buffer_info)];
        // SAFETY: descriptor set, binding, and buffer are live on the same
        // device. Vulkan copies the descriptor info during the call.
        unsafe {
            self.device.raw().update_descriptor_sets(&write, &[]);
        }
    }
}
