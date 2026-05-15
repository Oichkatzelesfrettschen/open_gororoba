//! Physical-device selection.
//!
//! Consolidates the `enumerate_physical_devices` + queue-family-iteration
//! pattern from:
//!   - lbm_vulkan/src/lib.rs:140-158 (requires COMPUTE | GRAPHICS)
//!   - cd_kernel/turboquant/vulkan/quantizer.rs:109-119 (requires COMPUTE only)
//!
//! Both sites used `find_map` over the device list and the queue family
//! list with a hard-coded flag bitmask. This module exposes those flags as
//! a `QueueFamilyRequirement` enum and a single picker.

use ash::vk;

use crate::{
    error::{Result, VulkanError},
    instance::Instance,
};

/// What kind of queue family is needed?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueFamilyRequirement {
    /// Compute-only path (cd_kernel turboquant). Matches any queue family
    /// with `vk::QueueFlags::COMPUTE`. Does not require graphics support;
    /// suits headless compute servers.
    Compute,
    /// Compute + graphics (lbm_vulkan main path). Matches any queue family
    /// with both `COMPUTE` and `GRAPHICS` bits set; needed when the same
    /// queue submits both simulation kernels and presentation work.
    ComputeAndGraphics,
}

impl QueueFamilyRequirement {
    /// Required Vulkan queue-flag bitmask.
    pub fn flags(self) -> vk::QueueFlags {
        match self {
            Self::Compute => vk::QueueFlags::COMPUTE,
            Self::ComputeAndGraphics => vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS,
        }
    }
}

/// A picked physical device + its matched queue family index.
#[derive(Clone, Debug)]
pub struct Adapter {
    pub physical_device: vk::PhysicalDevice,
    pub queue_family_index: u32,
    pub properties: vk::PhysicalDeviceProperties,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl Adapter {
    /// Pick the first physical device whose queue families satisfy the
    /// requirement. Both prior sites used "first match" semantics; we
    /// preserve that behaviour. A future enhancement could rank devices
    /// by discrete-GPU preference, VRAM size, or compute-shared-memory.
    pub fn pick(instance: &Instance, requirement: QueueFamilyRequirement) -> Result<Self> {
        // SAFETY: instance.raw() outlives this call; the returned Vec is
        // owned by the caller.
        let pdevices = unsafe { instance.raw().enumerate_physical_devices() }?;

        let needed_flags = requirement.flags();

        for pdev in pdevices {
            // SAFETY: pdev was just enumerated from this instance.
            let queue_families = unsafe {
                instance
                    .raw()
                    .get_physical_device_queue_family_properties(pdev)
            };
            if let Some((qfi, _props)) = queue_families
                .iter()
                .enumerate()
                .find(|(_, p)| p.queue_flags.contains(needed_flags))
            {
                // SAFETY: pdev was just enumerated from this instance.
                let properties = unsafe { instance.raw().get_physical_device_properties(pdev) };
                let memory_properties =
                    unsafe { instance.raw().get_physical_device_memory_properties(pdev) };
                return Ok(Self {
                    physical_device: pdev,
                    queue_family_index: u32::try_from(qfi).unwrap_or(u32::MAX),
                    properties,
                    memory_properties,
                });
            }
        }

        Err(VulkanError::NoMatchingPhysicalDevice(requirement))
    }

    /// Total device-local VRAM in bytes (sum of `DEVICE_LOCAL` heaps).
    pub fn device_local_vram_bytes(&self) -> u64 {
        let memory_properties = self.memory_properties;
        memory_properties
            .memory_heaps
            .iter()
            .take(memory_properties.memory_heap_count as usize)
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .sum()
    }

    /// Human-readable device name (from `properties.device_name`).
    pub fn device_name(&self) -> String {
        let bytes = self.properties.device_name.as_slice();
        let nul = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
        // properties.device_name is i8 in ash 0.38; reinterpret as u8 for
        // String conversion.
        let u8_slice: &[u8] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u8>(), nul) };
        String::from_utf8_lossy(u8_slice).into_owned()
    }
}
