//! Vulkan memory allocator wrapper.
//!
//! Thin wrapper over `gpu_allocator::vulkan::Allocator` that defers the
//! AllocatorCreateDesc boilerplate and pairs the allocator's lifetime with
//! an `Arc<Mutex>` so multiple subsystems can share it. Consolidates the
//! lbm_vulkan/src/lib.rs:263-270 pattern.

use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocator as GaAllocator, AllocatorCreateDesc};

use crate::{device::Device, error::Result, instance::Instance};

/// Shareable Vulkan allocator.
///
/// Wraps `gpu_allocator::vulkan::Allocator` in `Arc<Mutex<...>>` so the
/// allocator can be cloned across worker threads / subsystems without
/// requiring callers to do their own locking.
#[derive(Clone)]
pub struct Allocator {
    inner: Arc<Mutex<GaAllocator>>,
}

impl Allocator {
    /// Create a new allocator bound to the given instance + device +
    /// physical-device handle.
    ///
    /// Defaults: debug_settings=Default, buffer_device_address=false,
    /// allocation_sizes=Default. These match lbm_vulkan's prior choices;
    /// callers needing GPU-pointer descriptors should construct via
    /// `from_create_desc` directly.
    pub fn new(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let desc = AllocatorCreateDesc {
            instance: instance.raw().clone(),
            device: device.raw().clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        };
        let inner = GaAllocator::new(&desc)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Build from an explicit AllocatorCreateDesc, for callers that need
    /// buffer-device-address or custom debug settings.
    pub fn from_create_desc(desc: &AllocatorCreateDesc) -> Result<Self> {
        let inner = GaAllocator::new(desc)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Borrow the underlying allocator for one allocation call.
    pub fn with<R, F: FnOnce(&mut GaAllocator) -> R>(&self, f: F) -> R {
        let mut guard = self.inner.lock().expect("Allocator mutex poisoned");
        f(&mut guard)
    }
}
