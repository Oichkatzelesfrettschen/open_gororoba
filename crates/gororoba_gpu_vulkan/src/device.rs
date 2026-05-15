//! Logical device + compute queue.
//!
//! Consolidates the DeviceQueueCreateInfo + DeviceCreateInfo + optional
//! feature-chain (fp16, fp64, int8, dynamic rendering) pattern from:
//!   - lbm_vulkan/src/lib.rs:209-261 (with fp16/dynamic-rendering chain)
//!   - cd_kernel/turboquant/vulkan/quantizer.rs:121-134 (bare features)
//!
//! The feature-chain delta between sites is the primary source of
//! configuration divergence; this builder exposes it explicitly so callers
//! pick which extras to negotiate.

use std::sync::Arc;

use ash::vk;

use crate::adapter::Adapter;
use crate::error::Result;
use crate::instance::Instance;

/// Optional feature requests forwarded to vk::DeviceCreateInfo.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeviceFeatures {
    /// Request 16-bit shader-storage support (VK_KHR_16bit_storage).
    pub fp16_storage: bool,
    /// Request 8-bit integer storage (VK_KHR_8bit_storage).
    pub int8_storage: bool,
    /// Request shader-side fp16 arithmetic (VK_KHR_shader_float16_int8).
    pub fp16_arith: bool,
    /// Request shader-side fp64 arithmetic (built-in PhysicalDeviceFeatures.shaderFloat64).
    pub fp64_arith: bool,
    /// Request VK_KHR_dynamic_rendering (Vulkan 1.3 promoted to core; needed
    /// for lbm_vulkan's swapchain path on 1.2 drivers).
    pub dynamic_rendering: bool,
}

/// Logical device with a single primary compute queue.
#[derive(Clone)]
pub struct Device {
    #[allow(dead_code)] // keep instance alive
    instance: Instance,
    inner: Arc<DeviceInner>,
    queue: vk::Queue,
    queue_family_index: u32,
    features: DeviceFeatures,
}

struct DeviceInner {
    device: ash::Device,
}

/// Builder for a logical device.
pub struct DeviceBuilder {
    adapter: Adapter,
    features: DeviceFeatures,
    extensions: Vec<&'static std::ffi::CStr>,
}

impl DeviceBuilder {
    pub fn new(adapter: Adapter) -> Self {
        Self {
            adapter,
            features: DeviceFeatures::default(),
            extensions: Vec::new(),
        }
    }

    pub fn features(mut self, features: DeviceFeatures) -> Self {
        self.features = features;
        self
    }

    pub fn extension(mut self, name: &'static std::ffi::CStr) -> Self {
        self.extensions.push(name);
        self
    }

    pub fn enable_fp64(mut self) -> Self {
        self.features.fp64_arith = true;
        self
    }

    pub fn build(self, instance: &Instance) -> Result<Device> {
        let queue_family_index = self.adapter.queue_family_index;
        let priorities = [1.0_f32];

        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);
        let queue_cis = [queue_ci];

        // Built-in fp64 lives in PhysicalDeviceFeatures.
        let mut enabled = vk::PhysicalDeviceFeatures::default();
        if self.features.fp64_arith {
            enabled = enabled.shader_float64(true);
        }

        let extension_ptrs: Vec<*const i8> =
            self.extensions.iter().map(|n| n.as_ptr().cast()).collect();

        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_cis)
            .enabled_extension_names(&extension_ptrs)
            .enabled_features(&enabled);

        // SAFETY: adapter.physical_device was enumerated from the same
        // instance; queue_ci references priorities slice which outlives
        // this call.
        let device = unsafe {
            instance
                .raw()
                .create_device(self.adapter.physical_device, &device_ci, None)
        }?;

        // SAFETY: device was just created with at least one queue at
        // queue_family_index=0.
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Ok(Device {
            instance: instance.clone(),
            inner: Arc::new(DeviceInner { device }),
            queue,
            queue_family_index,
            features: self.features,
        })
    }
}

impl Device {
    /// The owning ash::Device handle.
    pub fn raw(&self) -> &ash::Device {
        &self.inner.device
    }

    /// The primary compute queue.
    pub fn queue(&self) -> vk::Queue {
        self.queue
    }

    /// Queue family index this device's queue was created on.
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// Negotiated features at build time.
    pub fn features(&self) -> DeviceFeatures {
        self.features
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        // SAFETY: device was created by DeviceBuilder::build above and has
        // not been destroyed; child handles (DescriptorSetLayout, Pipeline,
        // etc.) each hold an Arc<DeviceInner> so this Drop runs only after
        // all of them have released their reference.
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
