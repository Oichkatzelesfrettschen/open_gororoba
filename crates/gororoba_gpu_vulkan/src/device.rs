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

use crate::{adapter::Adapter, error::Result, instance::Instance};

/// Optional feature requests forwarded to vk::DeviceCreateInfo.
///
/// The two members of `VK_KHR_shader_float16_int8` (`shaderFloat16` and
/// `shaderInt8`) are tracked as separate flags because GPUs in the wild
/// support each one independently. Requesting `int8_arith` on a device
/// that only exposes `shaderFloat16` would make `vkCreateDevice` fail
/// with `VK_ERROR_FEATURE_NOT_PRESENT`, and vice versa.
///
/// Marked `#[non_exhaustive]` so future field additions are backwards-
/// compatible for callers that construct via `DeviceFeatures::default()`
/// or field-update syntax (`DeviceFeatures { fp16_arith: true, ..Default::default() }`).
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct DeviceFeatures {
    /// Request 16-bit shader-storage support (VK_KHR_16bit_storage).
    pub fp16_storage: bool,
    /// Request 8-bit integer storage (VK_KHR_8bit_storage).
    pub int8_storage: bool,
    /// Request shader-side fp16 arithmetic
    /// (VK_KHR_shader_float16_int8::shaderFloat16). Independent of
    /// `int8_arith` because a GPU may expose one without the other.
    pub fp16_arith: bool,
    /// Request shader-side int8 arithmetic
    /// (VK_KHR_shader_float16_int8::shaderInt8). Independent of
    /// `fp16_arith` because a GPU may expose one without the other.
    pub int8_arith: bool,
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

        // Extended-feature chains: each requested capability lives in a
        // dedicated VkPhysicalDeviceXxxFeatures struct that the caller
        // threads into VkDeviceCreateInfo.pNext via push_next. We
        // construct only the structs whose corresponding bit is set so
        // drivers that don't expose the feature still build cleanly.
        let mut storage_16bit = vk::PhysicalDeviceVulkan11Features::default();
        let mut storage_8bit_features = vk::PhysicalDevice8BitStorageFeatures::default();
        let mut shader_f16_i8_features = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default();

        if self.features.fp16_storage {
            storage_16bit.storage_buffer16_bit_access = vk::TRUE;
            storage_16bit.uniform_and_storage_buffer16_bit_access = vk::TRUE;
        }
        if self.features.int8_storage {
            storage_8bit_features.storage_buffer8_bit_access = vk::TRUE;
            storage_8bit_features.uniform_and_storage_buffer8_bit_access = vk::TRUE;
        }
        if self.features.fp16_arith {
            shader_f16_i8_features.shader_float16 = vk::TRUE;
        }
        if self.features.int8_arith {
            shader_f16_i8_features.shader_int8 = vk::TRUE;
        }
        if self.features.dynamic_rendering {
            dynamic_rendering_features.dynamic_rendering = vk::TRUE;
        }

        let extension_ptrs: Vec<*const i8> =
            self.extensions.iter().map(|n| n.as_ptr().cast()).collect();

        let mut device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_cis)
            .enabled_extension_names(&extension_ptrs)
            .enabled_features(&enabled);
        if self.features.fp16_storage {
            device_ci = device_ci.push_next(&mut storage_16bit);
        }
        if self.features.int8_storage {
            device_ci = device_ci.push_next(&mut storage_8bit_features);
        }
        // shader_f16_i8_features carries both shaderFloat16 and
        // shaderInt8 bits; push it once if either capability was
        // requested so the chain only appears in the pNext list a
        // single time.
        if self.features.fp16_arith || self.features.int8_arith {
            device_ci = device_ci.push_next(&mut shader_f16_i8_features);
        }
        if self.features.dynamic_rendering {
            device_ci = device_ci.push_next(&mut dynamic_rendering_features);
        }

        // SAFETY: adapter.physical_device was enumerated from the same
        // instance; queue_ci references priorities slice which outlives
        // this call. The pushed feature structs are stack-allocated above
        // and outlive create_device because device_ci borrows them and is
        // dropped at the end of this scope after the call returns.
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
