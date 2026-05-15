//! Shader module loading (WGSL via naga, or pre-compiled SPIR-V).
//!
//! Consolidates the lbm_vulkan/src/compute.rs:166-184 `compile_wgsl` helper
//! and standardises SPIR-V loading (cd_kernel turboquant embeds pre-compiled
//! SPIR-V bytes at quantizer.rs:140-153).

use std::sync::Arc;

use ash::vk;
use naga::{ShaderStage, back::spv, front::wgsl, valid::Validator};

use crate::device::Device;
use crate::error::{Result, VulkanError};

/// Owned VkShaderModule + the source SPIR-V bytecode it was built from.
pub struct ShaderModule {
    device: Arc<Device>,
    raw: vk::ShaderModule,
    entry_point: String,
}

impl ShaderModule {
    /// Compile WGSL source through naga (parse + validate + spv emit) and
    /// create a `vk::ShaderModule`.
    ///
    /// `entry_point` is the function name in the WGSL source that this
    /// module exposes (typically "main").
    pub fn from_wgsl(device: &Device, wgsl_source: &str, entry_point: &str) -> Result<Self> {
        let module = wgsl::parse_str(wgsl_source)
            .map_err(|e| VulkanError::WgslParse(e.to_string()))?;
        let info = Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| VulkanError::WgslValidation(e.to_string()))?;

        let options = spv::Options::default();
        let pipeline_options = spv::PipelineOptions {
            shader_stage: ShaderStage::Compute,
            entry_point: entry_point.to_string(),
        };
        let spirv = spv::write_vec(&module, &info, &options, Some(&pipeline_options))
            .map_err(|e| VulkanError::SpirvEmit(e.to_string()))?;

        Self::from_spv(device, &spirv, entry_point)
    }

    /// Create a shader module from pre-compiled SPIR-V words.
    pub fn from_spv(device: &Device, spv: &[u32], entry_point: &str) -> Result<Self> {
        let ci = vk::ShaderModuleCreateInfo::default().code(spv);
        // SAFETY: spv slice is well-formed (validated by naga in the WGSL
        // path; caller-provided in the SPV-direct path); ash copies the
        // payload before returning so the slice need not outlive the call.
        let raw = unsafe { device.raw().create_shader_module(&ci, None) }?;
        Ok(Self {
            device: Arc::new(device.clone()),
            raw,
            entry_point: entry_point.to_string(),
        })
    }

    /// Convenience for callers that already have SPIR-V as a byte slice
    /// (e.g. via `include_bytes!`). The slice must be 4-byte aligned;
    /// if not aligned, this method copies into a u32 buffer.
    pub fn from_spv_bytes(device: &Device, spv_bytes: &[u8], entry_point: &str) -> Result<Self> {
        if spv_bytes.len().is_multiple_of(4) {
            // Round-trip via copy to ensure alignment.
            let mut words = vec![0u32; spv_bytes.len() / 4];
            for (i, chunk) in spv_bytes.chunks_exact(4).enumerate() {
                words[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Self::from_spv(device, &words, entry_point)
        } else {
            Err(VulkanError::UnsupportedFeature(
                "SPIR-V byte slice not a multiple of 4",
            ))
        }
    }

    /// Raw `vk::ShaderModule` handle (caller does not own).
    pub fn raw(&self) -> vk::ShaderModule {
        self.raw
    }

    /// Entry-point name this module exposes.
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        // SAFETY: self.raw was created by create_shader_module above; the
        // Arc<Device> ensures the device is still alive at Drop time.
        unsafe {
            self.device.raw().destroy_shader_module(self.raw, None);
        }
    }
}
