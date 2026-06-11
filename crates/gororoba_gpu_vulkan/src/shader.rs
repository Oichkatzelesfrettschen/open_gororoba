//! Shader module loading (WGSL via naga, or pre-compiled SPIR-V).
//!
//! Consolidates the lbm_vulkan/src/compute.rs:166-184 `compile_wgsl` helper
//! and standardises SPIR-V loading (cd_kernel turboquant embeds pre-compiled
//! SPIR-V bytes at quantizer.rs:140-153).

use std::{collections::BTreeMap, sync::Arc};

use ash::vk;
use naga::{ShaderStage, back::spv, front::wgsl, valid::Validator};

use crate::{
    device::Device,
    error::{Result, VulkanError},
};

/// Owned VkShaderModule + the source SPIR-V bytecode it was built from.
pub struct ShaderModule {
    device: Arc<Device>,
    raw: vk::ShaderModule,
    entry_point: String,
    override_ids: BTreeMap<String, u32>,
}

impl ShaderModule {
    /// Compile WGSL source through naga (parse + validate + spv emit) and
    /// create a `vk::ShaderModule`.
    ///
    /// `entry_point` is the function name in the WGSL source that this
    /// module exposes (typically "main").
    pub fn from_wgsl(device: &Device, wgsl_source: &str, entry_point: &str) -> Result<Self> {
        let module =
            wgsl::parse_str(wgsl_source).map_err(|e| VulkanError::WgslParse(e.to_string()))?;
        let info = Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| VulkanError::WgslValidation(e.to_string()))?;

        let override_ids = collect_override_ids(&module)?;

        let options = spv::Options::default();
        let pipeline_options = spv::PipelineOptions {
            shader_stage: ShaderStage::Compute,
            entry_point: entry_point.to_string(),
        };
        let spirv = spv::write_vec(&module, &info, &options, Some(&pipeline_options))
            .map_err(|e| VulkanError::SpirvEmit(e.to_string()))?;

        Self::from_spv_with_overrides(device, &spirv, entry_point, override_ids)
    }

    /// Compile WGSL after replacing selected u32 `override` declarations with
    /// concrete `const` declarations.
    ///
    /// naga 29 parses and validates WGSL overrides, but its SPIR-V backend
    /// rejects `Expression::Override`. This path keeps the named-override
    /// contract check, then emits one SPIR-V module per override map.
    pub fn from_wgsl_with_u32_overrides(
        device: &Device,
        wgsl_source: &str,
        entry_point: &str,
        overrides: &[(String, u32)],
    ) -> Result<Self> {
        let original_module =
            wgsl::parse_str(wgsl_source).map_err(|e| VulkanError::WgslParse(e.to_string()))?;
        let override_ids = collect_override_ids(&original_module)?;
        for (name, _) in overrides {
            if !override_ids.contains_key(name) {
                return Err(VulkanError::PipelineOverride(format!(
                    "WGSL override `{name}` is missing or has no explicit @id"
                )));
            }
        }

        let specialized_source = Self::specialize_wgsl_u32_overrides(wgsl_source, overrides)?;
        let module = wgsl::parse_str(&specialized_source)
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

        Self::from_spv_with_overrides(device, &spirv, entry_point, override_ids)
    }

    /// Return WGSL with selected single-line u32 overrides replaced by consts.
    ///
    /// This intentionally accepts only the compact declaration shape used by
    /// the workspace shaders: optional attributes followed by
    /// `override NAME: u32;` on one line.
    pub fn specialize_wgsl_u32_overrides(
        wgsl_source: &str,
        overrides: &[(String, u32)],
    ) -> Result<String> {
        let override_values: BTreeMap<&str, u32> = overrides
            .iter()
            .map(|(name, value)| (name.as_str(), *value))
            .collect();
        let mut replaced: BTreeMap<&str, bool> = override_values
            .keys()
            .copied()
            .map(|name| (name, false))
            .collect();
        let mut output = String::with_capacity(wgsl_source.len());

        for line in wgsl_source.lines() {
            let trimmed = line.trim_start();
            let indent_len = line.len() - trimmed.len();
            let indent = &line[..indent_len];
            let Some(override_pos) = trimmed.find("override ") else {
                output.push_str(line);
                output.push('\n');
                continue;
            };
            let name_start = override_pos + "override ".len();
            let name_end = trimmed[name_start..]
                .find(|ch: char| !(ch == '_' || ch.is_ascii_alphanumeric()))
                .map(|offset| name_start + offset)
                .unwrap_or(trimmed.len());
            let name = &trimmed[name_start..name_end];
            let Some(value) = override_values.get(name).copied() else {
                output.push_str(line);
                output.push('\n');
                continue;
            };

            let after_name = trimmed[name_end..].trim_start();
            if !after_name.starts_with(": u32;") {
                return Err(VulkanError::PipelineOverride(format!(
                    "WGSL override `{name}` must use single-line `override {name}: u32;` form"
                )));
            }

            output.push_str(indent);
            output.push_str("const ");
            output.push_str(name);
            output.push_str(": u32 = ");
            output.push_str(&value.to_string());
            output.push_str("u;\n");
            replaced.insert(name, true);
        }

        for (name, was_replaced) in replaced {
            if !was_replaced {
                return Err(VulkanError::PipelineOverride(format!(
                    "WGSL override `{name}` was not found in a replaceable u32 declaration"
                )));
            }
        }

        Ok(output)
    }

    /// Create a shader module from pre-compiled SPIR-V words.
    pub fn from_spv(device: &Device, spv: &[u32], entry_point: &str) -> Result<Self> {
        Self::from_spv_with_overrides(device, spv, entry_point, BTreeMap::new())
    }

    fn from_spv_with_overrides(
        device: &Device,
        spv: &[u32],
        entry_point: &str,
        override_ids: BTreeMap<String, u32>,
    ) -> Result<Self> {
        let ci = vk::ShaderModuleCreateInfo::default().code(spv);
        // SAFETY: spv slice is well-formed (validated by naga in the WGSL
        // path; caller-provided in the SPV-direct path); ash copies the
        // payload before returning so the slice need not outlive the call.
        let raw = unsafe { device.raw().create_shader_module(&ci, None) }?;
        Ok(Self {
            device: Arc::new(device.clone()),
            raw,
            entry_point: entry_point.to_string(),
            override_ids,
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

    /// Vulkan specialization constant ID for a named WGSL override.
    pub fn override_constant_id(&self, name: &str) -> Option<u32> {
        self.override_ids.get(name).copied()
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

fn collect_override_ids(module: &naga::Module) -> Result<BTreeMap<String, u32>> {
    let mut override_ids = BTreeMap::new();
    for (_, pipeline_override) in module.overrides.iter() {
        let Some(name) = pipeline_override.name.as_deref() else {
            continue;
        };
        let Some(id) = pipeline_override.id else {
            return Err(VulkanError::PipelineOverride(format!(
                "WGSL override `{name}` needs an explicit @id for Vulkan specialization"
            )));
        };
        override_ids.insert(name.to_string(), u32::from(id));
    }
    Ok(override_ids)
}

#[cfg(test)]
mod tests {
    use super::collect_override_ids;

    #[test]
    fn collects_named_wgsl_override_ids() {
        let module = naga::front::wgsl::parse_str(
            r"
@id(7) override CD_DIM: u32;

@compute @workgroup_size(1)
fn main() {
    let _half = CD_DIM / 2u;
}
",
        )
        .expect("WGSL override fixture parses");

        let override_ids = collect_override_ids(&module).expect("override id is explicit");

        assert_eq!(override_ids.get("CD_DIM"), Some(&7));
    }

    #[test]
    fn rejects_named_wgsl_override_without_id() {
        let module = naga::front::wgsl::parse_str(
            r"
override CD_DIM: u32;

@compute @workgroup_size(1)
fn main() {
    let _half = CD_DIM / 2u;
}
",
        )
        .expect("WGSL override fixture parses");

        let error = collect_override_ids(&module).expect_err("missing id is rejected");

        assert!(error.to_string().contains("explicit @id"));
    }

    #[test]
    fn specializes_single_line_u32_wgsl_override() {
        let specialized = super::ShaderModule::specialize_wgsl_u32_overrides(
            r"
@id(0) override CD_DIM: u32;

@compute @workgroup_size(1)
fn main() {
    let _half = CD_DIM / 2u;
}
",
            &[("CD_DIM".to_string(), 64)],
        )
        .unwrap();
        assert!(specialized.contains("const CD_DIM: u32 = 64u;"));
        assert!(!specialized.contains("override CD_DIM"));

        let module = naga::front::wgsl::parse_str(&specialized).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let pipeline_options = naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: "main".to_string(),
        };
        let spirv = naga::back::spv::write_vec(
            &module,
            &info,
            &naga::back::spv::Options::default(),
            Some(&pipeline_options),
        )
        .unwrap();
        assert!(!spirv.is_empty());
    }
}
