//! Vulkan launcher scaffold for Cayley-Dickson eta matrix computation.
//!
//! The CUDA eta kernel has one specialization axis: the Cayley-Dickson
//! dimension. Vulkan carries that axis through a WGSL `override CD_DIM` so a
//! pipeline can be specialized once per dimension and reused for dispatches.

use gororoba_gpu_vulkan::{
    Adapter, ComputePipeline, ComputePipelineBuilder, DescriptorPool, DescriptorSetLayout,
    DescriptorSetLayoutSpec, Device, DeviceBuilder, DispatchScope, HostVisibleBuffer, Instance,
    InstanceBuilder, QueueFamilyRequirement, ShaderModule, ValidationPolicy,
};

pub const ETA_MATRIX_VULKAN_ENTRY_POINT: &str = "compute_eta_matrix";

pub const ETA_MATRIX_VULKAN_WGSL: &str = r#"
@id(0) override CD_DIM: u32;

struct EtaMatrix {
    values: array<u32>,
};

@group(0) @binding(0)
var<storage, read_write> eta_out: EtaMatrix;

fn cd_basis_mul_sign(dim: u32, p_input: u32, q_input: u32) -> i32 {
    var sign: i32 = 1;
    var p: u32 = p_input;
    var q: u32 = q_input;
    var half: u32 = dim / 2u;

    while (half > 0u) {
        let p_hi: u32 = select(0u, 1u, p >= half);
        let q_hi: u32 = select(0u, 1u, q >= half);
        let branch: u32 = (p_hi << 1u) | q_hi;

        if (branch == 1u) {
            let qh: u32 = q - half;
            q = p;
            p = qh;
        } else if (branch == 2u) {
            p = p - half;
            if (q != 0u) {
                sign = -sign;
            }
        } else if (branch == 3u) {
            let qh: u32 = q - half;
            let ph: u32 = p - half;
            if (qh == 0u) {
                return -sign;
            }
            p = qh;
            q = ph;
        }

        half = half >> 1u;
    }

    return sign;
}

fn psi(dim: u32, i: u32, j: u32) -> u32 {
    return select(1u, 0u, cd_basis_mul_sign(dim, i, j) == 1);
}

@compute @workgroup_size(256)
fn compute_eta_matrix(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim_half: u32 = CD_DIM / 2u;
    let total: u32 = dim_half * dim_half;
    let idx: u32 = gid.x;

    if (idx >= total) {
        return;
    }

    let i: u32 = idx / dim_half;
    let j: u32 = idx % dim_half;
    let psi_ij: u32 = psi(CD_DIM, i, j + dim_half);
    let psi_ji: u32 = psi(CD_DIM, j, i + dim_half);
    eta_out.values[idx] = psi_ij ^ psi_ji;
}
"#;

pub struct EtaMatrixVulkanPipeline {
    pub pipeline: ComputePipeline,
    pub descriptor_layout: DescriptorSetLayout,
}

pub struct EtaMatrixVulkanKernel;

impl EtaMatrixVulkanKernel {
    pub fn wgsl_source() -> &'static str {
        ETA_MATRIX_VULKAN_WGSL
    }

    pub fn output_len(dim: usize) -> Result<usize, String> {
        Self::validate_dim(dim)?;
        let dim_half = dim / 2;
        dim_half
            .checked_mul(dim_half)
            .ok_or_else(|| format!("eta matrix dimension {dim} overflows usize"))
    }

    pub fn dispatch_groups(dim: usize) -> Result<u32, String> {
        let output_len = Self::output_len(dim)?;
        let output_len_u32 = u32::try_from(output_len)
            .map_err(|_| format!("eta matrix output for dimension {dim} exceeds u32 dispatch"))?;
        Ok(output_len_u32.div_ceil(256))
    }

    pub fn is_available() -> bool {
        match Self::build_context() {
            Ok((_instance, _adapter, _device)) => true,
            Err(_) => false,
        }
    }

    pub fn compute(dim: usize) -> Result<Vec<u8>, String> {
        let output_len = Self::output_len(dim)?;
        let (_instance, adapter, device) = Self::build_context()?;
        let pipeline = Self::build_pipeline(&device, dim)?;
        let byte_len = output_len
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| format!("eta matrix output for dimension {dim} overflows bytes"))?
            as u64;
        let output_buffer = HostVisibleBuffer::storage(&device, &adapter, byte_len)
            .map_err(|e| format!("eta output buffer allocation failed: {e}"))?;
        let zeroed_output = vec![0u32; output_len];
        output_buffer
            .write_u32_slice(&zeroed_output)
            .map_err(|e| format!("eta output buffer initialization failed: {e}"))?;
        let descriptor_pool = DescriptorPool::for_layout(&device, &pipeline.descriptor_layout, 1)
            .map_err(|e| format!("eta descriptor pool allocation failed: {e}"))?;
        let descriptor_set = descriptor_pool
            .allocate_set(&pipeline.descriptor_layout)
            .map_err(|e| format!("eta descriptor set allocation failed: {e}"))?;
        descriptor_set.write_storage_buffer(0, &output_buffer);

        let dispatch =
            DispatchScope::new(&device).map_err(|e| format!("eta dispatch scope failed: {e}"))?;
        let groups = Self::dispatch_groups(dim)?;
        dispatch
            .dispatch(
                &pipeline.pipeline,
                descriptor_set.raw(),
                groups,
                1,
                1,
                5_000_000_000,
            )
            .map_err(|e| format!("eta Vulkan dispatch failed: {e}"))?;

        let values = output_buffer
            .read_u32_slice(output_len)
            .map_err(|e| format!("eta output readback failed: {e}"))?;
        let mut eta = Vec::with_capacity(output_len);
        for value in values {
            match value {
                0 => eta.push(0),
                1 => eta.push(1),
                other => {
                    return Err(format!("eta Vulkan shader wrote non-binary value {other}"));
                }
            }
        }

        Ok(eta)
    }

    pub fn build_pipeline(device: &Device, dim: usize) -> Result<EtaMatrixVulkanPipeline, String> {
        Self::validate_dim(dim)?;
        let dim_u32 = u32::try_from(dim)
            .map_err(|_| format!("Vulkan CD_DIM override does not fit u32: {dim}"))?;
        let overrides = [("CD_DIM".to_string(), dim_u32)];
        let shader = ShaderModule::from_wgsl_with_u32_overrides(
            device,
            ETA_MATRIX_VULKAN_WGSL,
            ETA_MATRIX_VULKAN_ENTRY_POINT,
            &overrides,
        )
        .map_err(|e| format!("eta WGSL compile failed: {e}"))?;
        let descriptor_layout = DescriptorSetLayoutSpec::new()
            .storage_buffer(0)
            .build(device)
            .map_err(|e| format!("eta descriptor layout failed: {e}"))?;
        let pipeline = ComputePipelineBuilder::new(device, &shader)
            .descriptor_layout(&descriptor_layout)
            .build()
            .map_err(|e| format!("eta compute pipeline build failed: {e}"))?;

        Ok(EtaMatrixVulkanPipeline {
            pipeline,
            descriptor_layout,
        })
    }

    fn validate_dim(dim: usize) -> Result<(), String> {
        if dim < 2 {
            return Err(format!("eta matrix dimension must be >= 2, got {dim}"));
        }
        if !dim.is_power_of_two() {
            return Err(format!(
                "eta matrix dimension must be a power of two, got {dim}"
            ));
        }
        Ok(())
    }

    fn build_context() -> Result<(Instance, Adapter, Device), String> {
        let instance = InstanceBuilder::new("gororoba_algebra_eta_matrix_vulkan")
            .validation(ValidationPolicy::Disable)
            .build()
            .map_err(|e| format!("eta Vulkan instance creation failed: {e}"))?;
        let adapter = Adapter::pick(&instance, QueueFamilyRequirement::Compute)
            .map_err(|e| format!("eta Vulkan adapter pick failed: {e}"))?;
        let device = DeviceBuilder::new(adapter.clone())
            .build(&instance)
            .map_err(|e| format!("eta Vulkan device creation failed: {e}"))?;
        Ok((instance, adapter, device))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn eta_vulkan_wgsl_uses_explicit_cd_dim_override() {
        let wgsl = EtaMatrixVulkanKernel::wgsl_source();
        assert!(wgsl.contains("@id(0) override CD_DIM: u32;"));
        assert!(wgsl.contains("@compute @workgroup_size(256)"));
        assert!(wgsl.contains("fn compute_eta_matrix"));
    }

    #[test]
    fn eta_vulkan_wgsl_parses_and_emits_compute_spirv() {
        let original_module =
            naga::front::wgsl::parse_str(EtaMatrixVulkanKernel::wgsl_source()).unwrap();
        let override_ids: BTreeMap<&str, u32> = original_module
            .overrides
            .iter()
            .filter_map(|(_, override_constant)| {
                Some((
                    override_constant.name.as_deref()?,
                    u32::from(override_constant.id?),
                ))
            })
            .collect();
        assert_eq!(override_ids.get("CD_DIM"), Some(&0));

        let specialized = ShaderModule::specialize_wgsl_u32_overrides(
            EtaMatrixVulkanKernel::wgsl_source(),
            &[("CD_DIM".to_string(), 64)],
        )
        .unwrap();
        let module = naga::front::wgsl::parse_str(&specialized).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();

        let pipeline_options = naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: ETA_MATRIX_VULKAN_ENTRY_POINT.to_string(),
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

    #[test]
    fn eta_vulkan_output_len_matches_cpu_shape() {
        assert_eq!(EtaMatrixVulkanKernel::output_len(16).unwrap(), 64);
        assert_eq!(EtaMatrixVulkanKernel::output_len(32).unwrap(), 256);
    }

    #[test]
    fn eta_vulkan_dispatch_groups_round_up_work_items() {
        assert_eq!(EtaMatrixVulkanKernel::dispatch_groups(16).unwrap(), 1);
        assert_eq!(EtaMatrixVulkanKernel::dispatch_groups(64).unwrap(), 4);
    }

    #[test]
    fn eta_vulkan_rejects_invalid_dimensions() {
        assert!(EtaMatrixVulkanKernel::output_len(0).is_err());
        assert!(EtaMatrixVulkanKernel::output_len(3).is_err());
    }

    #[test]
    #[ignore = "requires local Vulkan compute device"]
    fn eta_vulkan_context_smoke() {
        eprintln!("eta_vulkan_context_smoke: before build_context");
        let (_instance, adapter, _device) = EtaMatrixVulkanKernel::build_context().unwrap();
        eprintln!(
            "eta_vulkan_context_smoke: selected device {}",
            adapter.device_name()
        );
    }

    #[test]
    #[ignore = "requires local Vulkan compute device"]
    fn eta_vulkan_pipeline_smoke() {
        eprintln!("eta_vulkan_pipeline_smoke: before build_context");
        let (_instance, adapter, device) = EtaMatrixVulkanKernel::build_context().unwrap();
        eprintln!(
            "eta_vulkan_pipeline_smoke: selected device {}",
            adapter.device_name()
        );
        let _pipeline = EtaMatrixVulkanKernel::build_pipeline(&device, 8).unwrap();
        eprintln!("eta_vulkan_pipeline_smoke: pipeline built");
    }

    #[test]
    #[ignore = "requires local Vulkan compute device"]
    fn eta_vulkan_dim8_compute_smoke() {
        eprintln!("eta_vulkan_dim8_compute_smoke: before compute");
        let eta = EtaMatrixVulkanKernel::compute(8).unwrap();
        eprintln!("eta_vulkan_dim8_compute_smoke: values {}", eta.len());
        assert_eq!(eta.len(), 16);
    }
}
