//! Vulkan PEPS row contraction.
//!
//! The Vulkan path mirrors `peps_cubecl`: host values narrow from `faer::c64`
//! to FP32 before dispatch, and the readback widens to `c64`. CUDA remains the
//! FP64 backend when the `gpu` feature is enabled.

#![cfg(feature = "vulkan")]

use faer::c64;
use gororoba_gpu_vulkan::{
    Adapter, ComputePipeline, ComputePipelineBuilder, DescriptorPool, DescriptorSetLayout,
    DescriptorSetLayoutSpec, Device, DeviceBuilder, DispatchScope, HostVisibleBuffer, Instance,
    InstanceBuilder, QueueFamilyRequirement, ShaderModule, ValidationPolicy,
};

pub const PEPS_VULKAN_ENTRY_POINT: &str = "peps_contract_rows";
const WORKGROUP_SIZE: u32 = 256;
const DISPATCH_TIMEOUT_NS: u64 = 10_000_000_000;

pub const PEPS_VULKAN_WGSL: &str = r#"
struct F32Buffer {
    values: array<f32>,
};

struct Params {
    len: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0)
var<storage, read> upper_re: F32Buffer;
@group(0) @binding(1)
var<storage, read> upper_im: F32Buffer;
@group(0) @binding(2)
var<storage, read> lower_re: F32Buffer;
@group(0) @binding(3)
var<storage, read> lower_im: F32Buffer;
@group(0) @binding(4)
var<storage, read_write> result_re: F32Buffer;
@group(0) @binding(5)
var<storage, read_write> result_im: F32Buffer;
@group(0) @binding(6)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn peps_contract_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;
    if (idx >= params.len) {
        return;
    }

    let a_re: f32 = upper_re.values[idx];
    let a_im: f32 = upper_im.values[idx];
    let b_re: f32 = lower_re.values[idx];
    let b_im: f32 = lower_im.values[idx];

    result_re.values[idx] = a_re * b_re - a_im * b_im;
    result_im.values[idx] = a_re * b_im + a_im * b_re;
}
"#;

pub struct PepsVulkanPipeline {
    pipeline: ComputePipeline,
    descriptor_layout: DescriptorSetLayout,
}

pub struct PepsVulkanKernel;

impl PepsVulkanKernel {
    pub fn is_available() -> bool {
        peps_vulkan_available()
    }

    pub fn contract_rows_fp32(upper: &[c64], lower: &[c64]) -> Result<Vec<c64>, String> {
        let prepared = PreparedPepsRows::new(upper, lower)?;
        if prepared.len == 0 {
            return Ok(Vec::new());
        }

        let (_instance, adapter, device) = build_vulkan_context()?;
        let pipeline = build_vulkan_pipeline(&device)?;
        let run = PepsVulkanRun::new(&device, &adapter, &pipeline, &prepared)?;
        let dispatch = DispatchScope::new(&device)
            .map_err(|e| format!("PEPS Vulkan dispatch scope failed: {e}"))?;
        let groups = prepared.len.div_ceil(WORKGROUP_SIZE as usize) as u32;
        dispatch
            .dispatch(
                &pipeline.pipeline,
                run.descriptor_set.raw(),
                groups,
                1,
                1,
                DISPATCH_TIMEOUT_NS,
            )
            .map_err(|e| format!("PEPS Vulkan dispatch failed: {e}"))?;

        let result_re = run
            .result_re_buffer
            .read_f32_slice(prepared.len)
            .map_err(|e| format!("PEPS Vulkan result_re readback failed: {e}"))?;
        let result_im = run
            .result_im_buffer
            .read_f32_slice(prepared.len)
            .map_err(|e| format!("PEPS Vulkan result_im readback failed: {e}"))?;

        Ok(result_re
            .into_iter()
            .zip(result_im)
            .map(|(re, im)| c64::new(re as f64, im as f64))
            .collect())
    }
}

pub fn peps_vulkan_available() -> bool {
    match build_vulkan_context() {
        Ok((_instance, _adapter, _device)) => true,
        Err(_) => false,
    }
}

pub fn vulkan_contract_rows_peps_fp32(upper: &[c64], lower: &[c64]) -> Vec<c64> {
    if upper.len() != lower.len() {
        return lower.to_vec();
    }

    if let Ok(result) = PepsVulkanKernel::contract_rows_fp32(upper, lower) {
        return result;
    }

    peps_contract_rows_cpu(upper, lower)
}

pub fn peps_contract_rows_cpu(upper: &[c64], lower: &[c64]) -> Vec<c64> {
    upper
        .iter()
        .zip(lower.iter())
        .map(|(a, b)| c64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re))
        .collect()
}

struct PepsVulkanRun {
    #[allow(dead_code)]
    descriptor_pool: DescriptorPool,
    descriptor_set: gororoba_gpu_vulkan::DescriptorSet,
    #[allow(dead_code)]
    upper_re_buffer: HostVisibleBuffer,
    #[allow(dead_code)]
    upper_im_buffer: HostVisibleBuffer,
    #[allow(dead_code)]
    lower_re_buffer: HostVisibleBuffer,
    #[allow(dead_code)]
    lower_im_buffer: HostVisibleBuffer,
    result_re_buffer: HostVisibleBuffer,
    result_im_buffer: HostVisibleBuffer,
    #[allow(dead_code)]
    params_buffer: HostVisibleBuffer,
}

impl PepsVulkanRun {
    fn new(
        device: &Device,
        adapter: &Adapter,
        pipeline: &PepsVulkanPipeline,
        prepared: &PreparedPepsRows,
    ) -> Result<Self, String> {
        let descriptor_pool = DescriptorPool::for_layout(device, &pipeline.descriptor_layout, 1)
            .map_err(|e| format!("PEPS Vulkan descriptor pool failed: {e}"))?;
        let descriptor_set = descriptor_pool
            .allocate_set(&pipeline.descriptor_layout)
            .map_err(|e| format!("PEPS Vulkan descriptor set failed: {e}"))?;

        let upper_re_buffer = upload_storage_f32(device, adapter, &prepared.upper_re, "upper_re")?;
        let upper_im_buffer = upload_storage_f32(device, adapter, &prepared.upper_im, "upper_im")?;
        let lower_re_buffer = upload_storage_f32(device, adapter, &prepared.lower_re, "lower_re")?;
        let lower_im_buffer = upload_storage_f32(device, adapter, &prepared.lower_im, "lower_im")?;
        let result_re_buffer =
            upload_storage_f32(device, adapter, &vec![0.0f32; prepared.len], "result_re")?;
        let result_im_buffer =
            upload_storage_f32(device, adapter, &vec![0.0f32; prepared.len], "result_im")?;
        let params_buffer = HostVisibleBuffer::uniform(device, adapter, 16)
            .map_err(|e| format!("PEPS Vulkan params buffer allocation failed: {e}"))?;
        params_buffer
            .write_bytes(&encode_params(prepared))
            .map_err(|e| format!("PEPS Vulkan params upload failed: {e}"))?;

        descriptor_set.write_storage_buffer(0, &upper_re_buffer);
        descriptor_set.write_storage_buffer(1, &upper_im_buffer);
        descriptor_set.write_storage_buffer(2, &lower_re_buffer);
        descriptor_set.write_storage_buffer(3, &lower_im_buffer);
        descriptor_set.write_storage_buffer(4, &result_re_buffer);
        descriptor_set.write_storage_buffer(5, &result_im_buffer);
        descriptor_set.write_uniform_buffer(6, &params_buffer);

        Ok(Self {
            descriptor_pool,
            descriptor_set,
            upper_re_buffer,
            upper_im_buffer,
            lower_re_buffer,
            lower_im_buffer,
            result_re_buffer,
            result_im_buffer,
            params_buffer,
        })
    }
}

struct PreparedPepsRows {
    upper_re: Vec<f32>,
    upper_im: Vec<f32>,
    lower_re: Vec<f32>,
    lower_im: Vec<f32>,
    len: usize,
}

impl PreparedPepsRows {
    fn new(upper: &[c64], lower: &[c64]) -> Result<Self, String> {
        if upper.len() != lower.len() {
            return Err(format!(
                "PEPS Vulkan row lengths differ: upper {}, lower {}",
                upper.len(),
                lower.len()
            ));
        }
        if upper.len() > u32::MAX as usize {
            return Err(format!(
                "PEPS Vulkan row length {} exceeds u32 dispatch",
                upper.len()
            ));
        }

        let mut upper_re = Vec::with_capacity(upper.len());
        let mut upper_im = Vec::with_capacity(upper.len());
        let mut lower_re = Vec::with_capacity(lower.len());
        let mut lower_im = Vec::with_capacity(lower.len());

        for (index, value) in upper.iter().enumerate() {
            upper_re.push(narrow_component(value.re, index, "upper_re")?);
            upper_im.push(narrow_component(value.im, index, "upper_im")?);
        }
        for (index, value) in lower.iter().enumerate() {
            lower_re.push(narrow_component(value.re, index, "lower_re")?);
            lower_im.push(narrow_component(value.im, index, "lower_im")?);
        }

        Ok(Self {
            upper_re,
            upper_im,
            lower_re,
            lower_im,
            len: upper.len(),
        })
    }
}

fn build_vulkan_context() -> Result<(Instance, Adapter, Device), String> {
    let instance = InstanceBuilder::new("quantum_core_peps_vulkan")
        .validation(ValidationPolicy::Disable)
        .build()
        .map_err(|e| format!("PEPS Vulkan instance failed: {e}"))?;
    let adapter = Adapter::pick(&instance, QueueFamilyRequirement::Compute)
        .map_err(|e| format!("PEPS Vulkan adapter failed: {e}"))?;
    let device = DeviceBuilder::new(adapter.clone())
        .build(&instance)
        .map_err(|e| format!("PEPS Vulkan device failed: {e}"))?;
    Ok((instance, adapter, device))
}

fn build_vulkan_pipeline(device: &Device) -> Result<PepsVulkanPipeline, String> {
    let shader = ShaderModule::from_wgsl(device, PEPS_VULKAN_WGSL, PEPS_VULKAN_ENTRY_POINT)
        .map_err(|e| format!("PEPS Vulkan WGSL compile failed: {e}"))?;
    let descriptor_layout = DescriptorSetLayoutSpec::new()
        .storage_buffer(0)
        .storage_buffer(1)
        .storage_buffer(2)
        .storage_buffer(3)
        .storage_buffer(4)
        .storage_buffer(5)
        .uniform_buffer(6)
        .build(device)
        .map_err(|e| format!("PEPS Vulkan descriptor layout failed: {e}"))?;
    let pipeline = ComputePipelineBuilder::new(device, &shader)
        .descriptor_layout(&descriptor_layout)
        .build()
        .map_err(|e| format!("PEPS Vulkan pipeline build failed: {e}"))?;

    Ok(PepsVulkanPipeline {
        pipeline,
        descriptor_layout,
    })
}

fn upload_storage_f32(
    device: &Device,
    adapter: &Adapter,
    values: &[f32],
    label: &str,
) -> Result<HostVisibleBuffer, String> {
    let buffer = HostVisibleBuffer::storage(device, adapter, byte_len::<f32>(values.len(), label)?)
        .map_err(|e| format!("PEPS Vulkan {label} buffer allocation failed: {e}"))?;
    buffer
        .write_f32_slice(values)
        .map_err(|e| format!("PEPS Vulkan {label} buffer upload failed: {e}"))?;
    Ok(buffer)
}

fn byte_len<T>(len: usize, label: &str) -> Result<u64, String> {
    len.checked_mul(std::mem::size_of::<T>())
        .map(|bytes| bytes as u64)
        .ok_or_else(|| format!("PEPS Vulkan {label} buffer size overflows"))
}

fn encode_params(prepared: &PreparedPepsRows) -> [u8; 16] {
    let words = [
        (prepared.len as u32).to_le_bytes(),
        0u32.to_le_bytes(),
        0u32.to_le_bytes(),
        0u32.to_le_bytes(),
    ];
    let mut bytes = [0u8; 16];
    for (idx, word) in words.iter().enumerate() {
        let start = 4 * idx;
        bytes[start..start + 4].copy_from_slice(word);
    }
    bytes
}

fn narrow_component(value: f64, index: usize, label: &str) -> Result<f32, String> {
    if !value.is_finite() {
        return Err(format!("PEPS Vulkan {label}[{index}] is not finite"));
    }
    if value.abs() > f32::MAX as f64 {
        return Err(format!("PEPS Vulkan {label}[{index}] exceeds f32 range"));
    }
    Ok(value as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peps_vulkan_available_does_not_panic() {
        let _ = PepsVulkanKernel::is_available();
    }

    #[test]
    fn peps_cpu_reference_multiplies_complex_rows() {
        let upper = vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)];
        let lower = vec![c64::new(2.0, 1.0), c64::new(1.0, 1.0)];
        let result = peps_contract_rows_cpu(&upper, &lower);
        assert_eq!(result, vec![c64::new(0.0, 5.0), c64::new(-1.0, 7.0)]);
    }

    #[test]
    fn peps_preparation_rejects_invalid_components() {
        let upper = vec![c64::new(f64::INFINITY, 0.0)];
        let lower = vec![c64::new(1.0, 0.0)];
        assert!(PreparedPepsRows::new(&upper, &lower).is_err());
    }

    #[test]
    fn peps_vulkan_matches_cpu_when_adapter_available() {
        if !PepsVulkanKernel::is_available() {
            return;
        }

        let upper = vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)];
        let lower = vec![c64::new(2.0, 1.0), c64::new(1.0, 1.0)];
        let vulkan = PepsVulkanKernel::contract_rows_fp32(&upper, &lower).unwrap();
        let cpu = peps_contract_rows_cpu(&upper, &lower);

        assert_eq!(vulkan, cpu);
    }
}
