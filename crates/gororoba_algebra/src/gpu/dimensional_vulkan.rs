//! Vulkan classifier for dimensional APT census samples.
//!
//! The dimensional APT census samples node triples and classifies each
//! triangle by Cayley-Dickson eta values. This launcher prepares the same
//! deterministic sample triples as the CPU path, then runs the sign/eta
//! classification in a Vulkan compute shader.

use gororoba_gpu_vulkan::{
    Adapter, ComputePipeline, ComputePipelineBuilder, DescriptorPool, DescriptorSetLayout,
    DescriptorSetLayoutSpec, Device, DeviceBuilder, DispatchScope, HostVisibleBuffer, Instance,
    InstanceBuilder, QueueFamilyRequirement, ShaderModule, ValidationPolicy,
};

use super::GpuAptResult;

pub const DIMENSIONAL_VULKAN_ENTRY_POINT: &str = "classify_apt_samples";

pub const DIMENSIONAL_VULKAN_WGSL: &str = r#"
struct U32Array {
    values: array<u32>,
};

struct AptCounters {
    pure_count: atomic<u32>,
    mixed_count: atomic<u32>,
    fiber_00: atomic<u32>,
    fiber_01: atomic<u32>,
    fiber_10: atomic<u32>,
    fiber_11: atomic<u32>,
};

@group(0) @binding(0)
var<storage, read> packed_nodes: U32Array;

@group(0) @binding(1)
var<storage, read> sample_i: U32Array;

@group(0) @binding(2)
var<storage, read> sample_j: U32Array;

@group(0) @binding(3)
var<storage, read> sample_k: U32Array;

@group(0) @binding(4)
var<storage, read_write> counters: AptCounters;

@group(0) @binding(5)
var<storage, read> params: U32Array;

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
fn classify_apt_samples(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;
    let dim: u32 = params.values[0];
    let n_samples: u32 = params.values[1];

    if (idx >= n_samples) {
        return;
    }

    let node_i: u32 = packed_nodes.values[sample_i.values[idx]];
    let node_j: u32 = packed_nodes.values[sample_j.values[idx]];
    let node_k: u32 = packed_nodes.values[sample_k.values[idx]];

    let ai: u32 = node_i & 0xffffu;
    let bi: u32 = (node_i >> 16u) & 0xffffu;
    let aj: u32 = node_j & 0xffffu;
    let bj: u32 = (node_j >> 16u) & 0xffffu;
    let ak: u32 = node_k & 0xffffu;
    let bk: u32 = (node_k >> 16u) & 0xffffu;

    if (ai >= dim || bi >= dim || aj >= dim || bj >= dim || ak >= dim || bk >= dim) {
        return;
    }

    let eta_ij: u32 = psi(dim, ai, bj) ^ psi(dim, aj, bi);
    let eta_ik: u32 = psi(dim, ai, bk) ^ psi(dim, ak, bi);
    let eta_jk: u32 = psi(dim, aj, bk) ^ psi(dim, ak, bj);

    if (eta_ij == eta_ik && eta_ik == eta_jk) {
        atomicAdd(&counters.pure_count, 1u);
        atomicAdd(&counters.fiber_00, 1u);
    } else {
        atomicAdd(&counters.mixed_count, 1u);
        let f0: u32 = eta_ij ^ eta_jk;
        let f1: u32 = eta_jk ^ eta_ik;
        let fiber_idx: u32 = (f0 << 1u) | f1;
        if (fiber_idx == 1u) {
            atomicAdd(&counters.fiber_01, 1u);
        } else if (fiber_idx == 2u) {
            atomicAdd(&counters.fiber_10, 1u);
        } else if (fiber_idx == 3u) {
            atomicAdd(&counters.fiber_11, 1u);
        }
    }
}
"#;

pub struct DimensionalVulkanPipeline {
    pub pipeline: ComputePipeline,
    pub descriptor_layout: DescriptorSetLayout,
}

pub struct DimensionalVulkanKernel;

impl DimensionalVulkanKernel {
    pub fn wgsl_source() -> &'static str {
        DIMENSIONAL_VULKAN_WGSL
    }

    pub fn is_available() -> bool {
        match Self::build_context() {
            Ok((_instance, _adapter, _device)) => true,
            Err(_) => false,
        }
    }

    pub fn compute_apt(
        dim: usize,
        nodes: &[(u8, u8)],
        n_samples: usize,
        seed: u64,
    ) -> Result<GpuAptResult, String> {
        let prepared = PreparedDimensionalInput::new(dim, nodes, n_samples, seed)?;
        if prepared.n_samples == 0 {
            return Ok(prepared.result_from_counters(&[0; 6]));
        }

        let (_instance, adapter, device) = Self::build_context()?;
        let pipeline = Self::build_pipeline(&device)?;
        let node_buffer = upload_u32_storage(&device, &adapter, &prepared.nodes, "nodes")?;
        let sample_i_buffer =
            upload_u32_storage(&device, &adapter, &prepared.sample_i, "sample_i")?;
        let sample_j_buffer =
            upload_u32_storage(&device, &adapter, &prepared.sample_j, "sample_j")?;
        let sample_k_buffer =
            upload_u32_storage(&device, &adapter, &prepared.sample_k, "sample_k")?;
        let counter_buffer =
            upload_u32_storage(&device, &adapter, &[0, 0, 0, 0, 0, 0], "counters")?;
        let params_buffer = upload_u32_storage(&device, &adapter, &prepared.params, "params")?;

        let descriptor_pool =
            DescriptorPool::for_layout(&device, &pipeline.descriptor_layout, 1)
                .map_err(|e| format!("dimensional descriptor pool allocation failed: {e}"))?;
        let descriptor_set = descriptor_pool
            .allocate_set(&pipeline.descriptor_layout)
            .map_err(|e| format!("dimensional descriptor set allocation failed: {e}"))?;
        descriptor_set.write_storage_buffer(0, &node_buffer);
        descriptor_set.write_storage_buffer(1, &sample_i_buffer);
        descriptor_set.write_storage_buffer(2, &sample_j_buffer);
        descriptor_set.write_storage_buffer(3, &sample_k_buffer);
        descriptor_set.write_storage_buffer(4, &counter_buffer);
        descriptor_set.write_storage_buffer(5, &params_buffer);

        let dispatch = DispatchScope::new(&device)
            .map_err(|e| format!("dimensional dispatch scope failed: {e}"))?;
        dispatch
            .dispatch(
                &pipeline.pipeline,
                descriptor_set.raw(),
                prepared.dispatch_groups()?,
                1,
                1,
                5_000_000_000,
            )
            .map_err(|e| format!("dimensional Vulkan dispatch failed: {e}"))?;

        let counters = counter_buffer
            .read_u32_slice(6)
            .map_err(|e| format!("dimensional counter readback failed: {e}"))?;
        Ok(prepared.result_from_counters(&counters))
    }

    pub fn build_pipeline(device: &Device) -> Result<DimensionalVulkanPipeline, String> {
        let shader = ShaderModule::from_wgsl(
            device,
            DIMENSIONAL_VULKAN_WGSL,
            DIMENSIONAL_VULKAN_ENTRY_POINT,
        )
        .map_err(|e| format!("dimensional WGSL compile failed: {e}"))?;
        let descriptor_layout = DescriptorSetLayoutSpec::new()
            .storage_buffer(0)
            .storage_buffer(1)
            .storage_buffer(2)
            .storage_buffer(3)
            .storage_buffer(4)
            .storage_buffer(5)
            .build(device)
            .map_err(|e| format!("dimensional descriptor layout failed: {e}"))?;
        let pipeline = ComputePipelineBuilder::new(device, &shader)
            .descriptor_layout(&descriptor_layout)
            .build()
            .map_err(|e| format!("dimensional compute pipeline build failed: {e}"))?;

        Ok(DimensionalVulkanPipeline {
            pipeline,
            descriptor_layout,
        })
    }

    fn build_context() -> Result<(Instance, Adapter, Device), String> {
        let instance = InstanceBuilder::new("gororoba_algebra_dimensional_vulkan")
            .validation(ValidationPolicy::Disable)
            .build()
            .map_err(|e| format!("dimensional Vulkan instance creation failed: {e}"))?;
        let adapter = Adapter::pick(&instance, QueueFamilyRequirement::Compute)
            .map_err(|e| format!("dimensional Vulkan adapter pick failed: {e}"))?;
        let device = DeviceBuilder::new(adapter.clone())
            .build(&instance)
            .map_err(|e| format!("dimensional Vulkan device creation failed: {e}"))?;
        Ok((instance, adapter, device))
    }
}

struct PreparedDimensionalInput {
    dim: usize,
    n_nodes: usize,
    n_samples: usize,
    nodes: Vec<u32>,
    sample_i: Vec<u32>,
    sample_j: Vec<u32>,
    sample_k: Vec<u32>,
    params: Vec<u32>,
}

impl PreparedDimensionalInput {
    fn new(dim: usize, nodes: &[(u8, u8)], n_samples: usize, seed: u64) -> Result<Self, String> {
        if dim < 2 {
            return Err(format!("dimensional APT dimension must be >= 2, got {dim}"));
        }
        if !dim.is_power_of_two() {
            return Err(format!(
                "dimensional APT dimension must be a power of two, got {dim}"
            ));
        }
        if dim > u16::MAX as usize {
            return Err(format!(
                "dimensional APT dimension {dim} exceeds packed node range"
            ));
        }
        if nodes.len() < 3 && n_samples != 0 {
            return Err(format!(
                "dimensional APT needs at least 3 nodes for sampling, got {}",
                nodes.len()
            ));
        }
        if nodes.len() > u32::MAX as usize {
            return Err(format!(
                "dimensional APT node count {} exceeds u32",
                nodes.len()
            ));
        }
        if n_samples > u32::MAX as usize {
            return Err(format!(
                "dimensional APT sample count {n_samples} exceeds u32 dispatch"
            ));
        }

        let packed_nodes = nodes
            .iter()
            .map(|&(a, b)| u32::from(a) | (u32::from(b) << 16))
            .collect();
        let (sample_i, sample_j, sample_k) = sample_triples(nodes.len(), n_samples, seed);
        let params = vec![dim as u32, n_samples as u32];

        Ok(Self {
            dim,
            n_nodes: nodes.len(),
            n_samples,
            nodes: packed_nodes,
            sample_i,
            sample_j,
            sample_k,
            params,
        })
    }

    fn dispatch_groups(&self) -> Result<u32, String> {
        let samples = u32::try_from(self.n_samples)
            .map_err(|_| "dimensional APT sample count exceeds u32 dispatch".to_string())?;
        Ok(samples.div_ceil(256))
    }

    fn result_from_counters(&self, counters: &[u32]) -> GpuAptResult {
        let pure_count = counters[0] as usize;
        let mixed_count = counters[1] as usize;
        GpuAptResult {
            dim: self.dim,
            n_nodes: self.n_nodes,
            n_samples: self.n_samples,
            pure_count,
            mixed_count,
            fiber_00: counters[2] as usize,
            fiber_01: counters[3] as usize,
            fiber_10: counters[4] as usize,
            fiber_11: counters[5] as usize,
            pure_ratio: pure_count as f64 / self.n_samples.max(1) as f64,
        }
    }
}

fn sample_triples(n_nodes: usize, n_samples: usize, seed: u64) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut rng_state = seed;
    let mut sample_i = Vec::with_capacity(n_samples);
    let mut sample_j = Vec::with_capacity(n_samples);
    let mut sample_k = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let i = (next_rng(&mut rng_state) as usize) % n_nodes;
        let mut j = (next_rng(&mut rng_state) as usize) % n_nodes;
        while j == i {
            j = (next_rng(&mut rng_state) as usize) % n_nodes;
        }
        let mut k = (next_rng(&mut rng_state) as usize) % n_nodes;
        while k == i || k == j {
            k = (next_rng(&mut rng_state) as usize) % n_nodes;
        }
        sample_i.push(i as u32);
        sample_j.push(j as u32);
        sample_k.push(k as u32);
    }

    (sample_i, sample_j, sample_k)
}

fn next_rng(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let z = *state ^ (*state >> 30);
    let z_mul = z.wrapping_mul(0xbf58476d1ce4e5b9);
    z_mul ^ (z_mul >> 27)
}

fn upload_u32_storage(
    device: &Device,
    adapter: &Adapter,
    values: &[u32],
    label: &str,
) -> Result<HostVisibleBuffer, String> {
    let byte_len = values
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| format!("dimensional {label} buffer size overflows"))?
        as u64;
    let buffer = HostVisibleBuffer::storage(device, adapter, byte_len)
        .map_err(|e| format!("dimensional {label} buffer allocation failed: {e}"))?;
    buffer
        .write_u32_slice(values)
        .map_err(|e| format!("dimensional {label} buffer upload failed: {e}"))?;
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensional_vulkan_wgsl_parses_and_emits_compute_spirv() {
        let module = naga::front::wgsl::parse_str(DimensionalVulkanKernel::wgsl_source()).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();

        let pipeline_options = naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: DIMENSIONAL_VULKAN_ENTRY_POINT.to_string(),
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
    fn dimensional_vulkan_prepares_cpu_samples() {
        let nodes = [(1u8, 8u8), (2, 9), (3, 10), (4, 11)];
        let prepared = PreparedDimensionalInput::new(16, &nodes, 8, 42).unwrap();
        assert_eq!(
            prepared.nodes,
            vec![0x0008_0001, 0x0009_0002, 0x000a_0003, 0x000b_0004]
        );
        assert_eq!(prepared.params, vec![16, 8]);
        assert_eq!(prepared.sample_i.len(), 8);
        assert_eq!(prepared.sample_j.len(), 8);
        assert_eq!(prepared.sample_k.len(), 8);
        for idx in 0..8 {
            assert_ne!(prepared.sample_i[idx], prepared.sample_j[idx]);
            assert_ne!(prepared.sample_i[idx], prepared.sample_k[idx]);
            assert_ne!(prepared.sample_j[idx], prepared.sample_k[idx]);
        }
    }

    #[test]
    fn dimensional_vulkan_rejects_invalid_input() {
        assert!(PreparedDimensionalInput::new(0, &[], 0, 42).is_err());
        assert!(PreparedDimensionalInput::new(3, &[], 0, 42).is_err());
        assert!(PreparedDimensionalInput::new(8, &[(1, 4), (2, 5)], 1, 42).is_err());
    }

    #[test]
    #[ignore = "requires local Vulkan compute device"]
    fn dimensional_vulkan_context_smoke() {
        let (_instance, adapter, _device) = DimensionalVulkanKernel::build_context().unwrap();
        eprintln!(
            "dimensional_vulkan_context_smoke: selected device {}",
            adapter.device_name()
        );
    }

    #[test]
    #[ignore = "requires local Vulkan compute device"]
    fn dimensional_vulkan_pipeline_smoke() {
        let (_instance, adapter, device) = DimensionalVulkanKernel::build_context().unwrap();
        eprintln!(
            "dimensional_vulkan_pipeline_smoke: selected device {}",
            adapter.device_name()
        );
        let _pipeline = DimensionalVulkanKernel::build_pipeline(&device).unwrap();
    }
}
