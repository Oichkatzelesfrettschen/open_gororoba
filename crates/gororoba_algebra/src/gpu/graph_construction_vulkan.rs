//! Vulkan edge compaction for Cayley-Dickson component graph construction.
//!
//! The CPU graph constructor checks each upper-triangle pair of cross-assessor
//! nodes against four eta-matrix entries. This module runs the same predicate
//! in parallel and compacts matching node-index pairs with an atomic counter.

use gororoba_gpu_vulkan::{
    Adapter, ComputePipeline, ComputePipelineBuilder, DescriptorPool, DescriptorSetLayout,
    DescriptorSetLayoutSpec, Device, DeviceBuilder, DispatchScope, HostVisibleBuffer, Instance,
    InstanceBuilder, QueueFamilyRequirement, ShaderModule, ValidationPolicy,
};

pub const GRAPH_CONSTRUCTION_VULKAN_ENTRY_POINT: &str = "compact_graph_edges";

pub const GRAPH_CONSTRUCTION_VULKAN_WGSL: &str = r#"
struct U32Array {
    values: array<u32>,
};

struct EdgeCounter {
    count: atomic<u32>,
};

@group(0) @binding(0)
var<storage, read> eta_matrix: U32Array;

@group(0) @binding(1)
var<storage, read> packed_nodes: U32Array;

@group(0) @binding(2)
var<storage, read_write> edge_i_out: U32Array;

@group(0) @binding(3)
var<storage, read_write> edge_j_out: U32Array;

@group(0) @binding(4)
var<storage, read_write> edge_counter: EdgeCounter;

@group(0) @binding(5)
var<storage, read> params: U32Array;

fn decode_upper_triangle(idx_input: u32, n_nodes: u32) -> vec2<u32> {
    var i: u32 = 0u;
    var remaining: u32 = idx_input;
    loop {
        let row_len: u32 = n_nodes - i - 1u;
        if (remaining < row_len) {
            break;
        }
        remaining = remaining - row_len;
        i = i + 1u;
    }
    return vec2<u32>(i, i + 1u + remaining);
}

@compute @workgroup_size(256)
fn compact_graph_edges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;
    let dim_half: u32 = params.values[0];
    let n_nodes: u32 = params.values[1];
    let tri_total: u32 = params.values[2];

    if (idx >= tri_total) {
        return;
    }

    let pair: vec2<u32> = decode_upper_triangle(idx, n_nodes);
    let node_a: u32 = packed_nodes.values[pair.x];
    let node_b: u32 = packed_nodes.values[pair.y];
    let ai: u32 = node_a & 0xffffu;
    let bi: u32 = (node_a >> 16u) & 0xffffu;
    let aj: u32 = node_b & 0xffffu;
    let bj: u32 = (node_b >> 16u) & 0xffffu;

    if (ai >= dim_half || bi >= dim_half || aj >= dim_half || bj >= dim_half) {
        return;
    }

    let eta_sum: u32 =
        eta_matrix.values[ai * dim_half + aj] +
        eta_matrix.values[bi * dim_half + bj] +
        eta_matrix.values[ai * dim_half + bj] +
        eta_matrix.values[bi * dim_half + aj];

    if (eta_sum == 2u || eta_sum == 4u) {
        let out_idx: u32 = atomicAdd(&edge_counter.count, 1u);
        if (out_idx < tri_total) {
            edge_i_out.values[out_idx] = pair.x;
            edge_j_out.values[out_idx] = pair.y;
        }
    }
}
"#;

pub struct GraphConstructionVulkanPipeline {
    pub pipeline: ComputePipeline,
    pub descriptor_layout: DescriptorSetLayout,
}

pub struct GraphConstructionVulkanKernel;

impl GraphConstructionVulkanKernel {
    pub fn wgsl_source() -> &'static str {
        GRAPH_CONSTRUCTION_VULKAN_WGSL
    }

    pub fn is_available() -> bool {
        match Self::build_context() {
            Ok((_instance, _adapter, _device)) => true,
            Err(_) => false,
        }
    }

    pub fn find_edges(
        dim: usize,
        eta_matrix: &[u8],
        nodes: &[(u8, u8)],
    ) -> Result<Vec<(usize, usize)>, String> {
        let prepared = PreparedGraphInput::new(dim, eta_matrix, nodes)?;
        if prepared.tri_total == 0 {
            return Ok(Vec::new());
        }

        let (_instance, adapter, device) = Self::build_context()?;
        let pipeline = Self::build_pipeline(&device)?;
        let eta_buffer = upload_u32_storage(&device, &adapter, &prepared.eta, "eta")?;
        let node_buffer = upload_u32_storage(&device, &adapter, &prepared.nodes, "nodes")?;
        let output_i_buffer =
            upload_u32_storage(&device, &adapter, &prepared.output_zeros, "edge_i")?;
        let output_j_buffer =
            upload_u32_storage(&device, &adapter, &prepared.output_zeros, "edge_j")?;
        let count_buffer = upload_u32_storage(&device, &adapter, &[0], "edge_count")?;
        let params_buffer = upload_u32_storage(&device, &adapter, &prepared.params, "params")?;

        let descriptor_pool =
            DescriptorPool::for_layout(&device, &pipeline.descriptor_layout, 1)
                .map_err(|e| format!("graph descriptor pool allocation failed: {e}"))?;
        let descriptor_set = descriptor_pool
            .allocate_set(&pipeline.descriptor_layout)
            .map_err(|e| format!("graph descriptor set allocation failed: {e}"))?;
        descriptor_set.write_storage_buffer(0, &eta_buffer);
        descriptor_set.write_storage_buffer(1, &node_buffer);
        descriptor_set.write_storage_buffer(2, &output_i_buffer);
        descriptor_set.write_storage_buffer(3, &output_j_buffer);
        descriptor_set.write_storage_buffer(4, &count_buffer);
        descriptor_set.write_storage_buffer(5, &params_buffer);

        let dispatch =
            DispatchScope::new(&device).map_err(|e| format!("graph dispatch scope failed: {e}"))?;
        dispatch
            .dispatch(
                &pipeline.pipeline,
                descriptor_set.raw(),
                prepared.dispatch_groups()?,
                1,
                1,
                5_000_000_000,
            )
            .map_err(|e| format!("graph Vulkan dispatch failed: {e}"))?;

        let count = count_buffer
            .read_u32_slice(1)
            .map_err(|e| format!("graph edge count readback failed: {e}"))?[0]
            as usize;
        if count > prepared.tri_total {
            return Err(format!(
                "graph Vulkan shader wrote edge count {count}, but capacity is {}",
                prepared.tri_total
            ));
        }
        let edge_i = output_i_buffer
            .read_u32_slice(prepared.tri_total)
            .map_err(|e| format!("graph edge_i readback failed: {e}"))?;
        let edge_j = output_j_buffer
            .read_u32_slice(prepared.tri_total)
            .map_err(|e| format!("graph edge_j readback failed: {e}"))?;

        let mut edges: Vec<(usize, usize)> = edge_i[..count]
            .iter()
            .zip(edge_j[..count].iter())
            .map(|(&i, &j)| (i as usize, j as usize))
            .collect();
        edges.sort_unstable();
        Ok(edges)
    }

    pub fn build_pipeline(device: &Device) -> Result<GraphConstructionVulkanPipeline, String> {
        let shader = ShaderModule::from_wgsl(
            device,
            GRAPH_CONSTRUCTION_VULKAN_WGSL,
            GRAPH_CONSTRUCTION_VULKAN_ENTRY_POINT,
        )
        .map_err(|e| format!("graph WGSL compile failed: {e}"))?;
        let descriptor_layout = DescriptorSetLayoutSpec::new()
            .storage_buffer(0)
            .storage_buffer(1)
            .storage_buffer(2)
            .storage_buffer(3)
            .storage_buffer(4)
            .storage_buffer(5)
            .build(device)
            .map_err(|e| format!("graph descriptor layout failed: {e}"))?;
        let pipeline = ComputePipelineBuilder::new(device, &shader)
            .descriptor_layout(&descriptor_layout)
            .build()
            .map_err(|e| format!("graph compute pipeline build failed: {e}"))?;

        Ok(GraphConstructionVulkanPipeline {
            pipeline,
            descriptor_layout,
        })
    }

    fn build_context() -> Result<(Instance, Adapter, Device), String> {
        let instance = InstanceBuilder::new("gororoba_algebra_graph_construction_vulkan")
            .validation(ValidationPolicy::Disable)
            .build()
            .map_err(|e| format!("graph Vulkan instance creation failed: {e}"))?;
        let adapter = Adapter::pick(&instance, QueueFamilyRequirement::Compute)
            .map_err(|e| format!("graph Vulkan adapter pick failed: {e}"))?;
        let device = DeviceBuilder::new(adapter.clone())
            .build(&instance)
            .map_err(|e| format!("graph Vulkan device creation failed: {e}"))?;
        Ok((instance, adapter, device))
    }
}

struct PreparedGraphInput {
    eta: Vec<u32>,
    nodes: Vec<u32>,
    output_zeros: Vec<u32>,
    params: Vec<u32>,
    tri_total: usize,
}

impl PreparedGraphInput {
    fn new(dim: usize, eta_matrix: &[u8], nodes: &[(u8, u8)]) -> Result<Self, String> {
        if dim < 2 {
            return Err(format!(
                "graph construction dimension must be >= 2, got {dim}"
            ));
        }
        if !dim.is_power_of_two() {
            return Err(format!(
                "graph construction dimension must be a power of two, got {dim}"
            ));
        }
        let dim_half = dim / 2;
        let expected_eta_len = dim_half
            .checked_mul(dim_half)
            .ok_or_else(|| format!("graph construction dimension {dim} overflows eta shape"))?;
        if eta_matrix.len() != expected_eta_len {
            return Err(format!(
                "graph construction eta length {} does not match expected {} for dim {dim}",
                eta_matrix.len(),
                expected_eta_len
            ));
        }
        if dim_half > u32::MAX as usize {
            return Err(format!(
                "graph construction dim_half {dim_half} exceeds u32"
            ));
        }
        if nodes.len() > u32::MAX as usize {
            return Err(format!(
                "graph construction node count {} exceeds u32",
                nodes.len()
            ));
        }
        let tri_total = nodes
            .len()
            .checked_mul(nodes.len().saturating_sub(1))
            .and_then(|value| value.checked_div(2))
            .ok_or_else(|| "graph construction triangular pair count overflows".to_string())?;
        if tri_total > u32::MAX as usize {
            return Err(format!(
                "graph construction pair count {tri_total} exceeds u32 dispatch"
            ));
        }

        let mut eta = Vec::with_capacity(eta_matrix.len());
        for &value in eta_matrix {
            match value {
                0 => eta.push(0),
                1 => eta.push(1),
                other => {
                    return Err(format!(
                        "graph construction eta value must be 0 or 1, got {other}"
                    ));
                }
            }
        }
        let packed_nodes = nodes
            .iter()
            .map(|&(a, b)| u32::from(a) | (u32::from(b) << 16))
            .collect();
        let output_zeros = vec![0u32; tri_total];
        let params = vec![dim_half as u32, nodes.len() as u32, tri_total as u32];

        Ok(Self {
            eta,
            nodes: packed_nodes,
            output_zeros,
            params,
            tri_total,
        })
    }

    fn dispatch_groups(&self) -> Result<u32, String> {
        let total = u32::try_from(self.tri_total)
            .map_err(|_| "graph construction pair count exceeds u32 dispatch".to_string())?;
        Ok(total.div_ceil(256))
    }
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
        .ok_or_else(|| format!("graph {label} buffer size overflows"))? as u64;
    let buffer = HostVisibleBuffer::storage(device, adapter, byte_len)
        .map_err(|e| format!("graph {label} buffer allocation failed: {e}"))?;
    buffer
        .write_u32_slice(values)
        .map_err(|e| format!("graph {label} buffer upload failed: {e}"))?;
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_vulkan_wgsl_parses_and_emits_compute_spirv() {
        let module =
            naga::front::wgsl::parse_str(GraphConstructionVulkanKernel::wgsl_source()).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();

        let pipeline_options = naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: GRAPH_CONSTRUCTION_VULKAN_ENTRY_POINT.to_string(),
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
    fn graph_vulkan_prepares_inputs() {
        let eta = vec![0u8; 16];
        let nodes = vec![(0u8, 1u8), (2, 3), (4, 5)];
        let prepared = PreparedGraphInput::new(8, &eta, &nodes).unwrap();
        assert_eq!(prepared.nodes, vec![0x0001_0000, 0x0003_0002, 0x0005_0004]);
        assert_eq!(prepared.params, vec![4, 3, 3]);
        assert_eq!(prepared.output_zeros, vec![0, 0, 0]);
    }

    #[test]
    fn graph_vulkan_rejects_invalid_input() {
        assert!(PreparedGraphInput::new(0, &[], &[]).is_err());
        assert!(PreparedGraphInput::new(3, &[0; 1], &[]).is_err());
        assert!(PreparedGraphInput::new(8, &[0; 15], &[]).is_err());
        assert!(PreparedGraphInput::new(8, &[2; 16], &[]).is_err());
    }

    #[test]
    #[ignore = "requires local Vulkan compute device"]
    fn graph_vulkan_context_smoke() {
        let (_instance, adapter, _device) = GraphConstructionVulkanKernel::build_context().unwrap();
        eprintln!(
            "graph_vulkan_context_smoke: selected device {}",
            adapter.device_name()
        );
    }

    #[test]
    #[ignore = "requires local Vulkan compute device"]
    fn graph_vulkan_pipeline_smoke() {
        let (_instance, adapter, device) = GraphConstructionVulkanKernel::build_context().unwrap();
        eprintln!(
            "graph_vulkan_pipeline_smoke: selected device {}",
            adapter.device_name()
        );
        let _pipeline = GraphConstructionVulkanKernel::build_pipeline(&device).unwrap();
    }
}
