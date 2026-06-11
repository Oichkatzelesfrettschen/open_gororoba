//! Vulkan minimax-path distances for ultrametric graph analysis.
//!
//! This is the Vulkan sibling of `ultrametric_cubecl`: each source-row thread
//! computes bottleneck distances over a dense adjacency matrix. The host keeps
//! the same CPU reference so the backend contract is exact and deterministic.

#![cfg(feature = "vulkan")]

use gororoba_gpu_vulkan::{
    Adapter, ComputePipeline, ComputePipelineBuilder, DescriptorPool, DescriptorSetLayout,
    DescriptorSetLayoutSpec, Device, DeviceBuilder, DispatchScope, HostVisibleBuffer, Instance,
    InstanceBuilder, QueueFamilyRequirement, ShaderModule, ValidationPolicy,
};

pub const INF_DISTANCE: f32 = 1.0e30;
pub const ULTRAMETRIC_VULKAN_ENTRY_POINT: &str = "minimax_rows";
const WORKGROUP_SIZE: u32 = 256;
const DISPATCH_TIMEOUT_NS: u64 = 10_000_000_000;

pub const ULTRAMETRIC_VULKAN_WGSL: &str = r#"
struct F32Buffer {
    values: array<f32>,
};

struct U32Buffer {
    values: array<u32>,
};

struct Params {
    n_nodes: u32,
    matrix_len: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0)
var<storage, read> adjacency: F32Buffer;
@group(0) @binding(1)
var<storage, read_write> distances: F32Buffer;
@group(0) @binding(2)
var<storage, read_write> visited: U32Buffer;
@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn minimax_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let source: u32 = gid.x;
    if (source >= params.n_nodes) {
        return;
    }

    let row_base: u32 = source * params.n_nodes;
    for (var dest: u32 = 0u; dest < params.n_nodes; dest = dest + 1u) {
        distances.values[row_base + dest] = adjacency.values[row_base + dest];
        visited.values[row_base + dest] = 0u;
    }
    distances.values[row_base + source] = 0.0;

    for (var step: u32 = 0u; step < params.n_nodes; step = step + 1u) {
        var best_node: u32 = 0u;
        var best_distance: f32 = 1.0e30;
        var has_best: bool = false;

        for (var candidate: u32 = 0u; candidate < params.n_nodes; candidate = candidate + 1u) {
            let candidate_distance: f32 = distances.values[row_base + candidate];
            if (visited.values[row_base + candidate] == 0u
                && (!has_best || candidate_distance < best_distance)) {
                best_distance = candidate_distance;
                best_node = candidate;
                has_best = true;
            }
        }

        if (has_best) {
            visited.values[row_base + best_node] = 1u;
            for (var neighbor: u32 = 0u; neighbor < params.n_nodes; neighbor = neighbor + 1u) {
                let edge_weight: f32 = adjacency.values[best_node * params.n_nodes + neighbor];
                var path_weight: f32 = best_distance;
                if (edge_weight > path_weight) {
                    path_weight = edge_weight;
                }
                if (path_weight < distances.values[row_base + neighbor]) {
                    distances.values[row_base + neighbor] = path_weight;
                }
            }
        }
    }
}
"#;

pub struct UltrametricVulkanPipeline {
    pipeline: ComputePipeline,
    descriptor_layout: DescriptorSetLayout,
}

pub struct UltrametricVulkanKernel;

impl UltrametricVulkanKernel {
    pub fn is_available() -> bool {
        ultrametric_vulkan_available()
    }

    pub fn minimax_distance_matrix(adjacency: &[f32], n_nodes: usize) -> Result<Vec<f32>, String> {
        let prepared = PreparedMinimaxInput::new(adjacency, n_nodes)?;
        if prepared.n_nodes == 0 {
            return Ok(Vec::new());
        }

        let (_instance, adapter, device) = build_vulkan_context()?;
        let pipeline = build_vulkan_pipeline(&device)?;
        let run = UltrametricVulkanRun::new(&device, &adapter, &pipeline, &prepared)?;
        let dispatch = DispatchScope::new(&device)
            .map_err(|e| format!("ultrametric Vulkan dispatch scope failed: {e}"))?;
        let groups = prepared.n_nodes.div_ceil(WORKGROUP_SIZE);
        dispatch
            .dispatch(
                &pipeline.pipeline,
                run.descriptor_set.raw(),
                groups,
                1,
                1,
                DISPATCH_TIMEOUT_NS,
            )
            .map_err(|e| format!("ultrametric Vulkan dispatch failed: {e}"))?;

        run.distances_buffer
            .read_f32_slice(prepared.matrix_len as usize)
            .map_err(|e| format!("ultrametric Vulkan minimax readback failed: {e}"))
    }
}

pub fn ultrametric_vulkan_available() -> bool {
    match build_vulkan_context() {
        Ok((_instance, _adapter, _device)) => true,
        Err(_) => false,
    }
}

pub fn minimax_distance_matrix_cpu(adjacency: &[f32], n_nodes: usize) -> Result<Vec<f32>, String> {
    let prepared = PreparedMinimaxInput::new(adjacency, n_nodes)?;
    let n = prepared.n_nodes as usize;
    let mut output = vec![INF_DISTANCE; prepared.matrix_len as usize];

    for source in 0..n {
        let row_base = source * n;
        let mut distances = prepared.adjacency[row_base..row_base + n].to_vec();
        let mut visited = vec![false; n];
        distances[source] = 0.0;

        for _ in 0..n {
            let mut best_node = None;
            let mut best_distance = INF_DISTANCE;
            for candidate in 0..n {
                if !visited[candidate] && distances[candidate] < best_distance {
                    best_distance = distances[candidate];
                    best_node = Some(candidate);
                }
            }

            let Some(node) = best_node else {
                break;
            };
            visited[node] = true;

            for (neighbor, distance) in distances.iter_mut().enumerate().take(n) {
                let path_weight = best_distance.max(prepared.adjacency[node * n + neighbor]);
                if path_weight < *distance {
                    *distance = path_weight;
                }
            }
        }

        output[row_base..row_base + n].copy_from_slice(&distances);
    }

    Ok(output)
}

struct UltrametricVulkanRun {
    #[allow(dead_code)]
    descriptor_pool: DescriptorPool,
    descriptor_set: gororoba_gpu_vulkan::DescriptorSet,
    #[allow(dead_code)]
    adjacency_buffer: HostVisibleBuffer,
    distances_buffer: HostVisibleBuffer,
    #[allow(dead_code)]
    visited_buffer: HostVisibleBuffer,
    #[allow(dead_code)]
    params_buffer: HostVisibleBuffer,
}

impl UltrametricVulkanRun {
    fn new(
        device: &Device,
        adapter: &Adapter,
        pipeline: &UltrametricVulkanPipeline,
        prepared: &PreparedMinimaxInput<'_>,
    ) -> Result<Self, String> {
        let descriptor_pool = DescriptorPool::for_layout(device, &pipeline.descriptor_layout, 1)
            .map_err(|e| format!("ultrametric Vulkan descriptor pool failed: {e}"))?;
        let descriptor_set = descriptor_pool
            .allocate_set(&pipeline.descriptor_layout)
            .map_err(|e| format!("ultrametric Vulkan descriptor set failed: {e}"))?;

        let adjacency_buffer =
            upload_storage_f32(device, adapter, prepared.adjacency, "adjacency")?;
        let distances_buffer = upload_storage_f32(
            device,
            adapter,
            &vec![INF_DISTANCE; prepared.matrix_len as usize],
            "distances",
        )?;
        let visited_buffer = upload_storage_u32(
            device,
            adapter,
            &vec![0u32; prepared.matrix_len as usize],
            "visited",
        )?;
        let params_buffer = HostVisibleBuffer::uniform(device, adapter, 16)
            .map_err(|e| format!("ultrametric Vulkan params buffer allocation failed: {e}"))?;
        params_buffer
            .write_bytes(&encode_params(prepared)?)
            .map_err(|e| format!("ultrametric Vulkan params upload failed: {e}"))?;

        descriptor_set.write_storage_buffer(0, &adjacency_buffer);
        descriptor_set.write_storage_buffer(1, &distances_buffer);
        descriptor_set.write_storage_buffer(2, &visited_buffer);
        descriptor_set.write_uniform_buffer(3, &params_buffer);

        Ok(Self {
            descriptor_pool,
            descriptor_set,
            adjacency_buffer,
            distances_buffer,
            visited_buffer,
            params_buffer,
        })
    }
}

struct PreparedMinimaxInput<'a> {
    adjacency: &'a [f32],
    n_nodes: u32,
    matrix_len: u32,
}

impl<'a> PreparedMinimaxInput<'a> {
    fn new(adjacency: &'a [f32], n_nodes: usize) -> Result<Self, String> {
        if n_nodes > u32::MAX as usize {
            return Err(format!(
                "ultrametric minimax node count {n_nodes} exceeds u32"
            ));
        }
        let matrix_len = n_nodes
            .checked_mul(n_nodes)
            .ok_or_else(|| format!("ultrametric minimax node count {n_nodes} overflows matrix"))?;
        if adjacency.len() != matrix_len {
            return Err(format!(
                "ultrametric minimax adjacency length {} does not match {matrix_len}",
                adjacency.len()
            ));
        }
        if matrix_len > u32::MAX as usize {
            return Err(format!(
                "ultrametric minimax matrix length {matrix_len} exceeds u32 dispatch"
            ));
        }
        for (index, &weight) in adjacency.iter().enumerate() {
            if weight.is_nan() || weight.is_sign_negative() {
                return Err(format!(
                    "ultrametric minimax adjacency weight at {index} must be nonnegative and not NaN"
                ));
            }
            if weight > INF_DISTANCE {
                return Err(format!(
                    "ultrametric minimax adjacency weight at {index} exceeds INF_DISTANCE"
                ));
            }
        }

        Ok(Self {
            adjacency,
            n_nodes: n_nodes as u32,
            matrix_len: matrix_len as u32,
        })
    }
}

fn build_vulkan_context() -> Result<(Instance, Adapter, Device), String> {
    let instance = InstanceBuilder::new("stats_core_ultrametric_vulkan")
        .validation(ValidationPolicy::Disable)
        .build()
        .map_err(|e| format!("ultrametric Vulkan instance failed: {e}"))?;
    let adapter = Adapter::pick(&instance, QueueFamilyRequirement::Compute)
        .map_err(|e| format!("ultrametric Vulkan adapter failed: {e}"))?;
    let device = DeviceBuilder::new(adapter.clone())
        .build(&instance)
        .map_err(|e| format!("ultrametric Vulkan device failed: {e}"))?;
    Ok((instance, adapter, device))
}

fn build_vulkan_pipeline(device: &Device) -> Result<UltrametricVulkanPipeline, String> {
    let shader = ShaderModule::from_wgsl(
        device,
        ULTRAMETRIC_VULKAN_WGSL,
        ULTRAMETRIC_VULKAN_ENTRY_POINT,
    )
    .map_err(|e| format!("ultrametric Vulkan WGSL compile failed: {e}"))?;
    let descriptor_layout = DescriptorSetLayoutSpec::new()
        .storage_buffer(0)
        .storage_buffer(1)
        .storage_buffer(2)
        .uniform_buffer(3)
        .build(device)
        .map_err(|e| format!("ultrametric Vulkan descriptor layout failed: {e}"))?;
    let pipeline = ComputePipelineBuilder::new(device, &shader)
        .descriptor_layout(&descriptor_layout)
        .build()
        .map_err(|e| format!("ultrametric Vulkan pipeline build failed: {e}"))?;

    Ok(UltrametricVulkanPipeline {
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
        .map_err(|e| format!("ultrametric Vulkan {label} buffer allocation failed: {e}"))?;
    buffer
        .write_f32_slice(values)
        .map_err(|e| format!("ultrametric Vulkan {label} buffer upload failed: {e}"))?;
    Ok(buffer)
}

fn upload_storage_u32(
    device: &Device,
    adapter: &Adapter,
    values: &[u32],
    label: &str,
) -> Result<HostVisibleBuffer, String> {
    let buffer = HostVisibleBuffer::storage(device, adapter, byte_len::<u32>(values.len(), label)?)
        .map_err(|e| format!("ultrametric Vulkan {label} buffer allocation failed: {e}"))?;
    buffer
        .write_u32_slice(values)
        .map_err(|e| format!("ultrametric Vulkan {label} buffer upload failed: {e}"))?;
    Ok(buffer)
}

fn byte_len<T>(len: usize, label: &str) -> Result<u64, String> {
    len.checked_mul(std::mem::size_of::<T>())
        .map(|bytes| bytes as u64)
        .ok_or_else(|| format!("ultrametric Vulkan {label} buffer size overflows"))
}

fn encode_params(prepared: &PreparedMinimaxInput<'_>) -> Result<[u8; 16], String> {
    let words = [
        prepared.n_nodes.to_le_bytes(),
        prepared.matrix_len.to_le_bytes(),
        0u32.to_le_bytes(),
        0u32.to_le_bytes(),
    ];
    let mut bytes = [0u8; 16];
    for (idx, word) in words.iter().enumerate() {
        let start = 4 * idx;
        bytes[start..start + 4].copy_from_slice(word);
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ultrametric_vulkan_available_does_not_panic() {
        let _ = UltrametricVulkanKernel::is_available();
    }

    #[test]
    fn minimax_cpu_prefers_lower_bottleneck_path() {
        let adjacency = vec![0.0, 9.0, 2.0, 9.0, 0.0, 3.0, 2.0, 3.0, 0.0];
        let distances = minimax_distance_matrix_cpu(&adjacency, 3).unwrap();
        assert_eq!(distances[1], 3.0);
        assert_eq!(distances[3], 3.0);
    }

    #[test]
    fn minimax_rejects_invalid_input() {
        assert!(PreparedMinimaxInput::new(&[0.0, 1.0], 2).is_err());
        assert!(PreparedMinimaxInput::new(&[f32::NAN], 1).is_err());
        assert!(PreparedMinimaxInput::new(&[-1.0], 1).is_err());
    }
}
