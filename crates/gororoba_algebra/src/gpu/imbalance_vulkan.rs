//! Vulkan edge-validation launcher for imbalance ratio computation.
//!
//! The imbalance algorithm has a serial graph-labeling step and a parallel
//! edge-validation step. This module keeps BFS delta assignment on the CPU and
//! uses Vulkan to count edges whose eta label disagrees with the assigned
//! coboundary label.

use std::collections::VecDeque;

use gororoba_gpu_vulkan::{
    Adapter, ComputePipeline, ComputePipelineBuilder, DescriptorPool, DescriptorSetLayout,
    DescriptorSetLayoutSpec, Device, DeviceBuilder, DispatchScope, HostVisibleBuffer, Instance,
    InstanceBuilder, QueueFamilyRequirement, ShaderModule, ValidationPolicy,
};

use super::ImbalanceResult;

pub const IMBALANCE_VULKAN_ENTRY_POINT: &str = "validate_imbalance_edges";

pub const IMBALANCE_VULKAN_WGSL: &str = r#"
struct U32Array {
    values: array<u32>,
};

struct FrustratedCounter {
    count: atomic<u32>,
};

@group(0) @binding(0)
var<storage, read> edge_u: U32Array;

@group(0) @binding(1)
var<storage, read> edge_v: U32Array;

@group(0) @binding(2)
var<storage, read> eta_values: U32Array;

@group(0) @binding(3)
var<storage, read> delta_values: U32Array;

@group(0) @binding(4)
var<storage, read_write> frustrated: FrustratedCounter;

@compute @workgroup_size(256)
fn validate_imbalance_edges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;
    if (idx >= arrayLength(&eta_values.values)) {
        return;
    }

    let u: u32 = edge_u.values[idx];
    let v: u32 = edge_v.values[idx];
    let computed: u32 = delta_values.values[u] ^ delta_values.values[v];
    if (computed != eta_values.values[idx]) {
        atomicAdd(&frustrated.count, 1u);
    }
}
"#;

pub struct ImbalanceVulkanPipeline {
    pub pipeline: ComputePipeline,
    pub descriptor_layout: DescriptorSetLayout,
}

pub struct ImbalanceVulkanKernel;

impl ImbalanceVulkanKernel {
    pub fn wgsl_source() -> &'static str {
        IMBALANCE_VULKAN_WGSL
    }

    pub fn is_available() -> bool {
        match Self::build_context() {
            Ok((_instance, _adapter, _device)) => true,
            Err(_) => false,
        }
    }

    pub fn compute(
        edges: &[(usize, usize)],
        n_nodes: usize,
        eta_values: &[u8],
    ) -> Result<ImbalanceResult, String> {
        let prepared = PreparedImbalanceInput::new(edges, n_nodes, eta_values)?;
        if prepared.edge_u.is_empty() {
            return Ok(prepared.empty_result());
        }

        let (_instance, adapter, device) = Self::build_context()?;
        let pipeline = Self::build_pipeline(&device)?;
        let edge_u_buffer = upload_u32_storage(&device, &adapter, &prepared.edge_u, "edge_u")?;
        let edge_v_buffer = upload_u32_storage(&device, &adapter, &prepared.edge_v, "edge_v")?;
        let eta_buffer = upload_u32_storage(&device, &adapter, &prepared.eta, "eta")?;
        let delta_buffer = upload_u32_storage(&device, &adapter, &prepared.delta, "delta")?;
        let output_buffer = upload_u32_storage(&device, &adapter, &[0], "frustrated_count")?;

        let descriptor_pool =
            DescriptorPool::for_layout(&device, &pipeline.descriptor_layout, 1)
                .map_err(|e| format!("imbalance descriptor pool allocation failed: {e}"))?;
        let descriptor_set = descriptor_pool
            .allocate_set(&pipeline.descriptor_layout)
            .map_err(|e| format!("imbalance descriptor set allocation failed: {e}"))?;
        descriptor_set.write_storage_buffer(0, &edge_u_buffer);
        descriptor_set.write_storage_buffer(1, &edge_v_buffer);
        descriptor_set.write_storage_buffer(2, &eta_buffer);
        descriptor_set.write_storage_buffer(3, &delta_buffer);
        descriptor_set.write_storage_buffer(4, &output_buffer);

        let dispatch = DispatchScope::new(&device)
            .map_err(|e| format!("imbalance dispatch scope failed: {e}"))?;
        let groups = prepared.dispatch_groups()?;
        dispatch
            .dispatch(
                &pipeline.pipeline,
                descriptor_set.raw(),
                groups,
                1,
                1,
                5_000_000_000,
            )
            .map_err(|e| format!("imbalance Vulkan dispatch failed: {e}"))?;

        let frustrated = output_buffer
            .read_u32_slice(1)
            .map_err(|e| format!("imbalance output readback failed: {e}"))?[0]
            as usize;
        Ok(prepared.result(frustrated))
    }

    pub fn build_pipeline(device: &Device) -> Result<ImbalanceVulkanPipeline, String> {
        let shader =
            ShaderModule::from_wgsl(device, IMBALANCE_VULKAN_WGSL, IMBALANCE_VULKAN_ENTRY_POINT)
                .map_err(|e| format!("imbalance WGSL compile failed: {e}"))?;
        let descriptor_layout = DescriptorSetLayoutSpec::new()
            .storage_buffer(0)
            .storage_buffer(1)
            .storage_buffer(2)
            .storage_buffer(3)
            .storage_buffer(4)
            .build(device)
            .map_err(|e| format!("imbalance descriptor layout failed: {e}"))?;
        let pipeline = ComputePipelineBuilder::new(device, &shader)
            .descriptor_layout(&descriptor_layout)
            .build()
            .map_err(|e| format!("imbalance compute pipeline build failed: {e}"))?;

        Ok(ImbalanceVulkanPipeline {
            pipeline,
            descriptor_layout,
        })
    }

    fn build_context() -> Result<(Instance, Adapter, Device), String> {
        let instance = InstanceBuilder::new("gororoba_algebra_imbalance_vulkan")
            .validation(ValidationPolicy::Disable)
            .build()
            .map_err(|e| format!("imbalance Vulkan instance creation failed: {e}"))?;
        let adapter = Adapter::pick(&instance, QueueFamilyRequirement::Compute)
            .map_err(|e| format!("imbalance Vulkan adapter pick failed: {e}"))?;
        let device = DeviceBuilder::new(adapter.clone())
            .build(&instance)
            .map_err(|e| format!("imbalance Vulkan device creation failed: {e}"))?;
        Ok((instance, adapter, device))
    }
}

struct PreparedImbalanceInput {
    edge_u: Vec<u32>,
    edge_v: Vec<u32>,
    eta: Vec<u32>,
    delta: Vec<u32>,
    total_eta0: usize,
    total_eta1: usize,
    cycle_rank: usize,
}

impl PreparedImbalanceInput {
    fn new(edges: &[(usize, usize)], n_nodes: usize, eta_values: &[u8]) -> Result<Self, String> {
        if n_nodes == 0 {
            return Err("imbalance graph must have at least one node".to_string());
        }
        if edges.len() != eta_values.len() {
            return Err(format!(
                "imbalance edge count {} does not match eta count {}",
                edges.len(),
                eta_values.len()
            ));
        }
        if edges.len() > u32::MAX as usize {
            return Err(format!(
                "imbalance edge count {} exceeds u32 dispatch",
                edges.len()
            ));
        }
        if n_nodes > u32::MAX as usize {
            return Err(format!(
                "imbalance node count {n_nodes} exceeds u32 buffers"
            ));
        }

        let mut edge_u = Vec::with_capacity(edges.len());
        let mut edge_v = Vec::with_capacity(edges.len());
        for &(u, v) in edges {
            if u >= n_nodes || v >= n_nodes {
                return Err(format!(
                    "imbalance edge ({u}, {v}) exceeds node count {n_nodes}"
                ));
            }
            edge_u.push(u as u32);
            edge_v.push(v as u32);
        }

        let mut eta = Vec::with_capacity(eta_values.len());
        let mut total_eta0 = 0usize;
        let mut total_eta1 = 0usize;
        for &value in eta_values {
            match value {
                0 => {
                    eta.push(0);
                    total_eta0 += 1;
                }
                1 => {
                    eta.push(1);
                    total_eta1 += 1;
                }
                other => {
                    return Err(format!("imbalance eta value must be 0 or 1, got {other}"));
                }
            }
        }

        let delta = assign_delta(edges, n_nodes, eta_values);
        let cycle_rank = edges.len().saturating_sub(n_nodes - 1);

        Ok(Self {
            edge_u,
            edge_v,
            eta,
            delta,
            total_eta0,
            total_eta1,
            cycle_rank,
        })
    }

    fn dispatch_groups(&self) -> Result<u32, String> {
        let n_edges = u32::try_from(self.edge_u.len())
            .map_err(|_| "imbalance edge count exceeds u32 dispatch".to_string())?;
        Ok(n_edges.div_ceil(256))
    }

    fn empty_result(&self) -> ImbalanceResult {
        self.result(0)
    }

    fn result(&self, frustrated_count: usize) -> ImbalanceResult {
        ImbalanceResult {
            total_edges: self.edge_u.len(),
            total_eta0: self.total_eta0,
            total_eta1: self.total_eta1,
            cycle_rank: self.cycle_rank,
            frustrated_count,
            imbalance_ratio: frustrated_count as f64 / self.cycle_rank.max(1) as f64,
        }
    }
}

fn assign_delta(edges: &[(usize, usize)], n_nodes: usize, eta_values: &[u8]) -> Vec<u32> {
    let mut delta = vec![0u32; n_nodes];
    let mut visited = vec![false; n_nodes];
    let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n_nodes];

    for (idx, &(u, v)) in edges.iter().enumerate() {
        let eta = eta_values[idx];
        adj[u].push((v, eta));
        adj[v].push((u, eta));
    }

    visited[0] = true;
    let mut queue = VecDeque::new();
    queue.push_back(0usize);

    while let Some(u) = queue.pop_front() {
        for &(v, eta) in &adj[u] {
            if !visited[v] {
                visited[v] = true;
                delta[v] = delta[u] ^ u32::from(eta);
                queue.push_back(v);
            }
        }
    }

    delta
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
        .ok_or_else(|| format!("imbalance {label} buffer size overflows"))?
        as u64;
    let buffer = HostVisibleBuffer::storage(device, adapter, byte_len)
        .map_err(|e| format!("imbalance {label} buffer allocation failed: {e}"))?;
    buffer
        .write_u32_slice(values)
        .map_err(|e| format!("imbalance {label} buffer upload failed: {e}"))?;
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imbalance_vulkan_wgsl_parses_and_emits_compute_spirv() {
        let module = naga::front::wgsl::parse_str(ImbalanceVulkanKernel::wgsl_source()).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();

        let pipeline_options = naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: IMBALANCE_VULKAN_ENTRY_POINT.to_string(),
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
    fn imbalance_vulkan_prepares_triangle_delta() {
        let edges = [(0usize, 1usize), (1, 2), (2, 0)];
        let eta = [0u8, 0, 1];
        let prepared = PreparedImbalanceInput::new(&edges, 3, &eta).unwrap();
        assert_eq!(prepared.edge_u, vec![0, 1, 2]);
        assert_eq!(prepared.edge_v, vec![1, 2, 0]);
        assert_eq!(prepared.eta, vec![0, 0, 1]);
        assert_eq!(prepared.delta, vec![0, 0, 1]);
        assert_eq!(prepared.cycle_rank, 1);
    }

    #[test]
    fn imbalance_vulkan_rejects_invalid_input() {
        assert!(PreparedImbalanceInput::new(&[(0, 1)], 0, &[0]).is_err());
        assert!(PreparedImbalanceInput::new(&[(0, 1)], 2, &[]).is_err());
        assert!(PreparedImbalanceInput::new(&[(0, 2)], 2, &[0]).is_err());
        assert!(PreparedImbalanceInput::new(&[(0, 1)], 2, &[2]).is_err());
    }

    #[test]
    #[ignore = "requires local Vulkan compute device"]
    fn imbalance_vulkan_context_smoke() {
        let (_instance, adapter, _device) = ImbalanceVulkanKernel::build_context().unwrap();
        eprintln!(
            "imbalance_vulkan_context_smoke: selected device {}",
            adapter.device_name()
        );
    }

    #[test]
    #[ignore = "requires local Vulkan compute device"]
    fn imbalance_vulkan_pipeline_smoke() {
        let (_instance, adapter, device) = ImbalanceVulkanKernel::build_context().unwrap();
        eprintln!(
            "imbalance_vulkan_pipeline_smoke: selected device {}",
            adapter.device_name()
        );
        let _pipeline = ImbalanceVulkanKernel::build_pipeline(&device).unwrap();
    }
}
