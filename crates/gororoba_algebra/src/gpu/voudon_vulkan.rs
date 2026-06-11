//! Vulkan launcher for 256D Voudon frustration-field generation.
//!
//! The CUDA Voudon kernel estimates a per-cell frustration value from 32
//! spatially hashed basis-pair samples. This module emits the same integer
//! violation count per cell on Vulkan, then converts counts to the public
//! `f32` field values on the host.

use gororoba_gpu_vulkan::{
    Adapter, ComputePipeline, ComputePipelineBuilder, DescriptorPool, DescriptorSetLayout,
    DescriptorSetLayoutSpec, Device, DeviceBuilder, DispatchScope, HostVisibleBuffer, Instance,
    InstanceBuilder, QueueFamilyRequirement, ShaderModule, ValidationPolicy,
};

use super::voudon::Cd256FrustrationKernel;

pub const VOUDON_VULKAN_ENTRY_POINT: &str = "compute_voudon_frustration";

pub const VOUDON_VULKAN_WGSL: &str = r#"
struct U32Array {
    values: array<u32>,
};

@group(0) @binding(0)
var<storage, read_write> frustration_counts: U32Array;

@group(0) @binding(1)
var<storage, read> params: U32Array;

fn cd_basis_mul_sign_256(p_input: u32, q_input: u32) -> i32 {
    var sign: i32 = 1;
    var p: u32 = p_input;
    var q: u32 = q_input;
    var half: u32 = 128u;

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

fn spatial_hash(x: u32, y: u32, z: u32, seed: u32) -> u32 {
    var h: u32 = seed ^ (x * 73856093u) ^ (y * 19349663u) ^ (z * 83492791u);
    h = (h >> 13u) ^ h;
    h = h * 0x5bd1e995u;
    h = (h >> 15u) ^ h;
    return h;
}

fn spatial_index_256(x: u32, y: u32, z: u32, seed: u32) -> u32 {
    return ((spatial_hash(x, y, z, seed) & 0xffffu) * 255u) / 65535u;
}

@compute @workgroup_size(256)
fn compute_voudon_frustration(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;
    let nx: u32 = params.values[0];
    let ny: u32 = params.values[1];
    let nz: u32 = params.values[2];
    let seed: u32 = params.values[3];
    let total: u32 = nx * ny * nz;

    if (idx >= total) {
        return;
    }

    let x: u32 = idx % nx;
    let y: u32 = (idx / nx) % ny;
    let z: u32 = idx / (nx * ny);
    var local_frustration: u32 = 0u;

    for (var sample: u32 = 0u; sample < 32u; sample = sample + 1u) {
        let i: u32 = spatial_index_256(x, y, z, seed ^ sample);
        let j: u32 = spatial_index_256(x, y, z, seed ^ (sample + 100u));

        let s1: i32 = cd_basis_mul_sign_256(i, i);
        let ij_idx: u32 = i ^ j;
        let s2: i32 = cd_basis_mul_sign_256(i, j);
        let i_ij_sign: i32 = cd_basis_mul_sign_256(i, ij_idx) * s2;

        if (s1 != i_ij_sign) {
            local_frustration = local_frustration + 1u;
        }
    }

    frustration_counts.values[idx] = local_frustration;
}
"#;

pub struct VoudonVulkanPipeline {
    pub pipeline: ComputePipeline,
    pub descriptor_layout: DescriptorSetLayout,
}

pub struct VoudonVulkanKernel;

impl VoudonVulkanKernel {
    pub fn wgsl_source() -> &'static str {
        VOUDON_VULKAN_WGSL
    }

    pub fn is_available() -> bool {
        match Self::build_context() {
            Ok((_instance, _adapter, _device)) => true,
            Err(_) => false,
        }
    }

    pub fn compute_field(nx: usize, ny: usize, nz: usize, seed: u32) -> Result<Vec<f32>, String> {
        let prepared = PreparedVoudonField::new(nx, ny, nz, seed)?;
        if prepared.n_cells == 0 {
            return Ok(Vec::new());
        }

        let (_instance, adapter, device) = Self::build_context()?;
        let pipeline = Self::build_pipeline(&device)?;
        let output_buffer = upload_u32_storage(
            &device,
            &adapter,
            &prepared.output_zeros,
            "frustration_counts",
        )?;
        let params_buffer = upload_u32_storage(&device, &adapter, &prepared.params, "params")?;

        let descriptor_pool =
            DescriptorPool::for_layout(&device, &pipeline.descriptor_layout, 1)
                .map_err(|e| format!("Voudon descriptor pool allocation failed: {e}"))?;
        let descriptor_set = descriptor_pool
            .allocate_set(&pipeline.descriptor_layout)
            .map_err(|e| format!("Voudon descriptor set allocation failed: {e}"))?;
        descriptor_set.write_storage_buffer(0, &output_buffer);
        descriptor_set.write_storage_buffer(1, &params_buffer);

        let dispatch = DispatchScope::new(&device)
            .map_err(|e| format!("Voudon dispatch scope failed: {e}"))?;
        dispatch
            .dispatch(
                &pipeline.pipeline,
                descriptor_set.raw(),
                prepared.dispatch_groups()?,
                1,
                1,
                5_000_000_000,
            )
            .map_err(|e| format!("Voudon Vulkan dispatch failed: {e}"))?;

        let counts = output_buffer
            .read_u32_slice(prepared.n_cells)
            .map_err(|e| format!("Voudon output readback failed: {e}"))?;
        counts
            .into_iter()
            .map(|count| {
                if count > Cd256FrustrationKernel::SAMPLES_PER_CELL as u32 {
                    Err(format!(
                        "Voudon Vulkan count {count} exceeds sample count {}",
                        Cd256FrustrationKernel::SAMPLES_PER_CELL
                    ))
                } else {
                    Ok(count as f32 / Cd256FrustrationKernel::SAMPLES_PER_CELL as f32)
                }
            })
            .collect()
    }

    pub fn build_pipeline(device: &Device) -> Result<VoudonVulkanPipeline, String> {
        let shader = ShaderModule::from_wgsl(device, VOUDON_VULKAN_WGSL, VOUDON_VULKAN_ENTRY_POINT)
            .map_err(|e| format!("Voudon WGSL compile failed: {e}"))?;
        let descriptor_layout = DescriptorSetLayoutSpec::new()
            .storage_buffer(0)
            .storage_buffer(1)
            .build(device)
            .map_err(|e| format!("Voudon descriptor layout failed: {e}"))?;
        let pipeline = ComputePipelineBuilder::new(device, &shader)
            .descriptor_layout(&descriptor_layout)
            .build()
            .map_err(|e| format!("Voudon compute pipeline build failed: {e}"))?;

        Ok(VoudonVulkanPipeline {
            pipeline,
            descriptor_layout,
        })
    }

    fn build_context() -> Result<(Instance, Adapter, Device), String> {
        let instance = InstanceBuilder::new("gororoba_algebra_voudon_vulkan")
            .validation(ValidationPolicy::Disable)
            .build()
            .map_err(|e| format!("Voudon Vulkan instance creation failed: {e}"))?;
        let adapter = Adapter::pick(&instance, QueueFamilyRequirement::Compute)
            .map_err(|e| format!("Voudon Vulkan adapter pick failed: {e}"))?;
        let device = DeviceBuilder::new(adapter.clone())
            .build(&instance)
            .map_err(|e| format!("Voudon Vulkan device creation failed: {e}"))?;
        Ok((instance, adapter, device))
    }
}

struct PreparedVoudonField {
    n_cells: usize,
    output_zeros: Vec<u32>,
    params: Vec<u32>,
}

impl PreparedVoudonField {
    fn new(nx: usize, ny: usize, nz: usize, seed: u32) -> Result<Self, String> {
        if nx > u32::MAX as usize || ny > u32::MAX as usize || nz > u32::MAX as usize {
            return Err(format!("Voudon field shape {nx}x{ny}x{nz} exceeds u32"));
        }
        let n_cells = nx
            .checked_mul(ny)
            .and_then(|xy| xy.checked_mul(nz))
            .ok_or_else(|| format!("Voudon field shape {nx}x{ny}x{nz} overflows usize"))?;
        if n_cells > u32::MAX as usize {
            return Err(format!(
                "Voudon field cell count {n_cells} exceeds u32 dispatch"
            ));
        }
        Ok(Self {
            n_cells,
            output_zeros: vec![0u32; n_cells],
            params: vec![nx as u32, ny as u32, nz as u32, seed],
        })
    }

    fn dispatch_groups(&self) -> Result<u32, String> {
        let cells = u32::try_from(self.n_cells)
            .map_err(|_| "Voudon field cell count exceeds u32 dispatch".to_string())?;
        Ok(cells.div_ceil(256))
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
        .ok_or_else(|| format!("Voudon {label} buffer size overflows"))? as u64;
    let buffer = HostVisibleBuffer::storage(device, adapter, byte_len)
        .map_err(|e| format!("Voudon {label} buffer allocation failed: {e}"))?;
    buffer
        .write_u32_slice(values)
        .map_err(|e| format!("Voudon {label} buffer upload failed: {e}"))?;
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voudon_vulkan_wgsl_parses_and_emits_compute_spirv() {
        let module = naga::front::wgsl::parse_str(VoudonVulkanKernel::wgsl_source()).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();

        let pipeline_options = naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: VOUDON_VULKAN_ENTRY_POINT.to_string(),
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
    fn voudon_vulkan_prepares_field_shape() {
        let prepared = PreparedVoudonField::new(3, 2, 2, 42).unwrap();
        assert_eq!(prepared.n_cells, 12);
        assert_eq!(prepared.output_zeros, vec![0; 12]);
        assert_eq!(prepared.params, vec![3, 2, 2, 42]);
    }

    #[test]
    fn voudon_vulkan_rejects_oversized_field() {
        assert!(PreparedVoudonField::new(u32::MAX as usize + 1, 1, 1, 0).is_err());
    }

    #[test]
    #[ignore = "requires local Vulkan compute device"]
    fn voudon_vulkan_context_smoke() {
        let (_instance, adapter, _device) = VoudonVulkanKernel::build_context().unwrap();
        eprintln!(
            "voudon_vulkan_context_smoke: selected device {}",
            adapter.device_name()
        );
    }

    #[test]
    #[ignore = "requires local Vulkan compute device"]
    fn voudon_vulkan_pipeline_smoke() {
        let (_instance, adapter, device) = VoudonVulkanKernel::build_context().unwrap();
        eprintln!(
            "voudon_vulkan_pipeline_smoke: selected device {}",
            adapter.device_name()
        );
        let _pipeline = VoudonVulkanKernel::build_pipeline(&device).unwrap();
    }
}
