//! GPU-accelerated search for stable zero-divisor cycles in 256D Voudon algebra.

#[cfg(feature = "cubecl")]
use cubecl::prelude::*;
#[cfg(feature = "cubecl")]
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
#[cfg(feature = "gpu")]
use cudarc::driver::PushKernelArg;
#[cfg(feature = "gpu")]
use gororoba_gpu_cuda::{Buffer, CompileOptions, LaunchConfig, ModuleRegistry};
#[cfg(feature = "vulkan")]
use gororoba_gpu_vulkan::{
    Adapter, ComputePipeline, ComputePipelineBuilder, DescriptorPool, DescriptorSetLayout,
    DescriptorSetLayoutSpec, Device, DeviceBuilder, DispatchScope, HostVisibleBuffer, Instance,
    InstanceBuilder, QueueFamilyRequirement, ShaderModule, ValidationPolicy,
};

const VOUDON_DIM: usize = 256;
const VOUDON_DIM_U32: u32 = 256;
#[cfg(feature = "vulkan")]
const VOUDON_VULKAN_ENTRY_POINT: &str = "compute_voudon_stable_cycle_counts";

/// CUDA kernel to find stable (associative) triples in 256D.
#[cfg(feature = "gpu")]
const STABILIZER_KERNEL_SRC: &str = r#"
__device__ int cd_basis_mul_sign(unsigned int p, unsigned int q) {
    int sign = 1;
    unsigned int half = 128; 
    while (half > 0) {
        unsigned int p_hi = (p >= half) ? 1 : 0;
        unsigned int q_hi = (q >= half) ? 1 : 0;
        unsigned int branch = (p_hi << 1) | q_hi;
        if (branch == 1) {
            unsigned int qh = q - half;
            q = p; p = qh;
        } else if (branch == 2) {
            p -= half;
            if (q != 0) sign = -sign;
        } else if (branch == 3) {
            unsigned int qh = q - half;
            unsigned int ph = p - half;
            if (qh == 0) return -sign;
            p = qh; q = ph;
        }
        half >>= 1;
    }
    return sign;
}

extern "C" __global__ void find_stable_cycles_kernel(
    unsigned int* __restrict__ stable_triples, // [3 * max_triples]
    unsigned int* __restrict__ count,
    unsigned int max_triples
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 256) return;

    for (unsigned int j = 0; j < 256; j++) {
        if (i == j) continue;
        
        // Triple (i, i, j) is stable if [i, i, j] = 0 (alternativity holds)
        int s1 = cd_basis_mul_sign(i, i);
        int ij_idx = i ^ j;
        int s2 = cd_basis_mul_sign(i, j);
        int i_ij_sign = cd_basis_mul_sign(i, ij_idx) * s2;
        
        if (s1 == i_ij_sign) {
            unsigned int c = atomicAdd(count, 1);
            if (c < max_triples) {
                stable_triples[c * 3 + 0] = i;
                stable_triples[c * 3 + 1] = i;
                stable_triples[c * 3 + 2] = j;
            }
        }
    }
}
"#;

pub struct Cd256StabilizerKernel;

impl Cd256StabilizerKernel {
    pub fn find_stable_cycles_cpu(max_triples: usize) -> Vec<(usize, usize, usize)> {
        stable_cycles_from_row_counts(&stable_cycle_row_counts_cpu(), max_triples)
    }

    pub fn stable_cycle_row_counts_cpu() -> [u32; VOUDON_DIM] {
        stable_cycle_row_counts_cpu()
    }

    #[cfg(feature = "cubecl")]
    pub fn cubecl_available() -> bool {
        gororoba_gpu_cubecl::Runtime::probe()
    }

    #[cfg(not(feature = "cubecl"))]
    pub fn cubecl_available() -> bool {
        false
    }

    #[cfg(feature = "cubecl")]
    pub fn stable_cycle_row_counts_cubecl() -> Result<[u32; VOUDON_DIM], String> {
        if !Self::cubecl_available() {
            return Err("Voudon stabilizer cubecl adapter unavailable".to_string());
        }

        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let counts_handle = client.empty(VOUDON_DIM * std::mem::size_of::<u32>());
        let counts_readback = counts_handle.clone();
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::new_1d(1);

        // SAFETY: counts_handle has exactly 256 u32 slots. The kernel exits
        // every thread with ABSOLUTE_POS >= 256.
        unsafe {
            voudon_stable_cycle_counts_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(counts_handle, VOUDON_DIM),
            );
        }

        let bytes = client.read_one_unchecked(counts_readback);
        decode_u32_array_256(&bytes, "voudon_stable_cycle_counts")
    }

    #[cfg(not(feature = "cubecl"))]
    pub fn stable_cycle_row_counts_cubecl() -> Result<[u32; VOUDON_DIM], String> {
        Err("cubecl feature not enabled".to_string())
    }

    #[cfg(feature = "cubecl")]
    pub fn find_stable_cycles_cubecl(
        max_triples: usize,
    ) -> Result<Vec<(usize, usize, usize)>, String> {
        let row_counts = Self::stable_cycle_row_counts_cubecl()?;
        Ok(stable_cycles_from_row_counts(&row_counts, max_triples))
    }

    #[cfg(not(feature = "cubecl"))]
    pub fn find_stable_cycles_cubecl(
        _max_triples: usize,
    ) -> Result<Vec<(usize, usize, usize)>, String> {
        Err("cubecl feature not enabled".to_string())
    }

    #[cfg(feature = "vulkan")]
    pub fn vulkan_available() -> bool {
        match build_vulkan_context() {
            Ok((_instance, _adapter, _device)) => true,
            Err(_) => false,
        }
    }

    #[cfg(not(feature = "vulkan"))]
    pub fn vulkan_available() -> bool {
        false
    }

    #[cfg(feature = "vulkan")]
    pub fn stable_cycle_row_counts_vulkan() -> Result<[u32; VOUDON_DIM], String> {
        let (_instance, adapter, device) = build_vulkan_context()?;
        let pipeline = build_vulkan_pipeline(&device)?;
        let output_buffer =
            upload_u32_storage(&device, &adapter, &[0u32; VOUDON_DIM], "row_counts")?;

        let descriptor_pool =
            DescriptorPool::for_layout(&device, &pipeline.descriptor_layout, 1)
                .map_err(|e| format!("Voudon stabilizer descriptor pool failed: {e}"))?;
        let descriptor_set = descriptor_pool
            .allocate_set(&pipeline.descriptor_layout)
            .map_err(|e| format!("Voudon stabilizer descriptor set failed: {e}"))?;
        descriptor_set.write_storage_buffer(0, &output_buffer);

        let dispatch = DispatchScope::new(&device)
            .map_err(|e| format!("Voudon stabilizer dispatch scope failed: {e}"))?;
        dispatch
            .dispatch(
                &pipeline.pipeline,
                descriptor_set.raw(),
                1,
                1,
                1,
                5_000_000_000,
            )
            .map_err(|e| format!("Voudon stabilizer Vulkan dispatch failed: {e}"))?;

        let counts = output_buffer
            .read_u32_slice(VOUDON_DIM)
            .map_err(|e| format!("Voudon stabilizer Vulkan readback failed: {e}"))?;
        counts.try_into().map_err(|values: Vec<u32>| {
            format!("Voudon stabilizer Vulkan returned {} rows", values.len())
        })
    }

    #[cfg(not(feature = "vulkan"))]
    pub fn stable_cycle_row_counts_vulkan() -> Result<[u32; VOUDON_DIM], String> {
        Err("vulkan feature not enabled".to_string())
    }

    #[cfg(feature = "vulkan")]
    pub fn find_stable_cycles_vulkan(
        max_triples: usize,
    ) -> Result<Vec<(usize, usize, usize)>, String> {
        let row_counts = Self::stable_cycle_row_counts_vulkan()?;
        Ok(stable_cycles_from_row_counts(&row_counts, max_triples))
    }

    #[cfg(not(feature = "vulkan"))]
    pub fn find_stable_cycles_vulkan(
        _max_triples: usize,
    ) -> Result<Vec<(usize, usize, usize)>, String> {
        Err("vulkan feature not enabled".to_string())
    }

    #[cfg(feature = "gpu")]
    pub fn find_stable_cycles(max_triples: usize) -> Result<Vec<(usize, usize, usize)>, String> {
        // Delegate context acquisition to gpu_cuda::Context so the
        // get_count + ordinal-range checks live in one place across
        // the workspace.
        let ctx_wrapper = gororoba_gpu_cuda::Context::with_default_device()
            .map_err(|e| format!("CUDA init: {}", e))?;
        let stream = ctx_wrapper.default_stream();

        let opts = CompileOptions::empty();
        let ptx = CompileOptions::compile_ptx(STABILIZER_KERNEL_SRC, &opts)
            .map_err(|e| format!("NVRTC compile: {}", e))?;
        let registry = ModuleRegistry::load(ctx_wrapper.raw(), ptx, &["find_stable_cycles_kernel"])
            .map_err(|e| format!("Module load: {}", e))?;
        let kernel = registry
            .get("find_stable_cycles_kernel")
            .map_err(|e| format!("Kernel load: {}", e))?;

        let mut d_triples = Buffer::<u32>::alloc_zeros(&stream, max_triples * 3)
            .map_err(|e| format!("Alloc: {}", e))?;
        let mut d_count =
            Buffer::<u32>::alloc_zeros(&stream, 1).map_err(|e| format!("Alloc: {}", e))?;

        let cfg = LaunchConfig::launch_1d(256);

        let mut builder = stream.launch_builder(&kernel);
        builder.arg(d_triples.raw_mut());
        builder.arg(d_count.raw_mut());
        let max_triples_u32 = max_triples as u32;
        builder.arg(&max_triples_u32);

        unsafe {
            builder.launch(cfg).map_err(|e| format!("Launch: {}", e))?;
        }

        let count_vec = d_count
            .dtoh_vec()
            .map_err(|e| format!("Copy count: {}", e))?;
        let count = (count_vec[0] as usize).min(max_triples);

        let triples_vec = d_triples
            .dtoh_vec()
            .map_err(|e| format!("Copy triples: {}", e))?;
        let mut result = Vec::new();
        for i in 0..count {
            result.push((
                triples_vec[i * 3] as usize,
                triples_vec[i * 3 + 1] as usize,
                triples_vec[i * 3 + 2] as usize,
            ));
        }

        Ok(result)
    }

    #[cfg(not(feature = "gpu"))]
    pub fn find_stable_cycles(max_triples: usize) -> Result<Vec<(usize, usize, usize)>, String> {
        Ok(Self::find_stable_cycles_cpu(max_triples))
    }
}

pub fn stable_cycle_row_counts_cpu() -> [u32; VOUDON_DIM] {
    let mut row_counts = [0u32; VOUDON_DIM];
    for i in 0..VOUDON_DIM_U32 {
        let mut count = 0u32;
        for j in 0..VOUDON_DIM_U32 {
            if stable_cycle_predicate(i, j) {
                count += 1;
            }
        }
        row_counts[i as usize] = count;
    }
    row_counts
}

pub fn stable_cycles_from_row_counts(
    row_counts: &[u32; VOUDON_DIM],
    max_triples: usize,
) -> Vec<(usize, usize, usize)> {
    let mut triples = Vec::new();
    for i in 0..VOUDON_DIM_U32 {
        let mut emitted_for_row = 0u32;
        for j in 0..VOUDON_DIM_U32 {
            if stable_cycle_predicate(i, j) {
                if triples.len() < max_triples {
                    triples.push((i as usize, i as usize, j as usize));
                }
                emitted_for_row += 1;
            }
        }
        debug_assert_eq!(emitted_for_row, row_counts[i as usize]);
        if triples.len() >= max_triples {
            break;
        }
    }
    triples
}

pub fn stable_cycle_predicate(i: u32, j: u32) -> bool {
    if i == j {
        return false;
    }
    let s1 = cd_basis_mul_sign_256(i, i);
    let ij_idx = i ^ j;
    let s2 = cd_basis_mul_sign_256(i, j);
    let i_ij_sign = cd_basis_mul_sign_256(i, ij_idx) * s2;
    s1 == i_ij_sign
}

pub fn cd_basis_mul_sign_256(mut p: u32, mut q: u32) -> i32 {
    let mut sign = 1i32;
    let mut half = 128u32;
    while half > 0 {
        let p_hi = p >= half;
        let q_hi = q >= half;
        let mut next_half = half >> 1;
        if !p_hi && q_hi {
            let qh = q - half;
            q = p;
            p = qh;
        } else if p_hi && !q_hi {
            p -= half;
            if q != 0 {
                sign = -sign;
            }
        } else if p_hi && q_hi {
            let qh = q - half;
            let ph = p - half;
            if qh == 0 {
                sign = -sign;
                next_half = 0;
            } else {
                p = qh;
                q = ph;
            }
        }
        half = next_half;
    }
    sign
}

#[cfg(feature = "vulkan")]
const VOUDON_STABLE_CYCLE_COUNTS_WGSL: &str = r#"
struct U32Array {
    values: array<u32>,
};

@group(0) @binding(0)
var<storage, read_write> row_counts: U32Array;

fn cd_basis_mul_sign_256(p_input: u32, q_input: u32) -> i32 {
    var sign: i32 = 1;
    var p: u32 = p_input;
    var q: u32 = q_input;
    var half: u32 = 128u;

    while (half > 0u) {
        let p_hi: bool = p >= half;
        let q_hi: bool = q >= half;

        if (!p_hi && q_hi) {
            let qh: u32 = q - half;
            q = p;
            p = qh;
        } else if (p_hi && !q_hi) {
            p = p - half;
            if (q != 0u) {
                sign = -sign;
            }
        } else if (p_hi && q_hi) {
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

fn stable_cycle_predicate(i: u32, j: u32) -> bool {
    if (i == j) {
        return false;
    }

    let s1: i32 = cd_basis_mul_sign_256(i, i);
    let ij_idx: u32 = i ^ j;
    let s2: i32 = cd_basis_mul_sign_256(i, j);
    let i_ij_sign: i32 = cd_basis_mul_sign_256(i, ij_idx) * s2;
    return s1 == i_ij_sign;
}

@compute @workgroup_size(256)
fn compute_voudon_stable_cycle_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    if (i >= 256u) {
        return;
    }

    var count: u32 = 0u;
    for (var j: u32 = 0u; j < 256u; j = j + 1u) {
        if (stable_cycle_predicate(i, j)) {
            count = count + 1u;
        }
    }

    row_counts.values[i] = count;
}
"#;

#[cfg(feature = "vulkan")]
struct VoudonStabilizerVulkanPipeline {
    pipeline: ComputePipeline,
    descriptor_layout: DescriptorSetLayout,
}

#[cfg(feature = "vulkan")]
fn build_vulkan_pipeline(device: &Device) -> Result<VoudonStabilizerVulkanPipeline, String> {
    let shader = ShaderModule::from_wgsl(
        device,
        VOUDON_STABLE_CYCLE_COUNTS_WGSL,
        VOUDON_VULKAN_ENTRY_POINT,
    )
    .map_err(|e| format!("Voudon stabilizer WGSL compile failed: {e}"))?;
    let descriptor_layout = DescriptorSetLayoutSpec::new()
        .storage_buffer(0)
        .build(device)
        .map_err(|e| format!("Voudon stabilizer descriptor layout failed: {e}"))?;
    let pipeline = ComputePipelineBuilder::new(device, &shader)
        .descriptor_layout(&descriptor_layout)
        .build()
        .map_err(|e| format!("Voudon stabilizer pipeline build failed: {e}"))?;

    Ok(VoudonStabilizerVulkanPipeline {
        pipeline,
        descriptor_layout,
    })
}

#[cfg(feature = "vulkan")]
fn build_vulkan_context() -> Result<(Instance, Adapter, Device), String> {
    let instance = InstanceBuilder::new("algebra_experimental_voudon_stabilizer_vulkan")
        .validation(ValidationPolicy::Disable)
        .build()
        .map_err(|e| format!("Voudon stabilizer Vulkan instance failed: {e}"))?;
    let adapter = Adapter::pick(&instance, QueueFamilyRequirement::Compute)
        .map_err(|e| format!("Voudon stabilizer Vulkan adapter failed: {e}"))?;
    let device = DeviceBuilder::new(adapter.clone())
        .build(&instance)
        .map_err(|e| format!("Voudon stabilizer Vulkan device failed: {e}"))?;
    Ok((instance, adapter, device))
}

#[cfg(feature = "vulkan")]
fn upload_u32_storage(
    device: &Device,
    adapter: &Adapter,
    values: &[u32],
    label: &str,
) -> Result<HostVisibleBuffer, String> {
    let byte_len = values
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| format!("Voudon stabilizer {label} buffer size overflows"))?
        as u64;
    let buffer = HostVisibleBuffer::storage(device, adapter, byte_len)
        .map_err(|e| format!("Voudon stabilizer {label} buffer allocation failed: {e}"))?;
    buffer
        .write_u32_slice(values)
        .map_err(|e| format!("Voudon stabilizer {label} buffer upload failed: {e}"))?;
    Ok(buffer)
}

#[cfg(feature = "cubecl")]
#[cube(launch_unchecked)]
pub fn voudon_stable_cycle_counts_kernel(row_counts: &mut Array<u32>) {
    let i = ABSOLUTE_POS;
    if i >= row_counts.len() {
        terminate!();
    }

    let i_u32 = i as u32;
    let mut count = 0u32;
    let mut j = 0u32;
    while j < VOUDON_DIM_U32 {
        if stable_cycle_predicate_cube(i_u32, j) {
            count += 1u32;
        }
        j += 1u32;
    }
    row_counts[i] = count;
}

#[cfg(feature = "cubecl")]
#[cube]
fn stable_cycle_predicate_cube(i: u32, j: u32) -> bool {
    let mut is_stable = false;
    if i != j {
        let s1 = cd_basis_mul_sign_256_cube(i, i);
        let ij_idx = i ^ j;
        let s2 = cd_basis_mul_sign_256_cube(i, j);
        let i_ij_sign = cd_basis_mul_sign_256_cube(i, ij_idx) * s2;
        is_stable = s1 == i_ij_sign;
    }
    is_stable
}

#[cfg(feature = "cubecl")]
#[cube]
fn cd_basis_mul_sign_256_cube(mut p: u32, mut q: u32) -> i32 {
    let mut sign = 1i32;
    let mut half = 128u32;
    while half > 0u32 {
        let p_hi = p >= half;
        let q_hi = q >= half;
        let mut next_half = half >> 1u32;
        if !p_hi && q_hi {
            let qh = q - half;
            q = p;
            p = qh;
        } else if p_hi && !q_hi {
            p -= half;
            if q != 0u32 {
                sign = -sign;
            }
        } else if p_hi && q_hi {
            let qh = q - half;
            let ph = p - half;
            if qh == 0u32 {
                sign = -sign;
                next_half = 0u32;
            } else {
                p = qh;
                q = ph;
            }
        }
        half = next_half;
    }
    sign
}

#[cfg(feature = "cubecl")]
fn decode_u32_array_256(bytes: &[u8], label: &str) -> Result<[u32; VOUDON_DIM], String> {
    let expected_bytes = VOUDON_DIM
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| format!("Voudon stabilizer cubecl {label} length overflows bytes"))?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "Voudon stabilizer cubecl {label} readback returned {} bytes, expected {expected_bytes}",
            bytes.len()
        ));
    }

    let mut out = [0u32; VOUDON_DIM];
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        out[index] = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_predicate_rejects_diagonal() {
        assert!(!stable_cycle_predicate(7, 7));
    }

    #[test]
    fn cpu_row_counts_match_reconstructed_triples() {
        let row_counts = Cd256StabilizerKernel::stable_cycle_row_counts_cpu();
        let all_triples = stable_cycles_from_row_counts(&row_counts, usize::MAX);
        let total: usize = row_counts.iter().map(|&value| value as usize).sum();
        assert_eq!(all_triples.len(), total);
        assert!(all_triples.iter().all(|&(i, _, j)| i != j));
    }

    #[test]
    fn public_find_stable_cycles_truncates_cpu_reference() {
        let triples = Cd256StabilizerKernel::find_stable_cycles_cpu(16);
        assert_eq!(triples.len(), 16);
        assert!(triples.iter().all(|&(i, left, j)| i == left && i != j));
    }

    #[cfg(feature = "cubecl")]
    #[test]
    fn cubecl_available_does_not_panic() {
        let _ = Cd256StabilizerKernel::cubecl_available();
    }

    #[cfg(feature = "cubecl")]
    #[test]
    fn cubecl_row_counts_match_cpu_when_adapter_available() {
        if !Cd256StabilizerKernel::cubecl_available() {
            return;
        }

        let cpu = Cd256StabilizerKernel::stable_cycle_row_counts_cpu();
        let cubecl = Cd256StabilizerKernel::stable_cycle_row_counts_cubecl().unwrap();
        assert_eq!(cubecl, cpu);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_available_does_not_panic() {
        let _ = Cd256StabilizerKernel::vulkan_available();
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_row_counts_match_cpu_when_adapter_available() {
        if !Cd256StabilizerKernel::vulkan_available() {
            return;
        }

        let cpu = Cd256StabilizerKernel::stable_cycle_row_counts_cpu();
        let vulkan = Cd256StabilizerKernel::stable_cycle_row_counts_vulkan().unwrap();
        assert_eq!(vulkan, cpu);
    }
}
