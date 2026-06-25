#[cfg(feature = "vulkan")]
use gororoba_gpu_vulkan::{
    Adapter, ComputePipeline, ComputePipelineBuilder, DescriptorPool, DescriptorSetLayout,
    DescriptorSetLayoutSpec, Device, DeviceBuilder, DispatchScope, HostVisibleBuffer, Instance,
    InstanceBuilder, QueueFamilyRequirement, ShaderModule, ValidationPolicy,
};

#[cfg(feature = "vulkan")]
use super::TensorAVT;

#[cfg(feature = "vulkan")]
const WORKGROUP_SIZE: u32 = 256;
#[cfg(feature = "vulkan")]
const DISPATCH_TIMEOUT_NS: u64 = 10_000_000_000;
#[cfg(feature = "vulkan")]
const OP_CD_MUL_BATCH: u32 = 0;
#[cfg(feature = "vulkan")]
const OP_NORM_SQ_BATCH: u32 = 1;
#[cfg(feature = "vulkan")]
pub const TENSOR_AVT_VULKAN_ENTRY_POINT: &str = "tensor_avt";

#[cfg(feature = "vulkan")]
pub const TENSOR_AVT_VULKAN_WGSL: &str = r#"
struct F32Buffer {
    values: array<f32>,
};

struct U32Buffer {
    values: array<u32>,
};

@group(0) @binding(0)
var<storage, read> left: F32Buffer;
@group(0) @binding(1)
var<storage, read> right: F32Buffer;
@group(0) @binding(2)
var<storage, read_write> output: F32Buffer;
@group(0) @binding(3)
var<storage, read> params: U32Buffer;

fn cd_basis_mul_sign(dim: u32, p_input: u32, q_input: u32) -> i32 {
    var sign: i32 = 1;
    var p: u32 = p_input;
    var q: u32 = q_input;
    var half: u32 = dim / 2u;

    while (half > 0u) {
        let p_hi: bool = p >= half;
        let q_hi: bool = q >= half;
        let branch: u32 = (select(0u, 1u, p_hi) << 1u) | select(0u, 1u, q_hi);

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

@compute @workgroup_size(256)
fn tensor_avt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;
    let dim: u32 = params.values[0u];
    let count: u32 = params.values[1u];
    let op: u32 = params.values[2u];

    if (op == 0u) {
        let total: u32 = dim * count;
        if (idx >= total) {
            return;
        }

        let row: u32 = idx % dim;
        let batch: u32 = idx / dim;
        var acc: f32 = 0.0;
        var j: u32 = 0u;
        while (j < dim) {
            let src: u32 = row ^ j;
            let sign: i32 = cd_basis_mul_sign(dim, src, j);
            acc = acc + left.values[src] * f32(sign) * right.values[batch * dim + j];
            j = j + 1u;
        }
        output.values[idx] = acc;
    } else if (op == 1u) {
        if (idx >= count) {
            return;
        }

        var acc: f32 = 0.0;
        var j: u32 = 0u;
        while (j < dim) {
            let value: f32 = right.values[idx * dim + j];
            acc = acc + value * value;
            j = j + 1u;
        }
        output.values[idx] = acc;
    }
}
"#;

#[cfg(feature = "vulkan")]
pub struct TensorAvtVulkanPipeline {
    pipeline: ComputePipeline,
    descriptor_layout: DescriptorSetLayout,
}

#[cfg(feature = "vulkan")]
struct TensorAvtVulkanRuntime {
    pipeline: TensorAvtVulkanPipeline,
    device: Device,
    adapter: Adapter,
    _instance: Instance,
}

#[cfg(feature = "vulkan")]
pub struct TensorAvtMulVulkanWorkspace {
    left: HostVisibleBuffer,
    right: HostVisibleBuffer,
    output: HostVisibleBuffer,
    runtime: TensorAvtVulkanRuntime,
    dim: usize,
    max_batch_size: usize,
}

#[cfg(feature = "vulkan")]
pub struct TensorAvtNormVulkanWorkspace {
    vectors: HostVisibleBuffer,
    norms: HostVisibleBuffer,
    runtime: TensorAvtVulkanRuntime,
    dim: usize,
    max_vectors: usize,
}

#[cfg(feature = "vulkan")]
pub struct TensorAvtVulkanKernel;

#[cfg(feature = "vulkan")]
impl TensorAvtVulkanKernel {
    pub fn is_available() -> bool {
        Self::build_runtime().is_ok()
    }

    pub fn compute_cd_mul_batch(
        dim: usize,
        left: &[f32],
        right: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        validate_mul_inputs(dim, left, right, batch_size)?;
        let runtime = Self::build_runtime()?;
        let left_buffer = storage_f32(&runtime.device, &runtime.adapter, dim, "left")?;
        let right_buffer = storage_f32(&runtime.device, &runtime.adapter, right.len(), "right")?;
        let output_len = dim * batch_size;
        let output_buffer = storage_f32(&runtime.device, &runtime.adapter, output_len, "output")?;
        left_buffer
            .write_f32_slice(left)
            .map_err(|e| format!("TensorAVT Vulkan left upload failed: {e}"))?;
        right_buffer
            .write_f32_slice(right)
            .map_err(|e| format!("TensorAVT Vulkan right upload failed: {e}"))?;
        dispatch(
            &runtime,
            &left_buffer,
            &right_buffer,
            &output_buffer,
            dim,
            batch_size,
            OP_CD_MUL_BATCH,
            output_len,
        )?;
        output_buffer
            .read_f32_slice(output_len)
            .map_err(|e| format!("TensorAVT Vulkan output readback failed: {e}"))
    }

    pub fn compute_norm_sq_batch(
        dim: usize,
        vectors: &[f32],
        n_vectors: usize,
    ) -> Result<Vec<f32>, String> {
        validate_norm_inputs(dim, vectors, n_vectors)?;
        let runtime = Self::build_runtime()?;
        let left_buffer = storage_f32(&runtime.device, &runtime.adapter, dim, "left")?;
        let vector_buffer =
            storage_f32(&runtime.device, &runtime.adapter, vectors.len(), "vectors")?;
        let norm_buffer = storage_f32(&runtime.device, &runtime.adapter, n_vectors, "norms")?;
        left_buffer
            .write_f32_slice(&vec![0.0f32; dim])
            .map_err(|e| format!("TensorAVT Vulkan left initialization failed: {e}"))?;
        vector_buffer
            .write_f32_slice(vectors)
            .map_err(|e| format!("TensorAVT Vulkan vector upload failed: {e}"))?;
        dispatch(
            &runtime,
            &left_buffer,
            &vector_buffer,
            &norm_buffer,
            dim,
            n_vectors,
            OP_NORM_SQ_BATCH,
            n_vectors,
        )?;
        norm_buffer
            .read_f32_slice(n_vectors)
            .map_err(|e| format!("TensorAVT Vulkan norm readback failed: {e}"))
    }

    fn build_runtime() -> Result<TensorAvtVulkanRuntime, String> {
        let instance = InstanceBuilder::new("gororoba_algebra_tensor_avt_vulkan")
            .validation(ValidationPolicy::Disable)
            .build()
            .map_err(|e| format!("TensorAVT Vulkan instance creation failed: {e}"))?;
        let adapter = Adapter::pick(&instance, QueueFamilyRequirement::Compute)
            .map_err(|e| format!("TensorAVT Vulkan adapter pick failed: {e}"))?;
        let device = DeviceBuilder::new(adapter.clone())
            .build(&instance)
            .map_err(|e| format!("TensorAVT Vulkan device creation failed: {e}"))?;
        let pipeline = build_pipeline(&device)?;
        Ok(TensorAvtVulkanRuntime {
            pipeline,
            device,
            adapter,
            _instance: instance,
        })
    }
}

#[cfg(not(feature = "vulkan"))]
pub(crate) fn tensor_avt_vulkan_error() -> String {
    "TensorAVT Vulkan backend requires building gororoba_algebra with --features vulkan".into()
}

#[cfg(not(feature = "vulkan"))]
pub fn tensor_avt_vulkan_available() -> bool {
    false
}

#[cfg(feature = "vulkan")]
pub fn tensor_avt_vulkan_available() -> bool {
    TensorAvtVulkanKernel::is_available()
}

#[cfg(feature = "vulkan")]
impl TensorAVT {
    pub fn new_vulkan_mul_workspace(
        &self,
        max_batch_size: usize,
    ) -> Result<TensorAvtMulVulkanWorkspace, String> {
        if max_batch_size == 0 {
            return Err("max_batch_size must be > 0".into());
        }
        let runtime = TensorAvtVulkanKernel::build_runtime()?;
        Ok(TensorAvtMulVulkanWorkspace {
            left: storage_f32(&runtime.device, &runtime.adapter, self.dim, "left")?,
            right: storage_f32(
                &runtime.device,
                &runtime.adapter,
                self.dim * max_batch_size,
                "right",
            )?,
            output: storage_f32(
                &runtime.device,
                &runtime.adapter,
                self.dim * max_batch_size,
                "output",
            )?,
            runtime,
            dim: self.dim,
            max_batch_size,
        })
    }

    pub fn new_vulkan_norm_workspace(
        &self,
        max_vectors: usize,
    ) -> Result<TensorAvtNormVulkanWorkspace, String> {
        if max_vectors == 0 {
            return Err("max_vectors must be > 0".into());
        }
        let runtime = TensorAvtVulkanKernel::build_runtime()?;
        Ok(TensorAvtNormVulkanWorkspace {
            vectors: storage_f32(
                &runtime.device,
                &runtime.adapter,
                self.dim * max_vectors,
                "vectors",
            )?,
            norms: storage_f32(&runtime.device, &runtime.adapter, max_vectors, "norms")?,
            runtime,
            dim: self.dim,
            max_vectors,
        })
    }

    pub fn compute_cd_mul_vulkan(&self, a: &[f32], x: &[f32]) -> Result<Vec<f32>, String> {
        TensorAvtVulkanKernel::compute_cd_mul_batch(self.dim, a, x, 1)
    }

    pub fn compute_cd_mul_batch_vulkan(
        &self,
        a: &[f32],
        x_batch: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        TensorAvtVulkanKernel::compute_cd_mul_batch(self.dim, a, x_batch, batch_size)
    }

    pub fn compute_norm_sq_batch_vulkan(
        &self,
        vectors: &[f32],
        n_vectors: usize,
    ) -> Result<Vec<f32>, String> {
        TensorAvtVulkanKernel::compute_norm_sq_batch(self.dim, vectors, n_vectors)
    }

    pub fn launch_cd_mul_vulkan_with_workspace(
        &self,
        workspace: &mut TensorAvtMulVulkanWorkspace,
    ) -> Result<(), String> {
        workspace.ensure_dim(self.dim)?;
        dispatch(
            &workspace.runtime,
            &workspace.left,
            &workspace.right,
            &workspace.output,
            self.dim,
            1,
            OP_CD_MUL_BATCH,
            self.dim,
        )
    }

    pub fn launch_cd_mul_batch_vulkan_with_workspace(
        &self,
        batch_size: usize,
        workspace: &mut TensorAvtMulVulkanWorkspace,
    ) -> Result<(), String> {
        workspace.ensure_dim(self.dim)?;
        workspace.ensure_batch(batch_size)?;
        dispatch(
            &workspace.runtime,
            &workspace.left,
            &workspace.right,
            &workspace.output,
            self.dim,
            batch_size,
            OP_CD_MUL_BATCH,
            self.dim * batch_size,
        )
    }

    pub fn launch_norm_sq_batch_vulkan_with_workspace(
        &self,
        n_vectors: usize,
        workspace: &mut TensorAvtNormVulkanWorkspace,
    ) -> Result<(), String> {
        workspace.ensure_dim(self.dim)?;
        workspace.ensure_vectors(n_vectors)?;
        let left = storage_f32(
            &workspace.runtime.device,
            &workspace.runtime.adapter,
            self.dim,
            "left",
        )?;
        left.write_f32_slice(&vec![0.0f32; self.dim])
            .map_err(|e| format!("TensorAVT Vulkan left initialization failed: {e}"))?;
        dispatch(
            &workspace.runtime,
            &left,
            &workspace.vectors,
            &workspace.norms,
            self.dim,
            n_vectors,
            OP_NORM_SQ_BATCH,
            n_vectors,
        )
    }
}

#[cfg(feature = "vulkan")]
impl TensorAvtMulVulkanWorkspace {
    pub fn upload_a(&self, a: &[f32]) -> Result<(), String> {
        if a.len() != self.dim {
            return Err(format!(
                "left input length {} must equal dim {}",
                a.len(),
                self.dim
            ));
        }
        self.left
            .write_f32_slice(a)
            .map_err(|e| format!("TensorAVT Vulkan left upload failed: {e}"))
    }

    pub fn upload_x(&self, x: &[f32], batch_size: usize, dim: usize) -> Result<(), String> {
        self.ensure_dim(dim)?;
        self.ensure_batch(batch_size)?;
        let expected = batch_size * self.dim;
        if x.len() != expected {
            return Err(format!(
                "right input length {} must equal batch_size * dim {}",
                x.len(),
                expected
            ));
        }
        self.right
            .write_f32_slice(x)
            .map_err(|e| format!("TensorAVT Vulkan right upload failed: {e}"))
    }

    pub fn download_y(&self, len: usize) -> Result<Vec<f32>, String> {
        if len > self.dim * self.max_batch_size {
            return Err(format!(
                "output length {} exceeds session capacity {}",
                len,
                self.dim * self.max_batch_size
            ));
        }
        self.output
            .read_f32_slice(len)
            .map_err(|e| format!("TensorAVT Vulkan output readback failed: {e}"))
    }

    fn ensure_dim(&self, dim: usize) -> Result<(), String> {
        if dim != self.dim {
            return Err(format!(
                "workspace dim {} does not match TensorAVT dim {}",
                self.dim, dim
            ));
        }
        Ok(())
    }

    fn ensure_batch(&self, batch_size: usize) -> Result<(), String> {
        if batch_size == 0 || batch_size > self.max_batch_size {
            return Err(format!(
                "batch_size must be in 1..={}, got {}",
                self.max_batch_size, batch_size
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "vulkan")]
impl TensorAvtNormVulkanWorkspace {
    pub fn upload_vectors(
        &self,
        vectors: &[f32],
        n_vectors: usize,
        dim: usize,
    ) -> Result<(), String> {
        self.ensure_dim(dim)?;
        self.ensure_vectors(n_vectors)?;
        let expected = n_vectors * self.dim;
        if vectors.len() != expected {
            return Err(format!(
                "vectors length {} must equal n_vectors * dim {}",
                vectors.len(),
                expected
            ));
        }
        self.vectors
            .write_f32_slice(vectors)
            .map_err(|e| format!("TensorAVT Vulkan vector upload failed: {e}"))
    }

    pub fn download_norms(&self, n_vectors: usize) -> Result<Vec<f32>, String> {
        self.ensure_vectors(n_vectors)?;
        self.norms
            .read_f32_slice(n_vectors)
            .map_err(|e| format!("TensorAVT Vulkan norm readback failed: {e}"))
    }

    fn ensure_dim(&self, dim: usize) -> Result<(), String> {
        if dim != self.dim {
            return Err(format!(
                "workspace dim {} does not match TensorAVT dim {}",
                self.dim, dim
            ));
        }
        Ok(())
    }

    fn ensure_vectors(&self, n_vectors: usize) -> Result<(), String> {
        if n_vectors == 0 || n_vectors > self.max_vectors {
            return Err(format!(
                "n_vectors must be in 1..={}, got {}",
                self.max_vectors, n_vectors
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "vulkan")]
fn build_pipeline(device: &Device) -> Result<TensorAvtVulkanPipeline, String> {
    let shader = ShaderModule::from_wgsl(
        device,
        TENSOR_AVT_VULKAN_WGSL,
        TENSOR_AVT_VULKAN_ENTRY_POINT,
    )
    .map_err(|e| format!("TensorAVT WGSL compile failed: {e}"))?;
    let descriptor_layout = DescriptorSetLayoutSpec::new()
        .storage_buffer(0)
        .storage_buffer(1)
        .storage_buffer(2)
        .storage_buffer(3)
        .build(device)
        .map_err(|e| format!("TensorAVT descriptor layout failed: {e}"))?;
    let pipeline = ComputePipelineBuilder::new(device, &shader)
        .descriptor_layout(&descriptor_layout)
        .build()
        .map_err(|e| format!("TensorAVT compute pipeline build failed: {e}"))?;
    Ok(TensorAvtVulkanPipeline {
        pipeline,
        descriptor_layout,
    })
}

#[cfg(feature = "vulkan")]
#[allow(clippy::too_many_arguments)] // Dispatch binds three storage buffers plus TensorAVT shape and operation scalars.
fn dispatch(
    runtime: &TensorAvtVulkanRuntime,
    left_buffer: &HostVisibleBuffer,
    right_buffer: &HostVisibleBuffer,
    output_buffer: &HostVisibleBuffer,
    dim: usize,
    count: usize,
    op: u32,
    work_items: usize,
) -> Result<(), String> {
    validate_dim(dim)?;
    if count == 0 {
        return Err("TensorAVT Vulkan count must be > 0".to_string());
    }
    let dim_u32 =
        u32::try_from(dim).map_err(|_| format!("TensorAVT Vulkan dim {dim} exceeds u32"))?;
    let count_u32 =
        u32::try_from(count).map_err(|_| format!("TensorAVT Vulkan count {count} exceeds u32"))?;
    let work_items_u32 = u32::try_from(work_items)
        .map_err(|_| format!("TensorAVT Vulkan work item count {work_items} exceeds u32"))?;
    let params_buffer = storage_u32(&runtime.device, &runtime.adapter, 4, "params")?;
    params_buffer
        .write_u32_slice(&[dim_u32, count_u32, op, 0])
        .map_err(|e| format!("TensorAVT Vulkan params upload failed: {e}"))?;
    let descriptor_pool =
        DescriptorPool::for_layout(&runtime.device, &runtime.pipeline.descriptor_layout, 1)
            .map_err(|e| format!("TensorAVT descriptor pool allocation failed: {e}"))?;
    let descriptor_set = descriptor_pool
        .allocate_set(&runtime.pipeline.descriptor_layout)
        .map_err(|e| format!("TensorAVT descriptor set allocation failed: {e}"))?;
    descriptor_set.write_storage_buffer(0, left_buffer);
    descriptor_set.write_storage_buffer(1, right_buffer);
    descriptor_set.write_storage_buffer(2, output_buffer);
    descriptor_set.write_storage_buffer(3, &params_buffer);
    let dispatch_scope = DispatchScope::new(&runtime.device)
        .map_err(|e| format!("TensorAVT dispatch scope creation failed: {e}"))?;
    dispatch_scope
        .dispatch(
            &runtime.pipeline.pipeline,
            descriptor_set.raw(),
            work_items_u32.div_ceil(WORKGROUP_SIZE),
            1,
            1,
            DISPATCH_TIMEOUT_NS,
        )
        .map_err(|e| format!("TensorAVT Vulkan dispatch failed: {e}"))
}

#[cfg(feature = "vulkan")]
fn storage_f32(
    device: &Device,
    adapter: &Adapter,
    len: usize,
    label: &str,
) -> Result<HostVisibleBuffer, String> {
    let byte_len = len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("TensorAVT Vulkan {label} length overflows bytes"))?;
    HostVisibleBuffer::storage(device, adapter, byte_len as u64)
        .map_err(|e| format!("TensorAVT Vulkan {label} buffer allocation failed: {e}"))
}

#[cfg(feature = "vulkan")]
fn storage_u32(
    device: &Device,
    adapter: &Adapter,
    len: usize,
    label: &str,
) -> Result<HostVisibleBuffer, String> {
    let byte_len = len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| format!("TensorAVT Vulkan {label} length overflows bytes"))?;
    HostVisibleBuffer::storage(device, adapter, byte_len as u64)
        .map_err(|e| format!("TensorAVT Vulkan {label} buffer allocation failed: {e}"))
}

#[cfg(feature = "vulkan")]
fn validate_dim(dim: usize) -> Result<(), String> {
    if dim < 16 || !dim.is_power_of_two() {
        return Err(format!(
            "TensorAVT Vulkan dim must be a power of two >= 16, got {dim}"
        ));
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn validate_mul_inputs(
    dim: usize,
    left: &[f32],
    right: &[f32],
    batch_size: usize,
) -> Result<(), String> {
    validate_dim(dim)?;
    if batch_size == 0 {
        return Err("TensorAVT Vulkan batch_size must be > 0".to_string());
    }
    if left.len() != dim {
        return Err(format!(
            "left input length {} must equal dim {}",
            left.len(),
            dim
        ));
    }
    let expected = dim * batch_size;
    if right.len() != expected {
        return Err(format!(
            "right input length {} must equal batch_size * dim {}",
            right.len(),
            expected
        ));
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn validate_norm_inputs(dim: usize, vectors: &[f32], n_vectors: usize) -> Result<(), String> {
    validate_dim(dim)?;
    if n_vectors == 0 {
        return Err("TensorAVT Vulkan n_vectors must be > 0".to_string());
    }
    let expected = dim * n_vectors;
    if vectors.len() != expected {
        return Err(format!(
            "vectors length {} must equal n_vectors * dim {}",
            vectors.len(),
            expected
        ));
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires local Vulkan compute device"]
    fn tensor_avt_vulkan_available_does_not_panic() {
        let _ = TensorAvtVulkanKernel::is_available();
    }

    #[test]
    fn tensor_avt_vulkan_rejects_invalid_inputs() {
        let left = vec![1.0f32; 16];
        let right = vec![1.0f32; 16];
        assert!(validate_mul_inputs(12, &left, &right, 1).is_err());
        assert!(validate_mul_inputs(16, &left[..15], &right, 1).is_err());
        assert!(validate_mul_inputs(16, &left, &right[..15], 1).is_err());
        assert!(validate_norm_inputs(16, &right, 0).is_err());
    }
}
