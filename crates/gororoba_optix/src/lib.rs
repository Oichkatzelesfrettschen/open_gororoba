//! Reusable OptiX runtime and FFI boundary for gororoba GPU workloads.
//!
//! This crate owns the generic acceleration-layer concerns for OptiX:
//! - dynamic loading of `libnvoptix.so.1`
//! - querying the OptiX function table
//! - creating and destroying an OptiX device context on the current CUDA context
//! - exposing the minimal host-side FFI types required by downstream crates
//!
//! Domain-specific concerns such as LBM density thresholds, SBT payload layout,
//! BVH rebuild policy, and particle-tracing semantics should remain in the
//! simulation crates that consume this boundary.

use std::{ffi::c_void, sync::Arc};

/// OptiX result code (0 = success).
pub type OptixResult = i32;

/// Opaque OptiX device-context handle.
pub type OptixDeviceContext = *mut c_void;
/// Opaque OptiX module handle.
pub type OptixModule = *mut c_void;
/// Opaque OptiX program-group handle.
pub type OptixProgramGroup = *mut c_void;
/// Opaque OptiX pipeline handle.
pub type OptixPipeline = *mut c_void;
/// Opaque OptiX traversable handle.
pub type OptixTraversableHandle = u64;

/// CUDA stream handle (`CUstream`).
pub type CuStream = *mut c_void;

/// CUDA device pointer (`CUdeviceptr`).
pub type CuDevicePtr = u64;

/// OptiX success code.
pub const OPTIX_SUCCESS: OptixResult = 0;

/// Device-context options (mirrors `OptixDeviceContextOptions`).
#[repr(C)]
#[derive(Default)]
pub struct OptixDeviceContextOptions {
    pub log_callback_function: Option<unsafe extern "C" fn(u32, *const i8, usize, *mut c_void)>,
    pub log_callback_data: *mut c_void,
    pub log_callback_level: i32,
    pub validation_mode: i32,
}

/// Module compile options (mirrors `OptixModuleCompileOptions`).
#[repr(C)]
#[derive(Default)]
pub struct OptixModuleCompileOptions {
    pub max_register_count: i32,
    pub opt_level: i32,
    pub debug_level: i32,
    pub bound_values: *const c_void,
    pub num_bound_values: u32,
    pub num_payload_types: u32,
    pub payload_types: *const c_void,
}

/// Pipeline compile options (mirrors `OptixPipelineCompileOptions`).
#[repr(C)]
pub struct OptixPipelineCompileOptions {
    pub uses_motion_blur: i32,
    pub traversable_graph_flags: u32,
    pub num_payload_values: i32,
    pub num_attribute_values: i32,
    pub exception_flags: u32,
    pub pipeline_launch_params_variable_name: *const i8,
    pub uses_primitive_type_flags: u32,
    pub allow_opacity_micromaps: i32,
}

/// Pipeline link options (mirrors `OptixPipelineLinkOptions`).
#[repr(C)]
pub struct OptixPipelineLinkOptions {
    pub max_trace_depth: u32,
}

/// Program-group description kind.
#[repr(C)]
#[derive(Clone, Copy)]
pub enum OptixProgramGroupKind {
    Raygen = 0x2421,
    Miss = 0x2422,
    Hitgroup = 0x2423,
}

/// Raygen or miss program-group descriptor entry.
#[repr(C)]
pub struct OptixProgramGroupSingleModule {
    pub module: OptixModule,
    pub entry_function_name: *const i8,
}

/// Program-group options (currently empty in OptiX 9.x).
#[repr(C)]
#[derive(Default)]
pub struct OptixProgramGroupOptions {
    pub payload_type: *const c_void,
}

/// Shader Binding Table entry header.
#[repr(C, align(16))]
pub struct OptixSbtRecordHeader {
    pub data: [u8; 32],
}

/// Shader Binding Table structure.
#[repr(C)]
pub struct OptixShaderBindingTable {
    pub raygen_record: CuDevicePtr,
    pub exception_record: CuDevicePtr,
    pub miss_record_base: CuDevicePtr,
    pub miss_record_stride_in_bytes: u32,
    pub miss_record_count: u32,
    pub hitgroup_record_base: CuDevicePtr,
    pub hitgroup_record_stride_in_bytes: u32,
    pub hitgroup_record_count: u32,
    pub callables_record_base: CuDevicePtr,
    pub callables_record_stride_in_bytes: u32,
    pub callables_record_count: u32,
}

/// Build-input descriptor for custom-primitive AABB acceleration structures.
#[repr(C)]
pub struct OptixBuildInputCustomPrimitiveArray {
    pub aabb_buffers: *const CuDevicePtr,
    pub num_primitives: u32,
    pub stride_in_bytes: u32,
    pub flags: *const u32,
    pub num_sbt_records: u32,
    pub sbt_index_offset_buffer: CuDevicePtr,
    pub sbt_index_offset_size_in_bytes: u32,
    pub sbt_index_offset_stride_in_bytes: u32,
    pub primitive_index_offset: u32,
}

/// Acceleration-structure build options.
#[repr(C)]
pub struct OptixAccelBuildOptions {
    pub build_flags: u32,
    pub operation: u32,
    pub motion_options: [u32; 3],
}

/// Output sizes reported by `optixAccelComputeMemoryUsage`.
#[repr(C)]
#[derive(Default)]
pub struct OptixAccelBufferSizes {
    pub output_size_in_bytes: usize,
    pub temp_size_in_bytes: usize,
    pub temp_update_size_in_bytes: usize,
}

/// Subset of the OptiX function table required by current gororoba workloads.
pub struct OptixFunctionTable {
    pub optix_device_context_create: unsafe extern "C" fn(
        CuDevicePtr,
        *const OptixDeviceContextOptions,
        *mut OptixDeviceContext,
    ) -> OptixResult,
    pub optix_device_context_destroy: unsafe extern "C" fn(OptixDeviceContext) -> OptixResult,
    pub optix_module_create: unsafe extern "C" fn(
        OptixDeviceContext,
        *const OptixModuleCompileOptions,
        *const OptixPipelineCompileOptions,
        *const u8,
        usize,
        *mut i8,
        *mut usize,
        *mut OptixModule,
    ) -> OptixResult,
    pub optix_module_destroy: unsafe extern "C" fn(OptixModule) -> OptixResult,
    pub optix_pipeline_create: unsafe extern "C" fn(
        OptixDeviceContext,
        *const OptixPipelineCompileOptions,
        *const OptixPipelineLinkOptions,
        *const OptixProgramGroup,
        u32,
        *mut i8,
        *mut usize,
        *mut OptixPipeline,
    ) -> OptixResult,
    pub optix_pipeline_destroy: unsafe extern "C" fn(OptixPipeline) -> OptixResult,
    pub optix_launch: unsafe extern "C" fn(
        OptixPipeline,
        CuStream,
        CuDevicePtr,
        usize,
        *const OptixShaderBindingTable,
        u32,
        u32,
        u32,
    ) -> OptixResult,
    pub optix_sbt_record_pack_header:
        unsafe extern "C" fn(OptixProgramGroup, *mut OptixSbtRecordHeader) -> OptixResult,
    pub optix_accel_compute_memory_usage: unsafe extern "C" fn(
        OptixDeviceContext,
        *const OptixAccelBuildOptions,
        *const c_void,
        u32,
        *mut OptixAccelBufferSizes,
    ) -> OptixResult,
    pub optix_accel_build: unsafe extern "C" fn(
        OptixDeviceContext,
        CuStream,
        *const OptixAccelBuildOptions,
        *const c_void,
        u32,
        CuDevicePtr,
        usize,
        CuDevicePtr,
        usize,
        *mut OptixTraversableHandle,
        *const c_void,
        u32,
    ) -> OptixResult,
}

/// Loaded OptiX shared library and queried function table.
pub struct OptixApi {
    _lib: Arc<libloading::Library>,
    pub table: OptixFunctionTable,
}

type QueryFunctionTableFn =
    unsafe extern "C" fn(u32, u32, *mut c_void, *mut c_void, *mut c_void, usize) -> OptixResult;

impl OptixApi {
    /// Load `libnvoptix.so.1` and query the OptiX function table.
    ///
    /// # Safety
    ///
    /// The current process must have a valid CUDA driver/runtime stack.
    #[allow(unsafe_op_in_unsafe_fn, clippy::missing_transmute_annotations)]
    pub unsafe fn load() -> anyhow::Result<Self> {
        let lib_path = if std::path::Path::new("/usr/lib/libnvoptix.so.1").exists() {
            "/usr/lib/libnvoptix.so.1"
        } else if std::path::Path::new("/usr/lib64/libnvoptix.so.1").exists() {
            "/usr/lib64/libnvoptix.so.1"
        } else {
            anyhow::bail!("libnvoptix.so.1 not found. Install NVIDIA driver >= 530.");
        };

        let lib = libloading::Library::new(lib_path)?;
        let query_fn: libloading::Symbol<QueryFunctionTableFn> =
            lib.get(b"optixQueryFunctionTable\0")?;

        const ABI_TABLE_SIZES: &[(u32, usize)] =
            &[(91, 50 * 8), (86, 50 * 8), (78, 56 * 8), (55, 60 * 8)];

        let mut raw_table = vec![0u8; 512];
        let mut matched_abi = 0u32;

        for &(abi, size) in ABI_TABLE_SIZES {
            raw_table[..size].fill(0);
            let result = query_fn(
                abi,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                raw_table.as_mut_ptr() as *mut c_void,
                size,
            );
            if result == OPTIX_SUCCESS {
                matched_abi = abi;
                break;
            }
        }

        if matched_abi == 0 {
            for n_ptrs in 40..80 {
                let size = n_ptrs * 8;
                raw_table[..size].fill(0);
                let result = query_fn(
                    91,
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    raw_table.as_mut_ptr() as *mut c_void,
                    size,
                );
                if result == OPTIX_SUCCESS {
                    matched_abi = 91;
                    break;
                }
            }
        }

        if matched_abi == 0 {
            anyhow::bail!("optixQueryFunctionTable failed for all probed ABI versions");
        }

        let ptrs = raw_table.as_ptr() as *const usize;
        let table = OptixFunctionTable {
            optix_device_context_create: std::mem::transmute(*ptrs.add(0)),
            optix_device_context_destroy: std::mem::transmute(*ptrs.add(1)),
            optix_module_create: std::mem::transmute(*ptrs.add(4)),
            optix_module_destroy: std::mem::transmute(*ptrs.add(6)),
            optix_pipeline_create: std::mem::transmute(*ptrs.add(14)),
            optix_pipeline_destroy: std::mem::transmute(*ptrs.add(15)),
            optix_launch: std::mem::transmute(*ptrs.add(17)),
            optix_sbt_record_pack_header: std::mem::transmute(*ptrs.add(12)),
            optix_accel_compute_memory_usage: std::mem::transmute(*ptrs.add(18)),
            optix_accel_build: std::mem::transmute(*ptrs.add(19)),
        };

        if table.optix_device_context_create as usize == 0 {
            anyhow::bail!(
                "optixDeviceContextCreate is null -- ABI version mismatch? Expected ABI {matched_abi}"
            );
        }

        Ok(Self {
            _lib: Arc::new(lib),
            table,
        })
    }
}

/// Live OptiX runtime bound to the current CUDA context.
///
/// This is the clear acceleration boundary for downstream crates: load the
/// OptiX API, create a device context, and keep the two tied together for the
/// lifetime of higher-level pipelines.
pub struct OptixRuntime {
    pub api: OptixApi,
    pub context: OptixDeviceContext,
}

impl OptixRuntime {
    /// Initialize OptiX on the current CUDA context.
    ///
    /// # Safety
    ///
    /// A CUDA context must already be current on the calling thread.
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn init_current_context(log_level: u32) -> anyhow::Result<Self> {
        let api = OptixApi::load()?;
        let mut context: OptixDeviceContext = std::ptr::null_mut();
        let options = OptixDeviceContextOptions {
            log_callback_level: log_level as i32,
            ..Default::default()
        };
        let result = (api.table.optix_device_context_create)(0, &options, &mut context);
        if result != OPTIX_SUCCESS {
            anyhow::bail!("optixDeviceContextCreate failed with code {result}");
        }
        if context.is_null() {
            anyhow::bail!("optixDeviceContextCreate returned null context");
        }
        Ok(Self { api, context })
    }

    /// Probe whether the OptiX shared library is loadable.
    pub fn probe() -> anyhow::Result<String> {
        probe_optix().map_err(anyhow::Error::msg)
    }
}

impl Drop for OptixRuntime {
    fn drop(&mut self) {
        if !self.context.is_null() {
            unsafe {
                (self.api.table.optix_device_context_destroy)(self.context);
            }
            self.context = std::ptr::null_mut();
        }
    }
}

/// Check whether an OptiX shared library is present on the system.
pub fn optix_available() -> bool {
    std::path::Path::new("/usr/lib/libnvoptix.so.1").exists()
        || std::path::Path::new("/usr/lib64/libnvoptix.so.1").exists()
}

/// Probe OptiX availability and report status.
pub fn probe_optix() -> Result<String, String> {
    if !optix_available() {
        return Err("libnvoptix.so.1 not found. Install NVIDIA driver >= 530.".to_string());
    }
    match unsafe { OptixApi::load() } {
        Ok(_) => Ok("OptiX available (function table loaded)".to_string()),
        Err(err) => Err(format!("OptiX library found but failed to load: {err}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optix_probe() {
        let _ = probe_optix();
    }

    #[test]
    fn test_handle_sizes() {
        assert_eq!(
            std::mem::size_of::<OptixDeviceContext>(),
            std::mem::size_of::<*mut ()>()
        );
        assert_eq!(std::mem::size_of::<OptixTraversableHandle>(), 8);
    }

    #[test]
    fn test_sbt_header_alignment() {
        assert_eq!(std::mem::size_of::<OptixSbtRecordHeader>(), 32);
        assert!(std::mem::align_of::<OptixSbtRecordHeader>() >= 16);
    }

    #[test]
    fn test_runtime_probe_passthrough() {
        let _ = OptixRuntime::probe();
    }
}
