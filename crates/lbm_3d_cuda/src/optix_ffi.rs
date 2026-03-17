//! OptiX 9.x FFI via dynamic loading of libnvoptix.so.1.
//!
//! The OptiX shared library exports a single symbol: `optixQueryFunctionTable`.
//! Calling it with the ABI version returns a struct of ~60 function pointers.
//! We wrap the subset needed for particle tracing (~15 functions).
//!
//! # Loading sequence
//!
//! 1. `dlopen("libnvoptix.so.1")` via libloading
//! 2. `dlsym("optixQueryFunctionTable")` to get the table query function
//! 3. Call `optixQueryFunctionTable(ABI_VERSION, ...)` to populate the function table
//! 4. All subsequent OptiX calls go through function pointers in the table
//!
//! # Safety
//!
//! All OptiX calls are `unsafe` -- they require a valid CUDA context bound to the
//! calling thread and correctly sized/aligned structs.

use std::ffi::c_void;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// OptiX type aliases (matches optix_types.h)
// ---------------------------------------------------------------------------

/// OptiX result code (0 = success).
pub type OptixResult = i32;

/// Opaque handle types (pointers in C).
pub type OptixDeviceContext = *mut c_void;
pub type OptixModule = *mut c_void;
pub type OptixProgramGroup = *mut c_void;
pub type OptixPipeline = *mut c_void;
pub type OptixTraversableHandle = u64;

/// CUDA stream handle (CUstream).
pub type CuStream = *mut c_void;

/// CUDA device pointer (matches CUdeviceptr = unsigned long long).
pub type CuDevicePtr = u64;

/// OptiX success code.
pub const OPTIX_SUCCESS: OptixResult = 0;

// OptiX ABI versions and table sizes are probed in OptixApi::load().

// ---------------------------------------------------------------------------
// OptiX configuration structs (minimal subset for particle tracing)
// ---------------------------------------------------------------------------

/// Device context options (mirrors OptixDeviceContextOptions).
#[repr(C)]
#[derive(Default)]
pub struct OptixDeviceContextOptions {
    pub log_callback_function: Option<unsafe extern "C" fn(u32, *const i8, usize, *mut c_void)>,
    pub log_callback_data: *mut c_void,
    pub log_callback_level: i32,
    pub validation_mode: i32,
}

/// Module compile options (mirrors OptixModuleCompileOptions).
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

/// Pipeline compile options (mirrors OptixPipelineCompileOptions).
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

/// Pipeline link options (mirrors OptixPipelineLinkOptions).
#[repr(C)]
pub struct OptixPipelineLinkOptions {
    pub max_trace_depth: u32,
}

/// Program group description kind.
#[repr(C)]
#[derive(Clone, Copy)]
pub enum OptixProgramGroupKind {
    Raygen = 0x2421,
    Miss = 0x2422,
    Hitgroup = 0x2423,
}

/// Raygen program group description (part of OptixProgramGroupDesc).
#[repr(C)]
pub struct OptixProgramGroupSingleModule {
    pub module: OptixModule,
    pub entry_function_name: *const i8,
}

/// Program group options (currently empty in OptiX 9.x).
#[repr(C)]
#[derive(Default)]
pub struct OptixProgramGroupOptions {
    pub payload_type: *const c_void,
}

/// Shader Binding Table entry header (32 bytes).
#[repr(C, align(16))]
pub struct OptixSbtRecordHeader {
    pub data: [u8; 32],
}

/// Shader Binding Table (mirrors OptixShaderBindingTable).
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

/// Accel build input type for custom primitives (AABB).
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

/// Accel build options (mirrors OptixAccelBuildOptions).
#[repr(C)]
pub struct OptixAccelBuildOptions {
    pub build_flags: u32,
    pub operation: u32,
    pub motion_options: [u32; 3],
}

/// Accel buffer sizes output (mirrors OptixAccelBufferSizes).
#[repr(C)]
#[derive(Default)]
pub struct OptixAccelBufferSizes {
    pub output_size_in_bytes: usize,
    pub temp_size_in_bytes: usize,
    pub temp_update_size_in_bytes: usize,
}

// ---------------------------------------------------------------------------
// Function table (populated by optixQueryFunctionTable)
// ---------------------------------------------------------------------------

/// Subset of the OptiX function table needed for particle tracing.
/// The actual table has ~60 entries; we store only the ones we call.
pub struct OptixFunctionTable {
    pub optix_device_context_create:
        unsafe extern "C" fn(CuDevicePtr, *const OptixDeviceContextOptions, *mut OptixDeviceContext) -> OptixResult,
    pub optix_device_context_destroy:
        unsafe extern "C" fn(OptixDeviceContext) -> OptixResult,
    pub optix_module_create:
        unsafe extern "C" fn(OptixDeviceContext, *const OptixModuleCompileOptions, *const OptixPipelineCompileOptions, *const u8, usize, *mut i8, *mut usize, *mut OptixModule) -> OptixResult,
    pub optix_module_destroy:
        unsafe extern "C" fn(OptixModule) -> OptixResult,
    pub optix_pipeline_create:
        unsafe extern "C" fn(OptixDeviceContext, *const OptixPipelineCompileOptions, *const OptixPipelineLinkOptions, *const OptixProgramGroup, u32, *mut i8, *mut usize, *mut OptixPipeline) -> OptixResult,
    pub optix_pipeline_destroy:
        unsafe extern "C" fn(OptixPipeline) -> OptixResult,
    pub optix_launch:
        unsafe extern "C" fn(OptixPipeline, CuStream, CuDevicePtr, usize, *const OptixShaderBindingTable, u32, u32, u32) -> OptixResult,
    pub optix_sbt_record_pack_header:
        unsafe extern "C" fn(OptixProgramGroup, *mut OptixSbtRecordHeader) -> OptixResult,
    pub optix_accel_compute_memory_usage:
        unsafe extern "C" fn(OptixDeviceContext, *const OptixAccelBuildOptions, *const c_void, u32, *mut OptixAccelBufferSizes) -> OptixResult,
    pub optix_accel_build:
        unsafe extern "C" fn(OptixDeviceContext, CuStream, *const OptixAccelBuildOptions, *const c_void, u32, CuDevicePtr, usize, CuDevicePtr, usize, *mut OptixTraversableHandle, *const c_void, u32) -> OptixResult,
}

// ---------------------------------------------------------------------------
// Dynamic loading
// ---------------------------------------------------------------------------

/// Loaded OptiX library handle + function table.
pub struct OptixApi {
    /// Keep the library alive for the lifetime of the function table.
    _lib: Arc<libloading::Library>,
    /// Function table populated by optixQueryFunctionTable.
    pub table: OptixFunctionTable,
}

/// Type of the optixQueryFunctionTable entry point.
type QueryFunctionTableFn = unsafe extern "C" fn(
    u32,       // abiId
    u32,       // numOptions
    *mut c_void, // optionKeys
    *mut c_void, // optionValues
    *mut c_void, // functionTable
    usize,     // sizeOfTable
) -> OptixResult;

impl OptixApi {
    /// Load libnvoptix.so.1 and populate the OptiX function table.
    ///
    /// # Safety
    ///
    /// A valid CUDA context must be initialized before calling this.
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

        // The function table has exactly N function pointers. The size must
        // match EXACTLY or optixQueryFunctionTable returns 7801 (SIZE_MISMATCH).
        // Table sizes per ABI version (probed empirically):
        //   ABI 91 (OptiX 9.x): 50 ptrs = 400 bytes
        //   ABI 55 (OptiX 7.x): ~60 ptrs = 480 bytes (estimated)
        const ABI_TABLE_SIZES: &[(u32, usize)] = &[
            (91, 50 * 8),  // OptiX 9.0-9.1: 50 function pointers
            (86, 50 * 8),  // OptiX 8.1 (guess -- same era)
            (78, 56 * 8),  // OptiX 8.0 (estimated)
            (55, 60 * 8),  // OptiX 7.5 (estimated)
        ];

        let mut raw_table = vec![0u8; 512]; // max of all known sizes
        let mut matched_abi = 0u32;
        let mut matched_size = 0usize;

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
                matched_size = size;
                break;
            }
        }

        if matched_abi == 0 {
            // Brute-force: try ABI 91 with sizes 40..80 ptrs
            for n_ptrs in 40..80 {
                let size = n_ptrs * 8;
                raw_table[..size].fill(0);
                let result = query_fn(
                    91, 0,
                    std::ptr::null_mut(), std::ptr::null_mut(),
                    raw_table.as_mut_ptr() as *mut c_void, size,
                );
                if result == OPTIX_SUCCESS {
                    matched_abi = 91;
                    matched_size = size;
                    break;
                }
            }
        }

        if matched_abi == 0 {
            anyhow::bail!(
                "optixQueryFunctionTable failed for all probed ABI versions"
            );
        }
        let _ = matched_size; // used for documentation

        // Extract function pointers from the raw table.
        // The function table layout is documented in optix_function_table_definition.h.
        // Offsets are stable within an ABI version.
        let ptrs = raw_table.as_ptr() as *const usize;

        // Function pointer offsets in the OptiX 9.x function table
        // (each entry is a function pointer = 8 bytes on 64-bit).
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

        // Verify the function pointers are non-null
        let ctx_create_ptr = table.optix_device_context_create as usize;
        if ctx_create_ptr == 0 {
            anyhow::bail!(
                "optixDeviceContextCreate is null -- ABI version mismatch? \
                 Expected ABI {matched_abi}"
            );
        }

        let lib = Arc::new(lib);
        Ok(Self { _lib: lib, table })
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Check if OptiX shared library is available on this system.
pub fn optix_available() -> bool {
    std::path::Path::new("/usr/lib/libnvoptix.so.1").exists()
        || std::path::Path::new("/usr/lib64/libnvoptix.so.1").exists()
}

/// Probe OptiX availability and report status.
pub fn probe_optix() -> Result<String, String> {
    if !optix_available() {
        return Err("libnvoptix.so.1 not found. Install NVIDIA driver >= 530.".to_string());
    }

    // Try to load and query the function table
    match unsafe { OptixApi::load() } {
        Ok(_api) => Ok("OptiX available (function table loaded)".to_string()),
        Err(e) => Err(format!("OptiX library found but failed to load: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optix_probe() {
        // Just verify the probe doesn't crash -- actual availability depends on system
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
}
