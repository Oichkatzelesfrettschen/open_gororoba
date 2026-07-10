//! OptiX 9.1 pipeline initialization and management.
//!
//! Provides the full host-side OptiX API wrapper for particle tracing:
//! - Context creation from CUDA device
//! - Module compilation from PTX (NVRTC-compiled optix_tracer.cu)
//! - Program group creation (raygen, closesthit+intersection, miss)
//! - Pipeline linking
//! - SBT construction with zero-copy LBM device pointers
//! - Launch orchestration with CUDA stream interop
//!
//! # Architecture
//!
//! The OptiX pipeline runs on RT cores concurrently with LBM on SMs:
//! ```text
//! Stream A (SMs): LBM step N -> LBM step N+1 -> LBM step N+2 -> ...
//! Stream B (RT):              trace N      -> (idle)     -> trace N+2 -> ...
//! BVH rebuild:    (every 20 steps) -> scan + accelBuild -> (reuse GAS)
//! ```
//!
//! # Feature gate
//!
//! Requires `optix` feature flag. OptiX 9.1 SDK headers at `/usr/include/optix/`.

use crate::optix_tracer::{BrickResult, LbmSbtData, OptiXTracerConfig};
use gororoba_sparse_grid::{BrickGrid3d, BrickShape3d, LogicalGrid3d, OccupancyBitsetStats};

/// OptiX pipeline configuration.
#[derive(Debug, Clone)]
pub struct OptiXPipelineConfig {
    /// Maximum trace depth (1 for particle advection -- no recursive tracing).
    pub max_trace_depth: u32,
    /// Maximum number of tracer particles.
    pub max_particles: u32,
    /// OptiX log verbosity (0-4).
    pub log_level: u32,
    /// CUDA stream index for OptiX launches (0 = default stream).
    pub optix_stream_id: u32,
}

impl Default for OptiXPipelineConfig {
    fn default() -> Self {
        Self {
            max_trace_depth: 1,
            max_particles: 100_000,
            log_level: 0,
            optix_stream_id: 1, // separate stream from LBM
        }
    }
}

/// Represents the compiled OptiX pipeline ready for launch.
///
/// Holds all OptiX handles (context, module, pipeline, SBT buffers).
/// These are opaque u64 handles in Rust (matching OptixModule etc. in C).
///
/// The actual OptiX API calls require FFI to the OptiX shared library.
/// This struct provides the Rust-side state management; the FFI calls
/// are handled by the `optix_ffi` module, which binds optix_stubs.h
/// through bindgen or manual C FFI.
#[derive(Debug)]
pub struct OptiXPipeline {
    /// Pipeline configuration.
    pub config: OptiXPipelineConfig,
    /// Tracer configuration (grid, thresholds, timing).
    pub tracer_config: OptiXTracerConfig,
    /// SBT data for the hit group record.
    pub sbt_data: LbmSbtData,
    /// PTX source for the OptiX module (compiled from optix_tracer.cu).
    pub ptx_source: String,
    /// Current GAS (Geometry Acceleration Structure) validity.
    pub gas_valid: bool,
    /// Last BVH rebuild step number.
    pub last_rebuild_step: u64,
    /// Last measured occupancy fraction at rebuild time.
    pub last_rebuild_occupancy: f64,
    /// Cached brick AABBs from the last scan.
    pub cached_bricks: Vec<BrickResult>,
}

impl OptiXPipeline {
    /// Create a new OptiX pipeline configuration.
    ///
    /// Does NOT initialize the OptiX context -- call `initialize()` for that.
    /// This separation allows configuration before GPU resource allocation.
    pub fn new(
        pipeline_config: OptiXPipelineConfig,
        tracer_config: OptiXTracerConfig,
        velocity_device_ptr: u64,
        density_device_ptr: u64,
    ) -> Self {
        let sbt_data =
            LbmSbtData::from_config(&tracer_config, velocity_device_ptr, density_device_ptr);
        Self {
            config: pipeline_config,
            tracer_config,
            sbt_data,
            ptx_source: String::new(),
            gas_valid: false,
            last_rebuild_step: 0,
            last_rebuild_occupancy: 0.0,
            cached_bricks: Vec::new(),
        }
    }

    /// Set the PTX source (compiled from optix_tracer.cu via NVRTC).
    pub fn set_ptx(&mut self, ptx: String) {
        self.ptx_source = ptx;
    }

    /// Update the SBT device pointers (call when LBM buffers are reallocated).
    pub fn update_device_pointers(&mut self, velocity_device_ptr: u64, density_device_ptr: u64) {
        self.sbt_data =
            LbmSbtData::from_config(&self.tracer_config, velocity_device_ptr, density_device_ptr);
    }

    /// Check if the BVH needs rebuilding based on step count and occupancy delta.
    pub fn needs_rebuild(&self, current_step: u64, current_occupancy: f64) -> bool {
        crate::optix_tracer::should_rebuild_bvh(
            current_step,
            &self.tracer_config,
            current_occupancy,
            self.last_rebuild_occupancy,
        )
    }

    /// Update brick AABBs from a density field (CPU fallback path).
    /// For GPU path, use the CUDA brick scan kernel from optix_brick_scan.cu.
    pub fn rebuild_bvh_cpu(&mut self, rho: &[f32], current_step: u64) {
        self.cached_bricks = crate::optix_tracer::build_brick_aabbs(rho, &self.tracer_config);
        self.last_rebuild_step = current_step;
        self.last_rebuild_occupancy =
            crate::optix_tracer::occupancy_fraction(rho, self.tracer_config.rho_threshold);
        self.gas_valid = true;
    }

    /// Number of active bricks in the current BVH.
    pub fn active_brick_count(&self) -> usize {
        self.cached_bricks.len()
    }

    /// Occupancy stats for the current cached brick set.
    #[must_use]
    pub fn cached_brick_occupancy_stats(&self) -> OccupancyBitsetStats {
        let (nx, ny, nz) = self.tracer_config.grid_dim;
        let brick_grid = BrickGrid3d::from_logical_grid(
            LogicalGrid3d { nx, ny, nz },
            BrickShape3d {
                core_edge_cells: self.tracer_config.brick_size,
                halo_edge_cells: self.tracer_config.brick_size + 2,
            },
        );
        OccupancyBitsetStats {
            total_bricks: brick_grid.total_bricks(),
            active_bricks: self.cached_bricks.len() as u64,
        }
    }

    /// Extract AABB data as a flat array suitable for optixAccelBuild input.
    /// Returns (aabb_data, n_primitives) where aabb_data is 6 floats per AABB.
    pub fn aabb_build_input(&self) -> (Vec<f32>, usize) {
        let n = self.cached_bricks.len();
        let mut data = Vec::with_capacity(n * 6);
        for brick in &self.cached_bricks {
            data.push(brick.aabb.min_x);
            data.push(brick.aabb.min_y);
            data.push(brick.aabb.min_z);
            data.push(brick.aabb.max_x);
            data.push(brick.aabb.max_y);
            data.push(brick.aabb.max_z);
        }
        (data, n)
    }

    /// Generate the OptiX pipeline initialization sequence as a documentation string.
    ///
    /// This describes the exact API calls needed for the FFI implementation.
    /// Each step maps to one or more OptiX C API functions.
    pub fn initialization_sequence() -> &'static str {
        r#"OptiX Pipeline Initialization Sequence:

1. optixInit()
   - Loads the OptiX shared library and populates the function table.
   - Must be called once per process before any other OptiX call.

2. optixDeviceContextCreate(CUcontext, &options, &context)
   - Creates an OptiX context from the CUDA context.
   - CUcontext obtained from the current CUDA context.

3. NVRTC compile optix_tracer.cu -> PTX string
   - Use gororoba_gpu_cuda::CompileOptions with include paths for optix headers.
   - Pass -I/usr/include/optix as a compile option.

4. optixModuleCreate(context, &module_options, ptx, ptx_len, log, &module)
   - Creates an OptiX module from the PTX.
   - module_options: maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT.

5. optixProgramGroupCreate(context, &desc, 1, &options, log, &group) x3
   - Raygen group: entry = "__raygen__trace_particles"
   - HitGroup: closesthit = "__closesthit__cell", intersection = "__intersection__aabb"
   - Miss group: entry = "__miss__void"

6. optixPipelineCreate(context, &compile_options, &link_options, groups, 3, log, &pipeline)
   - compile_options: numPayloadValues = 2, numAttributeValues = 0.
   - link_options: maxTraceDepth = 1.

7. Allocate SBT buffers on device:
   - raygen_record: sizeof(SbtRecordHeader) + 0 data
   - hitgroup_record: sizeof(SbtRecordHeader) + sizeof(LbmSbtData)
   - miss_record: sizeof(SbtRecordHeader) + 0 data
   - Pack LbmSbtData into hitgroup_record after the header.

8. optixSbtRecordPackHeader(raygen_group, &raygen_record)
   optixSbtRecordPackHeader(hitgroup_group, &hitgroup_record)
   optixSbtRecordPackHeader(miss_group, &miss_record)

9. Build GAS:
   - optixAccelBuild(context, stream, &build_options, &build_input, 1,
                     temp_buffer, temp_size, output_buffer, output_size, &gas_handle, NULL, 0)
   - build_input: OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES with AABB data from aabb_build_input().

10. optixLaunch(pipeline, stream, launch_params_dptr, sizeof(LaunchParams), &sbt, n_particles, 1, 1)
    - launch_params_dptr: device pointer to LaunchParams struct (positions, velocities, brick_indices, traversable, n_particles, dt).
    - sbt: { raygen_record, miss_record, hitgroup_record }.
"#
    }
}

/// OptiX compilation options for the tracer module.
pub struct OptiXCompileOptions {
    /// Include paths for NVRTC (must include OptiX headers).
    pub include_paths: Vec<String>,
    /// CUDA architecture target (e.g., "sm_89").
    pub arch: String,
    /// Additional NVRTC flags.
    pub extra_flags: Vec<String>,
}

impl Default for OptiXCompileOptions {
    fn default() -> Self {
        Self {
            include_paths: vec!["/usr/include/optix".to_string(), "/usr/include".to_string()],
            arch: "sm_89".to_string(),
            extra_flags: vec!["--use_fast_math".to_string(), "-default-device".to_string()],
        }
    }
}

// ---------------------------------------------------------------------------
// Live OptiX pipeline (requires libnvoptix.so.1 at runtime)
// ---------------------------------------------------------------------------

use gororoba_optix::{self, OptixRuntime};

/// Live OptiX pipeline with loaded API handles.
///
/// Wraps the configuration-only `OptiXPipeline` with an actual OptiX runtime
/// obtained via `gororoba_optix::OptixRuntime`. Creating this struct
/// initializes the OptiX device context and validates the API is loadable.
///
/// The full pipeline build (module compile, program groups, SBT, GAS) is
/// deferred to `build()` because it requires OptiX SDK headers for NVRTC
/// compilation of `optix_tracer.cu`. Without the headers, the context
/// creation still validates that the driver and RT cores are accessible.
pub struct LiveOptiXPipeline {
    /// Generic OptiX runtime boundary (FFI + device context ownership).
    pub runtime: OptixRuntime,
    /// Configuration-side pipeline state (bricks, SBT data, etc.).
    pub config_pipeline: OptiXPipeline,
}

impl LiveOptiXPipeline {
    /// Initialize the OptiX API and create a device context.
    ///
    /// This performs steps 1-2 of the initialization sequence:
    /// 1. Load libnvoptix.so.1 and populate function table
    /// 2. Create OptiX device context from the current CUDA context
    ///
    /// # Safety
    ///
    /// A CUDA context must be current on the calling thread before creating
    /// the OptiX device context.
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn init(
        pipeline_config: OptiXPipelineConfig,
        tracer_config: OptiXTracerConfig,
        velocity_device_ptr: u64,
        density_device_ptr: u64,
    ) -> anyhow::Result<Self> {
        let runtime = OptixRuntime::init_current_context(pipeline_config.log_level)?;

        let config_pipeline = OptiXPipeline::new(
            pipeline_config,
            tracer_config,
            velocity_device_ptr,
            density_device_ptr,
        );

        Ok(Self {
            runtime,
            config_pipeline,
        })
    }

    /// Check if the OptiX API is available and loadable.
    ///
    /// Lighter than `init()` -- just probes the shared library without
    /// creating a device context.
    pub fn probe() -> anyhow::Result<String> {
        OptixRuntime::probe()
    }
}

/// The OptiX tracer module source (included at compile time).
pub const OPTIX_TRACER_CU_SRC: &str = include_str!("optix_tracer.cu");

/// The brick scan kernel source (included at compile time).
pub const OPTIX_BRICK_SCAN_CU_SRC: &str = include_str!("optix_brick_scan.cu");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let config = OptiXPipelineConfig::default();
        let tracer = OptiXTracerConfig::default();
        let pipeline = OptiXPipeline::new(config, tracer, 0xDEAD, 0xBEEF);
        assert!(!pipeline.gas_valid);
        assert_eq!(pipeline.active_brick_count(), 0);
        assert_eq!(pipeline.sbt_data.velocity_field_ptr, 0xDEAD);
    }

    #[test]
    fn test_aabb_build_input_empty() {
        let config = OptiXPipelineConfig::default();
        let tracer = OptiXTracerConfig::default();
        let pipeline = OptiXPipeline::new(config, tracer, 0, 0);
        let (data, n) = pipeline.aabb_build_input();
        assert_eq!(n, 0);
        assert!(data.is_empty());
    }

    #[test]
    fn test_rebuild_bvh_cpu() {
        let config = OptiXPipelineConfig::default();
        let tracer = OptiXTracerConfig {
            grid_dim: (16, 16, 16),
            brick_size: 8,
            rho_threshold: 0.5,
            ..Default::default()
        };
        let mut pipeline = OptiXPipeline::new(config, tracer, 0, 0);
        let rho = vec![1.0f32; 16 * 16 * 16];
        pipeline.rebuild_bvh_cpu(&rho, 0);
        assert!(pipeline.gas_valid);
        assert_eq!(pipeline.active_brick_count(), 8); // 2^3 bricks
        let (data, n) = pipeline.aabb_build_input();
        assert_eq!(n, 8);
        assert_eq!(data.len(), 48); // 8 * 6 floats
    }

    #[test]
    fn test_needs_rebuild() {
        let config = OptiXPipelineConfig::default();
        let tracer = OptiXTracerConfig {
            bvh_rebuild_interval: 20,
            rebuild_occupancy_delta: 0.05,
            ..Default::default()
        };
        let mut pipeline = OptiXPipeline::new(config, tracer, 0, 0);
        pipeline.last_rebuild_occupancy = 0.5;
        assert!(pipeline.needs_rebuild(0, 0.5)); // step 0 always rebuilds
        assert!(!pipeline.needs_rebuild(5, 0.52)); // delta < 0.05
        assert!(pipeline.needs_rebuild(5, 0.6)); // delta > 0.05
        assert!(pipeline.needs_rebuild(20, 0.5)); // periodic
    }

    #[test]
    fn test_optix_sources_included() {
        assert!(OPTIX_TRACER_CU_SRC.contains("__raygen__trace_particles"));
        assert!(OPTIX_BRICK_SCAN_CU_SRC.contains("scan_brick_occupancy"));
    }

    #[test]
    fn test_live_optix_probe() {
        // Initialize CUDA context first -- OptiX requires CUDA driver init.
        let _ctx = gororoba_gpu_cuda::Context::with_default_device();
        match LiveOptiXPipeline::probe() {
            Ok(msg) => eprintln!("OptiX probe: {msg}"),
            Err(e) => eprintln!("OptiX probe: {e} (expected without GPU/CI)"),
        }
    }

    #[test]
    fn test_initialization_sequence_documented() {
        let seq = OptiXPipeline::initialization_sequence();
        assert!(seq.contains("optixInit"));
        assert!(seq.contains("optixModuleCreate"));
        assert!(seq.contains("optixPipelineCreate"));
        assert!(seq.contains("optixLaunch"));
    }
}
