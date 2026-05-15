//! OptiX pipeline scaffolding.
//!
//! Lifts the reusable half of `lbm_3d_cuda/src/optix_pipeline.rs` --
//! the device-context + module-compile setup. SBT layout and
//! ray-tracing orchestration stay in lbm_3d_cuda (LBM-specific:
//! particle seeding, BVH rebuild policy, trajectory logging).
//!
//! This module is intentionally thin: only `PipelineBuilder` is
//! exposed today; expansion is gated on a second OptiX consumer
//! appearing. Until then the LBM pipeline carries the full machinery.

/// Builder for the OptiX device-context + module-compile half of a
/// pipeline. Concrete pipeline construction (program groups, SBT
/// layout, raygen + closesthit ordering) is domain-specific and lives
/// at the consumer.
#[derive(Default)]
pub struct PipelineBuilder {
    log_level: u32,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self { log_level: 0 }
    }

    /// Set the OptiX log callback level (0=disable, 1=fatal, 2=error,
    /// 3=warning, 4=print).
    pub fn log_level(mut self, level: u32) -> Self {
        self.log_level = level;
        self
    }

    /// Current log level. Domain-specific pipeline builders read this
    /// when calling `optixDeviceContextCreate`.
    pub fn current_log_level(&self) -> u32 {
        self.log_level
    }
}
