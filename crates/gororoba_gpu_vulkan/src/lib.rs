//! Shared Vulkan helpers for the open_gororoba workspace.
//!
//! WHY: Prior to this crate, Vulkan init boilerplate was duplicated across
//! 10+ sites (lbm_vulkan/src/lib.rs, cd_kernel/turboquant/vulkan/quantizer.rs,
//! 7 binaries under gororoba_cli_physics/src/bin/, plus zd_resonance_sweep).
//! Each site hand-rolled Entry::load() + create_instance() + physical-device
//! enumeration + queue-family scan + descriptor-set-layout creation + compute
//! pipeline construction + command-buffer + fence + readback. This crate
//! consolidates that surface behind narrow builders that match the existing
//! call patterns 1:1.
//!
//! WHAT: Eight helper modules behind the `ash` feature flag (default off).
//! Re-exports gororoba_gpu_bridge canonical types so consumers have one
//! import surface.
//!
//! HOW: Feature-gated dependencies mean `cargo build --workspace` with no
//! features compiles only the type re-exports (~10 lines of code). Enabling
//! `--features ash` pulls in ash + naga + gpu-allocator and exposes the
//! full helper surface.
//!
//! # Universal SAFETY argument for unsafe ash::Device calls
//!
//! ash exposes the Vulkan FFI verbatim. Every device call is `unsafe` because
//! Vulkan handle validity is the caller's responsibility. The contract this
//! crate maintains:
//!
//! 1. Handles are paired with their owner: `Instance` Drop destroys, `Device`
//!    Drop destroys, `Allocator` Drop destroys, etc.
//! 2. Builders enforce required-order semantics at the type level where
//!    possible (e.g. you cannot build a `Device` without first building an
//!    `Adapter`).
//! 3. Buffer mapping + readback is fence-gated through `DispatchScope`.
//!
//! Module-level SAFETY comments document the specific per-call argument
//! that justifies the unsafe block.

// Re-export the canonical type vocabulary unconditionally so non-ash builds
// still expose `ComputeBackend`, `StoragePrecision`, etc.
pub use gororoba_gpu_bridge::{
    BufferLayout, ComputeBackend, ExecutionProfile, FrameMode, HardwareCaps, MemoryResidency,
    SimdCaps, StoragePrecision, detect_best_backend, probe_simd,
};

#[cfg(feature = "ash")]
mod adapter;
#[cfg(feature = "ash")]
mod allocator;
#[cfg(feature = "ash")]
mod buffer;
#[cfg(feature = "ash")]
mod descriptor;
#[cfg(feature = "ash")]
mod device;
#[cfg(feature = "ash")]
mod dispatch;
#[cfg(feature = "ash")]
mod error;
#[cfg(feature = "ash")]
mod instance;
#[cfg(feature = "ash")]
mod pipeline;
#[cfg(feature = "ash")]
mod shader;

#[cfg(feature = "ash")]
pub use adapter::{Adapter, QueueFamilyRequirement};
#[cfg(feature = "ash")]
pub use allocator::Allocator;
#[cfg(feature = "ash")]
pub use buffer::HostVisibleBuffer;
#[cfg(feature = "ash")]
pub use descriptor::{
    DescriptorPool, DescriptorSet, DescriptorSetLayout, DescriptorSetLayoutSpec, DescriptorType,
};
#[cfg(feature = "ash")]
pub use device::{Device, DeviceBuilder, DeviceFeatures};
#[cfg(feature = "ash")]
pub use dispatch::DispatchScope;
#[cfg(feature = "ash")]
pub use error::VulkanError;
#[cfg(feature = "ash")]
pub use instance::{Instance, InstanceBuilder, ValidationPolicy};
#[cfg(feature = "ash")]
pub use pipeline::{ComputePipeline, ComputePipelineBuilder};
#[cfg(feature = "ash")]
pub use shader::ShaderModule;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn re_exports_compile() {
        // Sanity check: the canonical types from gororoba_gpu_bridge are
        // reachable through this crate even when `ash` is disabled.
        let _: ComputeBackend = ComputeBackend::CpuScalar;
        let _: StoragePrecision = StoragePrecision::Fp32;
    }
}
