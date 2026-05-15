//! Error type for Vulkan helper failures.
//!
//! WHY: ash returns `vk::Result` enums and `ash::LoadingError`; gpu-allocator
//! returns `AllocationError`; naga returns three different error types
//! (`ParseError`, `ValidationError`, `SpvOutError`). The workspace's prior
//! call sites variously wrapped these in `anyhow::Result`, `Result<T,String>`,
//! or `Option<T>`. This crate consolidates them under a single error type
//! with `From` impls so callers see one error surface.

use thiserror::Error;

/// All errors produced by the Vulkan helpers.
#[derive(Debug, Error)]
pub enum VulkanError {
    /// Failed to load the Vulkan loader (libvulkan.so / vulkan-1.dll absent).
    #[error("Vulkan loader missing or failed: {0}")]
    LoaderLoad(#[from] ash::LoadingError),

    /// Vulkan API returned a non-success code.
    #[error("Vulkan API error: {0}")]
    Vk(#[from] ash::vk::Result),

    /// gpu-allocator failed to allocate / free / map memory.
    #[error("gpu-allocator error: {0}")]
    Allocator(#[from] gpu_allocator::AllocationError),

    /// WGSL source failed to parse via naga.
    #[error("WGSL parse error: {0}")]
    WgslParse(String),

    /// WGSL parsed but failed naga validation.
    #[error("WGSL validation error: {0}")]
    WgslValidation(String),

    /// naga failed to emit SPIR-V from a valid WGSL module.
    #[error("SPIR-V emit error: {0}")]
    SpirvEmit(String),

    /// No physical device matched the requested QueueFamilyRequirement.
    #[error("No physical device matched queue family requirement: {0:?}")]
    NoMatchingPhysicalDevice(crate::QueueFamilyRequirement),

    /// No queue family on the picked device matched the requirement.
    /// (Should be unreachable since the picker already filters by this, but
    /// retained for defense-in-depth in builder code paths.)
    #[error("No queue family matched on picked device for requirement: {0:?}")]
    NoMatchingQueueFamily(crate::QueueFamilyRequirement),

    /// A required Vulkan extension or feature is unsupported on the device.
    #[error("Unsupported Vulkan feature: {0}")]
    UnsupportedFeature(&'static str),

    /// A timeout was hit waiting on a fence or queue submission.
    #[error("Vulkan operation timed out after {timeout_ns} ns: {context}")]
    Timeout {
        timeout_ns: u64,
        context: &'static str,
    },
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, VulkanError>;
