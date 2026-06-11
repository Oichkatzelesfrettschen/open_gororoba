//! Tensor Core AVT contraction for high-dimensional CD algebras.
//!
//! CD multiplication is NOT standard matrix multiplication. For any fixed
//! vector `a`, the map `x -> a*x` is linear and represented by the
//! Left-Multiplication Matrix `L_a` (dim x dim), where:
//!   `L_a[i][j] = a_{i XOR j} * gamma(i XOR j, j)`
//!
//! For dim=256, `L_a` is 256x256 = 128 KB in FP16, fitting in SM shared
//! memory. It decomposes into (dim/16)^2 WMMA 16x16 tiles on the RTX 4070 Ti
//! (SM 8.9, 4th-gen Tensor Cores).
//!
//! FP16 storage + FP32 accumulate: `wmma::mma_sync` does
//! A(FP16) x B(FP16) + C(FP32) -> D(FP32).
//!
//! For CD basis elements (entries are exactly 0 or +/-1), FP16 is EXACT.
//! The FP32 accumulator preserves full single-precision for dense vectors.

mod api;
mod cpu;
#[cfg(feature = "cubecl")]
mod cubecl;
mod cuda;
mod policy;
mod sessions;
#[cfg(test)]
mod tests;
mod vulkan;

pub use api::TensorAVT;
#[cfg(feature = "cubecl")]
pub use cubecl::{TensorAvtCubeclKernel, tensor_avt_cubecl_available};
#[cfg(feature = "gpu")]
pub use cuda::{TensorAvtMulGpuWorkspace, TensorAvtNormGpuWorkspace};
pub use policy::{
    TensorAvtAutoConfig, TensorAvtAutoResult, TensorAvtCalibrationMode, TensorAvtThresholdOverrides,
};
pub use sessions::{TensorAvtMulSession, TensorAvtNormSession};
#[cfg(feature = "vulkan")]
pub use vulkan::{TensorAvtVulkanKernel, tensor_avt_vulkan_available};
