//! GPU compute ecosystem evaluation: current CUDA+Vulkan vs cubecl.
//!
//! Current state: separate CUDA (cudarc JIT) and Vulkan (ash + GLSL) backends.
//! cubecl from the Burn team could unify them into one kernel definition.
//!
//! # Comparison
//!
//! | Feature                | CUDA+Vulkan (current) | cubecl           | WGPU/Burn      |
//! |------------------------|----------------------|------------------|----------------|
//! | Write kernel once      | No (separate .cu/.comp)| **Yes**         | Burn tensors   |
//! | CUDA support           | Yes (NVRTC JIT)      | Yes              | Via wgpu       |
//! | Vulkan support         | Yes (GLSL shaders)   | Yes (SPIR-V)     | Yes            |
//! | WebGPU support         | No                   | **Yes**          | **Yes**         |
//! | Metal support          | No                   | **Yes**          | **Yes**         |
//! | Tensor Core access     | Via PTX inline       | Auto-vectorize   | Via Burn       |
//! | Autotune               | Manual selector      | **Built-in**     | Via Burn       |
//! | JIT compilation        | Yes (NVRTC)          | Yes (multi-back) | Yes            |
//! | Integration effort     | Already done         | Medium           | High (Burn dep)|
//! | Dependency weight      | cudarc + ash         | cubecl           | wgpu + burn    |
//!
//! # Current GPU Backend Inventory
//!
//! Files that would be unified by cubecl:
//! - cuda/device.rs (29 LOC relevant) -> cubecl device probe
//! - cuda/jit.rs (65 LOC) -> cubecl JIT
//! - cuda/launch.rs (90 LOC) -> cubecl launch
//! - cuda/kernels/turboquant.cu (240 LOC) -> cubecl kernel definitions
//! - vulkan/context.rs (165 LOC) -> cubecl context
//! - vulkan/shaders/quantize.comp (55 LOC) -> merged into cubecl kernel
//! - vulkan/shaders/dequant_dot.comp (80 LOC) -> merged into cubecl kernel
//!
//! Total: ~725 LOC of backend-specific code that could become ~300 LOC of
//! cubecl kernel definitions with automatic multi-backend dispatch.
//!
//! # Recommendation
//!
//! cubecl is the right long-term choice:
//! 1. Eliminates ~400 LOC of backend duplication
//! 2. Adds WebGPU and Metal support for free
//! 3. Built-in autotune replaces our kernel_selector.rs
//! 4. JIT compilation is native (no NVRTC dependency)
//!
//! However, migration should wait until:
//! - cubecl reaches 1.0 (currently pre-1.0)
//! - Our CUDA/Vulkan kernels are proven in production
//! - The kernel API stabilizes
//!
//! Migration plan when ready:
//! 1. Add cubecl as optional dependency: `cubecl = ["dep:cubecl"]`
//! 2. Write cubecl kernel equivalents for the 5 CUDA kernels
//! 3. Add `Backend::CubeCL` variant to backend.rs
//! 4. Feature-gate: old backends stay available, cubecl is opt-in
//! 5. Once cubecl is proven, deprecate cuda/ and vulkan/ directories

/// Summary of GPU backend unification assessment.
pub struct GpuEvaluation {
    pub current_backends: Vec<&'static str>,
    pub current_kernel_count: usize,
    pub current_loc: usize,
    pub cubecl_estimated_loc: usize,
    pub loc_savings: usize,
    pub new_platforms_gained: Vec<&'static str>,
    pub recommendation: &'static str,
    pub migration_blocked_by: &'static str,
}

impl GpuEvaluation {
    pub fn assess() -> Self {
        GpuEvaluation {
            current_backends: vec!["CUDA (cudarc NVRTC)", "Vulkan (ash + GLSL)"],
            current_kernel_count: 5, // quantize, dequant_dot, sign_dot, fast_jl, dequant_dot_q16
            current_loc: 725,
            cubecl_estimated_loc: 300,
            loc_savings: 425,
            new_platforms_gained: vec!["WebGPU", "Metal", "ROCm (via cubecl)"],
            recommendation: "Adopt cubecl when it reaches 1.0; keep current backends as fallback",
            migration_blocked_by: "cubecl pre-1.0 API instability",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_evaluation() {
        let eval = GpuEvaluation::assess();
        println!("\n=== GPU Backend Evaluation ===");
        println!("Current backends: {:?}", eval.current_backends);
        println!("Current kernels: {}", eval.current_kernel_count);
        println!("Current LOC: {}", eval.current_loc);
        println!("cubecl estimated LOC: {}", eval.cubecl_estimated_loc);
        println!(
            "LOC savings: {} ({:.0}%)",
            eval.loc_savings,
            eval.loc_savings as f64 / eval.current_loc as f64 * 100.0
        );
        println!("New platforms: {:?}", eval.new_platforms_gained);
        println!("Recommendation: {}", eval.recommendation);
        println!("Blocked by: {}", eval.migration_blocked_by);

        assert!(eval.loc_savings > 0);
        assert!(eval.new_platforms_gained.len() >= 2);
    }
}
