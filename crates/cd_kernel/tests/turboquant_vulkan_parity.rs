//! Parity test: Backend::Cpu vs Backend::Vulkan for turboquant.quantize.
//!
//! Verifies that the Vulkan compute pipeline produces bit-identical
//! u8 indices to the CPU reference path for a synthetic 1024-element
//! input. Gated `#[ignore = "gpu"]` because the test requires a Vulkan
//! ICD (libvulkan.so + a compute-capable adapter); CI without GPU
//! support skips it via `cargo test -- --ignored` opt-in.
//!
//! Run with:
//!   cargo test -p cd_kernel --features vulkan --release \
//!     turboquant_vulkan_parity -- --ignored --nocapture

#![cfg(feature = "vulkan")]

use cd_kernel::{
    lloyd_max::get_codebook,
    turboquant::{
        backend::{Backend, BackendQuantizer},
        dispatch::SimdLevel,
        vulkan::VulkanQuantizer,
    },
};

#[test]
#[ignore = "gpu (Vulkan ICD + compute-capable adapter required)"]
fn vulkan_parity_3bit_1024() {
    // Skip cleanly if Vulkan is not available at runtime.
    if !VulkanQuantizer::is_available() {
        eprintln!(
            "skip: VulkanQuantizer::is_available() is false (no libvulkan / no \
             compute device / glslc was missing at build)"
        );
        return;
    }

    // d=128 + bits=3 is a typical TurboQuant configuration.
    let bits = 3;
    let codebook = get_codebook(128, bits);

    // Synthetic input: 1024 floats from a fixed seed (reproducibility).
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let normal = StandardNormal;
    let values: Vec<f32> = (0..1024)
        .map(|_| <StandardNormal as Distribution<f32>>::sample(&normal, &mut rng))
        .collect();

    // CPU reference.
    let cpu_bq = BackendQuantizer::with_backend(&codebook, bits, Backend::Cpu(SimdLevel::Scalar));
    let mut cpu_indices = vec![0u8; values.len()];
    cpu_bq.quantize(&values, &mut cpu_indices);

    // Vulkan path. The recommended_shader_tier is irrelevant for the
    // current single-shader implementation; pass any tier value.
    let vk_bq = BackendQuantizer::with_backend(
        &codebook,
        bits,
        Backend::Vulkan(cd_kernel::turboquant::vulkan::context::VulkanShaderTier::Portable),
    );
    let mut vk_indices = vec![0u8; values.len()];
    vk_bq.quantize(&values, &mut vk_indices);

    // Bit-identical assertion.
    if cpu_indices != vk_indices {
        let mismatches: Vec<(usize, u8, u8)> = cpu_indices
            .iter()
            .zip(vk_indices.iter())
            .enumerate()
            .filter(|(_, (c, v))| c != v)
            .map(|(i, (c, v))| (i, *c, *v))
            .take(10)
            .collect();
        panic!(
            "CPU vs Vulkan indices diverge at {} positions; first 10: {:?}",
            cpu_indices
                .iter()
                .zip(vk_indices.iter())
                .filter(|(c, v)| c != v)
                .count(),
            mismatches
        );
    }
}
