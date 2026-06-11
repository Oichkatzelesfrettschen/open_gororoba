#![cfg(feature = "vulkan")]

use gororoba_algebra::gpu::{DimensionalVulkanKernel, GpuAptResult, GpuDimensionalEngine};

type AptCase<'a> = (&'a str, usize, &'a [(u8, u8)], usize, u64);

fn assert_results_equal(cpu: &GpuAptResult, vulkan: &GpuAptResult, label: &str) {
    assert_eq!(vulkan.dim, cpu.dim, "{label}: dim");
    assert_eq!(vulkan.n_nodes, cpu.n_nodes, "{label}: n_nodes");
    assert_eq!(vulkan.n_samples, cpu.n_samples, "{label}: n_samples");
    assert_eq!(vulkan.pure_count, cpu.pure_count, "{label}: pure_count");
    assert_eq!(vulkan.mixed_count, cpu.mixed_count, "{label}: mixed_count");
    assert_eq!(vulkan.fiber_00, cpu.fiber_00, "{label}: fiber_00");
    assert_eq!(vulkan.fiber_01, cpu.fiber_01, "{label}: fiber_01");
    assert_eq!(vulkan.fiber_10, cpu.fiber_10, "{label}: fiber_10");
    assert_eq!(vulkan.fiber_11, cpu.fiber_11, "{label}: fiber_11");
    assert_eq!(vulkan.pure_ratio, cpu.pure_ratio, "{label}: pure_ratio");
}

#[test]
#[ignore = "requires local Vulkan compute device"]
fn dimensional_vulkan_matches_cpu_apt_cases() {
    if !DimensionalVulkanKernel::is_available() {
        eprintln!("Vulkan compute unavailable; skipping dimensional Vulkan parity");
        return;
    }

    let dim16_nodes = &[(1u8, 8u8), (2, 9), (3, 10), (4, 11), (5, 12)];
    let dim32_nodes = &[(1u8, 16u8), (2, 17), (3, 18), (4, 19), (5, 20), (6, 21)];
    let cases: &[AptCase<'_>] = &[
        ("dim16_seed42", 16, dim16_nodes, 256, 42),
        ("dim16_seed99", 16, dim16_nodes, 257, 99),
        ("dim32_seed42", 32, dim32_nodes, 384, 42),
    ];

    for (label, dim, nodes, n_samples, seed) in cases {
        let cpu = GpuDimensionalEngine::compute_apt_cpu(*dim, nodes, *n_samples, *seed)
            .unwrap_or_else(|err| panic!("{label}: CPU APT failed: {err}"));
        let vulkan = DimensionalVulkanKernel::compute_apt(*dim, nodes, *n_samples, *seed)
            .unwrap_or_else(|err| panic!("{label}: Vulkan APT failed: {err}"));
        assert_results_equal(&cpu, &vulkan, label);
    }
}
