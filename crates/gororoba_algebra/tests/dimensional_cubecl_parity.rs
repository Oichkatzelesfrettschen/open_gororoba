#![cfg(all(feature = "cubecl", feature = "vulkan"))]

use gororoba_algebra::gpu::{
    DimensionalCubeclKernel, DimensionalVulkanKernel, GpuAptResult, GpuDimensionalEngine,
};

type AptCase<'a> = (&'a str, usize, &'a [(u8, u8)], usize, u64);

fn assert_results_equal(expected: &GpuAptResult, actual: &GpuAptResult, label: &str) {
    assert_eq!(actual.dim, expected.dim, "{label}: dim");
    assert_eq!(actual.n_nodes, expected.n_nodes, "{label}: n_nodes");
    assert_eq!(actual.n_samples, expected.n_samples, "{label}: n_samples");
    assert_eq!(
        actual.pure_count, expected.pure_count,
        "{label}: pure_count"
    );
    assert_eq!(
        actual.mixed_count, expected.mixed_count,
        "{label}: mixed_count"
    );
    assert_eq!(actual.fiber_00, expected.fiber_00, "{label}: fiber_00");
    assert_eq!(actual.fiber_01, expected.fiber_01, "{label}: fiber_01");
    assert_eq!(actual.fiber_10, expected.fiber_10, "{label}: fiber_10");
    assert_eq!(actual.fiber_11, expected.fiber_11, "{label}: fiber_11");
    assert_eq!(
        actual.pure_ratio, expected.pure_ratio,
        "{label}: pure_ratio"
    );
}

#[test]
#[ignore = "requires local Vulkan compute device and cubecl-wgpu adapter"]
fn dimensional_cubecl_matches_cpu_and_vulkan_apt_cases() {
    if !DimensionalVulkanKernel::is_available() {
        eprintln!("Vulkan compute unavailable; skipping dimensional 3-way parity");
        return;
    }
    if !DimensionalCubeclKernel::is_available() {
        eprintln!("cubecl-wgpu unavailable; skipping dimensional 3-way parity");
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
        let cubecl = DimensionalCubeclKernel::compute_apt(*dim, nodes, *n_samples, *seed)
            .unwrap_or_else(|err| panic!("{label}: cubecl APT failed: {err}"));
        assert_results_equal(&cpu, &vulkan, label);
        assert_results_equal(&cpu, &cubecl, label);
        assert_results_equal(&vulkan, &cubecl, label);
    }
}
