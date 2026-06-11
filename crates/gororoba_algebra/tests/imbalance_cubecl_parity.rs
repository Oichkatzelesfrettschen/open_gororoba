#![cfg(all(feature = "cubecl", feature = "vulkan"))]

use gororoba_algebra::gpu::{
    ImbalanceCubeclKernel, ImbalanceGpu, ImbalanceResult, ImbalanceVulkanKernel,
};

type GraphCase<'a> = (&'a str, usize, &'a [(usize, usize)], &'a [u8]);

fn assert_results_equal(expected: &ImbalanceResult, actual: &ImbalanceResult, label: &str) {
    assert_eq!(
        actual.total_edges, expected.total_edges,
        "{label}: total_edges"
    );
    assert_eq!(
        actual.total_eta0, expected.total_eta0,
        "{label}: total_eta0"
    );
    assert_eq!(
        actual.total_eta1, expected.total_eta1,
        "{label}: total_eta1"
    );
    assert_eq!(
        actual.cycle_rank, expected.cycle_rank,
        "{label}: cycle_rank"
    );
    assert_eq!(
        actual.frustrated_count, expected.frustrated_count,
        "{label}: frustrated_count"
    );
    assert_eq!(
        actual.imbalance_ratio, expected.imbalance_ratio,
        "{label}: imbalance_ratio"
    );
}

#[test]
#[ignore = "requires local Vulkan compute device and cubecl-wgpu adapter"]
fn imbalance_cubecl_matches_cpu_and_vulkan_graph_cases() {
    if !ImbalanceVulkanKernel::is_available() {
        eprintln!("Vulkan compute unavailable; skipping imbalance 3-way parity");
        return;
    }
    if !ImbalanceCubeclKernel::is_available() {
        eprintln!("cubecl-wgpu unavailable; skipping imbalance 3-way parity");
        return;
    }

    let cases: &[GraphCase<'_>] = &[
        ("line_no_cycle", 4, &[(0, 1), (1, 2), (2, 3)], &[0, 0, 0]),
        (
            "triangle_frustrated",
            3,
            &[(0, 1), (1, 2), (2, 0)],
            &[0, 0, 1],
        ),
        (
            "square_balanced",
            4,
            &[(0, 1), (1, 2), (2, 3), (3, 0)],
            &[1, 0, 1, 0],
        ),
        (
            "two_cycle_component",
            5,
            &[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)],
            &[0, 1, 1, 1, 1, 1],
        ),
    ];

    for (label, n_nodes, edges, eta_values) in cases {
        let cpu = ImbalanceGpu::compute_imbalance_cpu(edges, *n_nodes, eta_values)
            .unwrap_or_else(|err| panic!("{label}: CPU imbalance failed: {err}"));
        let vulkan = ImbalanceVulkanKernel::compute(edges, *n_nodes, eta_values)
            .unwrap_or_else(|err| panic!("{label}: Vulkan imbalance failed: {err}"));
        let cubecl = ImbalanceCubeclKernel::compute(edges, *n_nodes, eta_values)
            .unwrap_or_else(|err| panic!("{label}: cubecl imbalance failed: {err}"));
        assert_results_equal(&cpu, &vulkan, label);
        assert_results_equal(&cpu, &cubecl, label);
        assert_results_equal(&vulkan, &cubecl, label);
    }
}
