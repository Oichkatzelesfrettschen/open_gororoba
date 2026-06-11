#![cfg(feature = "vulkan")]

use gororoba_algebra::gpu::{ImbalanceGpu, ImbalanceResult, ImbalanceVulkanKernel};

type GraphCase<'a> = (&'a str, usize, &'a [(usize, usize)], &'a [u8]);

fn assert_results_equal(cpu: &ImbalanceResult, vulkan: &ImbalanceResult, label: &str) {
    assert_eq!(vulkan.total_edges, cpu.total_edges, "{label}: total_edges");
    assert_eq!(vulkan.total_eta0, cpu.total_eta0, "{label}: total_eta0");
    assert_eq!(vulkan.total_eta1, cpu.total_eta1, "{label}: total_eta1");
    assert_eq!(vulkan.cycle_rank, cpu.cycle_rank, "{label}: cycle_rank");
    assert_eq!(
        vulkan.frustrated_count, cpu.frustrated_count,
        "{label}: frustrated_count"
    );
    assert_eq!(
        vulkan.imbalance_ratio, cpu.imbalance_ratio,
        "{label}: imbalance_ratio"
    );
}

#[test]
#[ignore = "requires local Vulkan compute device"]
fn imbalance_vulkan_matches_cpu_graph_cases() {
    if !ImbalanceVulkanKernel::is_available() {
        eprintln!("Vulkan compute unavailable; skipping imbalance Vulkan parity");
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
        assert_results_equal(&cpu, &vulkan, label);
    }
}
