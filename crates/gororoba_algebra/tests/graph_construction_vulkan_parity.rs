#![cfg(feature = "vulkan")]

use gororoba_algebra::gpu::{GraphConstructionVulkanKernel, GraphConstructorGpu};

type GraphCase<'a> = (&'a str, usize, &'a [u8], &'a [(u8, u8)]);

fn checkerboard_eta(dim: usize) -> Vec<u8> {
    let dim_half = dim / 2;
    let mut eta = vec![0u8; dim_half * dim_half];
    for row in 0..dim_half {
        for col in 0..dim_half {
            eta[row * dim_half + col] = ((row + col) % 2) as u8;
        }
    }
    eta
}

fn diagonal_eta(dim: usize) -> Vec<u8> {
    let dim_half = dim / 2;
    let mut eta = vec![0u8; dim_half * dim_half];
    for row in 0..dim_half {
        eta[row * dim_half + row] = 1;
    }
    eta
}

#[test]
#[ignore = "requires local Vulkan compute device"]
fn graph_construction_vulkan_matches_cpu_cases() {
    if !GraphConstructionVulkanKernel::is_available() {
        eprintln!("Vulkan compute unavailable; skipping graph construction Vulkan parity");
        return;
    }

    let zero_eta = vec![0u8; 16];
    let checkerboard = checkerboard_eta(8);
    let diagonal = diagonal_eta(8);
    let cases: &[GraphCase<'_>] = &[
        ("zero_eta", 8, &zero_eta, &[(0, 1), (2, 3), (1, 2)]),
        (
            "checkerboard",
            8,
            &checkerboard,
            &[(0, 1), (2, 3), (0, 2), (1, 3)],
        ),
        (
            "skips_out_of_half_nodes",
            8,
            &diagonal,
            &[(0, 1), (2, 3), (4, 0), (1, 2)],
        ),
    ];

    for (label, dim, eta, nodes) in cases {
        let mut cpu = GraphConstructorGpu::find_edges(*dim, eta, nodes)
            .unwrap_or_else(|err| panic!("{label}: CPU graph construction failed: {err}"));
        cpu.sort_unstable();
        let vulkan = GraphConstructionVulkanKernel::find_edges(*dim, eta, nodes)
            .unwrap_or_else(|err| panic!("{label}: Vulkan graph construction failed: {err}"));
        assert_eq!(vulkan, cpu, "{label}: edge list mismatch");
    }
}
