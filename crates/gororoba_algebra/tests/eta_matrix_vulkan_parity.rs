#![cfg(feature = "vulkan")]

use gororoba_algebra::{cd_basis_mul_sign, gpu::EtaMatrixVulkanKernel};

fn eta_cpu(dim: usize) -> Vec<u8> {
    let dim_half = dim / 2;
    let mut eta = vec![0u8; dim_half * dim_half];
    for i in 0..dim_half {
        for j in 0..dim_half {
            let psi_ij = psi(dim, i, j + dim_half);
            let psi_ji = psi(dim, j, i + dim_half);
            eta[i * dim_half + j] = psi_ij ^ psi_ji;
        }
    }
    eta
}

fn psi(dim: usize, i: usize, j: usize) -> u8 {
    if cd_basis_mul_sign(dim, i, j) == 1 {
        0
    } else {
        1
    }
}

#[test]
#[ignore = "requires local Vulkan compute device"]
fn eta_matrix_vulkan_matches_cpu_dims_8_to_64() {
    eprintln!("eta_matrix_vulkan_matches_cpu_dims_8_to_64: probing Vulkan availability");
    if !EtaMatrixVulkanKernel::is_available() {
        eprintln!("Vulkan compute unavailable; skipping eta matrix Vulkan parity");
        return;
    }
    eprintln!("eta_matrix_vulkan_matches_cpu_dims_8_to_64: Vulkan available");

    for dim in [8usize, 16, 32, 64] {
        eprintln!("eta_matrix_vulkan_matches_cpu_dims_8_to_64: CPU dim {dim}");
        let expected = eta_cpu(dim);
        eprintln!("eta_matrix_vulkan_matches_cpu_dims_8_to_64: Vulkan dim {dim}");
        let actual = EtaMatrixVulkanKernel::compute(dim)
            .unwrap_or_else(|err| panic!("eta Vulkan compute failed for dim {dim}: {err}"));
        eprintln!("eta_matrix_vulkan_matches_cpu_dims_8_to_64: compare dim {dim}");
        assert_eq!(actual, expected, "eta Vulkan parity mismatch at dim {dim}");
    }
}
