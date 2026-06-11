#![cfg(all(feature = "cubecl", feature = "vulkan"))]

use gororoba_algebra::{
    cd_basis_mul_sign,
    gpu::{EtaMatrixCubeclKernel, EtaMatrixVulkanKernel},
};

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
#[ignore = "requires local Vulkan compute device and cubecl-wgpu adapter"]
fn eta_matrix_cubecl_matches_cpu_and_vulkan_dims_8_to_64() {
    if !EtaMatrixVulkanKernel::is_available() {
        eprintln!("Vulkan compute unavailable; skipping eta matrix 3-way parity");
        return;
    }
    if !EtaMatrixCubeclKernel::is_available() {
        eprintln!("cubecl-wgpu unavailable; skipping eta matrix 3-way parity");
        return;
    }

    for dim in [8usize, 16, 32, 64] {
        let expected = eta_cpu(dim);
        let vulkan = EtaMatrixVulkanKernel::compute(dim)
            .unwrap_or_else(|err| panic!("eta Vulkan compute failed for dim {dim}: {err}"));
        let cubecl = EtaMatrixCubeclKernel::compute(dim)
            .unwrap_or_else(|err| panic!("eta cubecl compute failed for dim {dim}: {err}"));
        assert_eq!(vulkan, expected, "eta Vulkan parity mismatch at dim {dim}");
        assert_eq!(cubecl, expected, "eta cubecl parity mismatch at dim {dim}");
        assert_eq!(cubecl, vulkan, "eta cubecl/Vulkan mismatch at dim {dim}");
    }
}
