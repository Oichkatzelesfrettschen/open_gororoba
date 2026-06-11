#![cfg(feature = "vulkan")]

use faer::c64;
use quantum_core::gpu::peps_vulkan::{PepsVulkanKernel, peps_contract_rows_cpu};

#[test]
#[ignore = "requires local Vulkan compute adapter"]
fn peps_vulkan_matches_cpu_complex_row_product() {
    if !PepsVulkanKernel::is_available() {
        return;
    }

    let upper = vec![
        c64::new(1.0, 2.0),
        c64::new(3.0, 4.0),
        c64::new(-2.0, 0.5),
        c64::new(0.25, -0.75),
    ];
    let lower = vec![
        c64::new(2.0, 1.0),
        c64::new(1.0, 1.0),
        c64::new(0.5, -1.5),
        c64::new(-4.0, 2.0),
    ];

    let vulkan = PepsVulkanKernel::contract_rows_fp32(&upper, &lower).unwrap();
    let cpu = peps_contract_rows_cpu(&upper, &lower);

    for (index, (actual, expected)) in vulkan.iter().zip(cpu.iter()).enumerate() {
        assert!(
            (actual.re - expected.re).abs() <= 1.0e-6,
            "real component mismatch at {index}: {actual:?} vs {expected:?}"
        );
        assert!(
            (actual.im - expected.im).abs() <= 1.0e-6,
            "imag component mismatch at {index}: {actual:?} vs {expected:?}"
        );
    }
}
