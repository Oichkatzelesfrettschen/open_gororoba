#![cfg(feature = "vulkan")]

use gororoba_algebra::gpu::{VoudonVulkanKernel, voudon::Cd256FrustrationKernel};

type FieldCase = (&'static str, usize, usize, usize, u32);

#[test]
#[ignore = "requires local Vulkan compute device"]
fn voudon_vulkan_matches_cpu_fields() {
    if !VoudonVulkanKernel::is_available() {
        eprintln!("Vulkan compute unavailable; skipping Voudon Vulkan parity");
        return;
    }

    let cases: &[FieldCase] = &[
        ("single_cell", 1, 1, 1, 42),
        ("thin_line", 7, 1, 1, 1234),
        ("small_volume", 3, 2, 2, 99),
    ];

    for (label, nx, ny, nz, seed) in cases {
        let cpu = Cd256FrustrationKernel::compute_field_cpu(*nx, *ny, *nz, *seed)
            .unwrap_or_else(|err| panic!("{label}: CPU Voudon field failed: {err}"));
        let vulkan = VoudonVulkanKernel::compute_field(*nx, *ny, *nz, *seed)
            .unwrap_or_else(|err| panic!("{label}: Vulkan Voudon field failed: {err}"));
        assert_eq!(vulkan, cpu, "{label}: field mismatch");
    }
}
