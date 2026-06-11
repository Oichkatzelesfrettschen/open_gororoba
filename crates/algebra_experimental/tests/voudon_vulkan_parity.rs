#![cfg(feature = "vulkan")]

use algebra_experimental::voudon_stabilizer::Cd256StabilizerKernel;

#[test]
#[ignore = "requires local Vulkan compute device"]
fn voudon_vulkan_row_counts_match_cpu_reference() {
    if !Cd256StabilizerKernel::vulkan_available() {
        eprintln!("Vulkan compute unavailable; skipping Voudon stabilizer parity");
        return;
    }

    let cpu = Cd256StabilizerKernel::stable_cycle_row_counts_cpu();
    let vulkan = Cd256StabilizerKernel::stable_cycle_row_counts_vulkan().unwrap();

    assert_eq!(vulkan, cpu);
}
