#![cfg(feature = "cubecl")]

use algebra_experimental::voudon_stabilizer::Cd256StabilizerKernel;

#[test]
#[ignore = "requires local cubecl-wgpu adapter"]
fn voudon_cubecl_row_counts_match_cpu_reference() {
    gororoba_gpu_cubecl::test_support::skip_if_unavailable!();

    let cpu = Cd256StabilizerKernel::stable_cycle_row_counts_cpu();
    let cubecl = Cd256StabilizerKernel::stable_cycle_row_counts_cubecl().unwrap();

    assert_eq!(cubecl, cpu);
}
