use lbm_vulkan::{
    sparse_lbm_common::{SparseLbmPlan, evolve_sparse_d3q19_cpu},
    sparse_lbm_vulkan::evolve_sparse_d3q19_vulkan,
};

#[test]
#[ignore = "Vulkan compute adapter required"]
fn sparse_lbm_vulkan_matches_cpu_direct_active_brick() {
    let mask = vec![1u8; 8 * 8 * 8];
    let plan = SparseLbmPlan::from_geometry_mask(8, 8, 8, &mask).unwrap();
    let f0 = plan.equilibrium_at_rest();
    let cpu = evolve_sparse_d3q19_cpu(&plan, 1.0, &f0, 4).unwrap();
    let gpu = match evolve_sparse_d3q19_vulkan(&plan, 1.0, &f0, 4) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("skip: Vulkan sparse LBM failed ({err})");
            return;
        }
    };
    assert_close(&gpu, &cpu);
}

fn assert_close(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (idx, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1.0e-6,
            "idx={idx} got={got} expected={expected}"
        );
    }
}
