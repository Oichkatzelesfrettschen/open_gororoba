#![cfg(feature = "cubecl")]

use grmhd_core::{
    cubecl::{GrmhdCubeclConfig, GrmhdCubeclKernel},
    vulkan::{NPRIM, advance_conserved_cpu_reference},
};

fn acceptance_config() -> GrmhdCubeclConfig {
    GrmhdCubeclConfig::new(32, 32, 32, 2.5, 20.0, 0.0, 4.0 / 3.0).unwrap()
}

fn soa_index(n_total: usize, channel: usize, cell: usize) -> usize {
    channel * n_total + cell
}

fn fixture_prims(config: GrmhdCubeclConfig) -> Vec<f32> {
    let n_total = config.n_total();
    let mut prims = vec![0.0f32; NPRIM * n_total];
    for cell in 0..n_total {
        let k = cell % config.n3;
        prims[soa_index(n_total, 0, cell)] = 1.0 + 0.01 * (cell % 7) as f32;
        prims[soa_index(n_total, 1, cell)] = 0.02 + 0.001 * (cell % 5) as f32;
        prims[soa_index(n_total, 2, cell)] = 0.0003 * (cell % 3) as f32;
        prims[soa_index(n_total, 3, cell)] = -0.0002 * (cell % 4) as f32;
        prims[soa_index(n_total, 4, cell)] = 0.0001 * k as f32;
        prims[soa_index(n_total, 5, cell)] = 0.001;
        prims[soa_index(n_total, 6, cell)] = 0.0005;
        prims[soa_index(n_total, 7, cell)] = -0.00025;
    }
    prims
}

#[test]
#[ignore = "requires local cubecl-wgpu adapter"]
fn grmhd_cubecl_matches_cpu_reference_for_cuda_style_advance() {
    if !GrmhdCubeclKernel::is_available() {
        return;
    }

    let config = acceptance_config();
    let prims = fixture_prims(config);
    let cpu = advance_conserved_cpu_reference(config, &prims, 0.0001, 10).unwrap();
    let gpu = GrmhdCubeclKernel::advance_conserved(config, &prims, 0.0001, 10).unwrap();
    assert_eq!(cpu.len(), gpu.len());

    for (idx, (expected, observed)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let scale = expected.abs().max(1.0);
        let rel = (expected - observed).abs() / scale;
        assert!(
            rel < 1.0e-4,
            "GRMHD cubecl mismatch at {idx}: cpu={expected}, gpu={observed}, rel={rel}"
        );
    }
}
