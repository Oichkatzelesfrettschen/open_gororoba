#![cfg(feature = "cubecl")]

use grmhd_core::{
    cubecl::{GrmhdCubeclConfig, GrmhdCubeclKernel},
    vulkan::{NCONS, NPRIM},
};

fn fixture_config() -> GrmhdCubeclConfig {
    GrmhdCubeclConfig::new(4, 3, 2, 4.0, 12.0, 0.7, 4.0 / 3.0).unwrap()
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
fn grmhd_cubecl_advance_returns_finite_conserved_state() {
    if !GrmhdCubeclKernel::is_available() {
        eprintln!("No CubeCL wgpu adapter, skipping GRMHD advance test");
        return;
    }

    let config = fixture_config();
    let prims = fixture_prims(config);
    let observed = GrmhdCubeclKernel::advance_conserved(config, &prims, 0.0001, 1)
        .unwrap_or_else(|error| panic!("CubeCL advance failed on an available device: {error}"));
    assert_eq!(observed.len(), NCONS * config.n_total());
    assert!(observed.iter().all(|value| value.is_finite()));
}
