#![cfg(feature = "cubecl")]

use optics_core::{
    AlgebraicLensingCubecl, AlgebraicLensingCubeclConfig, GpuVec3, trace_rays_cpu_reference_cubecl,
};

#[test]
#[ignore = "requires local cubecl-wgpu adapter"]
fn algebraic_lensing_cubecl_matches_cpu_reference() {
    gororoba_gpu_cubecl::test_support::skip_if_unavailable!();

    let config = AlgebraicLensingCubeclConfig::new(8, 8, 8, 1.0, 0.1, 10).unwrap();
    let mut density = vec![0.0f32; config.n_cells()];
    for z in 0..config.nz {
        for y in 0..config.ny {
            for x in 0..config.nx {
                let idx = z * (config.nx * config.ny) + y * config.nx + x;
                density[idx] = 0.2 + 0.01 * x as f32 + 0.005 * y as f32;
            }
        }
    }
    let initial_pos = vec![
        GpuVec3 {
            x: 4.0,
            y: 4.0,
            z: 0.0,
        },
        GpuVec3 {
            x: 2.5,
            y: 3.5,
            z: 1.0,
        },
    ];
    let initial_dir = vec![
        GpuVec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        normalize(GpuVec3 {
            x: 0.1,
            y: 0.0,
            z: 1.0,
        }),
    ];

    let cpu =
        trace_rays_cpu_reference_cubecl(config, &density, &initial_pos, &initial_dir).unwrap();
    let cubecl =
        AlgebraicLensingCubecl::trace_rays(config, &density, &initial_pos, &initial_dir).unwrap();

    assert_vec3_close(&cpu.0, &cubecl.0);
    assert_vec3_close(&cpu.1, &cubecl.1);
}

fn normalize(value: GpuVec3) -> GpuVec3 {
    let len_sq = value.x * value.x + value.y * value.y + value.z * value.z;
    if len_sq < 1.0e-30 {
        value
    } else {
        let inv_len = len_sq.sqrt().recip();
        GpuVec3 {
            x: value.x * inv_len,
            y: value.y * inv_len,
            z: value.z * inv_len,
        }
    }
}

fn assert_vec3_close(expected: &[GpuVec3], observed: &[GpuVec3]) {
    assert_eq!(expected.len(), observed.len());
    for (idx, (lhs, rhs)) in expected.iter().zip(observed.iter()).enumerate() {
        for (component, lhs_value, rhs_value) in [
            ("x", lhs.x, rhs.x),
            ("y", lhs.y, rhs.y),
            ("z", lhs.z, rhs.z),
        ] {
            let scale = lhs_value.abs().max(1.0);
            let rel = (lhs_value - rhs_value).abs() / scale;
            assert!(
                rel < 2.0e-4,
                "ray {idx} component {component}: cpu={lhs_value}, cubecl={rhs_value}, rel={rel}"
            );
        }
    }
}
