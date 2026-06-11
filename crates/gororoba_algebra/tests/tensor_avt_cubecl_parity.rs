#![cfg(feature = "cubecl")]

use gororoba_algebra::gpu::{TensorAVT, TensorAvtCubeclKernel};

fn fixture_left(dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|idx| ((idx * 7 + 3) as f32 * 0.031).sin())
        .collect()
}

fn fixture_batch(dim: usize, batch_size: usize) -> Vec<f32> {
    (0..(dim * batch_size))
        .map(|idx| ((idx * 11 + 5) as f32 * 0.023).cos())
        .collect()
}

fn assert_close(got: &[f32], expected: &[f32], label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (idx, (lhs, rhs)) in got.iter().zip(expected.iter()).enumerate() {
        let scale = rhs.abs().max(1.0);
        let rel = (lhs - rhs).abs() / scale;
        assert!(
            rel < 1.0e-4,
            "{label}: mismatch at {idx}: got={lhs}, expected={rhs}, rel={rel}"
        );
    }
}

#[test]
#[ignore = "requires local cubecl-wgpu adapter"]
fn tensor_avt_cubecl_matches_cpu_mul_batch_and_norm() {
    if !TensorAvtCubeclKernel::is_available() {
        eprintln!("cubecl-wgpu unavailable; skipping Tensor AVT cubecl parity");
        return;
    }

    for dim in [16usize, 32] {
        let batch_size = 3;
        let avt = TensorAVT::new(dim);
        let left = fixture_left(dim);
        let right = fixture_batch(dim, batch_size);
        let cpu_single = avt.compute_cd_mul(&left, &right[..dim]).unwrap();
        let cube_single = avt.compute_cd_mul_cubecl(&left, &right[..dim]).unwrap();
        assert_close(&cube_single, &cpu_single, "cubecl single multiply");

        let cpu_batch = avt.compute_cd_mul_batch(&left, &right, batch_size).unwrap();
        let cube_batch = avt
            .compute_cd_mul_batch_cubecl(&left, &right, batch_size)
            .unwrap();
        assert_close(&cube_batch, &cpu_batch, "cubecl batched multiply");

        let cpu_norms = avt.compute_norm_sq_batch(&right, batch_size).unwrap();
        let cube_norms = avt
            .compute_norm_sq_batch_cubecl(&right, batch_size)
            .unwrap();
        assert_close(&cube_norms, &cpu_norms, "cubecl norms");
    }
}
