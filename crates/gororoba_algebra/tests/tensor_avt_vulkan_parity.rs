#![cfg(feature = "vulkan")]

use gororoba_algebra::gpu::{TensorAVT, TensorAvtVulkanKernel};

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
#[ignore = "requires local Vulkan compute device"]
fn tensor_avt_vulkan_matches_cpu_mul_batch_and_norm() {
    if !TensorAvtVulkanKernel::is_available() {
        eprintln!("Vulkan compute unavailable; skipping Tensor AVT Vulkan parity");
        return;
    }

    for dim in [16usize, 32] {
        let batch_size = 3;
        let avt = TensorAVT::new(dim);
        let left = fixture_left(dim);
        let right = fixture_batch(dim, batch_size);
        let cpu_single = avt.compute_cd_mul(&left, &right[..dim]).unwrap();
        let vk_single = avt.compute_cd_mul_vulkan(&left, &right[..dim]).unwrap();
        assert_close(&vk_single, &cpu_single, "Vulkan single multiply");

        let cpu_batch = avt.compute_cd_mul_batch(&left, &right, batch_size).unwrap();
        let vk_batch = avt
            .compute_cd_mul_batch_vulkan(&left, &right, batch_size)
            .unwrap();
        assert_close(&vk_batch, &cpu_batch, "Vulkan batched multiply");

        let cpu_norms = avt.compute_norm_sq_batch(&right, batch_size).unwrap();
        let vk_norms = avt
            .compute_norm_sq_batch_vulkan(&right, batch_size)
            .unwrap();
        assert_close(&vk_norms, &cpu_norms, "Vulkan norms");
    }
}
