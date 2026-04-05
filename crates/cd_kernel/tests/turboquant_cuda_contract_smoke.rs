use cd_kernel::{
    lloyd_max,
    turboquant::{
        backend::{Backend, BackendQuantizer},
        dispatch::SimdLevel,
    },
};

const DEQUANT_DOT_KERNEL_PARAM_ORDER: [&str; 9] = [
    "queries",
    "key_indices",
    "centroids",
    "key_norms",
    "scores",
    "d",
    "n_queries",
    "n_keys",
    "n_levels",
];

#[test]
fn dequant_dot_shape_guard_rejects_mismatch() {
    let cb = lloyd_max::get_codebook(128, 3);
    let q = BackendQuantizer::with_backend(&cb, 3, Backend::Cpu(SimdLevel::Scalar));
    let err = q
        .dequant_dot_batch(&[0.1f32, 0.2, 0.3], &[0u8, 1, 2, 3], &[1.0f32], 1, 1, 4)
        .expect_err("invalid query shape should fail");
    assert!(err.contains("queries length mismatch"));
}

#[test]
fn dequant_dot_kernel_signature_matches_host_contract() {
    let src = include_str!("../src/turboquant/cuda/kernels/turboquant.cu");
    let actual = extract_dequant_dot_kernel_param_names(src);
    assert_eq!(
        actual, DEQUANT_DOT_KERNEL_PARAM_ORDER,
        "turboquant_dequant_dot ABI drift detected"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_dequant_dot_parity_smoke_runtime_guarded() {
    let Some(props) = cd_kernel::turboquant::cuda::device::probe_device() else {
        println!("Skipping CUDA parity smoke: no CUDA device available");
        return;
    };

    let cb = lloyd_max::get_codebook(128, 3);
    let cpu_q = BackendQuantizer::with_backend(&cb, 3, Backend::Cpu(SimdLevel::Scalar));
    let cuda_q = BackendQuantizer::with_backend(&cb, 3, Backend::Cuda(props.recommended_tier()));

    let d = 4usize;
    let n_queries = 2usize;
    let n_keys = 3usize;
    let queries = vec![0.5, -0.25, 0.75, 0.1, -0.2, 0.3, 0.4, -0.5];
    let key_indices = vec![
        0, 1, 2, // c0 across keys
        3, 4, 5, // c1 across keys
        6, 7, 0, // c2 across keys
        1, 2, 3, // c3 across keys
    ];
    let key_norms = vec![1.0f32, 0.5, 2.0];

    let cpu_scores = cpu_q
        .dequant_dot_batch(&queries, &key_indices, &key_norms, n_queries, n_keys, d)
        .expect("CPU fused dequant-dot should succeed");

    let cuda_scores =
        match cuda_q.dequant_dot_batch(&queries, &key_indices, &key_norms, n_queries, n_keys, d) {
            Ok(scores) => scores,
            Err(err) if err.contains("CUDA kernel init") => {
                println!("Skipping CUDA parity smoke: {err}");
                return;
            }
            Err(err) => {
                panic!("CUDA fused dequant-dot should succeed when runtime is available: {err}")
            }
        };

    assert_eq!(
        cpu_scores.len(),
        cuda_scores.len(),
        "CPU/CUDA score length mismatch"
    );

    for (idx, (cpu, cuda)) in cpu_scores.iter().zip(cuda_scores.iter()).enumerate() {
        let abs_diff = (cpu - cuda).abs();
        assert!(
            abs_diff <= 1e-5,
            "CPU/CUDA parity mismatch at {}: cpu={}, cuda={}, |diff|={}",
            idx,
            cpu,
            cuda,
            abs_diff
        );
    }
}

fn extract_dequant_dot_kernel_param_names(kernel_src: &str) -> [&str; 9] {
    let marker = "__global__ void turboquant_dequant_dot(";
    let start = kernel_src
        .find(marker)
        .expect("turboquant_dequant_dot kernel declaration not found");
    let after_marker = &kernel_src[start + marker.len()..];
    let close = after_marker
        .find(')')
        .expect("turboquant_dequant_dot declaration missing ')'");
    let params = &after_marker[..close];

    let parsed: Vec<&str> = params
        .split(',')
        .map(|param| {
            param
                .split_whitespace()
                .last()
                .expect("parameter token")
                .trim_matches(|c: char| c == ',' || c == '*' || c == '&')
        })
        .collect();
    parsed
        .try_into()
        .expect("turboquant_dequant_dot parameter count should be 9")
}
