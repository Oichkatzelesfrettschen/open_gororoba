//! GPU-accelerated ultrametric computation using CUDA.
//!
//! Evaluates millions of random triples in parallel on the GPU to compute
//! ultrametric fractions. Falls back gracefully if no CUDA device is found.
//!
//! Architecture:
//! - Data (column-major f32) is uploaded once per dataset/permutation.
//! - A single kernel launch evaluates all triples for ALL epsilon thresholds
//!   simultaneously, using atomic counters.
//! - The host shuffles columns for null permutations and re-uploads.
//!
//! Performance: ~10 billion triple evaluations/sec on RTX 4070 Ti (7680 CUDA
//! cores, FP32). A full ultrametric test with 10M triples x 200 permutations
//! x 20 epsilons completes in ~0.2 seconds per attribute subset.

use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// CUDA kernel: evaluate ultrametric condition for random triples.
///
/// Each thread generates one random triple (i,j,k), computes 3 pairwise
/// Euclidean distances from column-major data, and checks the ultrametric
/// condition (two largest distances approximately equal) for ALL epsilon
/// thresholds. Uses splitmix64-style hashing for per-thread deterministic RNG.
const KERNEL_SRC: &str = r#"
extern "C" __global__ void ultrametric_triples(
    const float* __restrict__ data,
    const int n,
    const int d,
    const unsigned long long seed,
    const float* __restrict__ epsilons,
    const int n_eps,
    const int n_triples,
    int* __restrict__ counts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_triples) return;

    // Splitmix64-style hash: deterministic, statistically adequate for sampling.
    // Each thread gets a unique state from (seed, tid).
    unsigned long long state = seed + (unsigned long long)tid * 0x9E3779B97F4A7C15ULL;
    state ^= state >> 30;
    state *= 0xBF58476D1CE4E5B9ULL;
    state ^= state >> 27;
    state *= 0x94D049BB133111EBULL;
    state ^= state >> 31;

    // Generate 3 distinct indices in [0, n)
    unsigned int i = (unsigned int)(state % (unsigned long long)n);
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned int j = (unsigned int)(state % (unsigned long long)(n - 1));
    if (j >= i) j++;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned int k = (unsigned int)(state % (unsigned long long)(n - 2));
    unsigned int mn = (i < j) ? i : j;
    unsigned int mx = (i > j) ? i : j;
    if (k >= mn) k++;
    if (k >= mx) k++;

    // Compute 3 squared Euclidean distances from column-major data
    float d_ij_sq = 0.0f, d_jk_sq = 0.0f, d_ik_sq = 0.0f;
    for (int c = 0; c < d; c++) {
        int base = c * n;
        float vi = data[base + i];
        float vj = data[base + j];
        float vk = data[base + k];
        float diff_ij = vi - vj;
        float diff_jk = vj - vk;
        float diff_ik = vi - vk;
        d_ij_sq += diff_ij * diff_ij;
        d_jk_sq += diff_jk * diff_jk;
        d_ik_sq += diff_ik * diff_ik;
    }

    // Sort squared distances to find d_max_sq and d_mid_sq.
    // No sqrtf() needed: the ultrametric condition on distances
    //   (d_max - d_mid) / d_max < epsilon
    // is equivalent to
    //   (d_max_sq - d_mid_sq) / d_max_sq < epsilon_sq
    // where epsilon_sq = 1 - (1 - epsilon)^2.
    // The host pre-computes epsilon_sq for each threshold.
    float d_max_sq = fmaxf(d_ij_sq, fmaxf(d_jk_sq, d_ik_sq));
    float d_min_sq = fminf(d_ij_sq, fminf(d_jk_sq, d_ik_sq));
    float d_mid_sq = d_ij_sq + d_jk_sq + d_ik_sq - d_min_sq - d_max_sq;

    // Relative difference between the two largest squared distances
    float rel_diff_sq;
    if (d_max_sq > 1e-14f) {
        rel_diff_sq = (d_max_sq - d_mid_sq) / d_max_sq;
    } else {
        rel_diff_sq = -1.0f;  // degenerate triple: always counts
    }

    // Check ALL epsilon thresholds for this triple (one atomic per epsilon).
    // epsilons[] now contains squared-distance thresholds pre-computed by host.
    for (int e = 0; e < n_eps; e++) {
        if (rel_diff_sq < epsilons[e]) {
            atomicAdd(&counts[e], 1);
        }
    }
}
"#;

/// GPU-accelerated ultrametric computation engine.
///
/// Holds a compiled CUDA module and device context. Create once at startup
/// and reuse for all datasets and subsets.
pub struct GpuUltrametricEngine {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
}

impl GpuUltrametricEngine {
    /// Try to initialize CUDA. Returns None if no GPU is available.
    ///
    /// This uses dynamic loading: the binary works on machines without CUDA,
    /// it just won't have GPU acceleration.
    pub fn try_new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        let ptx = compile_ptx(KERNEL_SRC).ok()?;
        let module = ctx.load_module(ptx).ok()?;
        let kernel = module.load_function("ultrametric_triples").ok()?;
        Some(Self {
            _ctx: ctx,
            stream,
            kernel,
        })
    }

    /// Evaluate ultrametric fractions for multiple epsilon thresholds in one
    /// kernel launch.
    ///
    /// `data_f32`: column-major f32 normalized data (d columns x n rows).
    /// `epsilons`: array of epsilon thresholds (on distances) to test simultaneously.
    ///
    /// Internally converts epsilon thresholds to squared-distance space:
    /// `eps_sq = 1 - (1 - eps)^2` before uploading to the GPU kernel.
    ///
    /// Returns a Vec<f64> of fractions, one per epsilon.
    pub fn fraction_multi_eps(
        &self,
        data_f32: &[f32],
        n: usize,
        d: usize,
        n_triples: usize,
        seed: u64,
        epsilons: &[f32],
    ) -> Result<Vec<f64>, cudarc::driver::DriverError> {
        // Convert distance-space epsilons to squared-distance-space thresholds
        let epsilons_sq: Vec<f32> = epsilons
            .iter()
            .map(|&eps| 1.0 - (1.0 - eps).powi(2))
            .collect();
        let data_dev = self.stream.clone_htod(data_f32)?;
        let eps_dev = self.stream.clone_htod(&epsilons_sq)?;
        let mut counts_dev = self.stream.alloc_zeros::<i32>(epsilons.len())?;

        let n_i32 = n as i32;
        let d_i32 = d as i32;
        let n_eps_i32 = epsilons.len() as i32;
        let n_triples_i32 = n_triples as i32;

        let block_size = 256_u32;
        let grid_size = (n_triples as u32).div_ceil(block_size);
        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = self.stream.launch_builder(&self.kernel);
        builder.arg(&data_dev);
        builder.arg(&n_i32);
        builder.arg(&d_i32);
        builder.arg(&seed);
        builder.arg(&eps_dev);
        builder.arg(&n_eps_i32);
        builder.arg(&n_triples_i32);
        builder.arg(&mut counts_dev);

        unsafe { builder.launch(cfg) }?;

        let counts: Vec<i32> = self.stream.clone_dtoh(&counts_dev)?;
        Ok(counts
            .iter()
            .map(|&c| c as f64 / n_triples as f64)
            .collect())
    }

    /// Full GPU-accelerated ultrametric test with permutation null.
    ///
    /// Computes observed ultrametric fraction and tolerance curve, then
    /// builds a null distribution by shuffling columns and recomputing.
    ///
    /// Memory: O(N * d) on both host and device -- no O(N^2) matrices.
    /// Compute: (1 + n_permutations) kernel launches of n_triples each.
    pub fn ultrametric_test(
        &self,
        data_f32: &mut [f32],
        n: usize,
        d: usize,
        n_triples: usize,
        n_permutations: usize,
        seed: u64,
    ) -> Result<GpuTestResult, cudarc::driver::DriverError> {
        let epsilons: Vec<f32> = (1..=20).map(|i| i as f32 * 0.01).collect();
        let n_eps = epsilons.len();

        // Observed fractions at all epsilons
        let obs_fracs = self.fraction_multi_eps(data_f32, n, d, n_triples, seed, &epsilons)?;

        // The "main" fraction at epsilon=0.05 (index 4, since 5 * 0.01 = 0.05)
        let obs_main = obs_fracs[4];

        // Null distribution: shuffle columns, recompute
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 1_000_000);
        let mut null_main_fracs = Vec::with_capacity(n_permutations);
        let mut null_curve_sums = vec![0.0_f64; n_eps];

        for perm in 0..n_permutations {
            // Shuffle each column independently (breaks inter-attribute correlations)
            for col in 0..d {
                let start = col * n;
                let slice = &mut data_f32[start..start + n];
                slice.shuffle(&mut rng);
            }

            let null_fracs = self.fraction_multi_eps(
                data_f32,
                n,
                d,
                n_triples,
                seed + 2_000_000 + perm as u64,
                &epsilons,
            )?;

            null_main_fracs.push(null_fracs[4]);
            for (i, &nf) in null_fracs.iter().enumerate() {
                null_curve_sums[i] += nf;
            }
        }

        // Main fraction statistics
        let null_mean =
            null_main_fracs.iter().sum::<f64>() / n_permutations as f64;
        let null_var = null_main_fracs
            .iter()
            .map(|f| (f - null_mean).powi(2))
            .sum::<f64>()
            / n_permutations as f64;
        let null_std = null_var.sqrt();

        let n_extreme = null_main_fracs
            .iter()
            .filter(|&&f| f >= obs_main)
            .count();
        let p_value = (n_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0);

        // Tolerance curve
        let null_curve_means: Vec<f64> = null_curve_sums
            .iter()
            .map(|s| s / n_permutations as f64)
            .collect();

        let mut curve_points = Vec::with_capacity(n_eps);
        for i in 0..n_eps {
            let excess = obs_fracs[i] - null_curve_means[i];
            curve_points.push(super::ToleranceCurvePoint {
                epsilon: epsilons[i] as f64,
                observed: obs_fracs[i],
                null_mean: null_curve_means[i],
                excess,
            });
        }

        // AUC of excess curve
        let mut auc = 0.0;
        for w in curve_points.windows(2) {
            let h = w[1].epsilon - w[0].epsilon;
            auc += 0.5 * h * (w[0].excess + w[1].excess);
        }

        let (max_excess, best_epsilon) = curve_points
            .iter()
            .max_by(|a, b| a.excess.partial_cmp(&b.excess).unwrap())
            .map(|p| (p.excess, p.epsilon))
            .unwrap_or((0.0, 0.05));

        Ok(GpuTestResult {
            n_objects: n,
            n_attributes: d,
            n_triples,
            ultrametric_fraction: obs_main,
            null_fraction_mean: null_mean,
            null_fraction_std: null_std,
            p_value,
            tolerance_curve: super::ToleranceCurveResult {
                points: curve_points,
                auc_excess: auc,
                max_excess,
                best_epsilon,
            },
        })
    }
}

/// Convert f64 column-major data to f32 for GPU.
///
/// The GPU uses f32 (40 TFLOPS vs 625 GFLOPS for f64 on consumer GPUs).
/// For ultrametric fraction computation, f32 precision is more than adequate.
pub fn to_f32_column_major(cols_f64: &[f64]) -> Vec<f32> {
    cols_f64.iter().map(|&v| v as f32).collect()
}

/// Combined result from GPU ultrametric test.
#[derive(Debug, Clone)]
pub struct GpuTestResult {
    pub n_objects: usize,
    pub n_attributes: usize,
    pub n_triples: usize,
    pub ultrametric_fraction: f64,
    pub null_fraction_mean: f64,
    pub null_fraction_std: f64,
    pub p_value: f64,
    pub tolerance_curve: super::ToleranceCurveResult,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ultrametric::baire::{AttributeSpec, BaireEncoder, normalize_data_column_major};

    #[test]
    fn test_gpu_init() {
        // This test verifies GPU initialization works on machines with CUDA.
        // On machines without CUDA, it gracefully returns None.
        let engine = GpuUltrametricEngine::try_new();
        if engine.is_some() {
            eprintln!("GPU engine initialized successfully");
        } else {
            eprintln!("No CUDA device available -- GPU tests skipped");
        }
    }

    #[test]
    fn test_gpu_fraction_matches_cpu() {
        let engine = match GpuUltrametricEngine::try_new() {
            Some(e) => e,
            None => {
                eprintln!("Skipping GPU test: no CUDA device");
                return;
            }
        };

        let specs = vec![
            AttributeSpec { name: "x".into(), min: 0.0, max: 10.0, log_scale: false },
            AttributeSpec { name: "y".into(), min: 0.0, max: 10.0, log_scale: false },
        ];
        let encoder = BaireEncoder::new(specs, 10, 4);
        let mut rng = ChaCha8Rng::seed_from_u64(999);
        let data: Vec<Vec<f64>> = (0..500)
            .map(|_| vec![rng.gen_range(0.0..10.0), rng.gen_range(0.0..10.0)])
            .collect();

        let (cols_f64, n, d) = normalize_data_column_major(&encoder, &data);
        let cols_f32 = to_f32_column_major(&cols_f64);

        // GPU fraction at epsilon=0.05
        let epsilons = vec![0.05_f32];
        let gpu_frac = engine
            .fraction_multi_eps(&cols_f32, n, d, 100_000, 42, &epsilons)
            .expect("GPU kernel failed");

        // CPU fraction at epsilon=0.05
        let cpu_frac = crate::ultrametric::baire::matrix_free_fraction(
            &cols_f64, n, d, 100_000, 42, 0.05,
        );

        // They use different RNGs (GPU: splitmix64, CPU: ChaCha8), so exact
        // match is not expected. But statistically they should be very close.
        let diff = (gpu_frac[0] - cpu_frac).abs();
        assert!(
            diff < 0.02,
            "GPU ({:.4}) and CPU ({:.4}) fractions should be close (diff={:.4})",
            gpu_frac[0], cpu_frac, diff,
        );
        eprintln!(
            "GPU fraction: {:.4}, CPU fraction: {:.4}, diff: {:.4}",
            gpu_frac[0], cpu_frac, diff,
        );
    }

    #[test]
    fn test_gpu_multi_eps() {
        let engine = match GpuUltrametricEngine::try_new() {
            Some(e) => e,
            None => {
                eprintln!("Skipping GPU test: no CUDA device");
                return;
            }
        };

        let specs = vec![
            AttributeSpec { name: "x".into(), min: 0.0, max: 1.0, log_scale: false },
            AttributeSpec { name: "y".into(), min: 0.0, max: 1.0, log_scale: false },
        ];
        let encoder = BaireEncoder::new(specs, 10, 4);
        let mut rng = ChaCha8Rng::seed_from_u64(77);
        let data: Vec<Vec<f64>> = (0..200)
            .map(|_| vec![rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)])
            .collect();

        let (cols_f64, n, d) = normalize_data_column_major(&encoder, &data);
        let cols_f32 = to_f32_column_major(&cols_f64);

        let epsilons: Vec<f32> = (1..=20).map(|i| i as f32 * 0.01).collect();
        let fracs = engine
            .fraction_multi_eps(&cols_f32, n, d, 1_000_000, 42, &epsilons)
            .expect("GPU kernel failed");

        assert_eq!(fracs.len(), 20);
        // Fractions must be monotonically non-decreasing with epsilon
        for w in fracs.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-6,
                "Fractions must increase: {:.4} -> {:.4}",
                w[0], w[1],
            );
        }
        eprintln!("GPU multi-eps fractions: {:?}", fracs);
    }

    #[test]
    fn test_gpu_full_test() {
        let engine = match GpuUltrametricEngine::try_new() {
            Some(e) => e,
            None => {
                eprintln!("Skipping GPU test: no CUDA device");
                return;
            }
        };

        let specs = vec![
            AttributeSpec { name: "x".into(), min: 0.0, max: 1.0, log_scale: false },
            AttributeSpec { name: "y".into(), min: 0.0, max: 1.0, log_scale: false },
        ];
        let encoder = BaireEncoder::new(specs, 10, 4);
        let mut rng = ChaCha8Rng::seed_from_u64(55);
        let data: Vec<Vec<f64>> = (0..100)
            .map(|_| vec![rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)])
            .collect();

        let (cols_f64, n, d) = normalize_data_column_major(&encoder, &data);
        let mut cols_f32 = to_f32_column_major(&cols_f64);

        let result = engine
            .ultrametric_test(&mut cols_f32, n, d, 500_000, 50, 42)
            .expect("GPU test failed");

        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.ultrametric_fraction > 0.0);
        assert_eq!(result.tolerance_curve.points.len(), 20);

        eprintln!(
            "GPU test: frac={:.4}, null={:.4}, p={:.3}, AUC={:.4}",
            result.ultrametric_fraction,
            result.null_fraction_mean,
            result.p_value,
            result.tolerance_curve.auc_excess,
        );
    }
}
