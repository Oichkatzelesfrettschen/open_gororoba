//! High-dimensional Cayley-Dickson algebra analysis for dark matter field configurations.
//!
//! Sweeps CD dimensions 16..4096 measuring:
//! - Associator norm: ||(ab)c - a(bc)|| for random triples
//! - Zero-divisor density: fraction of pairs with ||ab|| < eps but ||a||,||b|| > eps
//! - Norm multiplicativity deviation: |N(ab) - N(a)*N(b)| / (N(a)*N(b))
//! - AVT computation time (TensorAVT L_a matmul)
//!
//! Outputs CSV with one row per dimension.

use algebra_core::construction::cayley_dickson::{cd_multiply, cd_norm_sq};
use algebra_core::gpu::TensorAVT;
use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// Sweep Cayley-Dickson dims measuring associator norms, zero-divisor density,
/// and norm multiplicativity via tensor_avt CUDA Tensor Core pipeline.
#[derive(Parser)]
#[command(name = "dm-cd-4096-analysis")]
struct Cli {
    /// Comma-separated dimensions to sweep (powers of 2, >= 16)
    #[arg(long, default_value = "16,32,64,128,256,512,1024,2048,4096")]
    dims: String,

    /// Random element pairs per dimension for associator/ZD/norm tests
    #[arg(long, default_value_t = 1000)]
    samples: usize,

    /// RNG seed for reproducibility
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Output CSV path (stdout if omitted)
    #[arg(long)]
    out: Option<PathBuf>,
}

/// Generate a random CD element of given dimension using the RNG.
fn random_cd_element(dim: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
    use rand::Rng as _;
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// Compute the associator [a,b,c] = (ab)c - a(bc).
fn associator(a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
    let ab = cd_multiply(a, b);
    let bc = cd_multiply(b, c);
    let ab_c = cd_multiply(&ab, c);
    let a_bc = cd_multiply(a, &bc);
    ab_c.iter().zip(a_bc.iter()).map(|(x, y)| x - y).collect()
}

struct DimResult {
    dim: usize,
    associator_norm_mean: f64,
    associator_norm_max: f64,
    zd_density: f64,
    norm_mul_deviation: f64,
    avt_time_ms: f64,
}

fn analyze_dimension(dim: usize, samples: usize, seed: u64) -> DimResult {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ dim as u64);
    let eps = 1e-10;

    // --- Associator norms ---
    let mut assoc_norms = Vec::with_capacity(samples);
    for _ in 0..samples {
        let a = random_cd_element(dim, &mut rng);
        let b = random_cd_element(dim, &mut rng);
        let c = random_cd_element(dim, &mut rng);
        let assoc = associator(&a, &b, &c);
        assoc_norms.push(cd_norm_sq(&assoc).sqrt());
    }
    let associator_norm_mean = assoc_norms.iter().sum::<f64>() / samples as f64;
    let associator_norm_max = assoc_norms
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);

    // --- Zero-divisor density ---
    let mut zd_count = 0usize;
    let mut rng_zd = ChaCha8Rng::seed_from_u64(seed ^ (dim as u64).wrapping_mul(7));
    for _ in 0..samples {
        let a = random_cd_element(dim, &mut rng_zd);
        let b = random_cd_element(dim, &mut rng_zd);
        let norm_a = cd_norm_sq(&a).sqrt();
        let norm_b = cd_norm_sq(&b).sqrt();
        if norm_a < eps || norm_b < eps {
            continue;
        }
        let ab = cd_multiply(&a, &b);
        let norm_ab = cd_norm_sq(&ab).sqrt();
        if norm_ab < eps {
            zd_count += 1;
        }
    }
    let zd_density = zd_count as f64 / samples as f64;

    // --- Norm multiplicativity deviation ---
    // For Hurwitz algebras (dim <= 8): N(ab) = N(a)*N(b) exactly.
    // For dim >= 16: deviation grows with dimension.
    let mut norm_dev_sum = 0.0_f64;
    let mut rng_norm = ChaCha8Rng::seed_from_u64(seed ^ (dim as u64).wrapping_mul(13));
    for _ in 0..samples {
        let a = random_cd_element(dim, &mut rng_norm);
        let b = random_cd_element(dim, &mut rng_norm);
        let na = cd_norm_sq(&a);
        let nb = cd_norm_sq(&b);
        let ab = cd_multiply(&a, &b);
        let nab = cd_norm_sq(&ab);
        let product = na * nb;
        if product > eps {
            norm_dev_sum += (nab - product).abs() / product;
        }
    }
    let norm_mul_deviation = norm_dev_sum / samples as f64;

    // --- AVT timing (TensorAVT construction + single matmul) ---
    // TensorAVT uses f32 internally; we measure the L_a matrix construction
    // and a single multiplication via the auto backend.
    let avt_time_ms = if dim >= 16 {
        let avt = TensorAVT::new(dim);
        // Build f32 test vectors from the same seed
        let a_f32: Vec<f32> = random_cd_element(dim, &mut ChaCha8Rng::seed_from_u64(seed))
            .iter()
            .map(|&v| v as f32)
            .collect();
        let x_f32: Vec<f32> = random_cd_element(dim, &mut ChaCha8Rng::seed_from_u64(seed ^ 1))
            .iter()
            .map(|&v| v as f32)
            .collect();

        // Warmup
        let _ = avt.compute_cd_mul_auto(&a_f32, &x_f32);

        let start = Instant::now();
        let n_avt_iters = 10;
        for _ in 0..n_avt_iters {
            let _ = avt.compute_cd_mul_auto(&a_f32, &x_f32);
        }
        start.elapsed().as_secs_f64() * 1000.0 / n_avt_iters as f64
    } else {
        0.0
    };

    DimResult {
        dim,
        associator_norm_mean,
        associator_norm_max,
        zd_density,
        norm_mul_deviation,
        avt_time_ms,
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let dims: Vec<usize> = cli
        .dims
        .split(',')
        .map(|s| {
            let d: usize = s.trim().parse().expect("dims must be comma-separated integers");
            assert!(d >= 16 && d.is_power_of_two(), "each dim must be power of 2, >= 16");
            d
        })
        .collect();

    // Reduce samples for very high dims to keep wall time reasonable
    let base_samples = cli.samples;

    let mut results = Vec::with_capacity(dims.len());
    for &dim in &dims {
        // Scale samples down for large dims: full samples up to 256,
        // then halve per doubling to keep total time bounded.
        let effective_samples = if dim <= 256 {
            base_samples
        } else {
            let doublings = (dim as f64 / 256.0).log2().ceil() as u32;
            (base_samples >> doublings).max(16)
        };

        eprintln!(
            "dim={dim:>5}  samples={effective_samples:>6}  ...",
        );
        let t0 = Instant::now();
        let r = analyze_dimension(dim, effective_samples, cli.seed);
        let wall_s = t0.elapsed().as_secs_f64();
        eprintln!(
            "  assoc_mean={:.6e}  assoc_max={:.6e}  zd={:.6e}  norm_dev={:.6e}  avt={:.3}ms  wall={:.1}s",
            r.associator_norm_mean,
            r.associator_norm_max,
            r.zd_density,
            r.norm_mul_deviation,
            r.avt_time_ms,
            wall_s,
        );
        results.push(r);
    }

    // Write CSV
    let header = "dim,associator_norm_mean,associator_norm_max,zd_density,norm_mul_deviation,avt_time_ms\n";
    let mut csv_data = String::from(header);
    for r in &results {
        csv_data.push_str(&format!(
            "{},{:.9e},{:.9e},{:.9e},{:.9e},{:.6}\n",
            r.dim,
            r.associator_norm_mean,
            r.associator_norm_max,
            r.zd_density,
            r.norm_mul_deviation,
            r.avt_time_ms,
        ));
    }

    if let Some(path) = &cli.out {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, &csv_data)?;
        eprintln!("wrote {}", path.display());
    } else {
        std::io::stdout().write_all(csv_data.as_bytes())?;
    }

    Ok(())
}
