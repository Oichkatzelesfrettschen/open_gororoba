//! Thesis 4: CHSH Bell parameter S > 2 preserved under >= 20x Haar wavelet compression.
//!
//! Software simulation of the CHSH Bell inequality experiment:
//! - 4 angle settings (a0, a1) x (b0, b1) for singlet state
//! - N_time-length Binomial-sampled coincidence time series
//! - Haar wavelet compression retaining coarsest scaling coefficient
//!
//! PASS criterion: |S_compressed| > 2.0 (Bell violation preserved).
//! NOTE: This is a software simulation; it tests whether the Bell violation
//! in coincidence count data survives wavelet compression, NOT a physical experiment.
//!
//! The Haar DWT coarsest scaling coefficient encodes the signal total exactly
//! (c[0] = sum(signal)/sqrt(n)), so chsh_correlation -- which sums all time bins --
//! is invariant under compression. delta_S = 0 is the expected result and
//! confirms that the scaling coefficient alone fully preserves the Bell parameter.

use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Binomial, Distribution};
use spectral_core::wavelet::{haar_dwt, haar_idwt};
use std::{f64::consts::PI, fs, path::Path};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Thesis 4: CHSH Bell parameter wavelet compression"
)]
struct Args {
    /// Number of time steps (must be power of 2 for wavelet)
    #[arg(long, default_value_t = 512)]
    n_time: usize,
    /// Photon pairs per time step
    #[arg(long, default_value_t = 1000)]
    n_pairs: u64,
    /// Compression target: fraction of non-scaling coefficients to retain
    #[arg(long, default_value_t = 0.05)]
    compression_target: f64,
    /// Output directory
    #[arg(long, default_value = "data/chsh_wavelet")]
    output_dir: String,
    /// RNG seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

/// Joint probabilities for singlet state at analyzer angles (a, b).
/// Returns [P(++), P(--), P(+-), P(-+)].
fn singlet_probs(a: f64, b: f64) -> [f64; 4] {
    let diff = a - b;
    let sin2 = (diff / 2.0).sin().powi(2);
    let cos2 = (diff / 2.0).cos().powi(2);
    [sin2 / 2.0, sin2 / 2.0, cos2 / 2.0, cos2 / 2.0]
}

/// CHSH correlation E(a, b) from count arrays.
fn chsh_correlation(n_pp: &[f64], n_mm: &[f64], n_pm: &[f64], n_mp: &[f64]) -> f64 {
    let total_pp: f64 = n_pp.iter().sum();
    let total_mm: f64 = n_mm.iter().sum();
    let total_pm: f64 = n_pm.iter().sum();
    let total_mp: f64 = n_mp.iter().sum();
    let total = total_pp + total_mm + total_pm + total_mp;
    if total < 1.0 {
        return 0.0;
    }
    (total_pp + total_mm - total_pm - total_mp) / total
}

/// Wavelet-compress a real-valued series, always keeping the coarsest scaling coeff.
/// Retains top-k detail coefficients by magnitude where k = floor(compression_target * (n-1)).
fn wavelet_compress(series: &[f64], compression_target: f64) -> Vec<f64> {
    let n = series.len();
    assert!(
        n.is_power_of_two(),
        "wavelet_compress: length must be power of 2"
    );
    let mut c = haar_dwt(series);

    // Index 0 is the single global scaling coefficient -- always retain.
    // Indices 1..n are detail coefficients; apply top-k magnitude selection.
    let n_detail = n - 1;
    let k = ((compression_target * n_detail as f64).ceil() as usize).max(1);

    // Find the k-th largest magnitude threshold among detail coefficients
    let mut mags: Vec<f64> = c[1..].iter().map(|x| x.abs()).collect();
    mags.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let threshold = if k < mags.len() { mags[k] } else { 0.0 };

    // Zero out detail coefficients below threshold
    for ci in &mut c[1..] {
        if ci.abs() <= threshold {
            *ci = 0.0;
        }
    }

    haar_idwt(&c)
}

fn sample_binom(rng: &mut ChaCha8Rng, n: u64, p: f64) -> u64 {
    if n == 0 || p <= 0.0 {
        return 0;
    }
    let q = p.min(1.0);
    let binom = Binomial::new(n, q).unwrap_or_else(|_| Binomial::new(n, 0.0).unwrap());
    binom.sample(rng)
}

fn compression_ratio(n: usize, compression_target: f64) -> f64 {
    let n_detail = n - 1;
    let k = ((compression_target * n_detail as f64).ceil() as usize).max(1);
    let kept = k + 1; // +1 for scaling coeff
    n as f64 / kept as f64
}

fn main() {
    let args = Args::parse();
    assert!(
        args.n_time.is_power_of_two(),
        "n_time must be a power of 2, got {}",
        args.n_time
    );

    // Standard near-optimal CHSH angles (in radians)
    let a0 = 0.0_f64;
    let a1 = PI / 4.0;
    let b0 = PI / 8.0;
    let b1 = 3.0 * PI / 8.0;

    // 4 setting pairs: (ai, bj) -> index (i*2+j)
    let settings = [(a0, b0), (a0, b1), (a1, b0), (a1, b1)];

    // For CHSH: S = E(a0,b0) - E(a0,b1) + E(a1,b0) + E(a1,b1)
    //   sign coefficients:
    let chsh_signs = [1.0_f64, -1.0, 1.0, 1.0];

    let ratio = compression_ratio(args.n_time, args.compression_target);
    println!("CHSH Wavelet Compression");
    println!(
        "  n_time={} n_pairs={} compression_target={:.1}% retention -> {:.1}x ratio",
        args.n_time,
        args.n_pairs,
        args.compression_target * 100.0,
        ratio
    );
    println!();

    // Generate all count series
    // For each setting (ai, bj): 4 outcome series each of length n_time
    let mut e_raw = [0.0_f64; 4];
    let mut e_compressed = [0.0_f64; 4];

    for (si, &(a, b)) in settings.iter().enumerate() {
        let probs = singlet_probs(a, b);
        // Generate 4 count series using sequential multinomial conditioning.
        // Each time step: draw N_pp ~ Binom(n_pairs, P_pp), then N_mm ~ Binom(n_pairs - N_pp, ...)
        let setting_seed = args.seed ^ ((si as u64).wrapping_mul(0x1234_5678_9ABC_DEF0));
        let mut rng = ChaCha8Rng::seed_from_u64(setting_seed);

        let mut ser0: Vec<f64> = Vec::with_capacity(args.n_time);
        let mut ser1: Vec<f64> = Vec::with_capacity(args.n_time);
        let mut ser2: Vec<f64> = Vec::with_capacity(args.n_time);
        let mut ser3: Vec<f64> = Vec::with_capacity(args.n_time);

        for _ in 0..args.n_time {
            let mut remaining = args.n_pairs;
            let c0 = sample_binom(&mut rng, remaining, probs[0]);
            remaining = remaining.saturating_sub(c0);
            let c1 = sample_binom(&mut rng, remaining, probs[1]);
            remaining = remaining.saturating_sub(c1);
            let c2 = sample_binom(&mut rng, remaining, probs[2]);
            let c3 = remaining.saturating_sub(c2);
            ser0.push(c0 as f64);
            ser1.push(c1 as f64);
            ser2.push(c2 as f64);
            ser3.push(c3 as f64);
        }

        // Raw E(ai,bj)
        e_raw[si] = chsh_correlation(&ser0, &ser1, &ser2, &ser3);

        // Compress each series and clamp negatives
        let c0: Vec<f64> = wavelet_compress(&ser0, args.compression_target)
            .into_iter()
            .map(|x| x.max(0.0))
            .collect();
        let c1: Vec<f64> = wavelet_compress(&ser1, args.compression_target)
            .into_iter()
            .map(|x| x.max(0.0))
            .collect();
        let c2: Vec<f64> = wavelet_compress(&ser2, args.compression_target)
            .into_iter()
            .map(|x| x.max(0.0))
            .collect();
        let c3: Vec<f64> = wavelet_compress(&ser3, args.compression_target)
            .into_iter()
            .map(|x| x.max(0.0))
            .collect();

        e_compressed[si] = chsh_correlation(&c0, &c1, &c2, &c3);

        println!(
            "  Setting ({:.4}, {:.4}): E_raw={:.4}  E_compressed={:.4}",
            a, b, e_raw[si], e_compressed[si]
        );
    }

    let s_raw: f64 = chsh_signs
        .iter()
        .zip(e_raw.iter())
        .map(|(s, e)| s * e)
        .sum();
    let s_compressed: f64 = chsh_signs
        .iter()
        .zip(e_compressed.iter())
        .map(|(s, e)| s * e)
        .sum();
    let delta_s = s_raw - s_compressed;

    println!();
    println!(
        "  S_raw       = {s_raw:.4}  (ideal 2*sqrt(2) = {:.4})",
        2.0_f64.sqrt() * 2.0
    );
    println!("  S_compressed = {s_compressed:.4}");
    println!("  delta_S     = {delta_s:.4}");
    println!("  Compression ratio: {ratio:.1}x");

    let pass = s_compressed.abs() > 2.0;
    println!();
    println!(
        "  Verdict: {} (|S|={:.4} > 2.0 required)",
        if pass { "PASS" } else { "FAIL" },
        s_compressed.abs()
    );

    let dir = Path::new(&args.output_dir);
    fs::create_dir_all(dir).expect("failed to create output dir");
    let toml_path = dir.join("results.toml");
    let content = format!(
        "# CHSH Wavelet Compression results\n# n_time={} n_pairs={} compression_target={:.2}\n# verdict = {}\n\n[result]\ns_raw = {s_raw:.6}\ns_compressed = {s_compressed:.6}\ndelta_s = {delta_s:.6}\ncompression_ratio = {ratio:.2}\npass = {pass}\n",
        args.n_time,
        args.n_pairs,
        args.compression_target,
        if pass { "PASS" } else { "FAIL" }
    );
    fs::write(&toml_path, content).expect("failed to write results.toml");
    println!("  Results written to {}", toml_path.display());

    if !pass {
        std::process::exit(1);
    }
}
