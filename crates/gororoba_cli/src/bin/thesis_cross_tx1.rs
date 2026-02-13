//! TX-1: Cross-Thesis T1 x T4 -- Frustration-Modulated Collision Dynamics
//!
//! Tests whether frustration topology modulates collision dynamics:
//! does spatially-varying frustration density change the return-time
//! scaling exponent (gamma) compared to a uniform reference?
//!
//! Pipeline:
//! 1. Generate SedenionField, compute local frustration density
//! 2. Uniform reference: simulate_shell_return_storm (no modulation)
//! 3. For each alpha: scale frustration -> collision noise modulation
//!    via simulate_frustration_modulated_storm
//! 4. Compare gamma exponents and latency law classifications
//! 5. Gate: |gamma_modulated - gamma_uniform| > 0.1 for any alpha

use clap::Parser;
use lattice_filtration::{
    classify_latency_law_detailed, simulate_frustration_modulated_storm,
    simulate_shell_return_storm, FrustrationStormConfig, LatencyLawDetail,
};
use std::fmt::Write as _;
use vacuum_frustration::bridge::SedenionField;

#[derive(Parser, Debug)]
#[command(name = "thesis-cross-tx1")]
#[command(about = "TX-1: Frustration-modulated collision dynamics (T1 x T4)")]
struct Args {
    /// Grid size per axis (N^3 cells for frustration field)
    #[arg(long, default_value = "16")]
    grid_size: usize,

    /// Number of random walk steps
    #[arg(long, default_value = "50000")]
    n_steps: usize,

    /// Number of radial shells for return-time binning
    #[arg(long, default_value = "16")]
    n_shells: usize,

    /// PRNG seed
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Alpha values for coupling sweep (comma-separated)
    #[arg(long, default_value = "0.0,0.5,1.0,2.0,5.0,10.0,50.0")]
    alphas: String,

    /// Output directory
    #[arg(long, default_value = "data/thesis_lab/tx1")]
    output_dir: String,
}

/// Result for one alpha value.
struct Tx1Result {
    alpha: f64,
    gamma: f64,
    power_law_r2: f64,
    latency_law: String,
    n_shells_populated: usize,
    n_unique_keys: usize,
    key_reuse_fraction: f64,
    detail: LatencyLawDetail,
}

/// Generate SedenionField with spatial variation (consistent with other binaries).
fn generate_sedenion_field(nx: usize) -> SedenionField {
    let mut field = SedenionField::uniform(nx, nx, nx);
    let mut state = 42_u64;
    let mut next_rand = || -> f64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64) / (u64::MAX as f64) * 2.0 - 1.0
    };
    let pi2 = std::f64::consts::PI * 2.0;
    for z in 0..nx {
        for y in 0..nx {
            for x in 0..nx {
                let s = field.get_mut(x, y, z);
                let xn = x as f64 / nx as f64;
                let yn = y as f64 / nx as f64;
                let zn = z as f64 / nx as f64;
                s[1] = 0.3 * (pi2 * xn).sin();
                s[3] = 0.2 * (pi2 * 2.0 * yn).cos();
                s[5] = 0.15 * (pi2 * xn + pi2 * zn).sin();
                s[7] = 0.15 * zn;
                s[9] = 0.1 * (pi2 * 3.0 * xn).cos();
                s[11] = 0.1 * (pi2 * yn * 2.0).sin();
                for component in s.iter_mut().take(16) {
                    *component += 0.05 * next_rand();
                }
            }
        }
    }
    field
}

/// Build (radius, return_time) samples from ShellReturnBins.
fn bins_to_samples(bins: &[lattice_filtration::ShellReturnBin]) -> Vec<(f64, f64)> {
    bins.iter()
        .filter(|b| b.n_returns > 0)
        .map(|b| (b.radius, b.mean_return_time))
        .collect()
}

/// Format a LatencyLaw variant as string.
fn latency_law_label(law: &lattice_filtration::LatencyLaw) -> &'static str {
    match law {
        lattice_filtration::LatencyLaw::InverseSquare => "InverseSquare",
        lattice_filtration::LatencyLaw::PowerLaw => "PowerLaw",
        lattice_filtration::LatencyLaw::Linear => "Linear",
        lattice_filtration::LatencyLaw::Exponential => "Exponential",
        lattice_filtration::LatencyLaw::Uniform => "Uniform",
        lattice_filtration::LatencyLaw::Undetermined => "Undetermined",
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let nx = args.grid_size;
    let n_cells = nx * nx * nx;

    let alphas: Vec<f64> = args
        .alphas
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("TX-1: Frustration-Modulated Collision Dynamics (T1 x T4)");
    println!("========================================================");
    println!("Grid: {}^3 ({} cells)", nx, n_cells);
    println!(
        "Walk steps: {}, shells: {}, seed: {}",
        args.n_steps, args.n_shells, args.seed
    );
    println!("Alpha sweep: {:?}", alphas);
    println!();

    // Step 1: Generate SedenionField and compute frustration
    println!("[1/4] Generating SedenionField and frustration density...");
    let field = generate_sedenion_field(nx);
    let frustration = field.local_frustration_density(16);
    drop(field);

    let mean_f = frustration.iter().sum::<f64>() / n_cells as f64;
    let f_min = frustration.iter().cloned().fold(f64::INFINITY, f64::min);
    let f_max = frustration
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  Frustration: mean={:.6}, min={:.6}, max={:.6}, range={:.6}",
        mean_f,
        f_min,
        f_max,
        f_max - f_min,
    );

    // Step 2: Uniform reference (no frustration modulation)
    println!("[2/4] Uniform reference walk...");
    let (ref_stats, ref_bins) =
        simulate_shell_return_storm(args.n_steps, 16, args.seed, args.n_shells);
    let ref_samples = bins_to_samples(&ref_bins);
    let ref_detail = classify_latency_law_detailed(&ref_samples);
    let ref_gamma = ref_stats.power_law_gamma;
    println!(
        "  Reference: gamma={:.4}, R2={:.4}, law={}, shells={}",
        ref_gamma,
        ref_stats.power_law_r2,
        latency_law_label(&ref_stats.latency_law),
        ref_stats.n_shells_populated,
    );

    // Step 3: Alpha sweep
    println!("[3/4] Alpha sweep ({} values)...", alphas.len());
    let mut results = Vec::new();

    for (i, &alpha) in alphas.iter().enumerate() {
        print!("  [{}/{}] alpha={:.1}... ", i + 1, alphas.len(), alpha);

        if alpha.abs() < 1e-12 {
            // alpha=0 is the reference case
            results.push(Tx1Result {
                alpha,
                gamma: ref_gamma,
                power_law_r2: ref_stats.power_law_r2,
                latency_law: latency_law_label(&ref_stats.latency_law).to_string(),
                n_shells_populated: ref_stats.n_shells_populated,
                n_unique_keys: ref_stats.n_unique_keys,
                key_reuse_fraction: ref_stats.key_reuse_fraction,
                detail: ref_detail,
            });
            println!(
                "gamma={:.4}, R2={:.4} (reference)",
                ref_gamma, ref_stats.power_law_r2,
            );
            continue;
        }

        let cfg = FrustrationStormConfig {
            n_steps: args.n_steps,
            seed: args.seed,
            n_shells: args.n_shells,
            frustration_field: &frustration,
            field_nx: nx,
            field_ny: nx,
            field_nz: nx,
            alpha,
        };
        let (stats, bins) = simulate_frustration_modulated_storm(&cfg);
        let samples = bins_to_samples(&bins);
        let detail = classify_latency_law_detailed(&samples);

        println!(
            "gamma={:.4}, R2={:.4}, law={}, delta_gamma={:.4}",
            stats.power_law_gamma,
            stats.power_law_r2,
            latency_law_label(&stats.latency_law),
            (stats.power_law_gamma - ref_gamma).abs(),
        );

        results.push(Tx1Result {
            alpha,
            gamma: stats.power_law_gamma,
            power_law_r2: stats.power_law_r2,
            latency_law: latency_law_label(&stats.latency_law).to_string(),
            n_shells_populated: stats.n_shells_populated,
            n_unique_keys: stats.n_unique_keys,
            key_reuse_fraction: stats.key_reuse_fraction,
            detail,
        });
    }

    // Step 4: Output
    println!("[4/4] Writing output...");
    std::fs::create_dir_all(&args.output_dir)?;

    let mut report = String::new();
    let _ = writeln!(report, "[metadata]");
    let _ = writeln!(report, "experiment = \"TX-1\"");
    let _ = writeln!(
        report,
        "description = \"Frustration-modulated collision dynamics (T1 x T4)\""
    );
    let _ = writeln!(report, "grid_size = {}", nx);
    let _ = writeln!(report, "n_steps = {}", args.n_steps);
    let _ = writeln!(report, "n_shells = {}", args.n_shells);
    let _ = writeln!(report, "seed = {}", args.seed);
    let _ = writeln!(report, "mean_frustration = {:.8}", mean_f);
    let _ = writeln!(report, "frustration_range = {:.8}", f_max - f_min);
    let _ = writeln!(report);

    let _ = writeln!(report, "[reference]");
    let _ = writeln!(report, "gamma = {:.6}", ref_gamma);
    let _ = writeln!(report, "power_law_r2 = {:.6}", ref_stats.power_law_r2);
    let _ = writeln!(
        report,
        "latency_law = \"{}\"",
        latency_law_label(&ref_stats.latency_law)
    );
    let _ = writeln!(report);

    for r in &results {
        let _ = writeln!(report, "[[alpha_sweep]]");
        let _ = writeln!(report, "alpha = {:.3}", r.alpha);
        let _ = writeln!(report, "gamma = {:.6}", r.gamma);
        let _ = writeln!(report, "power_law_r2 = {:.6}", r.power_law_r2);
        let _ = writeln!(report, "delta_gamma = {:.6}", (r.gamma - ref_gamma).abs());
        let _ = writeln!(report, "latency_law = \"{}\"", r.latency_law);
        let _ = writeln!(report, "n_shells_populated = {}", r.n_shells_populated);
        let _ = writeln!(report, "n_unique_keys = {}", r.n_unique_keys);
        let _ = writeln!(report, "key_reuse_fraction = {:.6}", r.key_reuse_fraction);
        let _ = writeln!(
            report,
            "r2_inverse_square = {:.6}",
            r.detail.r2_inverse_square
        );
        let _ = writeln!(report, "r2_power_law = {:.6}", r.detail.r2_power_law);
        let _ = writeln!(report, "r2_linear = {:.6}", r.detail.r2_linear);
        let _ = writeln!(report, "r2_exponential = {:.6}", r.detail.r2_exponential);
        let _ = writeln!(report);
    }

    // Summary
    let best_delta = results
        .iter()
        .filter(|r| r.alpha > 0.0)
        .map(|r| (r.gamma - ref_gamma).abs())
        .fold(0.0_f64, f64::max);
    let _ = writeln!(report, "[summary]");
    let _ = writeln!(report, "n_alphas = {}", results.len());
    let _ = writeln!(report, "reference_gamma = {:.6}", ref_gamma);
    let _ = writeln!(report, "max_delta_gamma = {:.6}", best_delta);
    let _ = writeln!(report, "significant = {}", best_delta > 0.1);

    let report_path = format!("{}/tx1_report.toml", args.output_dir);
    std::fs::write(&report_path, &report)?;
    println!("Report: {}", report_path);

    println!();
    println!("========================================================");
    println!(
        "TX-1 Summary: ref_gamma={:.4}, max_delta_gamma={:.4}, significant={}",
        ref_gamma,
        best_delta,
        best_delta > 0.1,
    );

    Ok(())
}
