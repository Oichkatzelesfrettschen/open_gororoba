use clap::Parser;
use lattice_filtration::{
    classify_latency_law_detailed, simulate_fibonacci_collision_storm,
    simulate_sedenion_collision_storm, LatencyLaw,
};
use lbm_core::viscosity_with_power_law_associator;
use neural_homotopy::{reference_hubble_curve, train_homotopy_surrogate, HomotopyTrainingConfig};
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use vacuum_frustration::{
    evaluate_frustration_star, FrustrationStarConfig, ScalarFrustrationMap,
    CASSINI_OMEGA_BD_LOWER_BOUND,
};

#[derive(Debug, Parser)]
#[command(name = "thesis-program-sweep")]
#[command(about = "Generate deterministic thesis evidence artifacts for STPT-006..009")]
struct Args {
    #[arg(long, default_value = "data/evidence")]
    output_dir: PathBuf,
    #[arg(long, default_value_t = false)]
    run_tov: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    fs::create_dir_all(&args.output_dir)?;

    let p1 = args.output_dir.join("thesis1_scalar_tov_sweep.toml");
    let p2 = args.output_dir.join("thesis2_power_law_viscosity_v2.toml");
    let p3 = args.output_dir.join("thesis3_epoch_alignment.toml");
    let p4 = args.output_dir.join("thesis4_latency_law_suite_v2.toml");

    fs::write(&p1, thesis1_scalar_tov_report(args.run_tov))?;
    println!("wrote {}", p1.display());

    fs::write(&p2, thesis2_thickening_report())?;
    println!("wrote {}", p2.display());

    fs::write(&p3, thesis3_epoch_alignment_report())?;
    println!("wrote {}", p3.display());

    fs::write(&p4, thesis4_latency_law_report())?;
    println!("wrote {}", p4.display());

    Ok(())
}

fn thesis1_scalar_tov_report(run_tov: bool) -> String {
    let lambdas = [0.5_f64, 1.0, 2.0, 4.0];
    let frustrations = [0.125_f64, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875];
    let cfg = FrustrationStarConfig {
        run_tov,
        ..FrustrationStarConfig::default()
    };

    let mut out = String::new();
    let mut pass_count = 0usize;
    let mut fail_count = 0usize;
    let mut sample_count = 0usize;

    let _ = writeln!(out, "[artifact]");
    let _ = writeln!(out, "id = \"STPT-006\"");
    let _ = writeln!(out, "updated = \"2026-02-12\"");
    let _ = writeln!(out, "run_tov = {}", run_tov);
    let _ = writeln!(
        out,
        "cassini_lower_bound = {:.1}",
        CASSINI_OMEGA_BD_LOWER_BOUND
    );
    let _ = writeln!(out, "lambda_values = [0.5, 1.0, 2.0, 4.0]");
    let _ = writeln!(
        out,
        "frustration_values = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]"
    );
    let _ = writeln!(out);

    for lambda in lambdas {
        let map = ScalarFrustrationMap::new(lambda);
        for frustration in frustrations {
            let field = vec![frustration; 64];
            let result = evaluate_frustration_star(&field, map, &cfg);
            let verdict = if result.cassini_violation {
                "fail"
            } else {
                "pass"
            };
            if result.cassini_violation {
                fail_count += 1;
            } else {
                pass_count += 1;
            }
            sample_count += 1;

            let _ = writeln!(out, "[[sample]]");
            let _ = writeln!(out, "lambda = {:.3}", lambda);
            let _ = writeln!(out, "mean_frustration = {:.6}", result.mean_frustration);
            let _ = writeln!(out, "mean_phi = {:.6}", result.mean_phi);
            let _ = writeln!(out, "omega_eff = {:.6}", result.omega_eff);
            let _ = writeln!(out, "cassini_verdict = \"{}\"", verdict);
            let _ = writeln!(out, "obstruction_norm = {:.6}", result.obstruction_norm);
            let _ = writeln!(out, "coupling = {:.6}", result.coupling);
            let _ = writeln!(out);
        }
    }

    let _ = writeln!(out, "[summary]");
    let _ = writeln!(out, "sample_count = {}", sample_count);
    let _ = writeln!(out, "pass_count = {}", pass_count);
    let _ = writeln!(out, "fail_count = {}", fail_count);
    let _ = writeln!(
        out,
        "pass_ratio = {:.6}",
        pass_count as f64 / sample_count.max(1) as f64
    );
    out
}

/// Power-law viscosity model sweep (replaces linear viscosity_with_associator).
///
/// Tests for non-Newtonian behavior: nu_eff(gamma_dot) should be CONVEX
/// (post_slope / pre_slope > 1.05) for shear-thickening fluids.
///
/// Parameter space:
/// - alpha: coupling strength [0.1, 0.5, 1.0, 2.0]
/// - beta: associator exponent [0.5, 1.0, 2.0]
/// - power_index: n > 1 = thickening [1.2, 1.5, 2.0]
fn thesis2_thickening_report() -> String {
    let nu_base = 0.05_f64;
    let alphas = [0.1_f64, 0.5, 1.0, 2.0];
    let betas = [0.5_f64, 1.0, 2.0];
    let power_indices = [1.2_f64, 1.5, 2.0];
    let associator_norms = [0.0_f64, 0.2, 0.5, 0.8, 1.0];
    // Strain rate sweep: low -> high
    let strain_rates = [0.001_f64, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0];

    let mut out = String::new();
    let _ = writeln!(out, "[artifact]");
    let _ = writeln!(out, "id = \"STPT-007-v2\"");
    let _ = writeln!(out, "updated = \"2026-02-12\"");
    let _ = writeln!(out, "model = \"power_law_associator\"");
    let _ = writeln!(out, "nu_base = {:.6}", nu_base);
    let _ = writeln!(
        out,
        "formula = \"nu_base * (1 + alpha * norm^beta * |gamma_dot|^(n-1))\""
    );
    let _ = writeln!(out);

    let mut crossing_count = 0usize;
    let mut series_count = 0usize;

    for &alpha in &alphas {
        for &beta in &betas {
            for &n in &power_indices {
                // Pick a representative associator norm (mid-range)
                let assoc_norm = 0.5;

                // Compute nu_eff at each strain rate
                let values: Vec<f64> = strain_rates
                    .iter()
                    .map(|&gamma_dot| {
                        viscosity_with_power_law_associator(
                            nu_base, alpha, beta, assoc_norm, gamma_dot, n,
                        )
                    })
                    .collect();

                // Pre-slope: (nu at strain=0.05) - (nu at strain=0.001) / delta
                let pre_slope = (values[2] - values[0]) / (strain_rates[2] - strain_rates[0]);
                // Post-slope: (nu at strain=1.0) - (nu at strain=0.2) / delta
                let post_slope = (values[6] - values[4]) / (strain_rates[6] - strain_rates[4]);

                let non_newtonian = post_slope > pre_slope * 1.05;
                if non_newtonian {
                    crossing_count += 1;
                }
                series_count += 1;

                // Also check convexity: second derivative of nu(gamma_dot) > 0
                let mut convex = true;
                for w in values.windows(3) {
                    let second_deriv = w[2] - 2.0 * w[1] + w[0];
                    if second_deriv < -1e-10 {
                        convex = false;
                        break;
                    }
                }

                let _ = writeln!(out, "[[series]]");
                let _ = writeln!(out, "alpha = {:.3}", alpha);
                let _ = writeln!(out, "beta = {:.3}", beta);
                let _ = writeln!(out, "power_index = {:.3}", n);
                let _ = writeln!(out, "associator_norm = {:.3}", assoc_norm);
                let _ = writeln!(out, "pre_slope = {:.8}", pre_slope);
                let _ = writeln!(out, "post_slope = {:.8}", post_slope);
                let _ = writeln!(
                    out,
                    "slope_ratio = {:.6}",
                    if pre_slope.abs() > 1e-12 {
                        post_slope / pre_slope
                    } else {
                        0.0
                    }
                );
                let _ = writeln!(out, "non_newtonian = {}", non_newtonian);
                let _ = writeln!(out, "convex = {}", convex);
                let _ = writeln!(out, "strain_rates = [{}]", join_f64(&strain_rates));
                let _ = writeln!(out, "nu_eff = [{}]", join_f64(&values));
                let _ = writeln!(out);

                // Also sweep across associator norms at fixed strain rate
                let gamma_dot_ref = 0.1;
                let norm_values: Vec<f64> = associator_norms
                    .iter()
                    .map(|&anorm| {
                        viscosity_with_power_law_associator(
                            nu_base,
                            alpha,
                            beta,
                            anorm,
                            gamma_dot_ref,
                            n,
                        )
                    })
                    .collect();
                let _ = writeln!(out, "[[norm_sweep]]");
                let _ = writeln!(out, "alpha = {:.3}", alpha);
                let _ = writeln!(out, "beta = {:.3}", beta);
                let _ = writeln!(out, "power_index = {:.3}", n);
                let _ = writeln!(out, "gamma_dot = {:.3}", gamma_dot_ref);
                let _ = writeln!(out, "associator_norms = [{}]", join_f64(&associator_norms));
                let _ = writeln!(out, "nu_eff = [{}]", join_f64(&norm_values));
                let _ = writeln!(out);
            }
        }
    }

    let _ = writeln!(out, "[summary]");
    let _ = writeln!(out, "series_count = {}", series_count);
    let _ = writeln!(out, "threshold_crossing_count = {}", crossing_count);
    let _ = writeln!(
        out,
        "crossing_ratio = {:.6}",
        crossing_count as f64 / series_count.max(1) as f64
    );
    let _ = writeln!(
        out,
        "verdict = \"{}\"",
        if crossing_count > 0 {
            "supported"
        } else {
            "refuted"
        }
    );
    out
}

fn thesis3_epoch_alignment_report() -> String {
    let cfg = HomotopyTrainingConfig::default();
    let trace = train_homotopy_surrogate(cfg);
    let hubble = reference_hubble_curve(cfg.epochs);

    let epoch_markers = [
        (6usize, "inflation_exit"),
        (16usize, "reheating"),
        (40usize, "matter_dark_energy_transition"),
        (52usize, "dark_energy_onset"),
    ];

    let (plateau_epoch, plateau_label) = match trace.plateau_epoch {
        Some(ep) => {
            let mut label = "unmapped";
            for (marker, name) in epoch_markers {
                let delta = marker.abs_diff(ep);
                if delta <= 5 {
                    label = name;
                    break;
                }
            }
            (ep as i64, label)
        }
        None => (-1, "none"),
    };

    let verdict = if trace.plateau_epoch.is_some() && plateau_label == "unmapped" {
        "refuted"
    } else {
        "supported"
    };

    let mut out = String::new();
    let _ = writeln!(out, "[artifact]");
    let _ = writeln!(out, "id = \"STPT-008\"");
    let _ = writeln!(out, "updated = \"2026-02-12\"");
    let _ = writeln!(out, "epochs = {}", cfg.epochs);
    let _ = writeln!(out, "learning_rate = {:.6}", cfg.learning_rate);
    let _ = writeln!(out, "plateau_tolerance = {:.8}", cfg.plateau_tolerance);
    let _ = writeln!(out, "pentagon_residual = {:.6}", trace.pentagon_residual);
    let _ = writeln!(out, "hubble_alignment = {:.6}", trace.hubble_alignment);
    let _ = writeln!(out, "plateau_epoch = {}", plateau_epoch);
    let _ = writeln!(out, "plateau_label = \"{}\"", plateau_label);
    let _ = writeln!(out, "verdict = \"{}\"", verdict);
    let _ = writeln!(out);

    let n = trace.losses.len().min(16);
    let _ = writeln!(out, "[trace_preview]");
    let _ = writeln!(out, "sample_count = {}", n);
    let _ = writeln!(out, "losses = [{}]", join_f64(&trace.losses[..n]));
    let _ = writeln!(out, "hubble = [{}]", join_f64(&hubble[..n]));
    out
}

/// Latency law report with production-scale parameters and convergence test.
///
/// Improvements over v1:
/// 1. Larger step counts: 2000, 5000, 10000, 50000 (vs 200, 300, 500)
/// 2. Prime bucket counts to avoid Pisano resonance: 61, 127, 251, 1021
/// 3. Both Fibonacci AND sedenion key streams for comparison
/// 4. Detailed latency law classifier with fitted exponents and all R^2 values
/// 5. Convergence criterion: R^2 must increase (or stay stable) with more steps
fn thesis4_latency_law_report() -> String {
    // Production-scale settings: (steps, buckets) with PRIME bucket counts
    let settings = [
        (2000_usize, 61_usize),
        (5000, 127),
        (10000, 251),
        (50000, 1021),
    ];

    let mut out = String::new();
    let _ = writeln!(out, "[artifact]");
    let _ = writeln!(out, "id = \"STPT-009-v2\"");
    let _ = writeln!(out, "updated = \"2026-02-12\"");
    let _ = writeln!(out, "latency_metric = \"return_time\"");
    let _ = writeln!(out, "key_streams = [\"fibonacci\", \"sedenion\"]");
    let _ = writeln!(out);

    let mut fib_inverse_square_hits = 0usize;
    let mut sed_inverse_square_hits = 0usize;
    let mut fib_r2_history: Vec<f64> = Vec::new();
    let mut sed_r2_history: Vec<f64> = Vec::new();

    // -- Fibonacci stream runs --
    let _ = writeln!(
        out,
        "# Fibonacci key stream (Pisano-limited, 32 distinct keys)"
    );
    let _ = writeln!(out);

    for &(steps, buckets) in &settings {
        let (stats, obs) = simulate_fibonacci_collision_storm(steps, buckets);
        let samples: Vec<(f64, f64)> = obs.iter().map(|o| (o.radius, o.latency)).collect();
        let detail = classify_latency_law_detailed(&samples);

        if matches!(detail.law, LatencyLaw::InverseSquare) {
            fib_inverse_square_hits += 1;
        }
        fib_r2_history.push(detail.r2_inverse_square);

        let _ = writeln!(out, "[[fibonacci_run]]");
        let _ = writeln!(out, "steps = {}", steps);
        let _ = writeln!(out, "buckets = {}", buckets);
        let _ = writeln!(out, "total_collisions = {}", stats.total_collisions);
        let _ = writeln!(
            out,
            "peak_bucket_occupancy = {}",
            stats.peak_bucket_occupancy
        );
        let _ = writeln!(out, "mean_latency = {:.8}", stats.mean_latency);
        let _ = writeln!(out, "r2_inverse_square = {:.8}", detail.r2_inverse_square);
        let _ = writeln!(out, "r2_power_law = {:.8}", detail.r2_power_law);
        let _ = writeln!(out, "power_law_exponent = {:.6}", detail.power_law_exponent);
        let _ = writeln!(out, "r2_linear = {:.8}", detail.r2_linear);
        let _ = writeln!(out, "r2_exponential = {:.8}", detail.r2_exponential);
        let _ = writeln!(out, "latency_law = \"{}\"", latency_law_name(detail.law));
        let _ = writeln!(out);
    }

    // -- Sedenion stream runs --
    let _ = writeln!(
        out,
        "# Sedenion key stream (high-dimensional, rich key space)"
    );
    let _ = writeln!(out);
    let seed = 42_u64;

    for &(steps, _buckets) in &settings {
        let (stats, obs) = simulate_sedenion_collision_storm(steps, 16, seed);
        let samples: Vec<(f64, f64)> = obs.iter().map(|o| (o.radius, o.latency)).collect();
        let detail = classify_latency_law_detailed(&samples);

        if matches!(detail.law, LatencyLaw::InverseSquare) {
            sed_inverse_square_hits += 1;
        }
        sed_r2_history.push(detail.r2_inverse_square);

        let _ = writeln!(out, "[[sedenion_run]]");
        let _ = writeln!(out, "steps = {}", steps);
        let _ = writeln!(out, "seed = {}", seed);
        let _ = writeln!(out, "total_collisions = {}", stats.total_collisions);
        let _ = writeln!(
            out,
            "peak_bucket_occupancy = {}",
            stats.peak_bucket_occupancy
        );
        let _ = writeln!(out, "mean_latency = {:.8}", stats.mean_latency);
        let _ = writeln!(out, "r2_inverse_square = {:.8}", detail.r2_inverse_square);
        let _ = writeln!(out, "r2_power_law = {:.8}", detail.r2_power_law);
        let _ = writeln!(out, "power_law_exponent = {:.6}", detail.power_law_exponent);
        let _ = writeln!(out, "r2_linear = {:.8}", detail.r2_linear);
        let _ = writeln!(out, "r2_exponential = {:.8}", detail.r2_exponential);
        let _ = writeln!(out, "latency_law = \"{}\"", latency_law_name(detail.law));
        let _ = writeln!(out);
    }

    // -- Convergence analysis --
    let fib_converging = is_r2_converging(&fib_r2_history);
    let sed_converging = is_r2_converging(&sed_r2_history);

    let total_hits = fib_inverse_square_hits + sed_inverse_square_hits;
    let total_runs = settings.len() * 2;
    let verdict = if total_hits > 0 && (fib_converging || sed_converging) {
        "supported"
    } else if total_hits > 0 {
        "weak_support"
    } else {
        "refuted"
    };

    let _ = writeln!(out, "[convergence]");
    let _ = writeln!(
        out,
        "fibonacci_r2_history = [{}]",
        join_f64(&fib_r2_history)
    );
    let _ = writeln!(out, "sedenion_r2_history = [{}]", join_f64(&sed_r2_history));
    let _ = writeln!(out, "fibonacci_converging = {}", fib_converging);
    let _ = writeln!(out, "sedenion_converging = {}", sed_converging);
    let _ = writeln!(out);

    let _ = writeln!(out, "[summary]");
    let _ = writeln!(out, "run_count = {}", total_runs);
    let _ = writeln!(
        out,
        "fibonacci_inverse_square_hits = {}",
        fib_inverse_square_hits
    );
    let _ = writeln!(
        out,
        "sedenion_inverse_square_hits = {}",
        sed_inverse_square_hits
    );
    let _ = writeln!(out, "total_inverse_square_hits = {}", total_hits);
    let _ = writeln!(out, "verdict = \"{}\"", verdict);
    out
}

/// Check if R^2 values are converging (non-decreasing trend).
///
/// R^2 should increase or stabilize with more steps, indicating that the
/// fit is genuine rather than a statistical fluke that washes out.
fn is_r2_converging(history: &[f64]) -> bool {
    if history.len() < 2 {
        return false;
    }
    // Allow small regression (0.02) but overall trend must be non-decreasing
    let mut decreases = 0;
    for w in history.windows(2) {
        if w[1] < w[0] - 0.02 {
            decreases += 1;
        }
    }
    decreases == 0
}

fn latency_law_name(law: LatencyLaw) -> &'static str {
    match law {
        LatencyLaw::InverseSquare => "inverse_square",
        LatencyLaw::PowerLaw => "power_law",
        LatencyLaw::Linear => "linear",
        LatencyLaw::Exponential => "exponential",
        LatencyLaw::Uniform => "uniform",
        LatencyLaw::Undetermined => "undetermined",
    }
}

fn join_f64(values: &[f64]) -> String {
    let mut out = String::new();
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{:.8}", v);
    }
    out
}

#[allow(dead_code)]
fn _assert_output_dir_exists(path: &Path) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        Ok(())
    } else {
        Err(format!("missing output directory: {}", path.display()).into())
    }
}
