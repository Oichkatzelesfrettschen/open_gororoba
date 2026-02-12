use clap::Parser;
use lattice_filtration::{simulate_fibonacci_collision_storm, LatencyLaw};
use lbm_core::viscosity_with_associator;
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
    let p2 = args.output_dir.join("thesis2_thickening_threshold.toml");
    let p3 = args.output_dir.join("thesis3_epoch_alignment.toml");
    let p4 = args.output_dir.join("thesis4_latency_law_suite.toml");

    fs::write(&p1, thesis1_scalar_tov_report(args.run_tov))?;
    fs::write(&p2, thesis2_thickening_report())?;
    fs::write(&p3, thesis3_epoch_alignment_report())?;
    fs::write(&p4, thesis4_latency_law_report())?;

    println!("wrote {}", p1.display());
    println!("wrote {}", p2.display());
    println!("wrote {}", p3.display());
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

fn thesis2_thickening_report() -> String {
    let alphas = [0.1_f64, 0.3, 0.6, 1.0];
    let norms = [0.0_f64, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2];
    let nu_base = 0.05_f64;
    let threshold_radians = 1.2_f64;

    let mut out = String::new();
    let _ = writeln!(out, "[artifact]");
    let _ = writeln!(out, "id = \"STPT-007\"");
    let _ = writeln!(out, "updated = \"2026-02-12\"");
    let _ = writeln!(out, "nu_base = {:.6}", nu_base);
    let _ = writeln!(out, "threshold_radians = {:.6}", threshold_radians);
    let _ = writeln!(out);

    let mut crossing_count = 0usize;

    for alpha in alphas {
        let mut values = Vec::with_capacity(norms.len());
        for n in norms {
            values.push(viscosity_with_associator(nu_base, alpha, n));
        }

        let pre_slope = (values[2] - values[0]) / (norms[2] - norms[0]);
        let post_slope = (values[6] - values[4]) / (norms[6] - norms[4]);
        let non_newtonian = post_slope > pre_slope * 1.05;
        if non_newtonian {
            crossing_count += 1;
        }

        let _ = writeln!(out, "[[series]]");
        let _ = writeln!(out, "alpha = {:.3}", alpha);
        let _ = writeln!(out, "pre_slope = {:.6}", pre_slope);
        let _ = writeln!(out, "post_slope = {:.6}", post_slope);
        let _ = writeln!(out, "non_newtonian = {}", non_newtonian);
        let _ = writeln!(out, "norms = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]");
        let _ = writeln!(
            out,
            "nu_eff = [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
            values[0], values[1], values[2], values[3], values[4], values[5], values[6]
        );
        let _ = writeln!(out);
    }

    let _ = writeln!(out, "[summary]");
    let _ = writeln!(out, "series_count = {}", alphas.len());
    let _ = writeln!(out, "threshold_crossing_count = {}", crossing_count);
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

fn thesis4_latency_law_report() -> String {
    let settings = [(200usize, 13usize), (300, 17), (500, 29)];
    let mut inverse_square_hits = 0usize;

    let mut out = String::new();
    let _ = writeln!(out, "[artifact]");
    let _ = writeln!(out, "id = \"STPT-009\"");
    let _ = writeln!(out, "updated = \"2026-02-12\"");
    let _ = writeln!(out);

    for (steps, buckets) in settings {
        let (stats, obs) = simulate_fibonacci_collision_storm(steps, buckets);
        if matches!(stats.latency_law, LatencyLaw::InverseSquare) {
            inverse_square_hits += 1;
        }

        let _ = writeln!(out, "[[run]]");
        let _ = writeln!(out, "steps = {}", steps);
        let _ = writeln!(out, "buckets = {}", buckets);
        let _ = writeln!(out, "total_collisions = {}", stats.total_collisions);
        let _ = writeln!(
            out,
            "peak_bucket_occupancy = {}",
            stats.peak_bucket_occupancy
        );
        let _ = writeln!(out, "mean_latency = {:.8}", stats.mean_latency);
        let _ = writeln!(out, "inverse_square_r2 = {:.8}", stats.inverse_square_r2);
        let _ = writeln!(
            out,
            "latency_law = \"{}\"",
            latency_law_name(stats.latency_law)
        );

        let preview = obs.len().min(12);
        let _ = writeln!(out, "preview_steps = {}", preview);
        let _ = writeln!(out, "preview = [");
        for o in obs.iter().take(preview) {
            let _ = writeln!(
                out,
                "  {{ step = {}, radius = {:.6}, latency = {:.8}, collisions = {} }},",
                o.step, o.radius, o.latency, o.collisions
            );
        }
        let _ = writeln!(out, "]");
        let _ = writeln!(out);
    }

    let verdict = if inverse_square_hits > 0 {
        "supported"
    } else {
        "refuted"
    };
    let _ = writeln!(out, "[summary]");
    let _ = writeln!(out, "run_count = {}", settings.len());
    let _ = writeln!(out, "inverse_square_hits = {}", inverse_square_hits);
    let _ = writeln!(out, "verdict = \"{}\"", verdict);
    out
}

fn latency_law_name(law: LatencyLaw) -> &'static str {
    match law {
        LatencyLaw::InverseSquare => "inverse_square",
        LatencyLaw::Linear => "linear",
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
