//! Entropy trap and C-010 projection assessment CLI (Sprint 51, NA-006).
//!
//! Combines absorber topology benchmarks with entropy-trap analysis to
//! assess the C-010 non-local coupling claim.
//!
//! Subcommands:
//! - `compare`: Run topology benchmark and entropy analysis, print summary
//! - `sweep`:   Sweep coupling strength kappa, report isolation vs kappa
//! - `gate`:    Run projection gate, exit 0 on PASS, 1 on FAIL

use clap::{Parser, Subcommand};
use optics_core::absorber_benchmark::{
    run_benchmark, reduced_c010_suite, standard_c010_suite, ProjectionGate,
};
use optics_core::tcmt::{InputField, KerrCavity};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "entropy-trap",
    about = "C-010 topology benchmark and entropy-trap assessment"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run topology comparison and print summary table.
    Compare {
        /// Coupling strength kappa (default 0.05).
        #[arg(long, default_value_t = 0.05)]
        kappa: f64,

        /// Use full 7xK6 suite (42 nodes) instead of reduced 2xK3.
        #[arg(long)]
        full: bool,

        /// Output directory for TOML results.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Sweep coupling strength and report isolation trend.
    Sweep {
        /// Minimum kappa value.
        #[arg(long, default_value_t = 0.01)]
        kappa_min: f64,

        /// Maximum kappa value.
        #[arg(long, default_value_t = 0.20)]
        kappa_max: f64,

        /// Number of steps in sweep.
        #[arg(long, default_value_t = 10)]
        n_steps: usize,

        /// Use full 7xK6 suite.
        #[arg(long)]
        full: bool,

        /// Output directory for TOML results.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Run projection gate: exit 0 on PASS, 1 on FAIL.
    Gate {
        /// Coupling strength kappa (default 0.05).
        #[arg(long, default_value_t = 0.05)]
        kappa: f64,

        /// Use full 7xK6 suite.
        #[arg(long)]
        full: bool,

        /// Minimum isolation threshold override.
        #[arg(long)]
        min_isolation: Option<f64>,

        /// Maximum crosstalk dB override.
        #[arg(long)]
        max_xtalk_db: Option<f64>,

        /// Minimum dominance margin override.
        #[arg(long)]
        min_dominance: Option<f64>,
    },
}

/// Default base cavity for benchmarks.
fn default_cavity() -> KerrCavity {
    KerrCavity::new(
        2.0 * std::f64::consts::PI * 193.0e12, // 193 THz (telecom C-band)
        1e6,                                     // Q_intrinsic
        5e5,                                     // Q_external
        1.45,                                    // n_linear (silica)
        2.6e-20,                                 // n2 (silica Kerr, m^2/W)
        1e-18,                                   // V_eff (cubic micron)
    )
}

/// Default drive field for benchmarks.
fn default_drive() -> InputField {
    InputField {
        amplitude: num_complex::Complex64::new(1e-3, 0.0),
        omega: 2.0 * std::f64::consts::PI * 193.0e12,
    }
}

/// Build topologies from a suite selector.
fn build_suite(kappa: f64, full: bool) -> Vec<optics_core::absorber_benchmark::CouplingTopology> {
    if full {
        standard_c010_suite(kappa)
    } else {
        reduced_c010_suite(kappa)
    }
}

/// Run a single benchmark at given kappa.
fn run_at_kappa(
    kappa: f64,
    full: bool,
) -> optics_core::absorber_benchmark::ComparativeBenchmark {
    let topos = build_suite(kappa, full);
    let cavity = default_cavity();
    let _drive = default_drive();

    // Drive first clique_size nodes (target clique)
    let clique_size = if full { 6 } else { 3 };
    let driven: Vec<usize> = (0..clique_size).collect();

    run_benchmark(
        &topos,
        &cavity,
        1e9,  // frequency spacing (1 GHz)
        &driven,
        1e-3, // drive amplitude
        1e-12, // dt (1 ps)
        2000,  // steps
    )
}

/// Write benchmark results as TOML to a directory.
fn write_results_toml(
    dir: &std::path::Path,
    benchmark: &optics_core::absorber_benchmark::ComparativeBenchmark,
    kappa: f64,
) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("benchmark_kappa_{:.4}.toml", kappa));

    let mut lines = Vec::new();
    lines.push(format!("# Entropy-trap benchmark at kappa = {:.4}", kappa));
    lines.push(format!("kappa = {:.6}", kappa));
    lines.push(format!("n_steps = {}", benchmark.n_steps));
    lines.push(format!(
        "driven_nodes = [{}]",
        benchmark
            .driven_nodes
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    lines.push(String::new());

    for (i, r) in benchmark.results.iter().enumerate() {
        lines.push("[[topology]]".to_string());
        lines.push(format!("name = {:?}", r.topology_name));
        lines.push(format!("n_nodes = {}", r.n_nodes));
        lines.push(format!("isolation_ratio = {:.6}", r.isolation_ratio));
        let xtalk = if r.crosstalk_db.is_infinite() {
            "\"-inf\"".to_string()
        } else {
            format!("{:.2}", r.crosstalk_db)
        };
        lines.push(format!("crosstalk_db = {}", xtalk));
        lines.push(format!("spectral_gap = {:.6}", r.spectral_gap));
        lines.push(format!("n_components = {}", r.n_components));
        if i < benchmark.results.len() - 1 {
            lines.push(String::new());
        }
    }

    fs::write(&path, lines.join("\n") + "\n")?;
    eprintln!("Wrote: {}", path.display());
    Ok(())
}

fn main() {
    let args = Args::parse();

    match args.command {
        Command::Compare {
            kappa,
            full,
            output,
        } => {
            let suite_label = if full { "full 7xK6" } else { "reduced 2xK3" };
            println!("=== Entropy-Trap Topology Comparison ===");
            println!("Suite: {}, kappa = {:.4}", suite_label, kappa);
            println!();

            let benchmark = run_at_kappa(kappa, full);
            println!("{}", benchmark.summary_table());
            println!();

            let gate = ProjectionGate::c010_default();
            let result = gate.evaluate(&benchmark);
            println!("Projection Gate: {}", result.verdict);
            println!(
                "  ZD isolation:      {:.6}",
                result.zd_isolation
            );
            println!(
                "  Best local:        {:.6}",
                result.best_local_isolation
            );
            println!(
                "  Dominance margin:  {:.6}",
                result.dominance_margin
            );
            println!(
                "  ZD crosstalk:      {:.1} dB",
                result.zd_crosstalk_db
            );

            if let Some(dir) = output {
                write_results_toml(&dir, &benchmark, kappa)
                    .expect("failed to write results TOML");
            }
        }

        Command::Sweep {
            kappa_min,
            kappa_max,
            n_steps,
            full,
            output,
        } => {
            let suite_label = if full { "full 7xK6" } else { "reduced 2xK3" };
            println!("=== Entropy-Trap Coupling Sweep ===");
            println!(
                "Suite: {}, kappa = [{:.4}, {:.4}], {} steps",
                suite_label, kappa_min, kappa_max, n_steps
            );
            println!();
            println!(
                "{:>8} {:>12} {:>12} {:>12} {:>8}",
                "kappa", "ZD_isol", "best_local", "margin", "verdict"
            );
            println!("{}", "-".repeat(56));

            for i in 0..n_steps {
                let t = if n_steps <= 1 {
                    0.5
                } else {
                    i as f64 / (n_steps - 1) as f64
                };
                let kappa = kappa_min + t * (kappa_max - kappa_min);

                let benchmark = run_at_kappa(kappa, full);
                let gate = ProjectionGate::c010_default();
                let result = gate.evaluate(&benchmark);

                let verdict_str = if result.pass { "PASS" } else { "FAIL" };
                println!(
                    "{:>8.4} {:>12.6} {:>12.6} {:>12.6} {:>8}",
                    kappa,
                    result.zd_isolation,
                    result.best_local_isolation,
                    result.dominance_margin,
                    verdict_str
                );

                if let Some(ref dir) = output {
                    let _ = write_results_toml(dir, &benchmark, kappa);
                }
            }
        }

        Command::Gate {
            kappa,
            full,
            min_isolation,
            max_xtalk_db,
            min_dominance,
        } => {
            let mut gate = ProjectionGate::c010_default();
            if let Some(v) = min_isolation {
                gate.min_isolation_threshold = v;
            }
            if let Some(v) = max_xtalk_db {
                gate.max_xtalk_db = v;
            }
            if let Some(v) = min_dominance {
                gate.min_dominance_margin = v;
            }

            let benchmark = run_at_kappa(kappa, full);
            let result = gate.evaluate(&benchmark);

            println!("Projection Gate: {}", result.verdict);
            println!(
                "  ZD isolation:      {:.6}",
                result.zd_isolation
            );
            println!(
                "  Dominance margin:  {:.6}",
                result.dominance_margin
            );

            if result.pass {
                std::process::exit(0);
            } else {
                std::process::exit(1);
            }
        }
    }
}
