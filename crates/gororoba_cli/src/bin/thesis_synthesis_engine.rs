//! Thesis Synthesis Engine: orchestrate all four grand synthesis thesis pipelines.
//!
//! Executes each thesis pipeline (T1-T4) with configurable parameters,
//! applies falsification gates, and writes structured TOML evidence artifacts.
//!
//! Usage:
//!   thesis-synthesis-engine --thesis all          # Run all 4
//!   thesis-synthesis-engine --thesis 1            # Run only T1
//!   thesis-synthesis-engine --thesis 2 --alpha 1.0 --power-index 2.0

use clap::Parser;
use gororoba_engine::{Thesis1Pipeline, Thesis2Pipeline, Thesis3Pipeline, Thesis4Pipeline};
use gororoba_engine::traits::ThesisPipeline;
use std::fmt::Write as _;

#[derive(Parser, Debug)]
#[command(name = "thesis-synthesis-engine")]
#[command(about = "Orchestrate all four grand synthesis thesis pipelines")]
struct Args {
    /// Which thesis to run: 1, 2, 3, 4, or "all"
    #[arg(long, default_value = "all")]
    thesis: String,

    // T1 parameters
    /// Grid size for T1 (N^3)
    #[arg(long, default_value = "16")]
    grid_size: usize,

    /// Lambda coupling for T1
    #[arg(long, default_value = "1.0")]
    lambda: f64,

    // T2 parameters
    /// Alpha coupling for T2
    #[arg(long, default_value = "0.5")]
    alpha: f64,

    /// Power-law index for T2
    #[arg(long, default_value = "1.5")]
    power_index: f64,

    // T3 parameters
    /// Training epochs for T3
    #[arg(long, default_value = "64")]
    epochs: usize,

    /// Pentagon optimization steps for T3 (0 = skip)
    #[arg(long, default_value = "500")]
    opt_steps: usize,

    // T4 parameters
    /// Collision steps for T4
    #[arg(long, default_value = "50000")]
    t4_steps: usize,

    /// Output directory
    #[arg(long, default_value = "data/evidence")]
    output_dir: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let theses_to_run: Vec<usize> = if args.thesis == "all" {
        vec![1, 2, 3, 4]
    } else {
        args.thesis
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect()
    };

    std::fs::create_dir_all(&args.output_dir)?;

    println!("Thesis Synthesis Engine");
    println!("=======================");
    println!("Running theses: {:?}", theses_to_run);
    println!();

    let mut summary = String::new();
    let _ = writeln!(summary, "[metadata]");
    let _ = writeln!(summary, "experiment = \"thesis_synthesis\"");
    let _ = writeln!(
        summary,
        "timestamp = \"{}\"",
        chrono_free_timestamp()
    );
    let _ = writeln!(summary, "theses = {:?}", theses_to_run);
    let _ = writeln!(summary);

    let mut n_pass = 0;
    let mut n_fail = 0;

    for &thesis_id in &theses_to_run {
        let evidence = match thesis_id {
            1 => {
                let pipeline = Thesis1Pipeline {
                    grid_size: args.grid_size,
                    lambda: args.lambda,
                    n_sub: 2,
                    p_threshold: 0.05,
                };
                println!("--- {} ---", pipeline.name());
                pipeline.execute()
            }
            2 => {
                let pipeline = Thesis2Pipeline {
                    alpha: args.alpha,
                    power_index: args.power_index,
                    ..Default::default()
                };
                println!("--- {} ---", pipeline.name());
                pipeline.execute()
            }
            3 => {
                let pipeline = Thesis3Pipeline {
                    epochs: args.epochs,
                    optimization_steps: args.opt_steps,
                    ..Default::default()
                };
                println!("--- {} ---", pipeline.name());
                pipeline.execute()
            }
            4 => {
                let pipeline = Thesis4Pipeline {
                    n_steps: args.t4_steps,
                    ..Default::default()
                };
                println!("--- {} ---", pipeline.name());
                pipeline.execute()
            }
            _ => {
                eprintln!("Unknown thesis: {}", thesis_id);
                continue;
            }
        };

        let gate_status = if evidence.passes_gate {
            n_pass += 1;
            "PASS"
        } else {
            n_fail += 1;
            "FAIL"
        };

        println!(
            "  Gate: {} (metric={:.4}, threshold={:.4})",
            gate_status, evidence.metric_value, evidence.threshold
        );
        for msg in &evidence.messages {
            println!("  {}", msg);
        }
        println!();

        // Write per-thesis TOML evidence
        let mut thesis_report = String::new();
        let _ = writeln!(thesis_report, "[evidence]");
        let _ = writeln!(thesis_report, "thesis_id = {}", evidence.thesis_id);
        let _ = writeln!(thesis_report, "label = \"{}\"", evidence.label);
        let _ = writeln!(thesis_report, "metric_value = {:.8}", evidence.metric_value);
        let _ = writeln!(thesis_report, "threshold = {:.8}", evidence.threshold);
        let _ = writeln!(thesis_report, "passes_gate = {}", evidence.passes_gate);
        let _ = writeln!(thesis_report);
        let _ = writeln!(thesis_report, "[messages]");
        for (i, msg) in evidence.messages.iter().enumerate() {
            let _ = writeln!(thesis_report, "m{} = \"{}\"", i, msg);
        }

        let thesis_path = format!(
            "{}/thesis{}_evidence.toml",
            args.output_dir, evidence.thesis_id
        );
        std::fs::write(&thesis_path, &thesis_report)?;
        println!("  Written: {}", thesis_path);
        println!();

        // Append to summary
        let _ = writeln!(summary, "[[thesis]]");
        let _ = writeln!(summary, "id = {}", evidence.thesis_id);
        let _ = writeln!(summary, "label = \"{}\"", evidence.label);
        let _ = writeln!(summary, "metric = {:.8}", evidence.metric_value);
        let _ = writeln!(summary, "threshold = {:.8}", evidence.threshold);
        let _ = writeln!(summary, "pass = {}", evidence.passes_gate);
        let _ = writeln!(summary);
    }

    let _ = writeln!(summary, "[summary]");
    let _ = writeln!(summary, "total = {}", n_pass + n_fail);
    let _ = writeln!(summary, "pass = {}", n_pass);
    let _ = writeln!(summary, "fail = {}", n_fail);

    let summary_path = format!("{}/synthesis_summary.toml", args.output_dir);
    std::fs::write(&summary_path, &summary)?;

    println!("=======================");
    println!(
        "Summary: {}/{} pass, {}/{} fail",
        n_pass,
        n_pass + n_fail,
        n_fail,
        n_pass + n_fail
    );
    println!("Summary written to: {}", summary_path);

    Ok(())
}

/// Generate a timestamp without pulling in chrono (deterministic enough for reports).
fn chrono_free_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Convert to approx ISO 8601 (good enough for TOML metadata)
    let days = secs / 86400;
    let years = 1970 + days / 365;
    let remaining_days = days % 365;
    let months = remaining_days / 30 + 1;
    let day = remaining_days % 30 + 1;
    format!("{:04}-{:02}-{:02}", years, months, day)
}
