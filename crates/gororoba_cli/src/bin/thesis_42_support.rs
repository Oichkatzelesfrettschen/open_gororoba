use clap::Parser;
use gororoba_cli::thesis_42_support::{
    DEFAULT_CALIBRATION_ID, Thesis42SupportConfig, generate_thesis_42_support_report,
    write_thesis_42_support_bundle,
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "thesis-42-support")]
#[command(about = "Generate an evidence-first thesis support bundle for the 42-lane synthesis")]
struct Args {
    #[arg(long, default_value = DEFAULT_CALIBRATION_ID)]
    calibration_id: String,

    #[arg(
        long,
        default_value = "data/csv/c010_nonlocal_material_calibrations.csv"
    )]
    calibration_csv: PathBuf,

    #[arg(long, default_value_t = 16)]
    dim: usize,

    #[arg(long, default_value_t = 20)]
    theta_steps: usize,

    #[arg(long, default_value_t = 0.0)]
    cp_phase_rad: f64,

    #[arg(long, default_value_t = 0.01)]
    alpha_zd: f64,

    #[arg(long, default_value = "data/evidence/thesis_42_support")]
    output_dir: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = Thesis42SupportConfig {
        calibration_id: args.calibration_id,
        calibration_csv: args.calibration_csv,
        dim: args.dim,
        theta_steps: args.theta_steps,
        cp_phase_rad: args.cp_phase_rad,
        alpha_zd: args.alpha_zd,
        output_dir: args.output_dir,
    };
    let report = generate_thesis_42_support_report(&config)?;
    write_thesis_42_support_bundle(&config.output_dir, &report)?;

    println!("thesis-42-support");
    println!(
        "summary: {}",
        config.output_dir.join("summary.toml").display()
    );
    println!(
        "labels: {}, {}, {}, {}",
        report.labels.nonlocal_metamaterial,
        report.labels.majorana,
        report.labels.dark_matter,
        report.labels.gravastar
    );
    println!(
        "majorana terminal normalized friction: {:.6}",
        report.majorana.friction_sweep.terminal_normalized_friction
    );
    println!(
        "harmonic halo alpha=0 recovery exact: {}",
        report.dark_matter.exact_nfw_recovery
    );
    println!(
        "gravastar physical/model status: {}/{}",
        report.gravastar.boundary.physical_claim_status,
        report.gravastar.boundary.model_claim_status
    );
    Ok(())
}
