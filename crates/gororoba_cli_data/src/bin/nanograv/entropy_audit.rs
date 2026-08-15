use algebra_analysis::{
    codebook::{enumerate_lambda_4096, is_in_lambda_1024},
    sky_mapping::project_sky_to_basis,
};
use algebra_experimental::higher_cd::{HigherAvt, SparseApeironState};
use anyhow::{Context, Result};
use csv::Writer;
use gororoba_cli_data::nanograv_timing::load_release;
use std::path::PathBuf;

#[derive(Debug, clap::Args)]
pub struct Args {
    #[arg(
        long,
        default_value = "data/external/nanograv_15yr_timing/NANOGrav15yr_PulsarTiming_v2.1.0"
    )]
    root: PathBuf,

    #[arg(long, default_value = "data/csv/nanograv_entropy_audit.csv")]
    csv_out: PathBuf,

    #[arg(long, default_value_t = 1_000_000)]
    n_samples: usize,
}

pub fn run(args: Args) -> Result<()> {
    println!("Loading NANOGrav 15-year dataset...");
    let release = load_release(&args.root).context("failed to load timing release")?;

    println!(
        "Generating 1024D DekaVoudon AVT ({} samples)...",
        args.n_samples
    );
    let avt_wrapper = HigherAvt::sampled(1024, args.n_samples, 42);
    let avt = &avt_wrapper.avt;

    let lattice_1024 = enumerate_lambda_4096()
        .into_iter()
        .filter(is_in_lambda_1024)
        .collect::<Vec<_>>();

    // 4. Calculate Shannon Entropy of AVT violation distribution
    // First, pre-calculate global violation participation counts per axis
    let mut global_participation = vec![0usize; 1024];
    for &(i, j, k, _, _) in &avt.violations {
        global_participation[i] += 1;
        global_participation[j] += 1;
        global_participation[k] += 1;
    }

    println!("Computing holographic entropy saturation for each pulsar...");
    let mut writer = Writer::from_path(&args.csv_out)?;
    writer.write_record([
        "pulsar",
        "dist_pc",
        "rms_us",
        "shannon_entropy",
        "bekenstein_bound",
        "saturation_ratio",
    ])?;

    // Planck length in parsecs (approx)
    let l_p_pc = 5.23e-52;

    for (name, data) in &release {
        let Some(sky) = data.sky_vector() else {
            continue;
        };
        let residuals: Vec<f64> = data.avg_residuals.iter().map(|p| p.residual_us).collect();
        if residuals.is_empty() {
            continue;
        }

        let rms_us = stats_core::metrics::rms(&residuals);

        // 1. Estimate distance from parallax
        let meta = data.preferred_metadata();
        let dist_pc = if let Some(px) = meta.px_mas {
            if px > 0.0 { 1000.0 / px } else { 1000.0 } // default if no px
        } else {
            1000.0 // Mean pulsar distance
        };

        // 2. Bekenstein Bound (Entropy proportional to area)
        // Area of a circle at distance d with radius of pulsar beam (approx)
        let area = std::f64::consts::PI * dist_pc * dist_pc;
        let bekenstein_bound = area / (4.0 * l_p_pc * l_p_pc);

        // 3. Project sky to 1024D basis
        let basis_vec = project_sky_to_basis(&sky, &lattice_1024, 1024);

        // 4. Calculate Shannon Entropy of AVT violation distribution
        // For each basis axis i, we sum its violation participations.
        let mut violations_per_axis = vec![0.0; 1024];
        for (i, &val) in basis_vec.iter().enumerate() {
            if val.abs() > 0.1 {
                violations_per_axis[i] = val.abs() * (global_participation[i] as f64);
            }
        }

        let sparse_state = SparseApeironState::from_dense(1024, &violations_per_axis, 1e-10);
        let h_avt = sparse_state.shannon_entropy();

        // 5. Saturation Ratio
        // We need to scale h_avt (nats) to bits or a comparable dimensionless number.
        let saturation_ratio = h_avt / bekenstein_bound.ln();

        writer.write_record([
            name,
            &format!("{:.2}", dist_pc),
            &format!("{:.12}", rms_us),
            &format!("{:.12}", h_avt),
            &format!("{:.2e}", bekenstein_bound),
            &format!("{:.12e}", saturation_ratio),
        ])?;
    }

    writer.flush()?;
    println!(
        "Entropy audit complete. Results saved to {:?}",
        args.csv_out
    );

    Ok(())
}
