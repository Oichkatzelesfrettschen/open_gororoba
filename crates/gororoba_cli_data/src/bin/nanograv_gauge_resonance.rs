use anyhow::{Context, Result};
use clap::Parser;
use gororoba_cli_data::nanograv_timing::load_release;
use algebra_analysis::codebook::{enumerate_lambda_4096, is_in_lambda_1024};
use algebra_analysis::sky_mapping::project_sky_to_basis;
use algebra_experimental::higher_cd::HigherAvt;
use algebra_experimental::particle_physics::StandardModelMapping;
use std::path::PathBuf;
use csv::Writer;

#[derive(Parser, Debug)]
#[command(
    name = "nanograv-gauge-resonance",
    about = "Correlates NANOGrav pulsar variance with 1024D Standard Model gauge sector cross-coupling"
)]
struct Args {
    #[arg(
        long,
        default_value = "data/external/nanograv_15yr_timing/NANOGrav15yr_PulsarTiming_v2.1.0"
    )]
    root: PathBuf,

    #[arg(long, default_value = "data/csv/nanograv_gauge_resonance_audit.csv")]
    csv_out: PathBuf,

    #[arg(long, default_value_t = 1_000_000)]
    n_samples: usize,

    #[arg(long)]
    gpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading NANOGrav 15-year dataset...");
    let release = load_release(&args.root).context("failed to load timing release")?;
    
    println!("Generating 1024D DekaVoudon AVT ({} samples)...", args.n_samples);
    let avt_wrapper = HigherAvt::sampled(1024, args.n_samples, 42);
    let avt = &avt_wrapper.avt;
    
    let sm = StandardModelMapping::new();
    let su3_axes = sm.su3_axes;
    let su2_axes = sm.su2_axes;

    println!("Calculating Gauge Cross-Coupling Density for each pulsar...");
    let lattice_1024 = enumerate_lambda_4096().into_iter().filter(is_in_lambda_1024).collect::<Vec<_>>();
    
    let mut writer = Writer::from_path(&args.csv_out)?;
    writer.write_record(&["pulsar", "rms_us", "gauge_frustration", "su3_su2_cross_coupling"])?;

    for (name, data) in &release {
        let Some(sky) = data.sky_vector() else { continue; };
        let residuals: Vec<f64> = data.avg_residuals.iter().map(|p| p.residual_us).collect();
        if residuals.is_empty() { continue; }
        
        let rms_us = stats_core::metrics::rms(&residuals);
        
        // Project sky to 1024D basis
        let basis_vec = project_sky_to_basis(&sky, &lattice_1024, 1024);
        
        // Find indices with significant projection
        let mut active_indices = Vec::new();
        for (i, &val) in basis_vec.iter().enumerate() {
            if val.abs() > 0.5 {
                active_indices.push(i);
            }
        }

        // Compute Gauge Frustration: count AVT violations connecting active basis indices 
        // to the SU(3) and SU(2) sectors.
        let mut cross_coupling_count = 0;
        for &idx in &active_indices {
            // Count violations where this index couples an SU3 axis to an SU2 axis
            // (i, j, k) where i=idx, j in SU3, k in SU2 (approximate)
            for &s3 in &su3_axes {
                for &s2 in &su2_axes {
                    if avt.check_violation(idx, s3, s2) {
                        cross_coupling_count += 1;
                    }
                }
            }
        }

        let gauge_frustration = cross_coupling_count as f64 / (args.n_samples as f64);

        writer.write_record(&[
            name,
            &format!("{:.12}", rms_us),
            &format!("{:.12}", gauge_frustration),
            &cross_coupling_count.to_string(),
        ])?;
    }

    writer.flush()?;
    println!("Gauge resonance audit complete. Results saved to {:?}", args.csv_out);

    Ok(())
}
