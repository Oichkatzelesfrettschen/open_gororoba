use anyhow::{Context, Result, bail};
use clap::Parser;
use csv::Writer;
use gororoba_cli_data::nanograv_refit::{
    build_refit_dataset, load_phase1_models_from_report, solve_refit,
};
use gororoba_cli_data::nanograv_timing_model::ReleaseBand;
use std::{
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(
    name = "nanograv-timing-phase1-refit",
    about = "Linearized WLS/GLS pilot refit over the six Phase 1 NANOGrav timing-model scout pulsars"
)]
struct Args {
    #[arg(
        long,
        default_value = "data/external/nanograv_15yr_timing/NANOGrav15yr_PulsarTiming_v2.1.0"
    )]
    root: PathBuf,

    #[arg(long, default_value = "reports/nanograv_15yr_timing_refit_preflight.toml")]
    phase1_report: PathBuf,

    #[arg(long, default_value = "data/csv/nanograv_phase1_refit_residuals.csv")]
    csv_out: PathBuf,

    #[arg(long, default_value = "reports/nanograv_phase1_refit_pilot.toml")]
    report_out: PathBuf,

    #[arg(long, default_value_t = 1.0e-8)]
    residual_tolerance_days: f64,

    #[arg(long, default_value_t = 120.0)]
    gls_corr_length_days: f64,

    #[arg(long, default_value_t = 0.5)]
    gls_red_noise_fraction: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.root.exists() {
        bail!("timing release root not found: {}", args.root.display());
    }
    let models =
        load_phase1_models_from_report(&args.root, &args.phase1_report, ReleaseBand::Wideband)?;
    if models.is_empty() {
        bail!("phase1 report yielded no models");
    }

    let mut rows = Vec::new();
    let mut report = String::new();
    writeln!(report, "[metadata]")?;
    writeln!(report, "title = \"NANOGrav Phase 1 timing refit pilot\"")?;
    writeln!(report, "root = {:?}", args.root.display().to_string())?;
    writeln!(
        report,
        "scope_note = \"This is a linearized release-around-solution pilot over released residual products and wideband DM estimates. It is a real WLS/GLS stacked fit, but not yet a full TOA timing engine with clock and ephemeris propagation.\""
    )?;
    writeln!(report, "phase1_solution_count = {}", models.len())?;
    writeln!(
        report,
        "residual_tolerance_days = {:.12}",
        args.residual_tolerance_days
    )?;
    writeln!(
        report,
        "gls_corr_length_days = {:.6}",
        args.gls_corr_length_days
    )?;
    writeln!(
        report,
        "gls_red_noise_fraction = {:.6}",
        args.gls_red_noise_fraction
    )?;

    for model in models {
        let dataset =
            build_refit_dataset(&args.root, &model, args.residual_tolerance_days).with_context(
                || format!("build refit dataset for {}", model.solution_id),
            )?;
        let fit = solve_refit(&dataset, args.gls_corr_length_days, args.gls_red_noise_fraction)
            .with_context(|| format!("solve pilot refit for {}", model.solution_id))?;

        writeln!(report)?;
        writeln!(report, "[[pulsar]]")?;
        writeln!(report, "solution_id = {:?}", model.solution_id)?;
        writeln!(report, "pulsar_id = {:?}", model.pulsar_id)?;
        writeln!(report, "matched_observations = {}", dataset.observations.len())?;
        writeln!(report, "channel_rows_used = {}", dataset.total_channel_rows_used)?;
        writeln!(
            report,
            "wls_toa_rms_before_us = {:.12}",
            fit.wls_summary.toa_rms_before_us
        )?;
        writeln!(
            report,
            "wls_toa_rms_after_us = {:.12}",
            fit.wls_summary.toa_rms_after_us
        )?;
        writeln!(
            report,
            "gls_toa_rms_before_us = {:.12}",
            fit.gls_summary.toa_rms_before_us
        )?;
        writeln!(
            report,
            "gls_toa_rms_after_us = {:.12}",
            fit.gls_summary.toa_rms_after_us
        )?;
        writeln!(report, "wls_dm_rms_before = {:.12}", fit.wls_summary.dm_rms_before)?;
        writeln!(report, "wls_dm_rms_after = {:.12}", fit.wls_summary.dm_rms_after)?;
        writeln!(report, "gls_dm_rms_before = {:.12}", fit.gls_summary.dm_rms_before)?;
        writeln!(report, "gls_dm_rms_after = {:.12}", fit.gls_summary.dm_rms_after)?;

        for row in fit.rows {
            rows.push(row);
        }
    }

    write_csv(&args.csv_out, &rows)?;
    write_report(&args.report_out, &report)?;

    println!("Phase 1 refit solutions: {}", rows.iter().map(|row| row.solution_id.as_str()).collect::<std::collections::BTreeSet<_>>().len());
    println!("Residual CSV: {}", args.csv_out.display());
    println!("Report: {}", args.report_out.display());
    Ok(())
}

fn write_csv(
    path: &Path,
    rows: &[gororoba_cli_data::nanograv_refit::RefitOutputRow],
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = Writer::from_path(path)?;
    writer.write_record([
        "solution_id",
        "pulsar_id",
        "mjd",
        "frequency_mhz",
        "matched_channel_rows",
        "residual_before_us",
        "residual_after_wls_us",
        "residual_after_gls_us",
        "residual_uncertainty_us",
        "dm_before",
        "dm_after_wls",
        "dm_after_gls",
        "dm_uncertainty",
    ])?;
    for row in rows {
        writer.write_record([
            row.solution_id.as_str(),
            row.pulsar_id.as_str(),
            &format!("{:.12}", row.mjd),
            &format!("{:.6}", row.frequency_mhz),
            &row.matched_channel_rows.to_string(),
            &format!("{:.12}", row.residual_before_us),
            &format!("{:.12}", row.residual_after_wls_us),
            &format!("{:.12}", row.residual_after_gls_us),
            &format!("{:.12}", row.residual_uncertainty_us),
            &format!("{:.12}", row.dm_before),
            &format!("{:.12}", row.dm_after_wls),
            &format!("{:.12}", row.dm_after_gls),
            &row.dm_uncertainty.map_or_else(String::new, |value| format!("{value:.12}")),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_report(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, content)?;
    Ok(())
}
