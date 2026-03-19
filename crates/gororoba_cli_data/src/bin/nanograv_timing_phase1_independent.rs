use anyhow::{Result, bail};
use clap::Parser;
use csv::Writer;
use gororoba_cli_data::nanograv_timing_engine::{
    TimingEphemeris, build_phase1_independent_datasets, solve_independent_refit,
};
use std::{
    collections::BTreeSet,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(
    name = "nanograv-timing-phase1-independent",
    about = "Independent TOA-driven Phase 1 NANOGrav timing-engine pilot over dominant frontend/backend subgroups"
)]
struct Args {
    #[arg(
        long,
        default_value = "data/external/nanograv_15yr_timing/NANOGrav15yr_PulsarTiming_v2.1.0"
    )]
    root: PathBuf,

    #[arg(long, default_value = "reports/nanograv_15yr_timing_refit_preflight.toml")]
    phase1_report: PathBuf,

    #[arg(long, default_value = "data/csv/nanograv_phase1_independent_residuals.csv")]
    csv_out: PathBuf,

    #[arg(long, default_value = "reports/nanograv_phase1_independent_refit.toml")]
    report_out: PathBuf,

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

    let ephemeris = TimingEphemeris::load_default()?;
    let datasets = build_phase1_independent_datasets(&args.root, &args.phase1_report, &ephemeris)?;
    if datasets.is_empty() {
        bail!("phase1 report yielded no independent datasets");
    }

    let mut rows = Vec::new();
    let mut report = String::new();
    writeln!(report, "[metadata]")?;
    writeln!(report, "title = \"NANOGrav Phase 1 independent timing-engine pilot\"")?;
    writeln!(report, "root = {:?}", args.root.display().to_string())?;
    writeln!(report, "phase1_solution_count = {}", datasets.len())?;
    writeln!(report, "ephemeris_used = {:?}", ephemeris.ephemeris_name())?;
    writeln!(
        report,
        "scope_note = \"This lane reconstructs phase residuals directly from topocentric .tim TOAs, hifitime clock conversion, cached NAIF Earth orientation, family-specific binary forward models, and a joint phase-plus-DM fit. It does not consume release *.res rows.\""
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

    for dataset in datasets {
        let fit = solve_independent_refit(
            &dataset,
            &ephemeris,
            args.gls_corr_length_days,
            args.gls_red_noise_fraction,
        )?;
        writeln!(report)?;
        writeln!(report, "[[pulsar]]")?;
        writeln!(report, "solution_id = {:?}", fit.dataset.model.solution_id)?;
        writeln!(report, "pulsar_id = {:?}", fit.dataset.model.pulsar_id)?;
        writeln!(
            report,
            "binary_family = {:?}",
            fit.dataset
                .model
                .binary_family
                .as_ref()
                .map(|family| family.as_str())
                .unwrap_or("isolated")
        )?;
        writeln!(report, "requested_ephem = {:?}", fit.dataset.requested_ephem)?;
        writeln!(report, "ephem_used = {:?}", fit.dataset.ephem_used)?;
        writeln!(report, "dominant_subgroup = {:?}", fit.dataset.dominant_subgroup)?;
        writeln!(
            report,
            "dominant_subgroup_count = {}",
            fit.dataset.dominant_subgroup_count
        )?;
        writeln!(report, "total_toa_count = {}", fit.dataset.total_toa_count)?;
        writeln!(report, "observation_count = {}", fit.summary.observation_count)?;
        writeln!(report, "dm_observation_count = {}", fit.summary.dm_observation_count)?;
        writeln!(
            report,
            "residual_rms_before_us = {:.12}",
            fit.summary.residual_rms_before_us
        )?;
        writeln!(
            report,
            "residual_rms_after_wls_us = {:.12}",
            fit.summary.residual_rms_after_wls_us
        )?;
        writeln!(
            report,
            "residual_rms_after_gls_us = {:.12}",
            fit.summary.residual_rms_after_gls_us
        )?;
        writeln!(
            report,
            "weighted_rms_before_us = {:.12}",
            fit.summary.weighted_rms_before_us
        )?;
        writeln!(
            report,
            "weighted_rms_after_wls_us = {:.12}",
            fit.summary.weighted_rms_after_wls_us
        )?;
        writeln!(
            report,
            "weighted_rms_after_gls_us = {:.12}",
            fit.summary.weighted_rms_after_gls_us
        )?;
        if let Some(value) = fit.summary.dm_rms_before {
            writeln!(report, "dm_rms_before = {:.12}", value)?;
        }
        if let Some(value) = fit.summary.dm_rms_after_wls {
            writeln!(report, "dm_rms_after_wls = {:.12}", value)?;
        }
        if let Some(value) = fit.summary.dm_rms_after_gls {
            writeln!(report, "dm_rms_after_gls = {:.12}", value)?;
        }
        writeln!(report, "gls_ridge_factor = {:.12e}", fit.summary.gls_ridge_factor)?;
        writeln!(report, "simplification_note = {:?}", fit.dataset.simplification_notes)?;
        let wls_parameters = fit
            .parameter_names
            .iter()
            .zip(fit.wls_coefficients.iter())
            .map(|(name, coefficient)| format!("{name}={coefficient:.12e}"))
            .collect::<Vec<_>>();
        writeln!(report, "wls_parameter = {:?}", wls_parameters)?;
        for row in fit.rows {
            rows.push(row);
        }
    }

    write_csv(&args.csv_out, &rows)?;
    write_report(&args.report_out, &report)?;

    println!(
        "Independent Phase 1 solutions: {}",
        rows.iter()
            .map(|row| row.solution_id.as_str())
            .collect::<BTreeSet<_>>()
            .len()
    );
    println!("Residual CSV: {}", args.csv_out.display());
    println!("Report: {}", args.report_out.display());
    Ok(())
}

fn write_csv(
    path: &Path,
    rows: &[gororoba_cli_data::nanograv_timing_engine::IndependentRefitRow],
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = Writer::from_path(path)?;
    writer.write_record([
        "solution_id",
        "pulsar_id",
        "site",
        "subgroup",
        "mjd_utc",
        "mjd_tdb",
        "frequency_mhz",
        "uncertainty_us",
        "residual_before_us",
        "residual_after_wls_us",
        "residual_after_gls_us",
        "dm_model",
        "pp_dm",
        "pp_dme",
        "dm_residual_before",
        "dm_residual_after_wls",
        "dm_residual_after_gls",
    ])?;
    for row in rows {
        writer.write_record([
            row.solution_id.as_str(),
            row.pulsar_id.as_str(),
            row.site.as_str(),
            row.subgroup.as_str(),
            &format!("{:.12}", row.mjd_utc),
            &format!("{:.12}", row.mjd_tdb),
            &format!("{:.6}", row.frequency_mhz),
            &format!("{:.12}", row.uncertainty_us),
            &format!("{:.12}", row.residual_before_us),
            &format!("{:.12}", row.residual_after_wls_us),
            &format!("{:.12}", row.residual_after_gls_us),
            &format!("{:.12}", row.dm_model),
            &row.pp_dm.map_or_else(String::new, |value| format!("{value:.12}")),
            &row.pp_dme.map_or_else(String::new, |value| format!("{value:.12}")),
            &row
                .dm_residual_before
                .map_or_else(String::new, |value| format!("{value:.12}")),
            &row
                .dm_residual_after_wls
                .map_or_else(String::new, |value| format!("{value:.12}")),
            &row
                .dm_residual_after_gls
                .map_or_else(String::new, |value| format!("{value:.12}")),
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
