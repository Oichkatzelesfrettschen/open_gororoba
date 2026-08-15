use anyhow::{Result, bail};
use csv::Writer;
use gororoba_cli_data::{
    nanograv_timing_engine::{
        IndependentRefitResult, TimingEphemeris, build_phase1_independent_datasets,
        solve_independent_refit,
    },
    nanograv_timing_model::{BinaryFamily, TimingModel},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fmt::Write as _,
    fs,
    fs::OpenOptions,
    os::fd::AsRawFd,
    path::{Path, PathBuf},
};

const LOCK_EX: i32 = 2;
const LOCK_UN: i32 = 8;

unsafe extern "C" {
    fn flock(fd: i32, operation: i32) -> i32;
}

#[derive(Debug, clap::Args)]
pub struct Args {
    #[arg(
        long,
        default_value = "data/external/nanograv_15yr_timing/NANOGrav15yr_PulsarTiming_v2.1.0"
    )]
    root: PathBuf,

    #[arg(
        long,
        default_value = "reports/nanograv_15yr_timing_refit_preflight.toml"
    )]
    phase1_report: PathBuf,

    #[arg(
        long,
        default_value = "data/csv/nanograv_phase1_independent_residuals.csv"
    )]
    csv_out: PathBuf,

    #[arg(long, default_value = "reports/nanograv_phase1_independent_refit.toml")]
    report_out: PathBuf,

    #[arg(
        long,
        default_value = "reports/nanograv_phase1_independent_refit_reference_2026_03_19.toml"
    )]
    baseline_report_out: PathBuf,

    #[arg(
        long,
        default_value = "data/csv/nanograv_phase1_independent_residuals_reference_2026_03_19.csv"
    )]
    baseline_csv_out: PathBuf,

    #[arg(
        long,
        default_value = "reports/nanograv_phase1_independent_comparison.toml"
    )]
    comparison_report_out: PathBuf,

    #[arg(
        long,
        default_value = "data/csv/nanograv_phase1_independent_pairwise.csv"
    )]
    pairwise_csv_out: PathBuf,

    #[arg(
        long,
        default_value = "reports/nanograv_phase1_independent_pairwise.toml"
    )]
    pairwise_report_out: PathBuf,

    #[arg(long, default_value_t = 120.0)]
    gls_corr_length_days: f64,

    #[arg(long, default_value_t = 0.5)]
    gls_red_noise_fraction: f64,

    #[arg(long, default_value_t = 30.0)]
    pair_bin_days: f64,

    #[arg(long, default_value_t = 8)]
    min_pair_overlap_bins: usize,
}

#[derive(Debug, Clone)]
struct ComparisonRow {
    solution_id: String,
    pulsar_id: String,
    binary_family: String,
    tactical_lane: String,
    covariance_preset: String,
    provisional_acceptance_policy: String,
    recommended_solver: String,
    acceptance_track: String,
    metric_conflict: bool,
    ecorr_status: String,
    family_equivalence_target: String,
    family_equivalence_status: String,
    raw_improvement_wls_frac: f64,
    raw_improvement_gls_frac: f64,
    weighted_improvement_wls_frac: f64,
    weighted_improvement_gls_frac: f64,
    dm_improvement_wls_frac: Option<f64>,
    dm_improvement_gls_frac: Option<f64>,
    synthesis_score_wls: f64,
    synthesis_score_gls: f64,
    wls_plausibility_status: String,
    gls_plausibility_status: String,
    wls_plausibility_flags: Vec<String>,
    gls_plausibility_flags: Vec<String>,
}

#[derive(Debug, Clone)]
struct PairwiseIndependentRow {
    pulsar_a: String,
    pulsar_b: String,
    separation_deg: f64,
    hellings_downs: f64,
    overlap_bins_wls: usize,
    overlap_bins_gls: usize,
    overlap_bins_policy: usize,
    wls_residual_pearson: Option<f64>,
    gls_residual_pearson: Option<f64>,
    policy_residual_pearson: Option<f64>,
}

#[derive(Debug, Clone)]
struct PreparedPairwisePulsar {
    pulsar: String,
    sky_vector: Option<[f64; 3]>,
    wls_bins: BTreeMap<i64, f64>,
    gls_bins: BTreeMap<i64, f64>,
    policy_bins: BTreeMap<i64, f64>,
}

#[derive(Debug, Clone)]
struct PlausibilityAssessment {
    status: String,
    flags: Vec<String>,
}

#[derive(Debug, Clone)]
struct FamilyEquivalenceAssessment {
    target: String,
    status: String,
}

pub fn run(args: Args) -> Result<()> {
    let _run_lock = RunLock::acquire()?;
    if !args.root.exists() {
        bail!("timing release root not found: {}", args.root.display());
    }

    let ephemeris = TimingEphemeris::load_default()?;
    let datasets = build_phase1_independent_datasets(&args.root, &args.phase1_report, &ephemeris)?;
    if datasets.is_empty() {
        bail!("phase1 report yielded no independent datasets");
    }

    // WHY: the timing engine already parallelizes the expensive inner loops
    // (dataset construction, finite-difference columns, and covariance feature
    // assembly). Keeping the top-level Phase 1 sweep sequential avoids nested
    // Rayon oversubscription across only six pulsars.
    let mut fits: Vec<IndependentRefitResult> = datasets
        .iter()
        .map(|dataset| {
            solve_independent_refit(
                dataset,
                &ephemeris,
                args.gls_corr_length_days,
                args.gls_red_noise_fraction,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    // Sort by solution_id so report ordering is stable and deterministic.
    fits.sort_by(|a, b| {
        a.dataset
            .model
            .solution_id
            .cmp(&b.dataset.model.solution_id)
    });
    let comparison_rows = fits
        .iter()
        .map(build_comparison_row)
        .collect::<Result<Vec<_>>>()?;
    let pairwise_rows = build_pairwise_rows(&fits, args.pair_bin_days, args.min_pair_overlap_bins);

    let mut rows = Vec::new();
    let mut report = String::new();
    writeln!(report, "[metadata]")?;
    writeln!(
        report,
        "title = \"NANOGrav Phase 1 independent timing-engine pilot\""
    )?;
    writeln!(report, "root = {:?}", args.root.display().to_string())?;
    writeln!(report, "phase1_solution_count = {}", fits.len())?;
    writeln!(report, "ephemeris_used = {:?}", ephemeris.ephemeris_name())?;
    writeln!(
        report,
        "scope_note = \"This lane reconstructs phase residuals directly from topocentric .tim TOAs, hifitime clock conversion, cached NAIF Earth orientation, family-specific binary forward models, and a joint phase-plus-DM fit. Its GLS path now uses a structured low-rank-plus-diagonal covariance: calibrated white-noise floors, Fourier and kernel basis terms for long-timescale phase/DM processes, and ECORR group columns, without consuming release *.res rows.\""
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

    for fit in fits {
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
        writeln!(
            report,
            "requested_ephem = {:?}",
            fit.dataset.requested_ephem
        )?;
        writeln!(report, "ephem_used = {:?}", fit.dataset.ephem_used)?;
        writeln!(
            report,
            "tactical_lane = {:?}",
            fit.covariance_calibration.tactical_lane
        )?;
        writeln!(
            report,
            "covariance_preset = {:?}",
            fit.covariance_calibration.preset_name
        )?;
        writeln!(
            report,
            "provisional_acceptance_policy = {:?}",
            fit.covariance_calibration.provisional_acceptance_policy
        )?;
        writeln!(
            report,
            "dominant_subgroup = {:?}",
            fit.dataset.dominant_subgroup
        )?;
        writeln!(
            report,
            "dominant_subgroup_count = {}",
            fit.dataset.dominant_subgroup_count
        )?;
        writeln!(report, "all_subgroups = {:?}", fit.dataset.all_subgroups)?;
        writeln!(
            report,
            "subgroup_count = {}",
            fit.dataset.all_subgroups.len()
        )?;
        writeln!(report, "total_toa_count = {}", fit.dataset.total_toa_count)?;
        writeln!(
            report,
            "observation_count = {}",
            fit.summary.observation_count
        )?;
        writeln!(
            report,
            "dm_observation_count = {}",
            fit.summary.dm_observation_count
        )?;
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
        writeln!(
            report,
            "raw_improvement_wls_frac = {:.12}",
            fit.summary.raw_improvement_wls_frac
        )?;
        writeln!(
            report,
            "raw_improvement_gls_frac = {:.12}",
            fit.summary.raw_improvement_gls_frac
        )?;
        writeln!(
            report,
            "weighted_improvement_wls_frac = {:.12}",
            fit.summary.weighted_improvement_wls_frac
        )?;
        writeln!(
            report,
            "weighted_improvement_gls_frac = {:.12}",
            fit.summary.weighted_improvement_gls_frac
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
        if let Some(value) = fit.summary.dm_improvement_wls_frac {
            writeln!(report, "dm_improvement_wls_frac = {:.12}", value)?;
        }
        if let Some(value) = fit.summary.dm_improvement_gls_frac {
            writeln!(report, "dm_improvement_gls_frac = {:.12}", value)?;
        }
        writeln!(
            report,
            "synthesis_score_wls = {:.12}",
            fit.summary.synthesis_score_wls
        )?;
        writeln!(
            report,
            "synthesis_score_gls = {:.12}",
            fit.summary.synthesis_score_gls
        )?;
        writeln!(
            report,
            "recommended_solver = {:?}",
            fit.summary.recommended_solver
        )?;
        writeln!(
            report,
            "acceptance_track = {:?}",
            fit.summary.acceptance_track
        )?;
        writeln!(report, "metric_conflict = {}", fit.summary.metric_conflict)?;
        writeln!(
            report,
            "gls_ridge_factor = {:.12e}",
            fit.summary.gls_ridge_factor
        )?;
        writeln!(
            report,
            "corr_length_days = {:.12}",
            fit.covariance_calibration.corr_length_days
        )?;
        writeln!(
            report,
            "red_noise_fraction = {:.12}",
            fit.covariance_calibration.red_noise_fraction
        )?;
        writeln!(
            report,
            "phase_white_floor_s = {:.12e}",
            fit.covariance_calibration.phase_white_floor_s
        )?;
        writeln!(
            report,
            "dm_white_floor = {:.12e}",
            fit.covariance_calibration.dm_white_floor
        )?;
        writeln!(
            report,
            "phase_amp_s = {:.12e}",
            fit.covariance_calibration.phase_amp_s
        )?;
        writeln!(
            report,
            "dm_amp = {:.12e}",
            fit.covariance_calibration.dm_amp
        )?;
        writeln!(
            report,
            "phase_fourier_harmonics = {}",
            fit.covariance_calibration.phase_fourier_harmonics
        )?;
        writeln!(
            report,
            "phase_short_basis_count = {}",
            fit.covariance_calibration.phase_short_basis_count
        )?;
        writeln!(
            report,
            "phase_long_basis_count = {}",
            fit.covariance_calibration.phase_long_basis_count
        )?;
        writeln!(
            report,
            "dm_fourier_harmonics = {}",
            fit.covariance_calibration.dm_fourier_harmonics
        )?;
        writeln!(
            report,
            "dm_short_basis_count = {}",
            fit.covariance_calibration.dm_short_basis_count
        )?;
        writeln!(
            report,
            "dm_long_basis_count = {}",
            fit.covariance_calibration.dm_long_basis_count
        )?;
        writeln!(
            report,
            "ecorr_basis_count = {}",
            fit.covariance_calibration.ecorr_basis_count
        )?;
        writeln!(report, "ecorr_status = {:?}", ecorr_status_for_fit(&fit))?;
        writeln!(
            report,
            "simplification_note = {:?}",
            fit.dataset.simplification_notes
        )?;
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
    freeze_reference_if_missing(&args.report_out, &args.baseline_report_out)?;
    freeze_reference_if_missing(&args.csv_out, &args.baseline_csv_out)?;
    write_comparison_report(&args.comparison_report_out, &comparison_rows)?;
    write_pairwise_csv(&args.pairwise_csv_out, &pairwise_rows)?;
    write_pairwise_report(
        &args.pairwise_report_out,
        &args.pairwise_csv_out,
        &pairwise_rows,
        args.pair_bin_days,
        args.min_pair_overlap_bins,
    )?;

    println!(
        "Independent Phase 1 solutions: {}",
        rows.iter()
            .map(|row| row.solution_id.as_str())
            .collect::<BTreeSet<_>>()
            .len()
    );
    println!("Residual CSV: {}", args.csv_out.display());
    println!("Report: {}", args.report_out.display());
    println!(
        "Comparison report: {}",
        args.comparison_report_out.display()
    );
    println!("Pairwise CSV: {}", args.pairwise_csv_out.display());
    println!("Pairwise report: {}", args.pairwise_report_out.display());
    Ok(())
}

struct RunLock {
    file: fs::File,
}

impl RunLock {
    fn acquire() -> Result<Self> {
        let lock_path =
            env::temp_dir().join("open_gororoba_nanograv_timing_phase1_independent.lock");
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)?;
        // SAFETY: flock(2) operates on a valid open file descriptor. We use a blocking
        // exclusive advisory lock so concurrent solver launches queue instead of thrashing CPU.
        let result = unsafe { flock(file.as_raw_fd(), LOCK_EX) };
        if result != 0 {
            bail!(
                "failed to acquire timing-engine run lock {}",
                lock_path.display()
            );
        }
        Ok(Self { file })
    }
}

impl Drop for RunLock {
    fn drop(&mut self) {
        // SAFETY: unlocking the same still-open descriptor is the corresponding cleanup path.
        let _ = unsafe { flock(self.file.as_raw_fd(), LOCK_UN) };
    }
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
            &row.pp_dm
                .map_or_else(String::new, |value| format!("{value:.12}")),
            &row.pp_dme
                .map_or_else(String::new, |value| format!("{value:.12}")),
            &row.dm_residual_before
                .map_or_else(String::new, |value| format!("{value:.12}")),
            &row.dm_residual_after_wls
                .map_or_else(String::new, |value| format!("{value:.12}")),
            &row.dm_residual_after_gls
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

fn freeze_reference_if_missing(source: &Path, destination: &Path) -> Result<()> {
    if destination.exists() {
        return Ok(());
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(source, destination)?;
    Ok(())
}

fn build_comparison_row(fit: &IndependentRefitResult) -> Result<ComparisonRow> {
    let wls_plausibility = assess_parameter_plausibility(
        &fit.dataset.model,
        &fit.parameter_names,
        &fit.wls_coefficients,
    );
    let gls_plausibility = assess_parameter_plausibility(
        &fit.dataset.model,
        &fit.parameter_names,
        &fit.gls_coefficients,
    );
    let equivalence = assess_family_equivalence(fit);
    Ok(ComparisonRow {
        solution_id: fit.dataset.model.solution_id.clone(),
        pulsar_id: fit.dataset.model.pulsar_id.clone(),
        binary_family: fit
            .dataset
            .model
            .binary_family
            .as_ref()
            .map(|family| family.as_str().to_string())
            .unwrap_or_else(|| "isolated".to_string()),
        tactical_lane: fit.covariance_calibration.tactical_lane.clone(),
        covariance_preset: fit.covariance_calibration.preset_name.clone(),
        provisional_acceptance_policy: fit
            .covariance_calibration
            .provisional_acceptance_policy
            .clone(),
        recommended_solver: fit.summary.recommended_solver.clone(),
        acceptance_track: fit.summary.acceptance_track.clone(),
        metric_conflict: fit.summary.metric_conflict,
        ecorr_status: ecorr_status_for_fit(fit),
        family_equivalence_target: equivalence.target,
        family_equivalence_status: equivalence.status,
        raw_improvement_wls_frac: fit.summary.raw_improvement_wls_frac,
        raw_improvement_gls_frac: fit.summary.raw_improvement_gls_frac,
        weighted_improvement_wls_frac: fit.summary.weighted_improvement_wls_frac,
        weighted_improvement_gls_frac: fit.summary.weighted_improvement_gls_frac,
        dm_improvement_wls_frac: fit.summary.dm_improvement_wls_frac,
        dm_improvement_gls_frac: fit.summary.dm_improvement_gls_frac,
        synthesis_score_wls: fit.summary.synthesis_score_wls,
        synthesis_score_gls: fit.summary.synthesis_score_gls,
        wls_plausibility_status: wls_plausibility.status,
        gls_plausibility_status: gls_plausibility.status,
        wls_plausibility_flags: wls_plausibility.flags,
        gls_plausibility_flags: gls_plausibility.flags,
    })
}

fn write_comparison_report(path: &Path, rows: &[ComparisonRow]) -> Result<()> {
    let mut out = String::new();
    writeln!(out, "[metadata]")?;
    writeln!(
        out,
        "title = \"NANOGrav Phase 1 independent timing-engine comparison surface\""
    )?;
    writeln!(out, "solution_count = {}", rows.len())?;
    writeln!(
        out,
        "tactical_lane_note = \"GLS-first is reserved for J1903+0327 and J2214+3000; the remaining Phase 1 systems are currently WLS-first pending tighter family-equivalence closure.\""
    )?;
    for row in rows {
        writeln!(out)?;
        writeln!(out, "[[pulsar]]")?;
        writeln!(out, "solution_id = {:?}", row.solution_id)?;
        writeln!(out, "pulsar_id = {:?}", row.pulsar_id)?;
        writeln!(out, "binary_family = {:?}", row.binary_family)?;
        writeln!(out, "tactical_lane = {:?}", row.tactical_lane)?;
        writeln!(out, "covariance_preset = {:?}", row.covariance_preset)?;
        writeln!(
            out,
            "provisional_acceptance_policy = {:?}",
            row.provisional_acceptance_policy
        )?;
        writeln!(out, "recommended_solver = {:?}", row.recommended_solver)?;
        writeln!(out, "acceptance_track = {:?}", row.acceptance_track)?;
        writeln!(out, "metric_conflict = {}", row.metric_conflict)?;
        writeln!(out, "ecorr_status = {:?}", row.ecorr_status)?;
        writeln!(
            out,
            "family_equivalence_target = {:?}",
            row.family_equivalence_target
        )?;
        writeln!(
            out,
            "family_equivalence_status = {:?}",
            row.family_equivalence_status
        )?;
        writeln!(
            out,
            "raw_improvement_wls_frac = {:.12}",
            row.raw_improvement_wls_frac
        )?;
        writeln!(
            out,
            "raw_improvement_gls_frac = {:.12}",
            row.raw_improvement_gls_frac
        )?;
        writeln!(
            out,
            "weighted_improvement_wls_frac = {:.12}",
            row.weighted_improvement_wls_frac
        )?;
        writeln!(
            out,
            "weighted_improvement_gls_frac = {:.12}",
            row.weighted_improvement_gls_frac
        )?;
        if let Some(value) = row.dm_improvement_wls_frac {
            writeln!(out, "dm_improvement_wls_frac = {:.12}", value)?;
        }
        if let Some(value) = row.dm_improvement_gls_frac {
            writeln!(out, "dm_improvement_gls_frac = {:.12}", value)?;
        }
        writeln!(out, "synthesis_score_wls = {:.12}", row.synthesis_score_wls)?;
        writeln!(out, "synthesis_score_gls = {:.12}", row.synthesis_score_gls)?;
        writeln!(
            out,
            "wls_plausibility_status = {:?}",
            row.wls_plausibility_status
        )?;
        writeln!(
            out,
            "gls_plausibility_status = {:?}",
            row.gls_plausibility_status
        )?;
        writeln!(
            out,
            "wls_plausibility_flags = {:?}",
            row.wls_plausibility_flags
        )?;
        writeln!(
            out,
            "gls_plausibility_flags = {:?}",
            row.gls_plausibility_flags
        )?;
    }
    write_report(path, &out)
}

fn write_pairwise_csv(path: &Path, rows: &[PairwiseIndependentRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = Writer::from_path(path)?;
    writer.write_record([
        "pulsar_a",
        "pulsar_b",
        "separation_deg",
        "hellings_downs",
        "overlap_bins_wls",
        "overlap_bins_gls",
        "overlap_bins_policy",
        "wls_residual_pearson",
        "gls_residual_pearson",
        "policy_residual_pearson",
    ])?;
    for row in rows {
        writer.write_record([
            row.pulsar_a.as_str(),
            row.pulsar_b.as_str(),
            &format!("{:.12}", row.separation_deg),
            &format!("{:.12}", row.hellings_downs),
            &row.overlap_bins_wls.to_string(),
            &row.overlap_bins_gls.to_string(),
            &row.overlap_bins_policy.to_string(),
            &format_opt_f64(row.wls_residual_pearson),
            &format_opt_f64(row.gls_residual_pearson),
            &format_opt_f64(row.policy_residual_pearson),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_pairwise_report(
    path: &Path,
    csv_path: &Path,
    rows: &[PairwiseIndependentRow],
    pair_bin_days: f64,
    min_pair_overlap_bins: usize,
) -> Result<()> {
    let mut out = String::new();
    writeln!(out, "[metadata]")?;
    writeln!(
        out,
        "title = \"NANOGrav Phase 1 independent pairwise regenerated-residual audit\""
    )?;
    writeln!(out, "pairwise_csv = {:?}", csv_path.display().to_string())?;
    writeln!(out, "pair_count = {}", rows.len())?;
    writeln!(out, "pair_bin_days = {:.12}", pair_bin_days)?;
    writeln!(out, "min_pair_overlap_bins = {}", min_pair_overlap_bins)?;
    for row in rows {
        writeln!(out)?;
        writeln!(out, "[[pair]]")?;
        writeln!(out, "pulsar_a = {:?}", row.pulsar_a)?;
        writeln!(out, "pulsar_b = {:?}", row.pulsar_b)?;
        writeln!(out, "separation_deg = {:.12}", row.separation_deg)?;
        writeln!(out, "hellings_downs = {:.12}", row.hellings_downs)?;
        writeln!(out, "overlap_bins_wls = {}", row.overlap_bins_wls)?;
        writeln!(out, "overlap_bins_gls = {}", row.overlap_bins_gls)?;
        writeln!(out, "overlap_bins_policy = {}", row.overlap_bins_policy)?;
        if let Some(value) = row.wls_residual_pearson {
            writeln!(out, "wls_residual_pearson = {:.12}", value)?;
        }
        if let Some(value) = row.gls_residual_pearson {
            writeln!(out, "gls_residual_pearson = {:.12}", value)?;
        }
        if let Some(value) = row.policy_residual_pearson {
            writeln!(out, "policy_residual_pearson = {:.12}", value)?;
        }
    }
    write_report(path, &out)
}

fn build_pairwise_rows(
    fits: &[IndependentRefitResult],
    pair_bin_days: f64,
    min_pair_overlap_bins: usize,
) -> Vec<PairwiseIndependentRow> {
    let prepared = fits
        .iter()
        .map(|fit| PreparedPairwisePulsar {
            pulsar: fit.dataset.model.pulsar_id.clone(),
            sky_vector: model_sky_vector(&fit.dataset.model),
            wls_bins: bin_residual_rows(&fit.rows, pair_bin_days, SolverTrack::Wls),
            gls_bins: bin_residual_rows(&fit.rows, pair_bin_days, SolverTrack::Gls),
            policy_bins: bin_residual_rows(
                &fit.rows,
                pair_bin_days,
                if fit.summary.recommended_solver == "gls" {
                    SolverTrack::Gls
                } else {
                    SolverTrack::Wls
                },
            ),
        })
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    for left_index in 0..prepared.len() {
        for right_index in (left_index + 1)..prepared.len() {
            let left = &prepared[left_index];
            let right = &prepared[right_index];
            let (Some(left_vec), Some(right_vec)) = (left.sky_vector, right.sky_vector) else {
                continue;
            };
            let wls_overlap = overlapping_values(&left.wls_bins, &right.wls_bins);
            let gls_overlap = overlapping_values(&left.gls_bins, &right.gls_bins);
            let policy_overlap = overlapping_values(&left.policy_bins, &right.policy_bins);
            let max_overlap = wls_overlap
                .0
                .len()
                .max(gls_overlap.0.len())
                .max(policy_overlap.0.len());
            if max_overlap < min_pair_overlap_bins {
                continue;
            }
            let separation_rad = angular_separation(left_vec, right_vec);
            rows.push(PairwiseIndependentRow {
                pulsar_a: left.pulsar.clone(),
                pulsar_b: right.pulsar.clone(),
                separation_deg: separation_rad.to_degrees(),
                hellings_downs: hellings_downs(separation_rad),
                overlap_bins_wls: wls_overlap.0.len(),
                overlap_bins_gls: gls_overlap.0.len(),
                overlap_bins_policy: policy_overlap.0.len(),
                wls_residual_pearson: pearson_correlation(&wls_overlap.0, &wls_overlap.1),
                gls_residual_pearson: pearson_correlation(&gls_overlap.0, &gls_overlap.1),
                policy_residual_pearson: pearson_correlation(&policy_overlap.0, &policy_overlap.1),
            });
        }
    }
    rows.sort_by_key(|row| std::cmp::Reverse(row.overlap_bins_policy));
    rows
}

#[derive(Debug, Clone, Copy)]
enum SolverTrack {
    Wls,
    Gls,
}

fn bin_residual_rows(
    rows: &[gororoba_cli_data::nanograv_timing_engine::IndependentRefitRow],
    bin_days: f64,
    track: SolverTrack,
) -> BTreeMap<i64, f64> {
    let mut accumulators: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
    for row in rows {
        let value = match track {
            SolverTrack::Wls => row.residual_after_wls_us,
            SolverTrack::Gls => row.residual_after_gls_us,
        };
        let bin = (row.mjd_utc / bin_days).floor() as i64;
        let weight = if row.uncertainty_us > 0.0 {
            1.0 / (row.uncertainty_us * row.uncertainty_us)
        } else {
            1.0
        };
        let entry = accumulators.entry(bin).or_insert((0.0, 0.0));
        entry.0 += weight * value;
        entry.1 += weight;
    }
    accumulators
        .into_iter()
        .filter_map(|(bin, (weighted_sum, weight_sum))| {
            (weight_sum > 0.0).then_some((bin, weighted_sum / weight_sum))
        })
        .collect()
}

fn overlapping_values(
    left: &BTreeMap<i64, f64>,
    right: &BTreeMap<i64, f64>,
) -> (Vec<f64>, Vec<f64>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (bin, left_value) in left {
        if let Some(right_value) = right.get(bin) {
            xs.push(*left_value);
            ys.push(*right_value);
        }
    }
    (xs, ys)
}

fn pearson_correlation(xs: &[f64], ys: &[f64]) -> Option<f64> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return None;
    }
    let x_mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let y_mean = ys.iter().sum::<f64>() / ys.len() as f64;
    let mut numerator = 0.0_f64;
    let mut denom_x = 0.0_f64;
    let mut denom_y = 0.0_f64;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let dx = *x - x_mean;
        let dy = *y - y_mean;
        numerator += dx * dy;
        denom_x += dx * dx;
        denom_y += dy * dy;
    }
    let denom = (denom_x * denom_y).sqrt();
    (denom > 0.0).then_some(numerator / denom)
}

fn model_sky_vector(model: &TimingModel) -> Option<[f64; 3]> {
    if let (Some(elong), Some(elat)) = (
        model.parameter_value("ELONG"),
        model.parameter_value("ELAT"),
    ) {
        return Some(ecliptic_to_equatorial_vector(
            elong.to_radians(),
            elat.to_radians(),
        ));
    }
    let raj = model.parameter_term("RAJ")?.raw_value.as_str();
    let decj = model.parameter_term("DECJ")?.raw_value.as_str();
    let ra_rad = parse_hms_radians(raj)?;
    let dec_rad = parse_dms_radians(decj)?;
    Some([
        dec_rad.cos() * ra_rad.cos(),
        dec_rad.cos() * ra_rad.sin(),
        dec_rad.sin(),
    ])
}

fn ecliptic_to_equatorial_vector(longitude_rad: f64, latitude_rad: f64) -> [f64; 3] {
    let epsilon = 23.439_291_1_f64.to_radians();
    let x_ecl = latitude_rad.cos() * longitude_rad.cos();
    let y_ecl = latitude_rad.cos() * longitude_rad.sin();
    let z_ecl = latitude_rad.sin();
    [
        x_ecl,
        y_ecl * epsilon.cos() - z_ecl * epsilon.sin(),
        y_ecl * epsilon.sin() + z_ecl * epsilon.cos(),
    ]
}

fn parse_hms_radians(value: &str) -> Option<f64> {
    let fields = value.split(':').collect::<Vec<_>>();
    if fields.len() != 3 {
        return None;
    }
    let hours = fields[0].parse::<f64>().ok()?;
    let minutes = fields[1].parse::<f64>().ok()?;
    let seconds = fields[2].parse::<f64>().ok()?;
    Some(((hours + minutes / 60.0 + seconds / 3600.0) * 15.0).to_radians())
}

fn parse_dms_radians(value: &str) -> Option<f64> {
    let fields = value.split(':').collect::<Vec<_>>();
    if fields.len() != 3 {
        return None;
    }
    let degrees = fields[0].parse::<f64>().ok()?;
    let minutes = fields[1].parse::<f64>().ok()?;
    let seconds = fields[2].parse::<f64>().ok()?;
    let sign = if degrees < 0.0 { -1.0 } else { 1.0 };
    let magnitude = degrees.abs() + minutes / 60.0 + seconds / 3600.0;
    Some((sign * magnitude).to_radians())
}

fn angular_separation(left: [f64; 3], right: [f64; 3]) -> f64 {
    let dot = (left[0] * right[0] + left[1] * right[1] + left[2] * right[2]).clamp(-1.0, 1.0);
    dot.acos()
}

fn hellings_downs(separation_rad: f64) -> f64 {
    let x = (1.0 - separation_rad.cos()) / 2.0;
    if x <= 0.0 {
        0.5
    } else {
        1.5 * x * x.ln() - 0.25 * x + 0.5
    }
}

fn ecorr_status_for_fit(fit: &IndependentRefitResult) -> String {
    let ecorr_terms = fit
        .dataset
        .model
        .noise_terms
        .iter()
        .filter(|term| term.name == "ECORR")
        .count();
    if ecorr_terms == 0 {
        "no_release_ecorr_terms".to_string()
    } else if fit.covariance_calibration.ecorr_basis_count == 0 {
        "selectors_unmatched".to_string()
    } else {
        "active".to_string()
    }
}

fn assess_parameter_plausibility(
    model: &TimingModel,
    parameter_names: &[String],
    coefficients: &[f64],
) -> PlausibilityAssessment {
    let mut severe = Vec::new();
    let mut advisory = Vec::new();
    let mut dmx_max_abs_delta = 0.0_f64;
    for (name, coefficient) in parameter_names.iter().zip(coefficients.iter()) {
        let absolute = resolved_parameter_value(model, name, *coefficient);
        match name.as_str() {
            "PX" => {
                if let Some(value) = absolute
                    && value < 0.0
                {
                    severe.push(format!("PX_negative:{value:.6e}"));
                }
            }
            "M2" => {
                if let Some(value) = absolute
                    && (value <= 0.0 || value > 5.0)
                {
                    severe.push(format!("M2_out_of_range:{value:.6e}"));
                }
            }
            "SINI" => {
                if let Some(value) = absolute
                    && !(0.0..=1.0).contains(&value)
                {
                    severe.push(format!("SINI_out_of_range:{value:.6e}"));
                }
            }
            "KIN" => {
                if let Some(value) = absolute
                    && !(0.0..=180.0).contains(&value)
                {
                    severe.push(format!("KIN_out_of_range:{value:.6e}"));
                }
            }
            "KOM" => {
                if let Some(value) = absolute
                    && !(0.0..360.0).contains(&value)
                {
                    advisory.push(format!("KOM_unwrapped:{value:.6e}"));
                }
            }
            "OMDOT" | "PBDOT" if exceeds_uncertainty_gate(model, name, *coefficient, 20.0) => {
                advisory.push(format!("{name}_large_delta:{coefficient:.6e}"));
            }
            value if value.starts_with("DMX_") => {
                dmx_max_abs_delta = dmx_max_abs_delta.max(coefficient.abs());
                if exceeds_uncertainty_gate(model, name, *coefficient, 25.0) {
                    advisory.push(format!("{name}_large_delta:{coefficient:.6e}"));
                }
            }
            _ => {}
        }
    }
    if dmx_max_abs_delta > 5.0e-3 {
        advisory.push(format!("DMX_runaway_delta_max:{dmx_max_abs_delta:.6e}"));
    }
    let status = if !severe.is_empty() {
        "reject"
    } else if !advisory.is_empty() {
        "advisory"
    } else {
        "accept"
    };
    let mut flags = severe;
    flags.extend(advisory);
    PlausibilityAssessment {
        status: status.to_string(),
        flags,
    }
}

fn resolved_parameter_value(model: &TimingModel, name: &str, coefficient: f64) -> Option<f64> {
    if name == "PHASE_OFFSET" {
        return Some(coefficient);
    }
    if let Some(index) = parse_selector_parameter_index(name, "JUMP@") {
        return model
            .jumps
            .get(index)
            .and_then(|term| term.value)
            .map(|value| value + coefficient)
            .or(Some(coefficient));
    }
    if let Some(index) = parse_selector_parameter_index(name, "DMJUMP@") {
        return model
            .dmjumps
            .get(index)
            .and_then(|term| term.value)
            .map(|value| value + coefficient)
            .or(Some(coefficient));
    }
    model.parameter_value(name).map(|value| value + coefficient)
}

fn exceeds_uncertainty_gate(
    model: &TimingModel,
    name: &str,
    coefficient: f64,
    sigma_multiple: f64,
) -> bool {
    let Some(term) = model.parameter_term(name) else {
        return false;
    };
    if let Some(uncertainty) = term.uncertainty.filter(|value| *value > 0.0) {
        coefficient.abs() > sigma_multiple * uncertainty
    } else {
        false
    }
}

fn assess_family_equivalence(fit: &IndependentRefitResult) -> FamilyEquivalenceAssessment {
    let available = fit.parameter_names.iter().cloned().collect::<BTreeSet<_>>();
    let model = &fit.dataset.model;
    match model.binary_family.as_ref() {
        Some(BinaryFamily::Dd) if model.pulsar_id == "J1903+0327" => {
            let required = [
                "A1", "PB", "ECC", "T0", "OM", "OMDOT", "PBDOT", "GAMMA", "M2", "SINI",
            ];
            equivalence_from_required("DD_targeted_surface", &available, &required)
        }
        Some(BinaryFamily::Ell1) if model.pulsar_id == "J2214+3000" => {
            let has_fb = available.iter().any(|name| name.starts_with("FB"));
            let base = ["A1", "TASC", "EPS1", "EPS2"];
            let mut assessment =
                equivalence_from_required("ELL1_targeted_surface", &available, &base);
            if !has_fb {
                assessment.status.push_str("_missing_FBn");
            }
            assessment
        }
        Some(BinaryFamily::Ddk) if model.pulsar_id == "J1713+0747" => {
            let required = [
                "A1", "PB", "ECC", "T0", "OM", "OMDOT", "PBDOT", "GAMMA", "KIN", "KOM", "M2",
            ];
            let mut assessment =
                equivalence_from_required("DDK_closure_surface", &available, &required);
            if model.parameter_bool("K96") != Some(true) {
                assessment.status.push_str("_k96_off_or_missing");
            }
            assessment
        }
        Some(BinaryFamily::Ell1h) if model.pulsar_id == "J2317+1439" => {
            let required = ["A1", "TASC", "EPS1", "EPS2", "H3"];
            let mut assessment =
                equivalence_from_required("ELL1H_closure_surface", &available, &required);
            let has_stigma = available.contains("STIGMA");
            let has_h4 = available.contains("H4");
            if !(has_stigma || has_h4) {
                assessment.status.push_str("_missing_orthometric_companion");
            }
            assessment
        }
        Some(BinaryFamily::Dd) => {
            equivalence_from_required("DD_surface", &available, &["A1", "PB", "ECC", "T0", "OM"])
        }
        Some(BinaryFamily::Bt) => {
            equivalence_from_required("BT_surface", &available, &["A1", "PB", "ECC", "T0", "OM"])
        }
        Some(BinaryFamily::Ell1) => {
            equivalence_from_required("ELL1_surface", &available, &["A1", "TASC", "EPS1", "EPS2"])
        }
        Some(BinaryFamily::Ell1h) => equivalence_from_required(
            "ELL1H_surface",
            &available,
            &["A1", "TASC", "EPS1", "EPS2", "H3"],
        ),
        Some(BinaryFamily::Ddk) => equivalence_from_required(
            "DDK_surface",
            &available,
            &["A1", "PB", "ECC", "T0", "OM", "KIN", "KOM"],
        ),
        _ => FamilyEquivalenceAssessment {
            target: "isolated_or_unspecified".to_string(),
            status: "not_applicable".to_string(),
        },
    }
}

fn equivalence_from_required(
    target: &str,
    available: &BTreeSet<String>,
    required: &[&str],
) -> FamilyEquivalenceAssessment {
    let missing = required
        .iter()
        .filter(|name| !available.contains(**name))
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();
    let status = if missing.is_empty() {
        "surface_complete_pending_numeric_crosscheck".to_string()
    } else {
        format!("missing_required_surface:{missing:?}")
    };
    FamilyEquivalenceAssessment {
        target: target.to_string(),
        status,
    }
}

fn parse_selector_parameter_index(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(prefix)?.parse::<usize>().ok()
}

fn format_opt_f64(value: Option<f64>) -> String {
    value.map_or_else(String::new, |inner| format!("{inner:.12}"))
}
