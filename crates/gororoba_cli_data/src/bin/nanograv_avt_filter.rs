use algebra_analysis::{
    codebook::{
        LatticeVector, enumerate_lambda_4096, is_in_lambda_256, is_in_lambda_512, is_in_lambda_1024,
    },
    sky_mapping::project_sky_to_basis,
};
use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use csv::Writer;
use data_core::spatial::equatorial_to_galactic_lat;
use gororoba_algebra::gpu::{ComputeBackend, TensorAVT};
use gororoba_cli_data::nanograv_timing::{PairAuditRow, PulsarTimingData, load_release};
use serde::Deserialize;
use stats_core::{
    helpers::mean,
    metrics::{centered_rms, pearson, percent_drop, rms},
};
use std::{
    collections::BTreeMap,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

const DIMENSIONS: [usize; 7] = [16, 32, 64, 128, 256, 512, 1024];

#[derive(Parser, Debug)]
#[command(
    name = "nanograv-avt-filter",
    about = "Cross-validated AVT mean-shift audit over NANOGrav timing residuals"
)]
struct Args {
    #[arg(
        long,
        default_value = "data/external/nanograv_15yr_timing/NANOGrav15yr_PulsarTiming_v2.1.0"
    )]
    root: PathBuf,

    #[arg(long, default_value = "data/csv/nanograv_avt_whitening_sweep.csv")]
    csv_out: PathBuf,

    #[arg(long, default_value = "data/csv/nanograv_pulsar_frustrations_512d.csv")]
    frustration_csv_out: PathBuf,

    #[arg(long, default_value = "data/csv/nanograv_15yr_pairwise_hd_audit.csv")]
    pairwise_csv: PathBuf,

    #[arg(long, default_value = "reports/nanograv_avt_filter.toml")]
    report_out: PathBuf,

    #[arg(long)]
    independent_residual_csv: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = ResidualSurface::Avg)]
    surface: ResidualSurface,

    #[arg(long, value_enum, default_value_t = IndependentResidualSurface::Policy)]
    independent_surface: IndependentResidualSurface,

    #[arg(long, default_value_t = 4)]
    folds: usize,

    #[arg(long, default_value_t = 8)]
    null_shifts: usize,

    #[arg(long, default_value_t = -2.0)]
    lambda_min: f64,

    #[arg(long, default_value_t = 2.0)]
    lambda_max: f64,

    #[arg(long, default_value_t = 1601)]
    lambda_steps: usize,

    #[arg(long)]
    lat_min: Option<f64>,

    #[arg(long)]
    lat_max: Option<f64>,

    #[arg(long)]
    lat_exclude: bool,

    #[arg(long)]
    gpu: bool,

    #[arg(long)]
    mjd_start: Option<f64>,

    #[arg(long)]
    mjd_end: Option<f64>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum ResidualSurface {
    Avg,
    Full,
}

impl ResidualSurface {
    fn as_str(self) -> &'static str {
        match self {
            Self::Avg => "avg",
            Self::Full => "full",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum IndependentResidualSurface {
    Before,
    Wls,
    Gls,
    Policy,
}

#[derive(Debug, Clone)]
struct PulsarProjection {
    pulsar: String,
    residuals: Vec<f64>,
    raw_mean_us: f64,
    raw_rms_us: f64,
    centered_rms_us: f64,
    frustration_z: f64,
    avg_uncertainty_mean_us: f64,
}

#[derive(Debug, Deserialize)]
struct IndependentResidualRow {
    pulsar_id: String,
    mjd_utc: f64,
    uncertainty_us: f64,
    residual_before_us: f64,
    residual_after_wls_us: f64,
    residual_after_gls_us: f64,
    residual_after_policy_us: f64,
}

#[derive(Debug, Deserialize)]
struct PairwiseInputRow {
    pulsar_a: String,
    pulsar_b: String,
    hellings_downs: f64,
    #[serde(default)]
    overlap_bins: Option<usize>,
    #[serde(default)]
    avg_residual_pearson: Option<f64>,
    #[serde(default)]
    overlap_bins_policy: Option<usize>,
    #[serde(default)]
    policy_residual_pearson: Option<f64>,
}

#[derive(Debug, Clone)]
struct PerDimensionSummary {
    dimension: usize,
    backend: &'static str,
    pulsar_count: usize,
    nonzero_frustration_count: usize,
    frustration_min: f64,
    frustration_max: f64,
    frustration_std: f64,
    mean_residual_corr: Option<f64>,
    lambda_mean: f64,
    baseline_raw_rms_us: f64,
    fitted_raw_rms_us: f64,
    raw_drop_pct: f64,
    baseline_centered_rms_us: f64,
    fitted_centered_rms_us: f64,
    centered_drop_pct: f64,
    null_mean_drop_pct: f64,
    null_max_drop_pct: f64,
    null_pvalue: f64,
    bonferroni_pvalue: f64,
    bh_q_value: f64,
    static_field_centered_invariant: bool,
    per_pulsar_drop_pct: f64,
    cross_corr_drop_pct: f64,
    cv_cross_corr_drop_pct: f64,
}

#[derive(Debug, Clone, Default)]
struct RunningStats {
    count: usize,
    mean: f64,
    m2: f64,
    min: Option<f64>,
    max: Option<f64>,
}

impl RunningStats {
    fn push(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        if self.min.is_none_or(|current| value < current) {
            self.min = Some(value);
        }
        if self.max.is_none_or(|current| value > current) {
            self.max = Some(value);
        }
    }

    fn stddev(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            (self.m2 / (self.count as f64 - 1.0)).sqrt()
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.folds < 2 {
        bail!("--folds must be >= 2");
    }
    if args.lambda_steps < 2 {
        bail!("--lambda-steps must be >= 2");
    }
    if !args.root.exists() {
        bail!("timing release root not found: {}", args.root.display());
    }

    let release = load_release(&args.root).context("failed to load timing release")?;
    if release.is_empty() {
        bail!("no pulsars found under {}", args.root.display());
    }

    let mut pairwise_rows = Vec::new();
    if args.pairwise_csv.exists() {
        let mut rdr = csv::Reader::from_path(&args.pairwise_csv)?;
        for result in rdr.deserialize() {
            let record: PairwiseInputRow = result?;
            pairwise_rows.push(PairAuditRow {
                pulsar_a: record.pulsar_a,
                pulsar_b: record.pulsar_b,
                hellings_downs: record.hellings_downs,
                overlap_bins: record
                    .overlap_bins_policy
                    .or(record.overlap_bins)
                    .unwrap_or(0),
                avg_residual_pearson: record
                    .policy_residual_pearson
                    .or(record.avg_residual_pearson)
                    .unwrap_or(0.0),
            });
        }
    }

    let backend = if args.gpu {
        ComputeBackend::Cuda
    } else {
        ComputeBackend::CpuScalar
    };
    let lambda_grid = build_lambda_grid(&args);
    let mut summaries = Vec::new();

    for &dimension in &DIMENSIONS {
        let lattice = lattice_for_dimension(dimension);
        let prepared = prepare_pulsars(&release, &args, &lattice, dimension, backend)?;
        if prepared.len() < args.folds {
            bail!(
                "dimension {} produced only {} prepared pulsars, fewer than {} folds",
                dimension,
                prepared.len(),
                args.folds
            );
        }
        summaries.push(analyze_dimension(
            &prepared,
            &lambda_grid,
            &args,
            dimension,
            backend,
            &pairwise_rows,
        ));
    }
    apply_multiple_testing_corrections(&mut summaries);

    write_csv(&args.csv_out, &summaries)?;
    write_report(&args.report_out, &args, &summaries)?;
    let lattice_512 = lattice_for_dimension(512);
    let prepared_512 = prepare_pulsars(&release, &args, &lattice_512, 512, backend)?;
    ensure_parent_dir(&args.frustration_csv_out)?;
    let mut frust_writer = csv::Writer::from_path(&args.frustration_csv_out)?;
    frust_writer.write_record(["pulsar", "frustration_z", "ra_deg", "dec_deg"])?;
    for p in prepared_512 {
        if let Some(sky) = release.get(&p.pulsar).and_then(|d| d.sky_vector()) {
            let ra = sky[1].atan2(sky[0]).to_degrees();
            let dec = sky[2].asin().to_degrees();
            frust_writer.write_record([
                p.pulsar,
                format!("{:.12}", p.frustration_z),
                format!("{:.12}", ra),
                format!("{:.12}", dec),
            ])?;
        }
    }
    frust_writer.flush()?;
    println!(
        "Resonance data saved to {}",
        args.frustration_csv_out.display()
    );

    println!("AVT audit dimensions: {}", summaries.len());
    println!("CSV: {}", args.csv_out.display());
    println!("Report: {}", args.report_out.display());
    Ok(())
}

fn build_lambda_grid(args: &Args) -> Vec<f64> {
    let step = (args.lambda_max - args.lambda_min) / (args.lambda_steps as f64 - 1.0);
    (0..args.lambda_steps)
        .map(|idx| args.lambda_min + step * idx as f64)
        .collect()
}

fn lattice_for_dimension(dimension: usize) -> Vec<LatticeVector> {
    match dimension {
        1024 => enumerate_lambda_1024(),
        512 => enumerate_lambda_512(),
        256 => enumerate_lambda_256(),
        _ => enumerate_lambda_generic(dimension),
    }
}

fn enumerate_lambda_1024() -> Vec<LatticeVector> {
    enumerate_lambda_4096()
        .into_iter()
        .filter(is_in_lambda_1024)
        .collect()
}

fn enumerate_lambda_512() -> Vec<LatticeVector> {
    enumerate_lambda_4096()
        .into_iter()
        .filter(is_in_lambda_512)
        .collect()
}

fn enumerate_lambda_256() -> Vec<LatticeVector> {
    enumerate_lambda_4096()
        .into_iter()
        .filter(is_in_lambda_256)
        .collect()
}

fn enumerate_lambda_generic(dimension: usize) -> Vec<LatticeVector> {
    enumerate_lambda_4096()
        .into_iter()
        .take(dimension)
        .collect()
}

fn prepare_pulsars(
    release: &std::collections::BTreeMap<String, PulsarTimingData>,
    args: &Args,
    lattice: &[LatticeVector],
    dimension: usize,
    backend: ComputeBackend,
) -> Result<Vec<PulsarProjection>> {
    if let Some(path) = &args.independent_residual_csv {
        return prepare_pulsars_from_independent_csv(release, path, args, lattice, dimension, backend);
    }
    let mut names = Vec::new();
    let mut residual_sets = Vec::new();
    let mut uncertainty_sets = Vec::new();
    let mut packed_vectors = Vec::new();

    for (name, data) in release {
        let residuals = residual_series(data, args);
        if residuals.is_empty() {
            continue;
        }
        let Some(sky) = data.sky_vector() else {
            continue;
        };

        let ra_deg = sky[1].atan2(sky[0]).to_degrees();
        let dec_deg = sky[2].asin().to_degrees();
        let b = equatorial_to_galactic_lat(ra_deg, dec_deg);

        let mut inside = true;
        if let Some(min) = args.lat_min && b < min {
            inside = false;
        }
        if let Some(max) = args.lat_max && b > max {
            inside = false;
        }

        if args.lat_exclude {
            if inside {
                continue;
            }
        } else {
            if !inside {
                continue;
            }
        }

        let uncertainties: Vec<f64> = data
            .avg_residuals
            .iter()
            .filter_map(|p| p.uncertainty_us)
            .collect();
        let avg_unc = if uncertainties.is_empty() {
            1.0
        } else {
            stats_core::helpers::mean(&uncertainties)
        };

        let projected = project_sky_to_basis(&sky, lattice, dimension);
        packed_vectors.extend(projected.iter().map(|value| *value as f32));
        names.push(name.clone());
        residual_sets.push(residuals);
        uncertainty_sets.push(avg_unc);
    }

    if names.is_empty() {
        bail!("no pulsars with usable residuals and sky metadata for AVT audit");
    }

    let frustrations =
        compute_frustration_scores(&packed_vectors, names.len(), dimension, backend)?;
    let standardized = zscore(&frustrations);
    let mut out = Vec::new();
    for (((name, residuals), frustration_z), avg_unc) in names
        .into_iter()
        .zip(residual_sets)
        .zip(standardized)
        .zip(uncertainty_sets)
    {
        let raw_mean_us = mean(&residuals);
        let raw_rms_us = rms(&residuals);
        let centered_rms_us = centered_rms(&residuals, raw_mean_us);
        out.push(PulsarProjection {
            pulsar: name,
            residuals,
            raw_mean_us,
            raw_rms_us,
            centered_rms_us,
            frustration_z,
            avg_uncertainty_mean_us: avg_unc,
        });
    }
    out.sort_by(|left, right| left.pulsar.cmp(&right.pulsar));
    Ok(out)
}

fn prepare_pulsars_from_independent_csv(
    release: &std::collections::BTreeMap<String, PulsarTimingData>,
    csv_path: &Path,
    args: &Args,
    lattice: &[LatticeVector],
    dimension: usize,
    backend: ComputeBackend,
) -> Result<Vec<PulsarProjection>> {
    let mut grouped: BTreeMap<String, Vec<IndependentResidualRow>> = BTreeMap::new();
    let mut reader = csv::Reader::from_path(csv_path)?;
    for result in reader.deserialize() {
        let row: IndependentResidualRow = result?;
        if let Some(start) = args.mjd_start && row.mjd_utc < start {
            continue;
        }
        if let Some(end) = args.mjd_end && row.mjd_utc > end {
            continue;
        }
        grouped.entry(row.pulsar_id.clone()).or_default().push(row);
    }

    let mut names = Vec::new();
    let mut residual_sets = Vec::new();
    let mut uncertainty_sets = Vec::new();
    let mut packed_vectors = Vec::new();
    for (name, rows) in grouped {
        let Some(data) = release.get(&name) else {
            continue;
        };
        let Some(sky) = data.sky_vector() else {
            continue;
        };
        let residuals = rows
            .iter()
            .map(|row| match args.independent_surface {
                IndependentResidualSurface::Before => row.residual_before_us,
                IndependentResidualSurface::Wls => row.residual_after_wls_us,
                IndependentResidualSurface::Gls => row.residual_after_gls_us,
                IndependentResidualSurface::Policy => row.residual_after_policy_us,
            })
            .collect::<Vec<_>>();
        if residuals.is_empty() {
            continue;
        }

        let ra_deg = sky[1].atan2(sky[0]).to_degrees();
        let dec_deg = sky[2].asin().to_degrees();
        let b = equatorial_to_galactic_lat(ra_deg, dec_deg);
        let mut inside = true;
        if let Some(min) = args.lat_min && b < min {
            inside = false;
        }
        if let Some(max) = args.lat_max && b > max {
            inside = false;
        }
        if args.lat_exclude {
            if inside {
                continue;
            }
        } else if !inside {
            continue;
        }

        let avg_unc = mean(
            &rows.iter()
                .map(|row| row.uncertainty_us.max(1.0e-6))
                .collect::<Vec<_>>(),
        );
        let projected = project_sky_to_basis(&sky, lattice, dimension);
        packed_vectors.extend(projected.iter().map(|value| *value as f32));
        names.push(name);
        residual_sets.push(residuals);
        uncertainty_sets.push(avg_unc);
    }

    if names.is_empty() {
        bail!(
            "no pulsars with usable independent residuals and sky metadata for AVT audit"
        );
    }

    let frustrations =
        compute_frustration_scores(&packed_vectors, names.len(), dimension, backend)?;
    let standardized = zscore(&frustrations);
    let mut out = Vec::new();
    for (((name, residuals), frustration_z), avg_unc) in names
        .into_iter()
        .zip(residual_sets)
        .zip(standardized)
        .zip(uncertainty_sets)
    {
        let raw_mean_us = mean(&residuals);
        let raw_rms_us = rms(&residuals);
        let centered_rms_us = centered_rms(&residuals, raw_mean_us);
        out.push(PulsarProjection {
            pulsar: name,
            residuals,
            raw_mean_us,
            raw_rms_us,
            centered_rms_us,
            frustration_z,
            avg_uncertainty_mean_us: avg_unc,
        });
    }
    out.sort_by(|left, right| left.pulsar.cmp(&right.pulsar));
    Ok(out)
}

fn compute_frustration_scores(
    packed_vectors: &[f32],
    count: usize,
    dimension: usize,
    backend: ComputeBackend,
) -> Result<Vec<f64>> {
    let avt = TensorAVT::new(dimension);
    let mut session = avt
        .new_norm_session(backend, count)
        .map_err(|err| anyhow::anyhow!("failed to initialize {:?} TensorAVT: {}", backend, err))?;
    session
        .load_vectors(packed_vectors, count)
        .map_err(|err| anyhow::anyhow!("failed to load AVT vectors: {}", err))?;
    session
        .run_norms(&avt, count)
        .map_err(|err| anyhow::anyhow!("failed to run AVT norms: {}", err))?;
    session
        .download_norms(count)
        .map(|values| values.into_iter().map(f64::from).collect())
        .map_err(|err| anyhow::anyhow!("failed to download AVT norms: {}", err))
}

fn analyze_dimension(
    pulsars: &[PulsarProjection],
    lambda_grid: &[f64],
    args: &Args,
    dimension: usize,
    backend: ComputeBackend,
    pairwise_rows: &[PairAuditRow],
) -> PerDimensionSummary {
    let baseline_raw_rms_us = global_raw_rms(pulsars, 0.0);
    let baseline_centered_rms_us = global_centered_rms(pulsars, 0.0);

    let (lambda_mean, fitted_raw_rms_us, fitted_centered_rms_us) =
        cross_validated_fit(pulsars, args.folds, lambda_grid);

    let raw_drop_pct = percent_drop(baseline_raw_rms_us, fitted_raw_rms_us);
    let centered_drop_pct = percent_drop(baseline_centered_rms_us, fitted_centered_rms_us);

    let null_drops = shifted_null_drops(pulsars, args.folds, lambda_grid, args.null_shifts);
    let null_mean_drop_pct = mean(&null_drops);
    let null_max_drop_pct = null_drops.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let null_hits = null_drops
        .iter()
        .filter(|&&drop| drop >= raw_drop_pct)
        .count();
    let null_pvalue = if null_drops.is_empty() {
        1.0
    } else {
        (null_hits as f64 + 1.0) / (null_drops.len() as f64 + 1.0)
    };

    // --- 1. Per-Pulsar Whitening ---
    let mut total_var_raw = 0.0;
    let mut total_var_whitened = 0.0;
    for pulsar in pulsars {
        let mut sum_sq_raw = 0.0;
        for r in &pulsar.residuals {
            sum_sq_raw += r * r;
        }

        let mut best_sum_sq_whitened = f64::MAX;
        for &lambda in lambda_grid {
            let mut sum_sq = 0.0;
            for r in &pulsar.residuals {
                let w = r - lambda * pulsar.frustration_z;
                sum_sq += w * w;
            }
            if sum_sq < best_sum_sq_whitened {
                best_sum_sq_whitened = sum_sq;
            }
        }

        let weight = if pulsar.avg_uncertainty_mean_us > 0.0 {
            1.0 / (pulsar.avg_uncertainty_mean_us * pulsar.avg_uncertainty_mean_us)
        } else {
            1.0
        };

        total_var_raw += sum_sq_raw * weight;
        total_var_whitened += best_sum_sq_whitened * weight;
    }
    let per_pulsar_drop_pct = percent_drop(total_var_raw, total_var_whitened);

    // --- 2. Cross-Correlation Matrix Analysis ---
    let mut cc_var_raw = 0.0;
    let mut best_cc_var_whitened = f64::MAX;

    if !pairwise_rows.is_empty() {
        let mut frust_map = std::collections::HashMap::new();
        for p in pulsars {
            frust_map.insert(p.pulsar.clone(), p.frustration_z);
        }

        for row in pairwise_rows {
            if frust_map.contains_key(&row.pulsar_a) && frust_map.contains_key(&row.pulsar_b) {
                let diff_raw = row.avg_residual_pearson - row.hellings_downs;
                let weight = row.overlap_bins as f64;
                cc_var_raw += diff_raw * diff_raw * weight;
            }
        }

        for &lambda in lambda_grid {
            let mut current_cc_var = 0.0;
            for row in pairwise_rows {
                if let (Some(&f_a), Some(&f_b)) =
                    (frust_map.get(&row.pulsar_a), frust_map.get(&row.pulsar_b))
                {
                    let diff_whitened =
                        (row.avg_residual_pearson - lambda * f_a * f_b) - row.hellings_downs;
                    let weight = row.overlap_bins as f64;
                    current_cc_var += diff_whitened * diff_whitened * weight;
                }
            }
            if current_cc_var < best_cc_var_whitened {
                best_cc_var_whitened = current_cc_var;
            }
        }
    }
    let cross_corr_drop_pct = if cc_var_raw > 0.0 && best_cc_var_whitened < f64::MAX {
        percent_drop(cc_var_raw, best_cc_var_whitened)
    } else {
        0.0
    };

    // --- 3. Cross-Validated Cross-Correlation ---
    let mut cv_cc_drop_sum = 0.0;
    let mut cv_folds_counted = 0;

    if !pairwise_rows.is_empty() {
        let assignments = fold_assignments(pulsars, args.folds);
        let mut frust_map = std::collections::HashMap::new();
        for p in pulsars {
            frust_map.insert(p.pulsar.clone(), p.frustration_z);
        }

        for fold in 0..args.folds {
            // Train pairs: both pulsars in training set (not in current fold)
            let mut train_pairs = Vec::new();
            // Test pairs: both pulsars in test set (in current fold)
            let mut test_pairs = Vec::new();

            for row in pairwise_rows {
                let Some(&f_a) = frust_map.get(&row.pulsar_a) else {
                    continue;
                };
                let Some(&f_b) = frust_map.get(&row.pulsar_b) else {
                    continue;
                };

                let a_fold = assignments[pulsars
                    .iter()
                    .position(|p| p.pulsar == row.pulsar_a)
                    .unwrap()];
                let b_fold = assignments[pulsars
                    .iter()
                    .position(|p| p.pulsar == row.pulsar_b)
                    .unwrap()];

                if a_fold != fold && b_fold != fold {
                    train_pairs.push((row, f_a, f_b));
                } else if a_fold == fold && b_fold == fold {
                    test_pairs.push((row, f_a, f_b));
                }
            }

            if train_pairs.is_empty() || test_pairs.is_empty() {
                continue;
            }

            // Optimize lambda on train pairs
            let mut best_lambda = 0.0;
            let mut min_train_var = f64::MAX;
            for &lambda in lambda_grid {
                let mut current_var = 0.0;
                for (row, f_a, f_b) in &train_pairs {
                    let diff = (row.avg_residual_pearson - lambda * f_a * f_b) - row.hellings_downs;
                    current_var += diff * diff * row.overlap_bins as f64;
                }
                if current_var < min_train_var {
                    min_train_var = current_var;
                    best_lambda = lambda;
                }
            }

            // Evaluate on test pairs
            let mut test_var_raw = 0.0;
            let mut test_var_whitened = 0.0;
            for (row, f_a, f_b) in &test_pairs {
                let diff_raw = row.avg_residual_pearson - row.hellings_downs;
                let diff_whitened =
                    (row.avg_residual_pearson - best_lambda * f_a * f_b) - row.hellings_downs;
                let weight = row.overlap_bins as f64;
                test_var_raw += diff_raw * diff_raw * weight;
                test_var_whitened += diff_whitened * diff_whitened * weight;
            }

            if test_var_raw > 0.0 {
                cv_cc_drop_sum += percent_drop(test_var_raw, test_var_whitened);
                cv_folds_counted += 1;
            }
        }
    }
    let cv_cross_corr_drop_pct = if cv_folds_counted > 0 {
        cv_cc_drop_sum / cv_folds_counted as f64
    } else {
        0.0
    };

    let mut frustration_stats = RunningStats::default();
    let mut nonzero_frustration_count = 0;
    let mut mean_values = Vec::new();
    let mut frustration_values = Vec::new();
    for pulsar in pulsars {
        frustration_stats.push(pulsar.frustration_z);
        if pulsar.frustration_z.abs() > 1.0e-12 {
            nonzero_frustration_count += 1;
        }
        mean_values.push(pulsar.raw_mean_us);
        frustration_values.push(pulsar.frustration_z);
    }

    PerDimensionSummary {
        dimension,
        backend: backend_label(backend),
        pulsar_count: pulsars.len(),
        nonzero_frustration_count,
        frustration_min: frustration_stats.min.unwrap_or(0.0),
        frustration_max: frustration_stats.max.unwrap_or(0.0),
        frustration_std: frustration_stats.stddev(),
        mean_residual_corr: pearson(&mean_values, &frustration_values),
        lambda_mean,
        baseline_raw_rms_us,
        fitted_raw_rms_us,
        raw_drop_pct,
        baseline_centered_rms_us,
        fitted_centered_rms_us,
        centered_drop_pct,
        null_mean_drop_pct,
        null_max_drop_pct: if null_max_drop_pct.is_finite() {
            null_max_drop_pct
        } else {
            0.0
        },
        null_pvalue,
        bonferroni_pvalue: 0.0,
        bh_q_value: 0.0,
        static_field_centered_invariant: centered_drop_pct.abs() < 1.0e-12,
        per_pulsar_drop_pct,
        cross_corr_drop_pct,
        cv_cross_corr_drop_pct,
    }
}

fn cross_validated_fit(
    pulsars: &[PulsarProjection],
    folds: usize,
    lambda_grid: &[f64],
) -> (f64, f64, f64) {
    let assignments = fold_assignments(pulsars, folds);
    let mut lambda_sum = 0.0;
    let mut fold_count = 0usize;
    let mut raw_sumsq = 0.0;
    let mut raw_count = 0usize;
    let mut centered_sumsq = 0.0;
    let mut centered_count = 0usize;

    for fold in 0..folds {
        let train = assignments
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != fold)
            .map(|(idx, _)| &pulsars[idx])
            .collect::<Vec<_>>();
        let test = assignments
            .iter()
            .enumerate()
            .filter(|(_, value)| **value == fold)
            .map(|(idx, _)| &pulsars[idx])
            .collect::<Vec<_>>();
        if train.is_empty() || test.is_empty() {
            continue;
        }

        let lambda = optimize_lambda(&train, lambda_grid);
        lambda_sum += lambda;
        fold_count += 1;

        let raw_eval = raw_eval_refs(&test, lambda);
        raw_sumsq += raw_eval.0;
        raw_count += raw_eval.1;

        let centered_eval = centered_eval_refs(&test, lambda);
        centered_sumsq += centered_eval.0;
        centered_count += centered_eval.1;
    }

    let lambda_mean = if fold_count == 0 {
        0.0
    } else {
        lambda_sum / fold_count as f64
    };
    let fitted_raw_rms_us = if raw_count == 0 {
        0.0
    } else {
        (raw_sumsq / raw_count as f64).sqrt()
    };
    let fitted_centered_rms_us = if centered_count == 0 {
        0.0
    } else {
        (centered_sumsq / centered_count as f64).sqrt()
    };
    (lambda_mean, fitted_raw_rms_us, fitted_centered_rms_us)
}

fn shifted_null_drops(
    pulsars: &[PulsarProjection],
    folds: usize,
    lambda_grid: &[f64],
    null_shifts: usize,
) -> Vec<f64> {
    if pulsars.len() < 2 || null_shifts == 0 {
        return Vec::new();
    }
    let max_shifts = null_shifts.min(pulsars.len().saturating_sub(1));
    let baseline_raw_rms_us = global_raw_rms(pulsars, 0.0);
    let mut out = Vec::new();
    for shift in 1..=max_shifts {
        let shifted = rotate_frustrations(pulsars, shift);
        let (_, fitted_raw_rms_us, _) = cross_validated_fit(&shifted, folds, lambda_grid);
        out.push(percent_drop(baseline_raw_rms_us, fitted_raw_rms_us));
    }
    out
}

fn optimize_lambda(pulsars: &[&PulsarProjection], lambda_grid: &[f64]) -> f64 {
    let mut best_lambda = 0.0;
    let mut best_rms = f64::INFINITY;
    for &lambda in lambda_grid {
        let (sum_sq, count) = raw_eval_refs(pulsars, lambda);
        if count == 0 {
            continue;
        }
        let rms_value = (sum_sq / count as f64).sqrt();
        if rms_value < best_rms {
            best_rms = rms_value;
            best_lambda = lambda;
        }
    }
    best_lambda
}

fn raw_eval_refs(pulsars: &[&PulsarProjection], lambda: f64) -> (f64, usize) {
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for pulsar in pulsars {
        for &residual in &pulsar.residuals {
            let shifted = residual - lambda * pulsar.frustration_z;
            sum_sq += shifted * shifted;
            count += 1;
        }
    }
    (sum_sq, count)
}

fn centered_eval_refs(pulsars: &[&PulsarProjection], lambda: f64) -> (f64, usize) {
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for pulsar in pulsars {
        let shifted_mean = pulsar.raw_mean_us - lambda * pulsar.frustration_z;
        for &residual in &pulsar.residuals {
            let shifted = residual - lambda * pulsar.frustration_z;
            let centered = shifted - shifted_mean;
            sum_sq += centered * centered;
            count += 1;
        }
    }
    (sum_sq, count)
}

fn global_raw_rms(pulsars: &[PulsarProjection], lambda: f64) -> f64 {
    let refs = pulsars.iter().collect::<Vec<_>>();
    let (sum_sq, count) = raw_eval_refs(&refs, lambda);
    if count == 0 {
        0.0
    } else {
        (sum_sq / count as f64).sqrt()
    }
}

fn global_centered_rms(pulsars: &[PulsarProjection], lambda: f64) -> f64 {
    let refs = pulsars.iter().collect::<Vec<_>>();
    let (sum_sq, count) = centered_eval_refs(&refs, lambda);
    if count == 0 {
        0.0
    } else {
        (sum_sq / count as f64).sqrt()
    }
}

fn fold_assignments(pulsars: &[PulsarProjection], folds: usize) -> Vec<usize> {
    pulsars
        .iter()
        .map(|pulsar| stable_hash(&pulsar.pulsar) % folds)
        .collect()
}

fn rotate_frustrations(pulsars: &[PulsarProjection], shift: usize) -> Vec<PulsarProjection> {
    let frustrations = pulsars
        .iter()
        .map(|pulsar| pulsar.frustration_z)
        .collect::<Vec<_>>();
    let len = frustrations.len();
    let mut shifted = frustrations.clone();
    shifted.rotate_left(shift % len);

    pulsars
        .iter()
        .zip(shifted)
        .map(|(pulsar, frustration_z)| PulsarProjection {
            pulsar: pulsar.pulsar.clone(),
            residuals: pulsar.residuals.clone(),
            raw_mean_us: pulsar.raw_mean_us,
            raw_rms_us: pulsar.raw_rms_us,
            centered_rms_us: pulsar.centered_rms_us,
            frustration_z,
            avg_uncertainty_mean_us: pulsar.avg_uncertainty_mean_us,
        })
        .collect()
}

fn residual_series(data: &PulsarTimingData, args: &Args) -> Vec<f64> {
    let points = match args.surface {
        ResidualSurface::Avg => &data.avg_residuals,
        ResidualSurface::Full => {
            if data.full_residuals.is_empty() {
                &data.avg_residuals
            } else {
                &data.full_residuals
            }
        }
    };
    points
        .iter()
        .filter(|p| {
            if let Some(start) = args.mjd_start && p.mjd < start {
                return false;
            }
            if let Some(end) = args.mjd_end && p.mjd > end {
                return false;
            }
            true
        })
        .map(|p| p.residual_us)
        .collect()
}

fn zscore(values: &[f64]) -> Vec<f64> {
    let mean_value = stats_core::helpers::mean(values);
    let mut stats = RunningStats::default();
    for &value in values {
        stats.push(value);
    }
    let stddev = stats.stddev();
    if stddev <= 1.0e-12 {
        return vec![0.0; values.len()];
    }
    values
        .iter()
        .map(|value| (value - mean_value) / stddev)
        .collect()
}

fn stable_hash(text: &str) -> usize {
    let mut hash: u64 = 1_469_598_103_934_665_603;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash as usize
}

fn backend_label(backend: ComputeBackend) -> &'static str {
    match backend {
        ComputeBackend::CpuScalar => "cpu_scalar",
        ComputeBackend::CpuSimd => "cpu_simd",
        ComputeBackend::Vulkan => "vulkan",
        ComputeBackend::Cuda => "cuda",
    }
}

fn apply_multiple_testing_corrections(summaries: &mut [PerDimensionSummary]) {
    if summaries.is_empty() {
        return;
    }
    let family_size = summaries.len() as f64;
    let mut ranked = summaries
        .iter()
        .enumerate()
        .map(|(idx, row)| (idx, row.null_pvalue))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| left.1.total_cmp(&right.1));

    let mut bh_q_values = vec![1.0; summaries.len()];
    let mut running_min: f64 = 1.0;
    for (reverse_rank, (idx, pvalue)) in ranked.iter().enumerate().rev() {
        let rank = (reverse_rank + 1) as f64;
        let adjusted = (*pvalue * family_size / rank).min(1.0);
        running_min = running_min.min(adjusted);
        bh_q_values[*idx] = running_min;
    }

    for (idx, row) in summaries.iter_mut().enumerate() {
        row.bonferroni_pvalue = (row.null_pvalue * family_size).min(1.0);
        row.bh_q_value = bh_q_values[idx];
    }
}

fn write_csv(path: &PathBuf, summaries: &[PerDimensionSummary]) -> Result<()> {
    ensure_parent_dir(path)?;
    let mut writer = Writer::from_path(path)?;
    writer.write_record([
        "dimension",
        "backend",
        "pulsars",
        "nonzero_frustrations",
        "frustration_min",
        "frustration_max",
        "frustration_std",
        "mean_residual_frustration_pearson",
        "lambda_mean",
        "baseline_raw_rms_us",
        "fitted_raw_rms_us",
        "raw_drop_pct",
        "baseline_centered_rms_us",
        "fitted_centered_rms_us",
        "centered_drop_pct",
        "null_mean_drop_pct",
        "null_max_drop_pct",
        "null_pvalue",
        "bonferroni_pvalue",
        "bh_q_value",
        "static_field_centered_invariant",
        "per_pulsar_drop_pct",
        "cross_corr_drop_pct",
        "cv_cross_corr_drop_pct",
    ])?;
    for row in summaries {
        writer.write_record([
            row.dimension.to_string(),
            row.backend.to_string(),
            row.pulsar_count.to_string(),
            row.nonzero_frustration_count.to_string(),
            format!("{:.12}", row.frustration_min),
            format!("{:.12}", row.frustration_max),
            format!("{:.12}", row.frustration_std),
            format_opt(row.mean_residual_corr),
            format!("{:.12}", row.lambda_mean),
            format!("{:.12}", row.baseline_raw_rms_us),
            format!("{:.12}", row.fitted_raw_rms_us),
            format!("{:.12}", row.raw_drop_pct),
            format!("{:.12}", row.baseline_centered_rms_us),
            format!("{:.12}", row.fitted_centered_rms_us),
            format!("{:.12}", row.centered_drop_pct),
            format!("{:.12}", row.null_mean_drop_pct),
            format!("{:.12}", row.null_max_drop_pct),
            format!("{:.12}", row.null_pvalue),
            format!("{:.12}", row.bonferroni_pvalue),
            format!("{:.12}", row.bh_q_value),
            row.static_field_centered_invariant.to_string(),
            format!("{:.12}", row.per_pulsar_drop_pct),
            format!("{:.12}", row.cross_corr_drop_pct),
            format!("{:.12}", row.cv_cross_corr_drop_pct),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_report(path: &PathBuf, args: &Args, summaries: &[PerDimensionSummary]) -> Result<()> {
    ensure_parent_dir(path)?;
    let mut out = String::new();
    let _ = writeln!(out, "[metadata]");
    let _ = writeln!(out, "title = \"NANOGrav AVT mean-shift audit\"");
    let _ = writeln!(out, "root = \"{}\"", args.root.display());
    let _ = writeln!(out, "surface = \"{}\"", args.surface.as_str());
    if let Some(path) = &args.independent_residual_csv {
        let _ = writeln!(
            out,
            "independent_residual_csv = \"{}\"",
            path.display()
        );
        let _ = writeln!(
            out,
            "independent_surface = \"{:?}\"",
            args.independent_surface
        );
    }
    let _ = writeln!(out, "folds = {}", args.folds);
    let _ = writeln!(out, "null_shifts = {}", args.null_shifts);
    let _ = writeln!(out, "lambda_min = {:.12}", args.lambda_min);
    let _ = writeln!(out, "lambda_max = {:.12}", args.lambda_max);
    let _ = writeln!(out, "lambda_steps = {}", args.lambda_steps);
    let _ = writeln!(out, "null_pvalue_method = \"plus_one_smoothed_shift_null\"");
    let _ = writeln!(
        out,
        "scope_note = \"This lane applies a sightline-static scalar field per pulsar. It can adjust per-pulsar means, but it cannot whiten centered intra-pulsar scatter without introducing time- or pair-dependent structure.\""
    );
    let _ = writeln!(out, "multiple_testing_family_size = {}", summaries.len());
    let _ = writeln!(
        out,
        "multiple_testing_method = \"bonferroni + benjamini-hochberg\""
    );
    let _ = writeln!(out, "claims = [\"C-1113\", \"C-458\"]");
    for row in summaries {
        let _ = writeln!(out);
        let _ = writeln!(out, "[[dimension]]");
        let _ = writeln!(out, "dim = {}", row.dimension);
        let _ = writeln!(out, "backend = \"{}\"", row.backend);
        let _ = writeln!(out, "pulsar_count = {}", row.pulsar_count);
        let _ = writeln!(
            out,
            "nonzero_frustration_count = {}",
            row.nonzero_frustration_count
        );
        let _ = writeln!(out, "frustration_min = {:.12}", row.frustration_min);
        let _ = writeln!(out, "frustration_max = {:.12}", row.frustration_max);
        let _ = writeln!(out, "frustration_std = {:.12}", row.frustration_std);
        let _ = writeln!(
            out,
            "mean_residual_frustration_pearson = {}",
            format_opt(row.mean_residual_corr)
        );
        let _ = writeln!(out, "lambda_mean = {:.12}", row.lambda_mean);
        let _ = writeln!(out, "baseline_raw_rms_us = {:.12}", row.baseline_raw_rms_us);
        let _ = writeln!(out, "fitted_raw_rms_us = {:.12}", row.fitted_raw_rms_us);
        let _ = writeln!(out, "raw_drop_pct = {:.12}", row.raw_drop_pct);
        let _ = writeln!(
            out,
            "baseline_centered_rms_us = {:.12}",
            row.baseline_centered_rms_us
        );
        let _ = writeln!(
            out,
            "fitted_centered_rms_us = {:.12}",
            row.fitted_centered_rms_us
        );
        let _ = writeln!(out, "centered_drop_pct = {:.12}", row.centered_drop_pct);
        let _ = writeln!(out, "null_mean_drop_pct = {:.12}", row.null_mean_drop_pct);
        let _ = writeln!(out, "null_max_drop_pct = {:.12}", row.null_max_drop_pct);
        let _ = writeln!(out, "null_pvalue = {:.12}", row.null_pvalue);
        let _ = writeln!(out, "bonferroni_pvalue = {:.12}", row.bonferroni_pvalue);
        let _ = writeln!(out, "bh_q_value = {:.12}", row.bh_q_value);
        let _ = writeln!(
            out,
            "static_field_centered_invariant = {}",
            row.static_field_centered_invariant
        );
        let _ = writeln!(out, "per_pulsar_drop_pct = {:.12}", row.per_pulsar_drop_pct);
        let _ = writeln!(out, "cross_corr_drop_pct = {:.12}", row.cross_corr_drop_pct);
        let _ = writeln!(
            out,
            "cv_cross_corr_drop_pct = {:.12}",
            row.cv_cross_corr_drop_pct
        );
    }
    fs::write(path, out)?;
    Ok(())
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    Ok(())
}

fn format_opt(value: Option<f64>) -> String {
    value
        .map(|inner| format!("{:.12}", inner))
        .unwrap_or_else(|| "\"nan\"".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_projection() -> PulsarProjection {
        PulsarProjection {
            pulsar: "J0000+0000".to_string(),
            residuals: vec![1.0, 3.0, 5.0, 7.0],
            raw_mean_us: 4.0,
            raw_rms_us: rms(&[1.0, 3.0, 5.0, 7.0]),
            centered_rms_us: centered_rms(&[1.0, 3.0, 5.0, 7.0], 4.0),
            frustration_z: 2.0,
            avg_uncertainty_mean_us: 1.0,
        }
    }

    #[test]
    fn centered_rms_is_invariant_under_static_offset() {
        let pulsars = vec![
            sample_projection(),
            PulsarProjection {
                pulsar: "J1111+1111".to_string(),
                residuals: vec![-2.0, -1.0, 0.0, 1.0],
                raw_mean_us: -0.5,
                raw_rms_us: rms(&[-2.0, -1.0, 0.0, 1.0]),
                centered_rms_us: centered_rms(&[-2.0, -1.0, 0.0, 1.0], -0.5),
                frustration_z: -3.0,
                avg_uncertainty_mean_us: 1.0,
            },
        ];
        let base = global_centered_rms(&pulsars, 0.0);
        let shifted = global_centered_rms(&pulsars, 1.75);
        assert!(
            (base - shifted).abs() < 1.0e-12,
            "static field should not change centered RMS"
        );
    }

    #[test]
    fn projection_is_deterministic() {
        let lattice = enumerate_lambda_generic(16);
        let sky = [0.25, -0.5, 0.75];
        let left = project_sky_to_basis(&sky, &lattice, 16);
        let right = project_sky_to_basis(&sky, &lattice, 16);
        assert_eq!(left, right);
    }

    #[test]
    fn rotation_null_reassigns_frustrations() {
        let left = sample_projection();
        let right = PulsarProjection {
            pulsar: "J1111+1111".to_string(),
            residuals: vec![0.0, 1.0],
            raw_mean_us: 0.5,
            raw_rms_us: rms(&[0.0, 1.0]),
            centered_rms_us: centered_rms(&[0.0, 1.0], 0.5),
            frustration_z: -4.0,
            avg_uncertainty_mean_us: 1.0,
        };
        let rotated = rotate_frustrations(&[left.clone(), right.clone()], 1);
        assert_eq!(rotated[0].frustration_z, right.frustration_z);
        assert_eq!(rotated[1].frustration_z, left.frustration_z);
    }

    #[test]
    fn parse_hms_and_dms_examples() {
        let ra = data_core::spatial::parse_hms_radians("12:00:00").unwrap();
        let dec = data_core::spatial::parse_dms_radians("-30:00:00").unwrap();
        assert!((ra - std::f64::consts::PI).abs() < 1.0e-12);
        assert!((dec + std::f64::consts::PI / 6.0).abs() < 1.0e-12);
    }

    #[test]
    fn zscore_zeroes_constant_inputs() {
        let values = vec![2.0, 2.0, 2.0];
        assert_eq!(zscore(&values), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn multiple_testing_corrections_are_monotone() {
        let mut summaries = vec![
            PerDimensionSummary {
                dimension: 16,
                backend: "cpu_scalar",
                pulsar_count: 1,
                nonzero_frustration_count: 1,
                frustration_min: 0.0,
                frustration_max: 0.0,
                frustration_std: 0.0,
                mean_residual_corr: None,
                lambda_mean: 0.0,
                baseline_raw_rms_us: 0.0,
                fitted_raw_rms_us: 0.0,
                raw_drop_pct: 0.0,
                baseline_centered_rms_us: 0.0,
                fitted_centered_rms_us: 0.0,
                centered_drop_pct: 0.0,
                null_mean_drop_pct: 0.0,
                null_max_drop_pct: 0.0,
                null_pvalue: 0.03,
                bonferroni_pvalue: 0.0,
                bh_q_value: 0.0,
                static_field_centered_invariant: true,
                per_pulsar_drop_pct: 0.0,
                cross_corr_drop_pct: 0.0,
                cv_cross_corr_drop_pct: 0.0,
            },
            PerDimensionSummary {
                dimension: 32,
                backend: "cpu_scalar",
                pulsar_count: 1,
                nonzero_frustration_count: 1,
                frustration_min: 0.0,
                frustration_max: 0.0,
                frustration_std: 0.0,
                mean_residual_corr: None,
                lambda_mean: 0.0,
                baseline_raw_rms_us: 0.0,
                fitted_raw_rms_us: 0.0,
                raw_drop_pct: 0.0,
                baseline_centered_rms_us: 0.0,
                fitted_centered_rms_us: 0.0,
                centered_drop_pct: 0.0,
                null_mean_drop_pct: 0.0,
                null_max_drop_pct: 0.0,
                null_pvalue: 0.01,
                bonferroni_pvalue: 0.0,
                bh_q_value: 0.0,
                static_field_centered_invariant: true,
                per_pulsar_drop_pct: 0.0,
                cross_corr_drop_pct: 0.0,
                cv_cross_corr_drop_pct: 0.0,
            },
            PerDimensionSummary {
                dimension: 64,
                backend: "cpu_scalar",
                pulsar_count: 1,
                nonzero_frustration_count: 1,
                frustration_min: 0.0,
                frustration_max: 0.0,
                frustration_std: 0.0,
                mean_residual_corr: None,
                lambda_mean: 0.0,
                baseline_raw_rms_us: 0.0,
                fitted_raw_rms_us: 0.0,
                raw_drop_pct: 0.0,
                baseline_centered_rms_us: 0.0,
                fitted_centered_rms_us: 0.0,
                centered_drop_pct: 0.0,
                null_mean_drop_pct: 0.0,
                null_max_drop_pct: 0.0,
                null_pvalue: 0.02,
                bonferroni_pvalue: 0.0,
                bh_q_value: 0.0,
                static_field_centered_invariant: true,
                per_pulsar_drop_pct: 0.0,
                cross_corr_drop_pct: 0.0,
                cv_cross_corr_drop_pct: 0.0,
            },
        ];
        apply_multiple_testing_corrections(&mut summaries);
        assert!(summaries[0].bonferroni_pvalue > summaries[1].bonferroni_pvalue);
        assert!(summaries[1].bh_q_value <= summaries[2].bh_q_value);
        assert!(
            summaries
                .iter()
                .all(|row| row.bh_q_value >= row.null_pvalue)
        );
    }
}
