use anyhow::{Context, Result};
use cd_kernel::{cd_associator_norm, cd_norm_sq};
use chrono::{DateTime, Datelike, Utc};
use clap::Parser;
use data_core::{CatalogFeatureChannel, CatalogFeatureCube, parse_catalog_feature_cube_json};
use nalgebra::{SMatrix, SymmetricEigen};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

const ALGEBRA_DIM: usize = 16;

#[derive(Parser, Debug)]
#[command(
    name = "catalog-feature-algebra",
    about = "Compute generic algebraic and manifold-shape summaries from a catalog feature cube"
)]
struct Cli {
    #[arg(long)]
    cube_json: PathBuf,

    #[arg(long, default_value = "raw")]
    feature_mode: String,

    #[arg(long = "dataset-filter")]
    dataset_filters: Vec<String>,

    #[arg(long)]
    max_rows_per_dataset: Option<usize>,

    #[arg(long, default_value_t = 32)]
    null_permutations: usize,

    #[arg(long)]
    ultrametric_csv: Option<PathBuf>,

    #[arg(long)]
    null_classification_out: Option<PathBuf>,

    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct DatasetSummary {
    dataset: String,
    row_count: usize,
    feature_channel_count: usize,
    modality_count: usize,
    mean_norm_sq: f64,
    mean_associator_norm: f64,
    max_associator_norm: f64,
    null_mean_associator_norm: f64,
    null_std_associator_norm: f64,
    null_p_value: f64,
    survives_residualized_null: bool,
    covariance_trace: f64,
    effective_rank: f64,
    participation_ratio: f64,
    centered_feature_energy: f64,
    dominant_channel: String,
    dominant_channel_mean_abs: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    generated_at_utc: String,
    cube_json: String,
    cube_name: String,
    feature_mode: String,
    dataset_count: usize,
    channel_names: Vec<String>,
    datasets: Vec<DatasetSummary>,
}

#[derive(Debug, Serialize)]
struct NullClassificationRow {
    dataset: String,
    algebra_survives_residualized_null: bool,
    ultrametric_survives_residualized_null: bool,
    classification: String,
    notes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct NullClassificationReport {
    generated_at_utc: String,
    cube_json: String,
    feature_mode: String,
    rows: Vec<NullClassificationRow>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let feature_mode = FeatureMode::parse(&cli.feature_mode)?;
    let out = cli.out.unwrap_or_else(|| {
        let date = Utc::now().date_naive();
        PathBuf::from("reports").join(format!("catalog_feature_algebra_survey-core_{}.toml", date))
    });
    let null_classification_out = cli.null_classification_out.unwrap_or_else(|| {
        let date = Utc::now().date_naive();
        PathBuf::from("reports").join(format!(
            "catalog_feature_null_classification_survey-core_{}.toml",
            date
        ))
    });
    let cube: CatalogFeatureCube = parse_catalog_feature_cube_json(
        &fs::read(&cli.cube_json).with_context(|| format!("read {}", cli.cube_json.display()))?,
    )
    .with_context(|| format!("parse {}", cli.cube_json.display()))?;
    let report = summarize_cube(
        &cube,
        &cli.cube_json,
        feature_mode,
        cli.null_permutations,
        &cli.dataset_filters,
        cli.max_rows_per_dataset,
    );
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, toml::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", out.display()))?;
    if let Some(ultrametric_csv) = &cli.ultrametric_csv {
        if let Some(parent) = null_classification_out.parent() {
            fs::create_dir_all(parent)?;
        }
        let classification = classify_null_results(
            &report,
            ultrametric_csv,
            &cli.cube_json,
            feature_mode,
        )?;
        fs::write(
            &null_classification_out,
            toml::to_string_pretty(&classification)?,
        )
        .with_context(|| format!("write {}", null_classification_out.display()))?;
    }
    println!("datasets = {}", report.dataset_count);
    println!("out = {}", out.display());
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeatureMode {
    Raw,
    Residualized,
}

impl FeatureMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "raw" => Ok(Self::Raw),
            "residualized" => Ok(Self::Residualized),
            other => anyhow::bail!("unsupported feature mode '{other}'; expected raw or residualized"),
        }
    }
}

fn summarize_cube(
    cube: &CatalogFeatureCube,
    cube_json: &Path,
    feature_mode: FeatureMode,
    null_permutations: usize,
    dataset_filters: &[String],
    max_rows_per_dataset: Option<usize>,
) -> Report {
    let channel_names = algebra_channel_names(&cube.manifest.channels, feature_mode);
    let mut grouped: BTreeMap<String, Vec<&data_core::CatalogFeatureRow>> = BTreeMap::new();
    for row in &cube.rows {
        if !dataset_selected(&row.dataset, dataset_filters) {
            continue;
        }
        grouped.entry(row.dataset.clone()).or_default().push(row);
    }
    let mut datasets = grouped
        .into_iter()
        .map(|(dataset, mut rows)| {
            if let Some(limit) = max_rows_per_dataset {
                rows.truncate(limit);
            }
            summarize_dataset(dataset, rows, &channel_names, feature_mode, null_permutations)
        })
        .collect::<Vec<_>>();
    datasets.sort_by(|a, b| a.dataset.cmp(&b.dataset));
    Report {
        generated_at_utc: Utc::now().to_rfc3339(),
        cube_json: cube_json.display().to_string(),
        cube_name: cube.manifest.cube_name.clone(),
        feature_mode: match feature_mode {
            FeatureMode::Raw => "raw".to_string(),
            FeatureMode::Residualized => "residualized".to_string(),
        },
        dataset_count: datasets.len(),
        channel_names,
        datasets,
    }
}

fn dataset_selected(dataset: &str, filters: &[String]) -> bool {
    if filters.is_empty() {
        return true;
    }
    let normalized_dataset = normalize_dataset_key(dataset);
    filters
        .iter()
        .map(|value| normalize_dataset_key(value))
        .any(|value| {
            value == normalized_dataset
                || normalized_dataset.contains(&value)
                || value.contains(&normalized_dataset)
        })
}

fn summarize_dataset(
    dataset: String,
    rows: Vec<&data_core::CatalogFeatureRow>,
    channel_names: &[String],
    feature_mode: FeatureMode,
    null_permutations: usize,
) -> DatasetSummary {
    let raw_vectors = rows
        .iter()
        .map(|row| algebra_vector(row, feature_mode))
        .collect::<Vec<_>>();
    let centered_vectors = center_vectors(&raw_vectors);
    let norms = centered_vectors
        .iter()
        .map(|vector| cd_norm_sq(vector))
        .collect::<Vec<_>>();
    let mut associators = Vec::new();
    for triple in centered_vectors.windows(3) {
        associators.push(cd_associator_norm(&triple[0], &triple[1], &triple[2]));
    }
    let null_distribution = null_associator_distribution(&centered_vectors, null_permutations);
    let null_mean_associator_norm = mean(&null_distribution);
    let null_std_associator_norm = stddev(&null_distribution, null_mean_associator_norm);
    let observed_associator = mean(&associators);
    let null_p_value = empirical_upper_tail_p_value(observed_associator, &null_distribution);
    let covariance = covariance_matrix(&centered_vectors);
    let covariance_trace = covariance.trace();
    let eigen = SymmetricEigen::new(covariance);
    let effective_rank = effective_rank(eigen.eigenvalues.as_slice());
    let participation_ratio = participation_ratio(eigen.eigenvalues.as_slice());
    let modality_count = rows
        .iter()
        .map(|row| row.modality.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let centered_feature_energy = mean(
        &centered_vectors
            .iter()
            .map(|vector| vector.iter().take(8).map(|v| v.abs()).sum::<f64>())
            .collect::<Vec<_>>(),
    );
    let channel_means = mean_abs_channels(&centered_vectors);
    let (dominant_idx, dominant_value) = channel_means
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap_or((0, 0.0));

    DatasetSummary {
        dataset,
        row_count: rows.len(),
        feature_channel_count: rows
            .first()
            .map(|row| selected_features(row, feature_mode).len())
            .unwrap_or(0),
        modality_count,
        mean_norm_sq: mean(&norms),
        mean_associator_norm: observed_associator,
        max_associator_norm: associators.into_iter().fold(0.0, f64::max),
        null_mean_associator_norm,
        null_std_associator_norm,
        null_p_value,
        survives_residualized_null: null_p_value <= 0.05,
        covariance_trace,
        effective_rank,
        participation_ratio,
        centered_feature_energy,
        dominant_channel: channel_names
            .get(dominant_idx)
            .cloned()
            .unwrap_or_else(|| format!("v{dominant_idx}")),
        dominant_channel_mean_abs: dominant_value,
    }
}

fn classify_null_results(
    report: &Report,
    ultrametric_csv: &Path,
    cube_json: &Path,
    feature_mode: FeatureMode,
) -> Result<NullClassificationReport> {
    let ultrametric = load_ultrametric_significance(ultrametric_csv)?;
    let mut rows = report
        .datasets
        .iter()
        .map(|dataset| {
            let ultrametric_survives = ultrametric
                .get(&normalize_dataset_key(&dataset.dataset))
                .copied()
                .unwrap_or(false);
            let mut notes = Vec::new();
            if !ultrametric.contains_key(&normalize_dataset_key(&dataset.dataset)) {
                notes.push("dataset_missing_from_ultrametric_report".to_string());
            }
            let classification = if dataset.survives_residualized_null && ultrametric_survives {
                "residual_astrophysical_candidate"
            } else if ultrametric.contains_key(&normalize_dataset_key(&dataset.dataset)) {
                "archive_structure_null"
            } else {
                "inconclusive"
            }
            .to_string();
            NullClassificationRow {
                dataset: dataset.dataset.clone(),
                algebra_survives_residualized_null: dataset.survives_residualized_null,
                ultrametric_survives_residualized_null: ultrametric_survives,
                classification,
                notes,
            }
        })
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| a.dataset.cmp(&b.dataset));
    Ok(NullClassificationReport {
        generated_at_utc: Utc::now().to_rfc3339(),
        cube_json: cube_json.display().to_string(),
        feature_mode: match feature_mode {
            FeatureMode::Raw => "raw".to_string(),
            FeatureMode::Residualized => "residualized".to_string(),
        },
        rows,
    })
}

fn algebra_vector(row: &data_core::CatalogFeatureRow, feature_mode: FeatureMode) -> [f64; ALGEBRA_DIM] {
    let features = selected_features(row, feature_mode);
    let mut out = [0.0_f64; ALGEBRA_DIM];
    for (idx, value) in features.iter().take(8).enumerate() {
        out[idx] = finite_or_zero(*value);
    }
    out[8] = row.ra_deg.map(|value| value / 180.0).unwrap_or(0.0);
    out[9] = row.dec_deg.map(|value| value / 90.0).unwrap_or(0.0);
    out[10] = row.time_utc.as_deref().map(normalized_time_year).unwrap_or(0.0);
    out[11] = finite_or_zero(row.redshift.unwrap_or(0.0));
    out[12] = finite_or_zero(signed_log1p(row.distance_proxy.unwrap_or(0.0)));
    out[13] = if row.program_id.as_deref().unwrap_or("").is_empty() {
        0.0
    } else {
        1.0
    };
    out[14] = if row.instrument.as_deref().unwrap_or("").is_empty() {
        0.0
    } else {
        1.0
    };
    out[15] = row
        .features
        .iter()
        .filter(|value| value.is_finite())
        .count() as f64
        / row.features.len().max(1) as f64;
    out
}

fn selected_features(row: &data_core::CatalogFeatureRow, feature_mode: FeatureMode) -> &[f64] {
    match feature_mode {
        FeatureMode::Raw => &row.features,
        FeatureMode::Residualized => row
            .residualized_features
            .as_deref()
            .unwrap_or(&row.features),
    }
}

fn algebra_channel_names(channels: &[CatalogFeatureChannel], feature_mode: FeatureMode) -> Vec<String> {
    let mut out = channels
        .iter()
        .take(8)
        .map(|channel| match feature_mode {
            FeatureMode::Raw => channel.name.clone(),
            FeatureMode::Residualized => format!("residualized:{}", channel.name),
        })
        .collect::<Vec<_>>();
    out.extend([
        "ra_deg".to_string(),
        "dec_deg".to_string(),
        "time_year".to_string(),
        "redshift".to_string(),
        "distance_proxy_log1p".to_string(),
        "program_present".to_string(),
        "instrument_present".to_string(),
        "feature_completeness".to_string(),
    ]);
    out
}

fn center_vectors(vectors: &[[f64; ALGEBRA_DIM]]) -> Vec<[f64; ALGEBRA_DIM]> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let mut means = [0.0_f64; ALGEBRA_DIM];
    let mut counts = [0usize; ALGEBRA_DIM];
    for vector in vectors {
        for (idx, value) in vector.iter().enumerate() {
            if value.is_finite() {
                means[idx] += value;
                counts[idx] += 1;
            }
        }
    }
    for idx in 0..ALGEBRA_DIM {
        if counts[idx] > 0 {
            means[idx] /= counts[idx] as f64;
        }
    }
    vectors
        .iter()
        .map(|vector| {
            let mut centered = [0.0_f64; ALGEBRA_DIM];
            for idx in 0..ALGEBRA_DIM {
                centered[idx] = if vector[idx].is_finite() {
                    vector[idx] - means[idx]
                } else {
                    0.0
                };
            }
            centered
        })
        .collect()
}

fn covariance_matrix(vectors: &[[f64; ALGEBRA_DIM]]) -> SMatrix<f64, ALGEBRA_DIM, ALGEBRA_DIM> {
    let mut covariance = SMatrix::<f64, ALGEBRA_DIM, ALGEBRA_DIM>::zeros();
    if vectors.is_empty() {
        return covariance;
    }
    for vector in vectors {
        for row in 0..ALGEBRA_DIM {
            for col in row..ALGEBRA_DIM {
                let product = vector[row] * vector[col];
                covariance[(row, col)] += product;
                if row != col {
                    covariance[(col, row)] += product;
                }
            }
        }
    }
    covariance / vectors.len() as f64
}

fn effective_rank(eigenvalues: &[f64]) -> f64 {
    let positive = eigenvalues
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if positive.is_empty() {
        return 0.0;
    }
    let total = positive.iter().sum::<f64>();
    let entropy = positive
        .iter()
        .map(|value| {
            let p = value / total;
            -p * p.ln()
        })
        .sum::<f64>();
    entropy.exp()
}

fn participation_ratio(eigenvalues: &[f64]) -> f64 {
    let positive = eigenvalues
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if positive.is_empty() {
        return 0.0;
    }
    let sum = positive.iter().sum::<f64>();
    let sum_sq = positive.iter().map(|value| value * value).sum::<f64>();
    if sum_sq == 0.0 {
        0.0
    } else {
        (sum * sum) / sum_sq
    }
}

fn mean(values: &[f64]) -> f64 {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.iter().sum::<f64>() / finite.len() as f64
}

fn stddev(values: &[f64], mean_value: f64) -> f64 {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return f64::NAN;
    }
    let variance = finite
        .iter()
        .map(|value| {
            let delta = *value - mean_value;
            delta * delta
        })
        .sum::<f64>()
        / finite.len() as f64;
    variance.sqrt()
}

fn null_associator_distribution(vectors: &[[f64; ALGEBRA_DIM]], n_perm: usize) -> Vec<f64> {
    if vectors.len() < 3 || n_perm == 0 {
        return Vec::new();
    }
    let mut distribution = Vec::with_capacity(n_perm);
    for perm_idx in 0..n_perm {
        let shuffled = deterministic_column_shuffle(vectors, perm_idx);
        let mut associators = Vec::new();
        for triple in shuffled.windows(3) {
            associators.push(cd_associator_norm(&triple[0], &triple[1], &triple[2]));
        }
        distribution.push(mean(&associators));
    }
    distribution
}

fn deterministic_column_shuffle(
    vectors: &[[f64; ALGEBRA_DIM]],
    perm_idx: usize,
) -> Vec<[f64; ALGEBRA_DIM]> {
    let n = vectors.len();
    let mut shuffled = vec![[0.0_f64; ALGEBRA_DIM]; n];
    for row_idx in 0..n {
        for col_idx in 0..ALGEBRA_DIM {
            let shift = ((perm_idx + 1) * (col_idx + 1)) % n;
            shuffled[row_idx][col_idx] = vectors[(row_idx + shift) % n][col_idx];
        }
    }
    shuffled
}

fn empirical_upper_tail_p_value(observed: f64, null_distribution: &[f64]) -> f64 {
    if null_distribution.is_empty() || !observed.is_finite() {
        return f64::NAN;
    }
    let exceed = null_distribution
        .iter()
        .filter(|value| value.is_finite() && **value >= observed)
        .count();
    (exceed as f64 + 1.0) / (null_distribution.len() as f64 + 1.0)
}

fn load_ultrametric_significance(path: &Path) -> Result<BTreeMap<String, bool>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("open {}", path.display()))?;
    let mut out = BTreeMap::new();
    for row in reader.records() {
        let row = row.with_context(|| format!("read {}", path.display()))?;
        let dataset = row.get(0).unwrap_or("").trim();
        let p_value = row.get(6).and_then(|value| value.parse::<f64>().ok());
        if dataset.is_empty() {
            continue;
        }
        out.insert(
            normalize_dataset_key(dataset),
            p_value.map(|value| value <= 0.05).unwrap_or(false),
        );
    }
    Ok(out)
}

fn normalize_dataset_key(value: &str) -> String {
    value
        .split('[')
        .next()
        .unwrap_or(value)
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '_', '/'], "-")
}

fn mean_abs_channels(vectors: &[[f64; ALGEBRA_DIM]]) -> [f64; ALGEBRA_DIM] {
    let mut sums = [0.0_f64; ALGEBRA_DIM];
    let mut counts = [0usize; ALGEBRA_DIM];
    for vector in vectors {
        for (idx, value) in vector.iter().enumerate() {
            if value.is_finite() {
                sums[idx] += value.abs();
                counts[idx] += 1;
            }
        }
    }
    let mut out = [0.0_f64; ALGEBRA_DIM];
    for idx in 0..ALGEBRA_DIM {
        if counts[idx] > 0 {
            out[idx] = sums[idx] / counts[idx] as f64;
        }
    }
    out
}

fn signed_log1p(value: f64) -> f64 {
    value.signum() * value.abs().ln_1p()
}

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() { value } else { 0.0 }
}

fn normalized_time_year(value: &str) -> f64 {
    DateTime::parse_from_rfc3339(value)
        .map(|dt| {
            let year = dt.year() as f64;
            (year - 2000.0) / 50.0
        })
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::{ALGEBRA_DIM, center_vectors, effective_rank};

    #[test]
    fn centering_removes_constant_bias() {
        let vectors = vec![[2.0_f64; ALGEBRA_DIM], [4.0_f64; ALGEBRA_DIM]];
        let centered = center_vectors(&vectors);
        assert_eq!(centered[0][0], -1.0);
        assert_eq!(centered[1][0], 1.0);
    }

    #[test]
    fn effective_rank_zero_for_empty_spectrum() {
        assert_eq!(effective_rank(&[]), 0.0);
        assert_eq!(effective_rank(&[0.0, -1.0]), 0.0);
    }
}
