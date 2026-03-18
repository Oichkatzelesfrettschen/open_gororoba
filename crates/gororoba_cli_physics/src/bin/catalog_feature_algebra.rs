use anyhow::{Context, Result};
use cd_kernel::{cd_associator_norm, cd_norm_sq};
use chrono::{DateTime, Datelike, Utc};
use clap::Parser;
use data_core::{CatalogFeatureChannel, CatalogFeatureCube, CatalogFeatureCubeManifest};
use nalgebra::{SMatrix, SymmetricEigen};
use serde::Deserialize;
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
    dataset_count: usize,
    channel_names: Vec<String>,
    datasets: Vec<DatasetSummary>,
}

#[derive(Debug, Deserialize)]
struct JsonCatalogFeatureCube {
    manifest: CatalogFeatureCubeManifest,
    rows: Vec<JsonCatalogFeatureRow>,
}

#[derive(Debug, Deserialize)]
struct JsonCatalogFeatureRow {
    cube_name: String,
    dataset: String,
    record_id: String,
    modality: String,
    ra_deg: Option<f64>,
    dec_deg: Option<f64>,
    time_utc: Option<String>,
    redshift: Option<f64>,
    distance_proxy: Option<f64>,
    program_id: Option<String>,
    instrument: Option<String>,
    features: Vec<Option<f64>>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let out = cli.out.unwrap_or_else(|| {
        let date = Utc::now().date_naive();
        PathBuf::from("reports").join(format!("catalog_feature_algebra_survey-core_{}.toml", date))
    });
    let json_cube: JsonCatalogFeatureCube = serde_json::from_slice(
        &fs::read(&cli.cube_json).with_context(|| format!("read {}", cli.cube_json.display()))?,
    )
    .with_context(|| format!("parse {}", cli.cube_json.display()))?;
    let cube = sanitize_cube(json_cube);
    let report = summarize_cube(&cube, &cli.cube_json);
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, toml::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", out.display()))?;
    println!("datasets = {}", report.dataset_count);
    println!("out = {}", out.display());
    Ok(())
}

fn sanitize_cube(json_cube: JsonCatalogFeatureCube) -> CatalogFeatureCube {
    CatalogFeatureCube {
        manifest: json_cube.manifest,
        rows: json_cube
            .rows
            .into_iter()
            .map(|row| data_core::CatalogFeatureRow {
                cube_name: row.cube_name,
                dataset: row.dataset,
                record_id: row.record_id,
                modality: row.modality,
                ra_deg: row.ra_deg,
                dec_deg: row.dec_deg,
                time_utc: row.time_utc,
                redshift: row.redshift,
                distance_proxy: row.distance_proxy,
                program_id: row.program_id,
                instrument: row.instrument,
                features: row
                    .features
                    .into_iter()
                    .map(|value| value.unwrap_or(f64::NAN))
                    .collect(),
            })
            .collect(),
    }
}

fn summarize_cube(cube: &CatalogFeatureCube, cube_json: &Path) -> Report {
    let channel_names = algebra_channel_names(&cube.manifest.channels);
    let mut grouped: BTreeMap<String, Vec<&data_core::CatalogFeatureRow>> = BTreeMap::new();
    for row in &cube.rows {
        grouped.entry(row.dataset.clone()).or_default().push(row);
    }
    let mut datasets = grouped
        .into_iter()
        .map(|(dataset, rows)| summarize_dataset(dataset, rows, &channel_names))
        .collect::<Vec<_>>();
    datasets.sort_by(|a, b| a.dataset.cmp(&b.dataset));
    Report {
        generated_at_utc: Utc::now().to_rfc3339(),
        cube_json: cube_json.display().to_string(),
        cube_name: cube.manifest.cube_name.clone(),
        dataset_count: datasets.len(),
        channel_names,
        datasets,
    }
}

fn summarize_dataset(
    dataset: String,
    rows: Vec<&data_core::CatalogFeatureRow>,
    channel_names: &[String],
) -> DatasetSummary {
    let raw_vectors = rows.iter().map(|row| algebra_vector(row)).collect::<Vec<_>>();
    let centered_vectors = center_vectors(&raw_vectors);
    let norms = centered_vectors
        .iter()
        .map(|vector| cd_norm_sq(vector))
        .collect::<Vec<_>>();
    let mut associators = Vec::new();
    for triple in centered_vectors.windows(3) {
        associators.push(cd_associator_norm(&triple[0], &triple[1], &triple[2]));
    }
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
        feature_channel_count: rows.first().map(|row| row.features.len()).unwrap_or(0),
        modality_count,
        mean_norm_sq: mean(&norms),
        mean_associator_norm: mean(&associators),
        max_associator_norm: associators.into_iter().fold(0.0, f64::max),
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

fn algebra_vector(row: &data_core::CatalogFeatureRow) -> [f64; ALGEBRA_DIM] {
    let mut out = [0.0_f64; ALGEBRA_DIM];
    for (idx, value) in row.features.iter().take(8).enumerate() {
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

fn algebra_channel_names(channels: &[CatalogFeatureChannel]) -> Vec<String> {
    let mut out = channels
        .iter()
        .take(8)
        .map(|channel| channel.name.clone())
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
