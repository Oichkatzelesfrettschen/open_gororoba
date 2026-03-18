use anyhow::{Context, Result, bail};
use chrono::{DateTime, Datelike, Utc};
use clap::Parser;
use data_core::{
    CatalogFeatureCube, CatalogNuisanceModel, NuisanceEffectReport, ResidualizedCatalogFeatureCube,
    parse_catalog_feature_cube_json,
};
use nalgebra::{DMatrix, DVector};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

#[derive(Parser, Debug)]
#[command(
    name = "catalog-feature-deconfound",
    about = "Residualize generic catalog feature cubes against dataset-specific nuisance metadata"
)]
struct Cli {
    #[arg(long)]
    cube_json: PathBuf,

    #[arg(long, default_value_t = 1.0)]
    ridge_lambda: f64,

    #[arg(long)]
    out_json: Option<PathBuf>,

    #[arg(long)]
    out_report: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct Report {
    generated_at_utc: String,
    cube_json: String,
    output_json: String,
    dataset_count: usize,
    nuisance_models: Vec<CatalogNuisanceModel>,
    nuisance_reports: Vec<NuisanceEffectReport>,
}

#[derive(Debug, Clone)]
struct Encoders {
    program_codes: BTreeMap<String, usize>,
    instrument_codes: BTreeMap<String, usize>,
    modality_codes: BTreeMap<String, usize>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let out_json = cli.out_json.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!(
            "catalog_feature_cube_deconfounded_{}.json",
            Utc::now().date_naive()
        ))
    });
    let out_report = cli.out_report.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!(
            "catalog_feature_deconfound_{}.toml",
            Utc::now().date_naive()
        ))
    });

    let mut cube: CatalogFeatureCube = parse_catalog_feature_cube_json(
        &fs::read(&cli.cube_json).with_context(|| format!("read {}", cli.cube_json.display()))?,
    )
    .with_context(|| format!("parse {}", cli.cube_json.display()))?;
    let (models, reports) = residualize_cube(&mut cube, cli.ridge_lambda)?;
    cube.manifest.notes.push(
        "Residualized features remove dataset-specific nuisance structure without dropping rows."
            .to_string(),
    );

    if let Some(parent) = out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = out_report.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&out_json, serde_json::to_vec_pretty(&cube)?)
        .with_context(|| format!("write {}", out_json.display()))?;
    let wrapped = ResidualizedCatalogFeatureCube {
        cube,
        nuisance_models: models.clone(),
        nuisance_reports: reports.clone(),
    };
    let report = Report {
        generated_at_utc: Utc::now().to_rfc3339(),
        cube_json: cli.cube_json.display().to_string(),
        output_json: out_json.display().to_string(),
        dataset_count: wrapped.cube.manifest.dataset_names.len(),
        nuisance_models: models,
        nuisance_reports: reports,
    };
    fs::write(&out_report, toml::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", out_report.display()))?;

    println!("datasets = {}", report.dataset_count);
    println!("json = {}", out_json.display());
    println!("report = {}", out_report.display());
    Ok(())
}

fn residualize_cube(
    cube: &mut CatalogFeatureCube,
    ridge_lambda: f64,
) -> Result<(Vec<CatalogNuisanceModel>, Vec<NuisanceEffectReport>)> {
    let mut grouped: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, row) in cube.rows.iter().enumerate() {
        grouped.entry(row.dataset.clone()).or_default().push(idx);
    }
    let mut models = Vec::new();
    let mut reports = Vec::new();
    for (dataset, indices) in grouped {
        if indices.is_empty() {
            continue;
        }
        let encoders = build_encoders(cube, &indices);
        let nuisance_terms = nuisance_terms(&dataset);
        let design_rows = indices
            .iter()
            .map(|idx| nuisance_vector(&cube.rows[*idx], &dataset, &encoders))
            .collect::<Vec<_>>();
        let col_count = cube.rows[indices[0]].features.len();
        let k = design_rows.first().map(Vec::len).unwrap_or(0);
        if k == 0 {
            bail!("nuisance design matrix for {dataset} is empty");
        }
        let mut feature_r2 = vec![0.0; col_count];
        let mut removed_energy = 0.0;
        let mut retained_energy = 0.0;
        let mut valid_columns = 0usize;
        for feature_idx in 0..col_count {
            let valid_positions = indices
                .iter()
                .enumerate()
                .filter_map(|(local_idx, global_idx)| {
                    let value = cube.rows[*global_idx].features[feature_idx];
                    value.is_finite().then_some((local_idx, value))
                })
                .collect::<Vec<_>>();
            if valid_positions.len() < 4 {
                for global_idx in &indices {
                    let row = &mut cube.rows[*global_idx];
                    let slot = row
                        .residualized_features
                        .get_or_insert_with(|| vec![f64::NAN; col_count]);
                    slot[feature_idx] = row.features[feature_idx];
                }
                continue;
            }

            let x_valid = DMatrix::from_row_slice(
                valid_positions.len(),
                k,
                &valid_positions
                    .iter()
                    .flat_map(|(local_idx, _)| design_rows[*local_idx].iter().copied())
                    .collect::<Vec<_>>(),
            );
            let y_valid = DVector::from_iterator(
                valid_positions.len(),
                valid_positions.iter().map(|(_, value)| *value),
            );
            let xt_valid = x_valid.transpose();
            let solve_matrix = &xt_valid * &x_valid + DMatrix::<f64>::identity(k, k) * ridge_lambda;
            let Some(beta) = solve_matrix.lu().solve(&(xt_valid * &y_valid)) else {
                bail!("ridge solve failed for {dataset} feature {feature_idx}");
            };

            let predictions = &x_valid * &beta;
            let mean_y = y_valid.iter().sum::<f64>() / y_valid.len() as f64;
            let mut sse = 0.0;
            let mut sst = 0.0;
            for ((local_idx, observed), predicted) in valid_positions.iter().zip(predictions.iter())
            {
                let residual = *observed - *predicted;
                sse += residual * residual;
                let centered = *observed - mean_y;
                sst += centered * centered;
                let global_idx = indices[*local_idx];
                let row = &mut cube.rows[global_idx];
                let slot = row
                    .residualized_features
                    .get_or_insert_with(|| vec![f64::NAN; col_count]);
                slot[feature_idx] = residual;
                removed_energy += *predicted * *predicted;
                retained_energy += residual * residual;
            }
            feature_r2[feature_idx] = if sst > 0.0 {
                (1.0 - sse / sst).clamp(0.0, 1.0)
            } else {
                0.0
            };
            valid_columns += valid_positions.len();
        }

        models.push(CatalogNuisanceModel {
            dataset: dataset.clone(),
            nuisance_terms,
            ridge_lambda,
            notes: dataset_notes(&dataset),
        });
        reports.push(NuisanceEffectReport {
            dataset,
            row_count: indices.len(),
            nuisance_term_count: k,
            feature_r2,
            mean_removed_energy: if valid_columns == 0 {
                0.0
            } else {
                removed_energy / valid_columns as f64
            },
            mean_retained_energy: if valid_columns == 0 {
                0.0
            } else {
                retained_energy / valid_columns as f64
            },
        });
    }
    Ok((models, reports))
}

fn build_encoders(cube: &CatalogFeatureCube, indices: &[usize]) -> Encoders {
    Encoders {
        program_codes: stable_codes(indices.iter().filter_map(|idx| {
            cube.rows[*idx]
                .program_id
                .as_ref()
                .filter(|value| !value.trim().is_empty())
                .cloned()
        })),
        instrument_codes: stable_codes(indices.iter().filter_map(|idx| {
            cube.rows[*idx]
                .instrument
                .as_ref()
                .filter(|value| !value.trim().is_empty())
                .cloned()
        })),
        modality_codes: stable_codes(indices.iter().map(|idx| cube.rows[*idx].modality.clone())),
    }
}

fn stable_codes<I>(values: I) -> BTreeMap<String, usize>
where
    I: IntoIterator<Item = String>,
{
    values
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .enumerate()
        .map(|(idx, value)| (value, idx))
        .collect()
}

fn nuisance_terms(dataset: &str) -> Vec<String> {
    match dataset {
        "JWST Public Metadata" | "HST Public Metadata" => vec![
            "intercept".to_string(),
            "release_year".to_string(),
            "program_numeric".to_string(),
            "program_code".to_string(),
            "instrument_code".to_string(),
            "calib_level_code".to_string(),
            "sky_ra_cell".to_string(),
            "sky_dec_cell".to_string(),
            "modality_code".to_string(),
            "feature_completeness".to_string(),
        ],
        "Gaia DR3 Nearby" => vec![
            "intercept".to_string(),
            "g_mag".to_string(),
            "distance_proxy_log1p".to_string(),
            "sky_ra_cell".to_string(),
            "sky_dec_cell".to_string(),
            "feature_completeness".to_string(),
        ],
        _ => vec![
            "intercept".to_string(),
            "redshift_or_distance".to_string(),
            "sky_ra_cell".to_string(),
            "sky_dec_cell".to_string(),
            "modality_code".to_string(),
            "feature_completeness".to_string(),
        ],
    }
}

fn dataset_notes(dataset: &str) -> Vec<String> {
    match dataset {
        "JWST Public Metadata" | "HST Public Metadata" => vec![
            "MAST program/instrument/release structure is treated as nuisance archive organization."
                .to_string(),
        ],
        "Gaia DR3 Nearby" => vec![
            "Magnitude and sky-cell structure are treated as nuisance completeness proxies."
                .to_string(),
        ],
        _ => vec![
            "Residualization uses lightweight nuisance proxies because the bounded lane does not yet expose a full selection-function model."
                .to_string(),
        ],
    }
}

fn nuisance_vector(
    row: &data_core::CatalogFeatureRow,
    dataset: &str,
    encoders: &Encoders,
) -> Vec<f64> {
    let release_year = release_year(row.time_utc.as_deref());
    let sky_ra_cell = row
        .ra_deg
        .map(|value| (value / 15.0).floor())
        .unwrap_or(0.0);
    let sky_dec_cell = row
        .dec_deg
        .map(|value| ((value + 90.0) / 15.0).floor())
        .unwrap_or(0.0);
    let feature_completeness = row
        .features
        .iter()
        .filter(|value| value.is_finite())
        .count() as f64
        / row.features.len().max(1) as f64;
    match dataset {
        "JWST Public Metadata" | "HST Public Metadata" => vec![
            1.0,
            release_year,
            row.program_id
                .as_deref()
                .and_then(|value| value.parse::<f64>().ok())
                .unwrap_or(0.0),
            code(
                encoders
                    .program_codes
                    .get(row.program_id.as_deref().unwrap_or_default()),
            ),
            code(
                encoders
                    .instrument_codes
                    .get(row.instrument.as_deref().unwrap_or_default()),
            ),
            finite_feature(row, 7),
            sky_ra_cell,
            sky_dec_cell,
            code(encoders.modality_codes.get(row.modality.as_str())),
            feature_completeness,
        ],
        "Gaia DR3 Nearby" => vec![
            1.0,
            finite_feature(row, 4),
            signed_log1p(row.distance_proxy.unwrap_or(0.0)),
            sky_ra_cell,
            sky_dec_cell,
            feature_completeness,
        ],
        _ => vec![
            1.0,
            row.redshift
                .or(row.distance_proxy)
                .map(signed_log1p)
                .unwrap_or(0.0),
            sky_ra_cell,
            sky_dec_cell,
            code(encoders.modality_codes.get(row.modality.as_str())),
            feature_completeness,
        ],
    }
}

fn finite_feature(row: &data_core::CatalogFeatureRow, idx: usize) -> f64 {
    row.features
        .get(idx)
        .copied()
        .filter(|value| value.is_finite())
        .unwrap_or(0.0)
}

fn code(value: Option<&usize>) -> f64 {
    value.map(|entry| *entry as f64).unwrap_or(0.0)
}

fn release_year(value: Option<&str>) -> f64 {
    value
        .and_then(|text| DateTime::parse_from_rfc3339(text).ok())
        .map(|time| time.year() as f64)
        .unwrap_or(0.0)
}

fn signed_log1p(value: f64) -> f64 {
    if value.is_sign_negative() {
        -(-value).ln_1p()
    } else {
        value.ln_1p()
    }
}
