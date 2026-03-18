use anyhow::{Context, Result, bail};
use chrono::{DateTime, Datelike, Duration, NaiveDateTime, TimeZone, Utc};
use clap::Parser;
use data_core::{
    CatalogFeatureChannel, CatalogFeatureCube, CatalogFeatureCubeManifest, CatalogFeatureRow,
    catalogs::{
        chime::parse_chime_csv, desi_bao::desi_dr2_bao, gaia::parse_gaia_csv,
        hst::parse_hst_public_metadata_csv, jwst::parse_jwst_public_metadata_csv,
        sdss::parse_sdss_quasar_csv,
    },
    encode_dictionary_value, pipe_count, stable_dictionary,
};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(
    name = "catalog-feature-cube",
    about = "Build a generic metadata-first catalog feature cube for bounded astronomy datasets"
)]
struct Cli {
    #[arg(long, default_value = "survey-core")]
    cube: String,

    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[arg(long)]
    out_json: Option<PathBuf>,

    #[arg(long)]
    out_manifest: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let date = Utc::now().date_naive();
    let out_json = cli.out_json.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!("catalog_feature_cube_{}_{}.json", cli.cube, date))
    });
    let out_manifest = cli.out_manifest.unwrap_or_else(|| {
        PathBuf::from("reports")
            .join(format!("catalog_feature_cube_{}_{}.toml", cli.cube, date))
    });

    let cube = build_cube(&cli.repo_root, &cli.cube)?;
    if let Some(parent) = out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = out_manifest.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_json, serde_json::to_vec_pretty(&cube)?)
        .with_context(|| format!("write {}", out_json.display()))?;
    fs::write(&out_manifest, toml::to_string_pretty(&cube.manifest)?)
        .with_context(|| format!("write {}", out_manifest.display()))?;

    println!("cube = {}", cube.manifest.cube_name);
    println!("rows = {}", cube.manifest.row_count);
    println!("datasets = {}", cube.manifest.dataset_names.join(", "));
    println!("json = {}", out_json.display());
    println!("manifest = {}", out_manifest.display());
    Ok(())
}

fn build_cube(repo_root: &Path, cube_name: &str) -> Result<CatalogFeatureCube> {
    match cube_name {
        "survey-core" => build_survey_core_cube(repo_root, cube_name),
        other => bail!("unsupported cube '{other}'; expected survey-core"),
    }
}

fn build_survey_core_cube(repo_root: &Path, cube_name: &str) -> Result<CatalogFeatureCube> {
    let jwst_path = repo_root.join("data/external/jwst_public_observations.csv");
    let hst_path = repo_root.join("data/external/hst_public_observations.csv");
    let gaia_path = repo_root.join("data/external/gaia_dr3_nearby.csv");
    let sdss_path = repo_root.join("data/external/sdss_dr18_quasars.csv");
    let chime_path = repo_root.join("data/external/chime_frb_cat2.csv");

    let jwst_rows = if jwst_path.exists() {
        parse_jwst_public_metadata_csv(&jwst_path)?
    } else {
        Vec::new()
    };
    let hst_rows = if hst_path.exists() {
        parse_hst_public_metadata_csv(&hst_path)?
    } else {
        Vec::new()
    };
    let gaia_rows = if gaia_path.exists() {
        parse_gaia_csv(&gaia_path)?
    } else {
        Vec::new()
    };
    let sdss_rows = if sdss_path.exists() {
        parse_sdss_quasar_csv(&sdss_path)?
    } else {
        Vec::new()
    };
    let chime_rows = if chime_path.exists() {
        parse_chime_csv(&chime_path)?
    } else {
        Vec::new()
    };
    let desi_rows = desi_dr2_bao();

    let instrument_dictionary = stable_dictionary(
        jwst_rows
            .iter()
            .map(|row| row.instrument_name.clone())
            .chain(hst_rows.iter().map(|row| row.instrument_name.clone())),
    );
    let calib_dictionary = stable_dictionary(
        jwst_rows
            .iter()
            .map(|row| row.calib_level.clone())
            .chain(hst_rows.iter().map(|row| row.calib_level.clone())),
    );

    let channels = vec![
        CatalogFeatureChannel {
            name: "f0".to_string(),
            description: "Primary numeric feature 0".to_string(),
            unit: None,
            role: "dataset_specific".to_string(),
            dictionary: Vec::new(),
        },
        CatalogFeatureChannel {
            name: "f1".to_string(),
            description: "Primary numeric feature 1".to_string(),
            unit: None,
            role: "dataset_specific".to_string(),
            dictionary: Vec::new(),
        },
        CatalogFeatureChannel {
            name: "f2".to_string(),
            description: "Primary numeric feature 2".to_string(),
            unit: None,
            role: "dataset_specific".to_string(),
            dictionary: Vec::new(),
        },
        CatalogFeatureChannel {
            name: "f3".to_string(),
            description: "Primary numeric feature 3".to_string(),
            unit: None,
            role: "dataset_specific".to_string(),
            dictionary: Vec::new(),
        },
        CatalogFeatureChannel {
            name: "f4".to_string(),
            description: "Primary numeric feature 4".to_string(),
            unit: None,
            role: "dataset_specific".to_string(),
            dictionary: Vec::new(),
        },
        CatalogFeatureChannel {
            name: "f5".to_string(),
            description: "Primary numeric feature 5".to_string(),
            unit: None,
            role: "dataset_specific".to_string(),
            dictionary: Vec::new(),
        },
        CatalogFeatureChannel {
            name: "instrument_code".to_string(),
            description: "Stable dictionary code for instrument metadata".to_string(),
            unit: None,
            role: "categorical_encoding".to_string(),
            dictionary: instrument_dictionary.clone(),
        },
        CatalogFeatureChannel {
            name: "calib_level_code".to_string(),
            description: "Stable dictionary code for calibration level metadata".to_string(),
            unit: None,
            role: "categorical_encoding".to_string(),
            dictionary: calib_dictionary.clone(),
        },
    ];

    let mut rows = Vec::new();
    let mut source_paths = Vec::new();
    let mut notes = Vec::new();

    if jwst_path.exists() {
        source_paths.push(jwst_path.display().to_string());
        for row in &jwst_rows {
            rows.push(CatalogFeatureRow {
                cube_name: cube_name.to_string(),
                dataset: "JWST Public Metadata".to_string(),
                record_id: row.obsid.clone(),
                modality: "sky_point_metadata".to_string(),
                ra_deg: Some(row.s_ra),
                dec_deg: Some(row.s_dec),
                time_utc: normalized_release_time(&row.t_obs_release),
                redshift: None,
                distance_proxy: None,
                program_id: Some(row.proposal_id.clone()),
                instrument: Some(row.instrument_name.clone()),
                features: vec![
                    release_year(&row.t_obs_release),
                    pipe_count(&row.filters),
                    parse_string_f64(&row.proposal_id),
                    parse_string_f64(&row.calib_level),
                    non_empty_flag(&row.target_name),
                    non_empty_flag(&row.dataproduct_type),
                    encode_dictionary_value(&instrument_dictionary, &row.instrument_name),
                    encode_dictionary_value(&calib_dictionary, &row.calib_level),
                ],
                residualized_features: None,
            });
        }
    }

    if hst_path.exists() {
        source_paths.push(hst_path.display().to_string());
        for row in &hst_rows {
            rows.push(CatalogFeatureRow {
                cube_name: cube_name.to_string(),
                dataset: "HST Public Metadata".to_string(),
                record_id: row.obsid.clone(),
                modality: "sky_point_metadata".to_string(),
                ra_deg: Some(row.s_ra),
                dec_deg: Some(row.s_dec),
                time_utc: normalized_release_time(&row.t_obs_release),
                redshift: None,
                distance_proxy: None,
                program_id: Some(row.proposal_id.clone()),
                instrument: Some(row.instrument_name.clone()),
                features: vec![
                    release_year(&row.t_obs_release),
                    pipe_count(&row.filters),
                    parse_string_f64(&row.proposal_id),
                    parse_string_f64(&row.calib_level),
                    non_empty_flag(&row.target_name),
                    non_empty_flag(&row.dataproduct_type),
                    encode_dictionary_value(&instrument_dictionary, &row.instrument_name),
                    encode_dictionary_value(&calib_dictionary, &row.calib_level),
                ],
                residualized_features: None,
            });
        }
    }

    if gaia_path.exists() {
        source_paths.push(gaia_path.display().to_string());
        for row in &gaia_rows {
            rows.push(CatalogFeatureRow {
                cube_name: cube_name.to_string(),
                dataset: "Gaia DR3 Nearby".to_string(),
                record_id: row.source_id.clone(),
                modality: "sky_point_metadata".to_string(),
                ra_deg: Some(row.ra),
                dec_deg: Some(row.dec),
                time_utc: Some("2016-01-01T00:00:00Z".to_string()),
                redshift: None,
                distance_proxy: finite_or_none(parallax_distance_pc(row.parallax)),
                program_id: None,
                instrument: None,
                features: vec![
                    row.parallax,
                    row.pmra,
                    row.pmdec,
                    row.radial_velocity,
                    row.phot_g_mean_mag,
                    row.bp_rp,
                    -1.0,
                    -1.0,
                ],
                residualized_features: None,
            });
        }
    }

    if sdss_path.exists() {
        source_paths.push(sdss_path.display().to_string());
        for row in &sdss_rows {
            rows.push(CatalogFeatureRow {
                cube_name: cube_name.to_string(),
                dataset: "SDSS DR18 Quasars".to_string(),
                record_id: row.objid.clone(),
                modality: "sky_point_metadata".to_string(),
                ra_deg: Some(row.ra),
                dec_deg: Some(row.dec),
                time_utc: None,
                redshift: Some(row.z),
                distance_proxy: Some(row.z),
                program_id: None,
                instrument: None,
                features: vec![
                    row.z,
                    row.mag_u,
                    row.mag_g,
                    row.mag_r,
                    row.mag_i,
                    row.mag_z,
                    -1.0,
                    -1.0,
                ],
                residualized_features: None,
            });
        }
    }

    if chime_path.exists() {
        source_paths.push(chime_path.display().to_string());
        for row in &chime_rows {
            rows.push(CatalogFeatureRow {
                cube_name: cube_name.to_string(),
                dataset: "CHIME FRB".to_string(),
                record_id: if row.tns_name.is_empty() {
                    row.repeater_name.clone()
                } else {
                    row.tns_name.clone()
                },
                modality: "sky_point_transient".to_string(),
                ra_deg: finite_or_none(row.ra),
                dec_deg: finite_or_none(row.dec),
                time_utc: mjd_utc(row.mjd_400),
                redshift: None,
                distance_proxy: finite_or_none(row.dm_exc_ne2001),
                program_id: None,
                instrument: Some("CHIME/FRB".to_string()),
                features: vec![
                    row.dm_exc_ne2001,
                    row.bonsai_snr,
                    row.flux,
                    row.fluence,
                    row.width_fitb,
                    row.scat_time,
                    -1.0,
                    -1.0,
                ],
                residualized_features: None,
            });
        }
    }

    notes.push(
        "Metadata-first generic cube spanning bounded survey and program metadata lanes.".to_string(),
    );
    notes.push(
        "Dataset-specific numeric features are carried in a manifest-declared vector rather than a fixed astrophysical schema."
            .to_string(),
    );

    let mut dataset_names = rows.iter().map(|row| row.dataset.clone()).collect::<Vec<_>>();
    dataset_names.sort();
    dataset_names.dedup();
    source_paths.sort();
    source_paths.dedup();

    for row in desi_rows {
        rows.push(CatalogFeatureRow {
            cube_name: cube_name.to_string(),
            dataset: "DESI DR2 BAO".to_string(),
            record_id: format!("{}-{:.3}", row.tracer, row.z_eff),
            modality: "non_spatial_summary".to_string(),
            ra_deg: None,
            dec_deg: None,
            time_utc: None,
            redshift: Some(row.z_eff),
            distance_proxy: Some(row.z_eff),
            program_id: None,
            instrument: None,
            features: vec![
                row.z_eff,
                row.dm_over_rd,
                row.dm_over_rd_err,
                row.dh_over_rd,
                row.dh_over_rd_err,
                row.rho,
                if row.is_isotropic { 1.0 } else { 0.0 },
                -1.0,
            ],
            residualized_features: None,
        });
    }

    dataset_names = rows.iter().map(|row| row.dataset.clone()).collect::<Vec<_>>();
    dataset_names.sort();
    dataset_names.dedup();

    Ok(CatalogFeatureCube {
        manifest: CatalogFeatureCubeManifest {
            cube_name: cube_name.to_string(),
            generated_at_utc: Utc::now().to_rfc3339(),
            row_count: rows.len(),
            dataset_names,
            source_paths,
            notes,
            channels,
        },
        rows,
    })
}

fn release_year(value: &str) -> f64 {
    if let Some(date) = parse_release_time(value) {
        return date.year() as f64;
    }
    if let Ok(date) = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S") {
        return date.year() as f64;
    }
    f64::NAN
}

fn parse_release_time(value: &str) -> Option<DateTime<Utc>> {
    if let Ok(date) = DateTime::parse_from_rfc3339(value) {
        return Some(date.with_timezone(&Utc));
    }
    let mjd = value.trim().parse::<f64>().ok()?;
    let epoch = Utc.with_ymd_and_hms(1858, 11, 17, 0, 0, 0).single()?;
    let millis = (mjd * 86_400_000.0).round() as i64;
    Some(epoch + Duration::milliseconds(millis))
}

fn normalized_release_time(value: &str) -> Option<String> {
    parse_release_time(value).map(|date| date.to_rfc3339())
}

fn parse_string_f64(value: &str) -> f64 {
    value.trim().parse::<f64>().unwrap_or(f64::NAN)
}

fn non_empty_flag(value: &str) -> f64 {
    if value.trim().is_empty() { 0.0 } else { 1.0 }
}

fn parallax_distance_pc(parallax_mas: f64) -> f64 {
    if parallax_mas.is_finite() && parallax_mas > 0.0 {
        1000.0 / parallax_mas
    } else {
        f64::NAN
    }
}

fn finite_or_none(value: f64) -> Option<f64> {
    if value.is_finite() { Some(value) } else { None }
}

fn mjd_utc(mjd: f64) -> Option<String> {
    if !mjd.is_finite() {
        return None;
    }
    let unix_seconds = (mjd - 40_587.0) * 86_400.0;
    chrono::DateTime::from_timestamp(unix_seconds as i64, 0).map(|dt| dt.to_rfc3339())
}
