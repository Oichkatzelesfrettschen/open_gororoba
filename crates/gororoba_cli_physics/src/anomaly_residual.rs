use crate::flyby::config::FlybyConfig;
use anyhow::{Context, Result, bail};
use csv::{ReaderBuilder, WriterBuilder};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FlybyObservedRecord {
    pub name: String,
    pub perigee_date_utc: String,
    pub perigee_jed: f64,
    pub perigee_alt_km: f64,
    pub v_inf_km_s: f64,
    pub inbound_dec_deg: f64,
    pub inbound_ra_deg: f64,
    pub outbound_dec_deg: f64,
    pub outbound_ra_deg: f64,
    pub observed_dv_mm_s: f64,
    pub source_bib: String,
    pub source_doi: String,
    pub notes: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FlybyResidualRecord {
    pub name: String,
    pub perigee_date_utc: String,
    pub perigee_jed: f64,
    pub observed_dv_mm_s: f64,
    pub classical_dv_mm_s: f64,
    pub nfw_dv_mm_s: f64,
    pub wake_dv_mm_s: f64,
    pub residual_wake_minus_observed_mm_s: f64,
    pub sign_match: bool,
    pub source_bib: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PioneerBenchmarkRecord {
    pub spacecraft: String,
    pub interval_start_utc: String,
    pub interval_end_utc: String,
    pub observed_anomalous_accel_m_s2: f64,
    pub observed_sigma_m_s2: f64,
    pub source_bib: String,
    pub source_doi: String,
    pub benchmark_kind: String,
    pub notes: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PioneerBenchmarkContextRecord {
    pub spacecraft: String,
    pub interval_start_utc: String,
    pub interval_end_utc: String,
    pub sample_time_utc: String,
    pub observed_anomalous_accel_m_s2: f64,
    pub observed_sigma_m_s2: f64,
    pub sampled_r_au: Option<f64>,
    pub sampled_speed_km_s: Option<f64>,
    pub sampled_density_cm3: Option<f64>,
    pub sampled_bmag_nt: Option<f64>,
    pub dynamic_pressure_npa: Option<f64>,
    pub fractal_predicted_accel_m_s2_df27: Option<f64>,
    pub provider_contract_satisfied: bool,
    pub source_label: String,
    pub source_bib: String,
    pub benchmark_kind: String,
    pub notes: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FitGridRecord {
    pub lane: String,
    pub subject: String,
    pub d_f: f64,
    pub r_disclination_km: f64,
    pub s_bulk: f64,
    pub observed_value: f64,
    pub predicted_value: f64,
    pub absolute_error: f64,
    pub relative_error: f64,
    pub sign_match: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BestFitSummary {
    pub verdict: String,
    pub best_lane: String,
    pub best_d_f: f64,
    pub best_r_disclination_km: f64,
    pub best_s_bulk: f64,
    pub flyby_sign_matches: usize,
    pub flyby_total: usize,
    pub mean_relative_error: f64,
    pub max_relative_error: f64,
    pub notes: String,
}

pub fn read_csv_records<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let mut rdr = ReaderBuilder::new()
        .flexible(false)
        .from_path(path)
        .with_context(|| format!("opening CSV {}", path.display()))?;
    let mut out = Vec::new();
    for row in rdr.deserialize() {
        let record: T =
            row.with_context(|| format!("deserializing row from {}", path.display()))?;
        out.push(record);
    }
    Ok(out)
}

pub fn write_csv_records<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    ensure_parent_dir(path)?;
    let mut wtr = WriterBuilder::new()
        .from_path(path)
        .with_context(|| format!("creating CSV {}", path.display()))?;
    for row in rows {
        wtr.serialize(row)
            .with_context(|| format!("serializing row into {}", path.display()))?;
    }
    wtr.flush()
        .with_context(|| format!("flushing CSV {}", path.display()))?;
    Ok(())
}

pub fn ensure_parent_dir(path: &Path) -> Result<()> {
    let Some(parent) = path.parent() else {
        bail!("path {} has no parent directory", path.display());
    };
    std::fs::create_dir_all(parent)
        .with_context(|| format!("creating parent directory {}", parent.display()))?;
    Ok(())
}

pub fn validate_flyby_catalog(
    observed: &[FlybyObservedRecord],
    expected: &[FlybyConfig],
) -> Result<()> {
    if observed.len() != expected.len() {
        bail!(
            "flyby catalog mismatch: observed rows={} expected rows={}",
            observed.len(),
            expected.len()
        );
    }
    let observed_by_name: BTreeMap<&str, &FlybyObservedRecord> = observed
        .iter()
        .map(|row| (row.name.as_str(), row))
        .collect();
    for cfg in expected {
        let Some(row) = observed_by_name.get(cfg.name) else {
            bail!("flyby catalog missing {}", cfg.name);
        };
        compare_float(
            "perigee_alt_km",
            row.perigee_alt_km,
            cfg.perigee_alt_km,
            1e-6,
        )?;
        compare_float("v_inf_km_s", row.v_inf_km_s, cfg.v_inf, 1e-6)?;
        compare_float(
            "inbound_dec_deg",
            row.inbound_dec_deg,
            cfg.inbound_dec_deg,
            1e-6,
        )?;
        compare_float(
            "inbound_ra_deg",
            row.inbound_ra_deg,
            cfg.inbound_ra_deg,
            1e-6,
        )?;
        compare_float(
            "outbound_dec_deg",
            row.outbound_dec_deg,
            cfg.outbound_dec_deg,
            1e-6,
        )?;
        compare_float(
            "outbound_ra_deg",
            row.outbound_ra_deg,
            cfg.outbound_ra_deg,
            1e-6,
        )?;
        compare_float(
            "observed_dv_mm_s",
            row.observed_dv_mm_s,
            cfg.observed_dv_mm_s,
            1e-6,
        )?;
        compare_float("perigee_jed", row.perigee_jed, cfg.perigee_jed, 1e-6)?;
    }
    Ok(())
}

pub fn compare_float(label: &str, observed: f64, expected: f64, tol: f64) -> Result<()> {
    if (observed - expected).abs() > tol {
        bail!(
            "catalog mismatch for {}: observed={} expected={} tol={}",
            label,
            observed,
            expected,
            tol
        );
    }
    Ok(())
}
