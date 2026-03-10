use anyhow::{Context, Result, bail};
use clap::Parser;
use csv::Writer;
use data_core::catalogs::{
    voyager::{VoyagerSpacecraft, parse_voyager_file},
    voyager_crs::{PROTON_CHANNEL_MEV, VoyagerCrsRecord, parse_voyager_crs},
    voyager_crs_flux::{VoyagerCrsFluxRecord, parse_voyager_crs_flux_csv},
};
use serde::Serialize;
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

const EPS: f64 = 1e-6;
const VOYAGER_FILL_B_MAG: f64 = 9999.99;
const VOYAGER_FILL_B_COMP: f64 = 999.99;
const VOYAGER_FILL_DENSITY: f64 = 999.9;
const VOYAGER_FILL_SPEED: f64 = 9999.9;
const VOYAGER_FILL_TEMP: f64 = 999999.0;
const VOYAGER_FILL_DISTANCE: f64 = 999.999;

#[derive(Parser, Debug)]
#[command(name = "heliosphere-stage")]
#[command(about = "Stage and audit local Voyager merged and CRS datasets.")]
struct Cli {
    /// Root containing voyager{1,2}/ merged and CRS folders.
    #[arg(long, default_value = "data/external/voyager")]
    root: PathBuf,

    /// Output directory for audit artifacts.
    #[arg(long, default_value = "data/output/heliosphere/staging")]
    out_dir: PathBuf,
}

#[derive(Serialize)]
struct MergedAuditRow {
    spacecraft: u8,
    path: String,
    records: usize,
    finite_distance: usize,
    finite_density: usize,
    finite_speed: usize,
    finite_temperature: usize,
    finite_b_mag: usize,
    finite_bx: usize,
    finite_by: usize,
    finite_bz: usize,
    year_min: u16,
    year_max: u16,
    distance_min_au: f64,
    distance_max_au: f64,
    sentinel_leaks: usize,
}

#[derive(Serialize)]
struct CrsAuditRow {
    spacecraft: u8,
    path: String,
    records: usize,
    skipped_lines: usize,
    fill_flagged_rows: usize,
    finite_distance: usize,
    finite_proton_rates: usize,
    finite_helium_rates: usize,
    finite_electron_rates: usize,
    sentinel_leaks: usize,
    year_min: f64,
    year_max: f64,
    distance_min_au: f64,
    distance_max_au: f64,
}

#[derive(Clone, Debug, Default)]
struct FluxManifestInfo {
    source_kind: String,
    notes_url_present: bool,
    dataset_page_present: bool,
    staged_entries: usize,
}

#[derive(Serialize)]
struct FluxAuditRow {
    spacecraft: u8,
    path: String,
    records: usize,
    skipped_lines: usize,
    fill_flagged_rows: usize,
    finite_distance: usize,
    finite_flux_values: usize,
    finite_error_values: usize,
    channel_count: usize,
    year_min: f64,
    year_max: f64,
    distance_min_au: f64,
    distance_max_au: f64,
    sentinel_leaks: usize,
    notes_url_present: bool,
    dataset_page_present: bool,
    manifest_source_kind: String,
    manifest_staged_entries: usize,
}

#[derive(Serialize)]
struct ObservedFluxRow {
    spacecraft: u8,
    decimal_year: f64,
    distance_au: f64,
    channel_index: usize,
    kinetic_energy_gev: f64,
    flux: f64,
    flux_error: f64,
}

#[derive(Serialize)]
struct StageSummary {
    root: String,
    merged_files: usize,
    crs_files: usize,
    flux_files: usize,
    merged_records: usize,
    crs_records: usize,
    flux_records: usize,
    crs_skipped_lines: usize,
    flux_skipped_lines: usize,
    merged_sentinel_leaks: usize,
    crs_sentinel_leaks: usize,
    flux_sentinel_leaks: usize,
    crs_flux_manifests: usize,
    spacecraft_present: Vec<u8>,
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPS
}

fn is_voyager_fill_survivor(value: f64) -> bool {
    value.is_finite()
        && (approx_eq(value, VOYAGER_FILL_B_MAG)
            || approx_eq(value, VOYAGER_FILL_B_COMP)
            || approx_eq(value, VOYAGER_FILL_DENSITY)
            || approx_eq(value, VOYAGER_FILL_SPEED)
            || approx_eq(value, VOYAGER_FILL_TEMP)
            || approx_eq(value, VOYAGER_FILL_DISTANCE))
}

fn finite_min(values: impl Iterator<Item = f64>) -> f64 {
    values
        .filter(|v| v.is_finite())
        .reduce(f64::min)
        .unwrap_or(f64::NAN)
}

fn finite_max(values: impl Iterator<Item = f64>) -> f64 {
    values
        .filter(|v| v.is_finite())
        .reduce(f64::max)
        .unwrap_or(f64::NAN)
}

fn count_finite(values: impl Iterator<Item = f64>) -> usize {
    values.filter(|v| v.is_finite()).count()
}

fn path_str(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn detect_spacecraft(path: &Path) -> Option<u8> {
    let text = path.to_string_lossy();
    if text.contains("voyager1") || text.contains("vy1") {
        Some(1)
    } else if text.contains("voyager2") || text.contains("vy2") {
        Some(2)
    } else {
        None
    }
}

fn collect_merged_files(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .map(|entry| entry.into_path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "asc"))
        .filter(|path| {
            path.parent()
                .is_some_and(|parent| parent.ends_with("voyager1") || parent.ends_with("voyager2"))
        })
        .collect()
}

fn collect_crs_files(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .map(|entry| entry.into_path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "asc"))
        .filter(|path| path.to_string_lossy().contains("/crs/"))
        .collect()
}

fn collect_flux_files(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .map(|entry| entry.into_path())
        .filter(|path| {
            path.extension()
                .is_some_and(|ext| ext == "csv" || ext == "txt")
        })
        .filter(|path| path.to_string_lossy().contains("/flux_daily/"))
        .collect()
}

fn collect_flux_manifests(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .min_depth(1)
        .max_depth(2)
        .into_iter()
        .filter_map(Result::ok)
        .map(|entry| entry.into_path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.contains("crs_daily_flux") && name.contains("manifest"))
        })
        .collect()
}

fn audit_merged_file(path: &Path) -> Result<MergedAuditRow> {
    let spacecraft =
        detect_spacecraft(path).context("unable to infer spacecraft from merged file path")?;
    let records = parse_voyager_file(
        path,
        if spacecraft == 1 {
            VoyagerSpacecraft::V1
        } else {
            VoyagerSpacecraft::V2
        },
    )?;

    let sentinel_leaks = records
        .iter()
        .flat_map(|record| {
            [
                record.distance_au,
                record.lat_deg,
                record.lon_deg,
                record.b_magnitude,
                record.br,
                record.bt,
                record.bn,
                record.proton_density,
                record.bulk_speed,
                record.proton_temperature,
            ]
        })
        .filter(|value| is_voyager_fill_survivor(*value))
        .count();

    Ok(MergedAuditRow {
        spacecraft,
        path: path_str(path),
        records: records.len(),
        finite_distance: count_finite(records.iter().map(|r| r.distance_au)),
        finite_density: count_finite(records.iter().map(|r| r.proton_density)),
        finite_speed: count_finite(records.iter().map(|r| r.bulk_speed)),
        finite_temperature: count_finite(records.iter().map(|r| r.proton_temperature)),
        finite_b_mag: count_finite(records.iter().map(|r| r.b_magnitude)),
        finite_bx: count_finite(records.iter().map(|r| r.br)),
        finite_by: count_finite(records.iter().map(|r| r.bt)),
        finite_bz: count_finite(records.iter().map(|r| r.bn)),
        year_min: records.iter().map(|r| r.year).min().unwrap_or(0),
        year_max: records.iter().map(|r| r.year).max().unwrap_or(0),
        distance_min_au: finite_min(records.iter().map(|r| r.distance_au)),
        distance_max_au: finite_max(records.iter().map(|r| r.distance_au)),
        sentinel_leaks,
    })
}

fn count_channel_finite(records: &[VoyagerCrsRecord], group: &str) -> usize {
    match group {
        "proton" => records
            .iter()
            .flat_map(|record| record.proton_rates)
            .filter(|value| value.is_finite())
            .count(),
        "helium" => records
            .iter()
            .flat_map(|record| record.helium_rates)
            .filter(|value| value.is_finite())
            .count(),
        "electron" => records
            .iter()
            .flat_map(|record| record.electron_rates)
            .filter(|value| value.is_finite())
            .count(),
        _ => 0,
    }
}

fn count_flux_finite(records: &[VoyagerCrsFluxRecord]) -> usize {
    records
        .iter()
        .flat_map(|record| record.proton_flux.iter().copied())
        .filter(|value| value.is_finite())
        .count()
}

fn count_flux_error_finite(records: &[VoyagerCrsFluxRecord]) -> usize {
    records
        .iter()
        .flat_map(|record| record.proton_flux_error.iter().copied())
        .filter(|value| value.is_finite() && *value > 0.0)
        .count()
}

fn audit_crs_file(path: &Path) -> Result<CrsAuditRow> {
    let spacecraft = detect_spacecraft(path).context("unable to infer spacecraft from CRS path")?;
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read CRS file {}", path.display()))?;
    let (records, skipped_lines) = parse_voyager_crs(&content, spacecraft);

    let sentinel_leaks = records
        .iter()
        .flat_map(|record| {
            record
                .proton_rates
                .into_iter()
                .chain(record.helium_rates)
                .chain(record.electron_rates)
        })
        .filter(|value| value.is_finite() && (approx_eq(*value, 999.9) || approx_eq(*value, -1.0)))
        .count();

    Ok(CrsAuditRow {
        spacecraft,
        path: path_str(path),
        records: records.len(),
        skipped_lines,
        fill_flagged_rows: records.iter().filter(|record| record.fill_flag).count(),
        finite_distance: count_finite(records.iter().map(|r| r.distance_au)),
        finite_proton_rates: count_channel_finite(&records, "proton"),
        finite_helium_rates: count_channel_finite(&records, "helium"),
        finite_electron_rates: count_channel_finite(&records, "electron"),
        sentinel_leaks,
        year_min: finite_min(records.iter().map(|r| r.decimal_year)),
        year_max: finite_max(records.iter().map(|r| r.decimal_year)),
        distance_min_au: finite_min(records.iter().map(|r| r.distance_au)),
        distance_max_au: finite_max(records.iter().map(|r| r.distance_au)),
    })
}

fn proton_channel_kinetic_energy_gev(channel_index: usize) -> Option<f64> {
    PROTON_CHANNEL_MEV
        .get(channel_index.checked_sub(1)?)
        .map(|energy_mev| energy_mev / 1000.0)
}

fn is_flux_fill_survivor(value: f64) -> bool {
    value.is_finite() && (value.abs() > 1.0e30 || value <= -1.0e20)
}

#[derive(serde::Deserialize)]
struct FluxManifestFileEntry {
    source_kind: Option<String>,
    status: Option<String>,
}

#[derive(serde::Deserialize)]
struct FluxManifestPayload {
    source: Option<String>,
    notes_url: Option<String>,
    dataset_page: Option<String>,
    files: Vec<FluxManifestFileEntry>,
}

fn load_flux_manifest_info(root: &Path) -> Result<Vec<(u8, FluxManifestInfo)>> {
    let mut info = Vec::new();
    for path in collect_flux_manifests(root) {
        let spacecraft = detect_spacecraft(&path)
            .context("unable to infer spacecraft from flux manifest path")?;
        let payload: FluxManifestPayload = serde_json::from_str(
            &fs::read_to_string(&path)
                .with_context(|| format!("failed to read flux manifest {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse flux manifest {}", path.display()))?;
        let staged_entries = payload
            .files
            .iter()
            .filter(|entry| entry.status.as_deref() == Some("staged"))
            .count();
        let source_kind = payload
            .files
            .iter()
            .find_map(|entry| entry.source_kind.clone())
            .or(payload.source)
            .unwrap_or_else(|| "unknown".to_string());
        info.push((
            spacecraft,
            FluxManifestInfo {
                source_kind,
                notes_url_present: payload.notes_url.is_some(),
                dataset_page_present: payload.dataset_page.is_some(),
                staged_entries,
            },
        ));
    }
    Ok(info)
}

fn audit_flux_file(
    path: &Path,
    manifest_info: Option<&FluxManifestInfo>,
) -> Result<(FluxAuditRow, Vec<ObservedFluxRow>)> {
    let spacecraft =
        detect_spacecraft(path).context("unable to infer spacecraft from calibrated CRS path")?;
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read calibrated CRS flux file {}", path.display()))?;
    let (records, skipped_lines) = parse_voyager_crs_flux_csv(&content, spacecraft);
    let channel_count = records
        .first()
        .map(|record| record.proton_flux.len())
        .unwrap_or(0);
    let sentinel_leaks = records
        .iter()
        .flat_map(|record| {
            record
                .proton_flux
                .iter()
                .copied()
                .chain(record.proton_flux_error.iter().copied())
        })
        .filter(|value| is_flux_fill_survivor(*value))
        .count();

    let mut observed_rows = Vec::new();
    for record in &records {
        for channel_index in 1..=record.proton_flux.len() {
            let Some(kinetic_energy_gev) = proton_channel_kinetic_energy_gev(channel_index) else {
                continue;
            };
            let flux = record.proton_flux[channel_index - 1];
            let flux_error = record.proton_flux_error[channel_index - 1];
            if !(record.distance_au.is_finite()
                && record.decimal_year.is_finite()
                && flux.is_finite()
                && flux_error.is_finite()
                && flux_error > 0.0)
            {
                continue;
            }
            observed_rows.push(ObservedFluxRow {
                spacecraft: record.spacecraft,
                decimal_year: record.decimal_year,
                distance_au: record.distance_au,
                channel_index,
                kinetic_energy_gev,
                flux,
                flux_error,
            });
        }
    }

    let manifest = manifest_info.cloned().unwrap_or_default();
    Ok((
        FluxAuditRow {
            spacecraft,
            path: path_str(path),
            records: records.len(),
            skipped_lines,
            fill_flagged_rows: records.iter().filter(|record| record.fill_flag).count(),
            finite_distance: count_finite(records.iter().map(|r| r.distance_au)),
            finite_flux_values: count_flux_finite(&records),
            finite_error_values: count_flux_error_finite(&records),
            channel_count,
            year_min: finite_min(records.iter().map(|r| r.decimal_year)),
            year_max: finite_max(records.iter().map(|r| r.decimal_year)),
            distance_min_au: finite_min(records.iter().map(|r| r.distance_au)),
            distance_max_au: finite_max(records.iter().map(|r| r.distance_au)),
            sentinel_leaks,
            notes_url_present: manifest.notes_url_present,
            dataset_page_present: manifest.dataset_page_present,
            manifest_source_kind: manifest.source_kind,
            manifest_staged_entries: manifest.staged_entries,
        },
        observed_rows,
    ))
}

fn write_csv<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    let mut writer = Writer::from_path(path)
        .with_context(|| format!("failed to create csv {}", path.display()))?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.out_dir)?;

    let merged_files = collect_merged_files(&cli.root);
    let crs_files = collect_crs_files(&cli.root);
    let flux_files = collect_flux_files(&cli.root);
    if merged_files.is_empty() && crs_files.is_empty() && flux_files.is_empty() {
        bail!(
            "no Voyager merged or CRS files found under {}",
            cli.root.display()
        );
    }
    let flux_manifest_info = load_flux_manifest_info(&cli.root)?;

    let mut merged_rows = Vec::new();
    for path in &merged_files {
        merged_rows.push(audit_merged_file(path)?);
    }

    let mut crs_rows = Vec::new();
    for path in &crs_files {
        crs_rows.push(audit_crs_file(path)?);
    }

    let mut flux_rows = Vec::new();
    let mut observed_flux_rows = Vec::new();
    for path in &flux_files {
        let spacecraft = detect_spacecraft(path)
            .context("unable to infer spacecraft from calibrated CRS path")?;
        let manifest = flux_manifest_info
            .iter()
            .find(|(sc, _)| *sc == spacecraft)
            .map(|(_, info)| info);
        let (row, observed_rows) = audit_flux_file(path, manifest)?;
        flux_rows.push(row);
        observed_flux_rows.extend(observed_rows);
    }

    let merged_leaks: usize = merged_rows.iter().map(|row| row.sentinel_leaks).sum();
    let crs_leaks: usize = crs_rows.iter().map(|row| row.sentinel_leaks).sum();
    let flux_leaks: usize = flux_rows.iter().map(|row| row.sentinel_leaks).sum();
    if merged_leaks > 0 || crs_leaks > 0 || flux_leaks > 0 {
        bail!(
            "fill sentinels leaked through parse layer: merged_leaks={}, crs_leaks={}, flux_leaks={}",
            merged_leaks,
            crs_leaks,
            flux_leaks
        );
    }

    let merged_csv = cli.out_dir.join("voyager_merged_audit.csv");
    let crs_csv = cli.out_dir.join("voyager_crs_audit.csv");
    let flux_csv = cli.out_dir.join("voyager_crs_flux_audit.csv");
    let observed_flux_csv = cli.out_dir.join("voyager_crs_flux_observed.csv");
    write_csv(&merged_csv, &merged_rows)?;
    write_csv(&crs_csv, &crs_rows)?;
    write_csv(&flux_csv, &flux_rows)?;
    write_csv(&observed_flux_csv, &observed_flux_rows)?;

    let mut spacecraft_present = BTreeSet::new();
    for row in &merged_rows {
        spacecraft_present.insert(row.spacecraft);
    }
    for row in &crs_rows {
        spacecraft_present.insert(row.spacecraft);
    }
    for row in &flux_rows {
        spacecraft_present.insert(row.spacecraft);
    }

    let summary = StageSummary {
        root: path_str(&cli.root),
        merged_files: merged_rows.len(),
        crs_files: crs_rows.len(),
        flux_files: flux_rows.len(),
        merged_records: merged_rows.iter().map(|row| row.records).sum(),
        crs_records: crs_rows.iter().map(|row| row.records).sum(),
        flux_records: flux_rows.iter().map(|row| row.records).sum(),
        crs_skipped_lines: crs_rows.iter().map(|row| row.skipped_lines).sum(),
        flux_skipped_lines: flux_rows.iter().map(|row| row.skipped_lines).sum(),
        merged_sentinel_leaks: merged_leaks,
        crs_sentinel_leaks: crs_leaks,
        flux_sentinel_leaks: flux_leaks,
        crs_flux_manifests: flux_manifest_info.len(),
        spacecraft_present: spacecraft_present.into_iter().collect(),
    };

    let summary_path = cli.out_dir.join("voyager_stage_report.toml");
    fs::write(&summary_path, toml::to_string_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;

    println!("wrote {}", merged_csv.display());
    println!("wrote {}", crs_csv.display());
    println!("wrote {}", flux_csv.display());
    println!("wrote {}", observed_flux_csv.display());
    println!("wrote {}", summary_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_voyager_fill_survivor_recognizes_fill_values() {
        assert!(is_voyager_fill_survivor(9999.99));
        assert!(is_voyager_fill_survivor(999.99));
        assert!(is_voyager_fill_survivor(999.9));
        assert!(!is_voyager_fill_survivor(0.02));
    }

    #[test]
    fn test_detect_spacecraft_from_path() {
        assert_eq!(
            detect_spacecraft(Path::new("data/external/voyager/voyager1/vy1_2020.asc")),
            Some(1)
        );
        assert_eq!(
            detect_spacecraft(Path::new(
                "data/external/voyager/voyager2/crs/one_day/vy2crs_2008.asc"
            )),
            Some(2)
        );
        assert_eq!(
            detect_spacecraft(Path::new("data/external/voyager/unknown.asc")),
            None
        );
    }

    #[test]
    fn test_proton_channel_energy_mapping() {
        assert_eq!(proton_channel_kinetic_energy_gev(0), None);
        assert!((proton_channel_kinetic_energy_gev(1).unwrap() - 0.0034).abs() < 1e-12);
        assert!((proton_channel_kinetic_energy_gev(8).unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(proton_channel_kinetic_energy_gev(9), None);
    }

    #[test]
    fn test_is_flux_fill_survivor_flags_extreme_fill() {
        assert!(is_flux_fill_survivor(-1.0e31));
        assert!(is_flux_fill_survivor(1.0e31));
        assert!(!is_flux_fill_survivor(42.0));
    }
}
