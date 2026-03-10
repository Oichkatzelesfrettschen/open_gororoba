use anyhow::{Context, Result, bail};
use chrono::NaiveDate;
use clap::Parser;
use csv::Writer;
use data_core::catalogs::{
    omni::OmniRecord,
    soho_celias::{SohoCeliasRecord, parse_soho_celias_bundle_file, soho_to_hourly_omni},
};
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(name = "soho-celias-stage")]
#[command(about = "Stage SOHO CELIAS Proton Monitor bundle into native and hourly CSV artifacts.")]
struct Cli {
    /// Path to the SOHO CELIAS Proton Monitor mission-long tar.gz bundle.
    #[arg(
        long,
        default_value = "data/external/soho/celias/CELIAS_Proton_Monitor_5min.tar.gz"
    )]
    input: PathBuf,

    /// CSV output path preserving the native CELIAS cadence.
    #[arg(
        long,
        default_value = "data/output/heliosphere/soho/celias_pm_native_5min.csv"
    )]
    native_out: PathBuf,

    /// CSV output path for hourly median-normalized records.
    #[arg(
        long,
        default_value = "data/output/heliosphere/soho/celias_pm_hourly.csv"
    )]
    hourly_out: PathBuf,

    /// JSON summary describing row counts, sizes, and coverage.
    #[arg(
        long,
        default_value = "data/output/heliosphere/soho/celias_pm_stage_summary.json"
    )]
    summary_out: PathBuf,
}

#[derive(Serialize)]
struct NativeCsvRow {
    timestamp: String,
    year: u16,
    doy: u16,
    hour: u8,
    minute: u8,
    second: u8,
    bulk_speed: f64,
    proton_density: f64,
    thermal_speed_kms: f64,
    proton_temperature: f64,
    r_au: f64,
    lat_deg: f64,
    lon_deg: f64,
}

#[derive(Serialize)]
struct HourlyCsvRow {
    timestamp: String,
    year: u16,
    doy: u16,
    hour: u8,
    proton_temperature: f64,
    proton_density: f64,
    bulk_speed: f64,
    r_au: f64,
    lat_deg: f64,
    lon_deg: f64,
}

#[derive(Serialize)]
struct StageSummary {
    input_bundle: String,
    input_bundle_size_bytes: u64,
    native_out: String,
    native_rows: usize,
    native_size_bytes: u64,
    hourly_out: String,
    hourly_rows: usize,
    hourly_size_bytes: u64,
    native_to_hourly_row_ratio: f64,
    native_to_hourly_size_ratio: f64,
    year_start: u16,
    year_end: u16,
    timestamp_start: String,
    timestamp_end: String,
}

fn timestamp_string(year: u16, doy: u16, hour: u8, minute: u8, second: u8) -> Result<String> {
    let date = NaiveDate::from_yo_opt(i32::from(year), u32::from(doy))
        .with_context(|| format!("invalid year/day pair: {year} DOY {doy}"))?;
    let time = date
        .and_hms_opt(u32::from(hour), u32::from(minute), u32::from(second))
        .with_context(|| format!("invalid time for {year} DOY {doy}: {hour}:{minute}:{second}"))?;
    Ok(format!("{}Z", time.format("%Y-%m-%dT%H:%M:%S")))
}

fn ensure_parent(path: &Path) -> Result<()> {
    let Some(parent) = path.parent() else {
        bail!("output path has no parent: {}", path.display());
    };
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create parent directory for {}", path.display()))?;
    Ok(())
}

fn write_native_csv(path: &Path, records: &[SohoCeliasRecord]) -> Result<()> {
    ensure_parent(path)?;
    let mut writer = Writer::from_path(path)
        .with_context(|| format!("failed to open native CSV {}", path.display()))?;
    for record in records {
        writer.serialize(NativeCsvRow {
            timestamp: timestamp_string(
                record.year,
                record.doy,
                record.hour,
                record.minute,
                record.second,
            )?,
            year: record.year,
            doy: record.doy,
            hour: record.hour,
            minute: record.minute,
            second: record.second,
            bulk_speed: record.bulk_speed,
            proton_density: record.proton_density,
            thermal_speed_kms: record.thermal_speed_kms,
            proton_temperature: record.proton_temperature,
            r_au: record.r_au,
            lat_deg: record.lat_deg,
            lon_deg: record.lon_deg,
        })?;
    }
    writer.flush()?;
    Ok(())
}

fn write_hourly_csv(path: &Path, records: &[OmniRecord]) -> Result<()> {
    ensure_parent(path)?;
    let mut writer = Writer::from_path(path)
        .with_context(|| format!("failed to open hourly CSV {}", path.display()))?;
    for record in records {
        writer.serialize(HourlyCsvRow {
            timestamp: timestamp_string(record.year, record.doy, record.hour, 0, 0)?,
            year: record.year,
            doy: record.doy,
            hour: record.hour,
            proton_temperature: record.proton_temperature,
            proton_density: record.proton_density,
            bulk_speed: record.bulk_speed,
            r_au: record.r_au,
            lat_deg: record.lat_deg,
            lon_deg: record.lon_deg,
        })?;
    }
    writer.flush()?;
    Ok(())
}

fn file_size(path: &Path) -> Result<u64> {
    Ok(fs::metadata(path)
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let native_records = parse_soho_celias_bundle_file(&cli.input)
        .with_context(|| format!("failed to parse CELIAS bundle {}", cli.input.display()))?;
    if native_records.is_empty() {
        bail!("no CELIAS records parsed from {}", cli.input.display());
    }
    let hourly_records = soho_to_hourly_omni(&native_records);
    if hourly_records.is_empty() {
        bail!("hourly CELIAS normalization produced no records");
    }

    write_native_csv(&cli.native_out, &native_records)?;
    write_hourly_csv(&cli.hourly_out, &hourly_records)?;
    ensure_parent(&cli.summary_out)?;

    let first = native_records
        .first()
        .context("missing first native record")?;
    let last = native_records
        .last()
        .context("missing last native record")?;

    let native_rows = native_records.len();
    let hourly_rows = hourly_records.len();
    let native_size_bytes = file_size(&cli.native_out)?;
    let hourly_size_bytes = file_size(&cli.hourly_out)?;
    let summary = StageSummary {
        input_bundle: cli.input.display().to_string(),
        input_bundle_size_bytes: file_size(&cli.input)?,
        native_out: cli.native_out.display().to_string(),
        native_rows,
        native_size_bytes,
        hourly_out: cli.hourly_out.display().to_string(),
        hourly_rows,
        hourly_size_bytes,
        native_to_hourly_row_ratio: native_rows as f64 / hourly_rows as f64,
        native_to_hourly_size_ratio: native_size_bytes as f64 / hourly_size_bytes as f64,
        year_start: first.year,
        year_end: last.year,
        timestamp_start: timestamp_string(
            first.year,
            first.doy,
            first.hour,
            first.minute,
            first.second,
        )?,
        timestamp_end: timestamp_string(last.year, last.doy, last.hour, last.minute, last.second)?,
    };
    let json = serde_json::to_string_pretty(&summary)?;
    fs::write(&cli.summary_out, json)
        .with_context(|| format!("failed to write summary {}", cli.summary_out.display()))?;

    println!(
        "staged native_rows={} hourly_rows={} native_size_bytes={} hourly_size_bytes={}",
        native_rows, hourly_rows, native_size_bytes, hourly_size_bytes
    );
    Ok(())
}
