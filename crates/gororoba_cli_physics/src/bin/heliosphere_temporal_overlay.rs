use anyhow::{Context, Result, bail};
use clap::Parser;
use data_core::{
    catalogs::{
        omni::{OmniRecord, parse_omni_file},
        spdf_merged::SpdfMergedRecord,
        voyager::{VoyagerSpacecraft, parse_voyager_file},
        voyager_crs_flux::{VoyagerCrsFluxRecord, parse_voyager_crs_flux_csv},
    },
    time_bounds::{TimeBounds, bounds_from_omni, format_timestamp_ms},
};
use serde::Serialize;
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(name = "heliosphere-temporal-overlay")]
#[command(about = "Time-aligned OMNI / Voyager / CRS overlay report")]
struct Cli {
    #[arg(long, default_value = "data/external/omni2")]
    omni_dir: PathBuf,

    #[arg(long, default_value = "data/external/voyager1")]
    voyager1_dir: PathBuf,

    #[arg(long, default_value = "data/external/voyager2")]
    voyager2_dir: PathBuf,

    #[arg(long, default_value = "data/external/voyager_crs/voyager1_crs_daily_flux_2020_2020.csv")]
    voyager1_crs: PathBuf,

    #[arg(long, default_value = "data/external/voyager_crs/voyager2_crs_daily_flux_2020_2020.csv")]
    voyager2_crs: PathBuf,

    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct DatasetWindow {
    label: String,
    path: String,
    availability: String,
    row_count: usize,
    start_utc: Option<String>,
    end_utc: Option<String>,
    cadence_seconds: Option<f64>,
    notes: Option<String>,
}

#[derive(Debug, Serialize)]
struct SpacecraftOverlay {
    spacecraft: String,
    omni_row_count: usize,
    voyager_row_count: usize,
    crs_row_count: usize,
    crs_availability: String,
    simultaneous_day_bins: usize,
    simultaneous_start_day: Option<String>,
    simultaneous_end_day: Option<String>,
    temporal_classification: String,
    mean_heliocentric_distance_au: Option<f64>,
    min_heliocentric_distance_au: Option<f64>,
    max_heliocentric_distance_au: Option<f64>,
    notes: Option<String>,
}

#[derive(Debug, Serialize)]
struct HeliosphereTemporalOverlayReport {
    generated_at_utc: String,
    overlay_definition: String,
    omni_dir: String,
    voyager1_dir: String,
    voyager2_dir: String,
    voyager1_crs_path: String,
    voyager2_crs_path: String,
    datasets: Vec<DatasetWindow>,
    overlays: Vec<SpacecraftOverlay>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let report_path = cli.report.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!(
            "heliosphere_temporal_overlay_{}.toml",
            chrono::Utc::now().date_naive()
        ))
    });

    let omni = load_omni_records(&cli.omni_dir)?;
    let voyager1 = load_voyager_records(&cli.voyager1_dir, VoyagerSpacecraft::V1)?;
    let voyager2 = load_voyager_records(&cli.voyager2_dir, VoyagerSpacecraft::V2)?;
    let voyager1_crs = load_crs_flux_records_optional(&cli.voyager1_crs, 1)?;
    let voyager2_crs = load_crs_flux_records_optional(&cli.voyager2_crs, 2)?;

    let omni_bounds = bounds_from_omni(&omni);
    let voyager1_bounds = bounds_from_voyager(&voyager1);
    let voyager2_bounds = bounds_from_voyager(&voyager2);
    let voyager1_crs_bounds = voyager1_crs.as_deref().and_then(bounds_from_crs_flux);
    let voyager2_crs_bounds = voyager2_crs.as_deref().and_then(bounds_from_crs_flux);

    let report = HeliosphereTemporalOverlayReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        overlay_definition: "4D overlay = shared observed epoch windows across heliocentric-point datasets, using true telemetry times rather than release dates or lookback proxies.".to_string(),
        omni_dir: cli.omni_dir.display().to_string(),
        voyager1_dir: cli.voyager1_dir.display().to_string(),
        voyager2_dir: cli.voyager2_dir.display().to_string(),
        voyager1_crs_path: cli.voyager1_crs.display().to_string(),
        voyager2_crs_path: cli.voyager2_crs.display().to_string(),
        datasets: vec![
            window_from_bounds(
                "OMNI hourly",
                &cli.omni_dir,
                omni.len(),
                omni_bounds.as_ref(),
                true,
                None,
            ),
            window_from_bounds(
                "Voyager 1 merged hourly",
                &cli.voyager1_dir,
                voyager1.len(),
                voyager1_bounds.as_ref(),
                true,
                None,
            ),
            window_from_bounds(
                "Voyager 2 merged hourly",
                &cli.voyager2_dir,
                voyager2.len(),
                voyager2_bounds.as_ref(),
                true,
                None,
            ),
            window_from_bounds(
                "Voyager 1 CRS daily flux",
                &cli.voyager1_crs,
                voyager1_crs.as_ref().map(|rows| rows.len()).unwrap_or(0),
                voyager1_crs_bounds.as_ref(),
                voyager1_crs.is_some(),
                voyager1_crs
                    .as_ref()
                    .map(|_| None)
                    .unwrap_or_else(|| Some("CRS daily-flux file is unavailable or not yet fetched.".to_string())),
            ),
            window_from_bounds(
                "Voyager 2 CRS daily flux",
                &cli.voyager2_crs,
                voyager2_crs.as_ref().map(|rows| rows.len()).unwrap_or(0),
                voyager2_crs_bounds.as_ref(),
                voyager2_crs.is_some(),
                voyager2_crs
                    .as_ref()
                    .map(|_| None)
                    .unwrap_or_else(|| Some("CRS daily-flux file is unavailable; current CDAWeb HAPI catalog did not expose a Voyager 2 daily-flux product ID during this run.".to_string())),
            ),
        ],
        overlays: vec![
            overlay_for_spacecraft("Voyager 1", &omni, &voyager1, voyager1_crs.as_deref()),
            overlay_for_spacecraft("Voyager 2", &omni, &voyager2, voyager2_crs.as_deref()),
        ],
    };
    write_toml_report(&report_path, &report)?;
    println!("Datasets: {}", report.datasets.len());
    for overlay in &report.overlays {
        println!(
            "{} simultaneous day bins: {} ({})",
            overlay.spacecraft, overlay.simultaneous_day_bins, overlay.temporal_classification
        );
    }
    println!("Report: {}", report_path.display());
    Ok(())
}

fn load_omni_records(dir: &Path) -> Result<Vec<OmniRecord>> {
    if !dir.exists() {
        bail!("OMNI directory not found: {}", dir.display());
    }
    let mut paths = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            let name = path.file_name().and_then(|v| v.to_str()).unwrap_or("");
            (name.starts_with("omni2_") && name.ends_with(".dat"))
                || (name.starts_with("omni2_") && name.ends_with("_amda_hourly.csv"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    let mut records = Vec::new();
    for path in paths {
        records.extend(parse_omni_file(&path)?);
    }
    if records.is_empty() {
        bail!("No OMNI files were parseable under {}", dir.display());
    }
    Ok(records)
}

fn load_voyager_records(dir: &Path, spacecraft: VoyagerSpacecraft) -> Result<Vec<SpdfMergedRecord>> {
    if !dir.exists() {
        bail!("Voyager directory not found: {}", dir.display());
    }
    let prefix = match spacecraft {
        VoyagerSpacecraft::V1 => "vy1_",
        VoyagerSpacecraft::V2 => "vy2_",
    };
    let mut paths = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|v| v.to_str())
                .map(|name| name.starts_with(prefix) && name.ends_with(".asc"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    paths.sort();
    let mut records = Vec::new();
    for path in paths {
        records.extend(parse_voyager_file(&path, spacecraft)?);
    }
    if records.is_empty() {
        bail!("No Voyager merged hourly files were parseable under {}", dir.display());
    }
    Ok(records)
}

fn load_crs_flux_records_optional(
    path: &Path,
    spacecraft: u8,
) -> Result<Option<Vec<VoyagerCrsFluxRecord>>> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(path)
        .with_context(|| format!("read CRS flux file {}", path.display()))?;
    let trimmed = raw.trim_start();
    if trimmed.starts_with('{')
        && trimmed.contains("\"code\": 1201")
        && trimmed.to_ascii_lowercase().contains("no data")
    {
        return Ok(None);
    }
    let (records, skipped) = parse_voyager_crs_flux_csv(&raw, spacecraft);
    if records.is_empty() {
        bail!(
            "No CRS flux rows were parseable from {} (skipped={})",
            path.display(),
            skipped
        );
    }
    Ok(Some(records))
}

fn voyager_timestamp_ms(record: &SpdfMergedRecord) -> Option<i64> {
    let date = chrono::NaiveDate::from_yo_opt(record.year as i32, record.doy as u32)?;
    let datetime = date.and_hms_opt(record.hour as u32, 0, 0)?;
    Some(datetime.and_utc().timestamp_millis())
}

fn bounds_from_voyager(records: &[SpdfMergedRecord]) -> Option<TimeBounds> {
    let mut timestamps = records
        .iter()
        .filter_map(voyager_timestamp_ms)
        .collect::<Vec<_>>();
    timestamps.sort();
    TimeBounds::from_sorted_epoch_ms(&timestamps)
}

fn decimal_year_to_day_key(decimal_year: f64) -> Option<i32> {
    if !decimal_year.is_finite() {
        return None;
    }
    let year = decimal_year.floor() as i32;
    let frac = decimal_year - year as f64;
    let days_in_year = if is_leap_year(year) { 366.0 } else { 365.0 };
    let doy = (frac * days_in_year).floor() as i32 + 1;
    if !(1..=366).contains(&doy) {
        return None;
    }
    Some(year * 1000 + doy)
}

fn day_key_to_date_string(day_key: i32) -> Option<String> {
    let year = day_key / 1000;
    let doy = (day_key % 1000) as u32;
    chrono::NaiveDate::from_yo_opt(year, doy).map(|date| date.to_string())
}

fn day_key_to_timestamp_ms(day_key: i32) -> Option<i64> {
    let year = day_key / 1000;
    let doy = (day_key % 1000) as u32;
    let date = chrono::NaiveDate::from_yo_opt(year, doy)?;
    let datetime = date.and_hms_opt(0, 0, 0)?;
    Some(datetime.and_utc().timestamp_millis())
}

fn bounds_from_crs_flux(records: &[VoyagerCrsFluxRecord]) -> Option<TimeBounds> {
    let mut timestamps = records
        .iter()
        .filter_map(|record| decimal_year_to_day_key(record.decimal_year))
        .filter_map(day_key_to_timestamp_ms)
        .collect::<Vec<_>>();
    timestamps.sort();
    TimeBounds::from_sorted_epoch_ms(&timestamps)
}

fn omni_day_keys(records: &[OmniRecord]) -> BTreeSet<i32> {
    records
        .iter()
        .map(|record| record.year as i32 * 1000 + record.doy as i32)
        .collect()
}

fn voyager_day_keys(records: &[SpdfMergedRecord]) -> BTreeSet<i32> {
    records
        .iter()
        .map(|record| record.year as i32 * 1000 + record.doy as i32)
        .collect()
}

fn crs_day_keys(records: &[VoyagerCrsFluxRecord]) -> BTreeSet<i32> {
    records
        .iter()
        .filter_map(|record| decimal_year_to_day_key(record.decimal_year))
        .collect()
}

fn overlay_for_spacecraft(
    label: &str,
    omni: &[OmniRecord],
    voyager: &[SpdfMergedRecord],
    crs: Option<&[VoyagerCrsFluxRecord]>,
) -> SpacecraftOverlay {
    let omni_days = omni_day_keys(omni);
    let voyager_days = voyager_day_keys(voyager);
    let crs_days = crs.map(crs_day_keys).unwrap_or_default();
    let simultaneous = if crs.is_some() {
        omni_days
            .intersection(&voyager_days)
            .copied()
            .collect::<BTreeSet<_>>()
            .intersection(&crs_days)
            .copied()
            .collect::<BTreeSet<_>>()
    } else {
        BTreeSet::new()
    };
    let mean_distance = mean_distance_on_days(voyager, &simultaneous);
    let min_distance = voyager
        .iter()
        .filter(|record| simultaneous.contains(&(record.year as i32 * 1000 + record.doy as i32)))
        .map(|record| record.distance_au)
        .filter(|value| value.is_finite())
        .reduce(f64::min);
    let max_distance = voyager
        .iter()
        .filter(|record| simultaneous.contains(&(record.year as i32 * 1000 + record.doy as i32)))
        .map(|record| record.distance_au)
        .filter(|value| value.is_finite())
        .reduce(f64::max);

    let classification = if crs.is_none() {
        "temporal_unknown"
    } else if !simultaneous.is_empty() {
        "simultaneous"
    } else if temporal_bounds_overlap(
        bounds_from_omni(omni),
        bounds_from_voyager(voyager),
        crs.and_then(bounds_from_crs_flux),
    ) {
        "near_contemporaneous"
    } else {
        "no_temporal_overlap"
    };

    SpacecraftOverlay {
        spacecraft: label.to_string(),
        omni_row_count: omni.len(),
        voyager_row_count: voyager.len(),
        crs_row_count: crs.map(|rows| rows.len()).unwrap_or(0),
        crs_availability: if crs.is_some() {
            "available".to_string()
        } else {
            "missing".to_string()
        },
        simultaneous_day_bins: simultaneous.len(),
        simultaneous_start_day: simultaneous.iter().next().and_then(|key| day_key_to_date_string(*key)),
        simultaneous_end_day: simultaneous.iter().next_back().and_then(|key| day_key_to_date_string(*key)),
        temporal_classification: classification.to_string(),
        mean_heliocentric_distance_au: mean_distance,
        min_heliocentric_distance_au: min_distance,
        max_heliocentric_distance_au: max_distance,
        notes: if crs.is_some() {
            None
        } else {
            Some("CRS daily-flux input missing; this overlay row is partial and uses OMNI plus Voyager merged hourly windows only.".to_string())
        },
    }
}

fn temporal_bounds_overlap(
    omni: Option<TimeBounds>,
    voyager: Option<TimeBounds>,
    crs: Option<TimeBounds>,
) -> bool {
    let Some(omni) = omni else { return false };
    let Some(voyager) = voyager else { return false };
    let Some(crs) = crs else { return false };
    TimeBounds::intersect_all(&[omni, voyager, crs]).is_some()
}

fn mean_distance_on_days(records: &[SpdfMergedRecord], day_keys: &BTreeSet<i32>) -> Option<f64> {
    let values = records
        .iter()
        .filter(|record| day_keys.contains(&(record.year as i32 * 1000 + record.doy as i32)))
        .map(|record| record.distance_au)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn window_from_bounds(
    label: &str,
    path: &Path,
    row_count: usize,
    bounds: Option<&TimeBounds>,
    available: bool,
    notes: Option<String>,
) -> DatasetWindow {
    DatasetWindow {
        label: label.to_string(),
        path: path.display().to_string(),
        availability: if available {
            "available".to_string()
        } else {
            "missing".to_string()
        },
        row_count,
        start_utc: bounds.map(|b| format_timestamp_ms(b.start_ms)),
        end_utc: bounds.map(|b| format_timestamp_ms(b.end_ms)),
        cadence_seconds: bounds.and_then(|b| b.cadence_seconds),
        notes,
    }
}

const fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn write_toml_report<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, toml::to_string_pretty(value)?)?;
    Ok(())
}
