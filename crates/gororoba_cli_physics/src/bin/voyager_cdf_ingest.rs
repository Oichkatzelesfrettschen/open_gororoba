//! Ingest Voyager 48-sec MAG CDF files into HeliosphereFeatureRow CSV format.
//!
//! Reads CDF files from SPDF (vim_48secmag dataset), extracts BR/BT/BN/F1
//! magnetic field components + ephemeris (Radius, lat, lon), and writes
//! the standard feature cube CSV for downstream CD analysis.
//!
//! Usage:
//!   voyager-cdf-ingest --spacecraft v1 --cdf-dir data/external/voyager_mag48_cdf/v1/ \
//!                       --out-csv data/output/heliosphere/ablations/v1_mag48_2019_2025.csv

use anyhow::{Context, Result};
use clap::Parser;
use data_core::cdf_support::{cdf_type_to_f64, cdf_type_to_ydh, cdf_variable_rows, read_cdf_file};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "voyager-cdf-ingest")]
struct Cli {
    /// Spacecraft: "v1" or "v2"
    #[arg(long)]
    spacecraft: String,

    /// Directory containing CDF files
    #[arg(long)]
    cdf_dir: PathBuf,

    /// Output CSV path
    #[arg(long)]
    out_csv: PathBuf,

    /// Minimum year to include (inclusive)
    #[arg(long, default_value_t = 2019)]
    year_min: u16,

    /// Maximum year to include (inclusive)
    #[arg(long, default_value_t = 2025)]
    year_max: u16,

    /// List zVariable names in the first CDF file and exit (diagnostic mode).
    #[arg(long)]
    list_vars: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mission = match cli.spacecraft.as_str() {
        "v1" => "Voyager 1",
        "v2" => "Voyager 2",
        _ => anyhow::bail!("spacecraft must be 'v1' or 'v2'"),
    };
    let product = format!("{}_mag48_cdf", cli.spacecraft);

    println!("=== Voyager CDF Ingest: {} ===", mission);
    println!("  dir: {}", cli.cdf_dir.display());

    // --list-vars mode: inspect CDF structure and exit
    if cli.list_vars {
        let mut cdf_paths: Vec<PathBuf> = std::fs::read_dir(&cli.cdf_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "cdf"))
            .collect();
        cdf_paths.sort();
        for path in cdf_paths.iter().take(1) {
            let fname = path.file_name().unwrap().to_string_lossy();
            let cdf = data_core::cdf_support::read_cdf_file(path)
                .with_context(|| format!("read {fname}"))?;
            let vars = data_core::cdf_support::cdf_list_zvariables(&cdf);
            println!("  File: {}", fname);
            println!("  zVariables ({}):", vars.len());
            for (i, name) in vars.iter().enumerate() {
                let row_count = data_core::cdf_support::cdf_variable_rows(&cdf, name)
                    .map(|r| r.len())
                    .unwrap_or(0);
                println!("    [{:>2}] {:30} ({} records)", i, name, row_count);
            }
        }
        return Ok(());
    }

    println!("  years: {}-{}", cli.year_min, cli.year_max);

    // Collect CDF files
    let mut cdf_paths: Vec<PathBuf> = std::fs::read_dir(&cli.cdf_dir)
        .with_context(|| format!("open dir {}", cli.cdf_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "cdf"))
        .collect();
    cdf_paths.sort();
    println!("  Found {} CDF files", cdf_paths.len());

    // Process each CDF file
    let mut total_records = 0usize;
    let mut writer = csv::WriterBuilder::new()
        .from_path(&cli.out_csv)
        .with_context(|| format!("create {}", cli.out_csv.display()))?;

    // Write header
    writer.write_record([
        "window_name", "mission", "product", "year", "doy", "hour",
        "r_au", "lat_deg", "lon_deg", "density_cm3", "speed_kms",
        "temperature_k", "bx", "by", "bz", "b_mag", "crs_flux",
        "spectral_mean", "spectral_peak", "map_flux_mean", "map_flux_std",
    ])?;

    for cdf_path in &cdf_paths {
        let fname = cdf_path.file_name().unwrap().to_string_lossy();
        println!("  Processing {}...", fname);

        let cdf = match read_cdf_file(cdf_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("    SKIP: {}", e);
                continue;
            }
        };

        // Read epoch (timestamps) and B-field components
        let epoch_rows = match cdf_variable_rows(&cdf, "Epoch") {
            Ok(r) => r,
            Err(e) => {
                eprintln!("    SKIP Epoch: {}", e);
                continue;
            }
        };
        let br_rows = cdf_variable_rows(&cdf, "BR").unwrap_or_default();
        let bt_rows = cdf_variable_rows(&cdf, "BT").unwrap_or_default();
        let bn_rows = cdf_variable_rows(&cdf, "BN").unwrap_or_default();
        let f1_rows = cdf_variable_rows(&cdf, "F1").unwrap_or_default();

        // Ephemeris: daily cadence, needs interpolation
        let radius_rows = cdf_variable_rows(&cdf, "Radius").unwrap_or_default();
        let lat_rows = cdf_variable_rows(&cdf, "hg_lat").unwrap_or_default();
        let lon_rows = cdf_variable_rows(&cdf, "hg_lon").unwrap_or_default();

        // Get single radius/lat/lon values (approximate -- daily ephemeris)
        let r_au = radius_rows
            .first()
            .and_then(|r| r.first())
            .and_then(|v| cdf_type_to_f64(v))
            .unwrap_or(f64::NAN);
        let lat = lat_rows
            .first()
            .and_then(|r| r.first())
            .and_then(|v| cdf_type_to_f64(v))
            .unwrap_or(f64::NAN);
        let lon = lon_rows
            .first()
            .and_then(|r| r.first())
            .and_then(|v| cdf_type_to_f64(v))
            .unwrap_or(f64::NAN);

        let n = epoch_rows.len();
        let mut file_count = 0usize;

        for i in 0..n {
            // Extract timestamp
            let Some(epoch_val) = epoch_rows[i].first() else { continue };
            let Some((year, doy, hour)) = cdf_type_to_ydh(epoch_val) else { continue };

            if year < cli.year_min || year > cli.year_max {
                continue;
            }

            // Extract B-field
            let br = br_rows.get(i)
                .and_then(|r| r.first())
                .and_then(|v| cdf_type_to_f64(v))
                .unwrap_or(f64::NAN);
            let bt = bt_rows.get(i)
                .and_then(|r| r.first())
                .and_then(|v| cdf_type_to_f64(v))
                .unwrap_or(f64::NAN);
            let bn = bn_rows.get(i)
                .and_then(|r| r.first())
                .and_then(|v| cdf_type_to_f64(v))
                .unwrap_or(f64::NAN);
            let b_mag = f1_rows.get(i)
                .and_then(|r| r.first())
                .and_then(|v| cdf_type_to_f64(v))
                .unwrap_or(f64::NAN);

            // Skip fill values (|B| > 999 or NaN)
            if !br.is_finite() || !bt.is_finite() || !bn.is_finite() || !b_mag.is_finite() {
                continue;
            }
            if br.abs() > 999.0 || b_mag.abs() > 999.0 {
                continue;
            }

            // Interpolate ephemeris by day-of-year index
            let day_idx = (doy as usize).saturating_sub(1).min(radius_rows.len().saturating_sub(1));
            let r_interp = radius_rows.get(day_idx)
                .and_then(|r| r.first())
                .and_then(|v| cdf_type_to_f64(v))
                .unwrap_or(r_au);
            let lat_interp = lat_rows.get(day_idx)
                .and_then(|r| r.first())
                .and_then(|v| cdf_type_to_f64(v))
                .unwrap_or(lat);
            let lon_interp = lon_rows.get(day_idx)
                .and_then(|r| r.first())
                .and_then(|v| cdf_type_to_f64(v))
                .unwrap_or(lon);

            let window_name = format!("{}_{}_{:03}_{:02}", cli.spacecraft, year, doy, hour);

            writer.write_record(&[
                &window_name,
                mission,
                &product,
                &year.to_string(),
                &doy.to_string(),
                &hour.to_string(),
                &format!("{:.3}", r_interp),
                &format!("{:.3}", lat_interp),
                &format!("{:.3}", lon_interp),
                "NaN", // density_cm3 (MAG-only)
                "NaN", // speed_kms
                "NaN", // temperature_k
                &format!("{:.6}", br),
                &format!("{:.6}", bt),
                &format!("{:.6}", bn),
                &format!("{:.6}", b_mag),
                "NaN", // crs_flux
                "NaN", // spectral_mean
                "NaN", // spectral_peak
                "NaN", // map_flux_mean
                "NaN", // map_flux_std
            ])?;
            file_count += 1;
        }

        println!("    {} records ({}-{})", file_count, cli.year_min, cli.year_max);
        total_records += file_count;
    }

    writer.flush()?;
    println!("  Total: {} records -> {}", total_records, cli.out_csv.display());

    Ok(())
}
