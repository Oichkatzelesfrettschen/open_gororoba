//! Ingest Voyager HAPI CSV data into HeliosphereFeatureRow format.
//!
//! Reads CDAWeb HAPI CSV output (Time,F1,B1,B2,B3,scDistance or similar)
//! and converts to the standard feature cube CSV for CD analysis.
//! Handles both 48-sec MAG and hourly COHO merged datasets.

use anyhow::{Context, Result};
use clap::Parser;
use csv::ReaderBuilder;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "voyager-hapi-ingest")]
struct Cli {
    /// Spacecraft name for mission column
    #[arg(long)]
    spacecraft: String,

    /// Input HAPI CSV file
    #[arg(long)]
    input: PathBuf,

    /// Output feature cube CSV
    #[arg(long)]
    out_csv: PathBuf,

    /// Column index for |B| (0-based, after Time)
    #[arg(long, default_value_t = 0)]
    col_bmag: usize,

    /// Column index for BR
    #[arg(long, default_value_t = 1)]
    col_br: usize,

    /// Column index for BT
    #[arg(long, default_value_t = 2)]
    col_bt: usize,

    /// Column index for BN
    #[arg(long, default_value_t = 3)]
    col_bn: usize,

    /// Column index for distance (AU), -1 to skip
    #[arg(long, default_value_t = 4)]
    col_dist: i32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== Voyager HAPI Ingest: {} ===", cli.spacecraft);

    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(&cli.input)
        .with_context(|| format!("open {}", cli.input.display()))?;

    let mut writer = csv::WriterBuilder::new()
        .from_path(&cli.out_csv)
        .with_context(|| format!("create {}", cli.out_csv.display()))?;

    writer.write_record([
        "window_name",
        "mission",
        "product",
        "year",
        "doy",
        "hour",
        "r_au",
        "lat_deg",
        "lon_deg",
        "density_cm3",
        "speed_kms",
        "temperature_k",
        "bx",
        "by",
        "bz",
        "b_mag",
        "crs_flux",
        "spectral_mean",
        "spectral_peak",
        "map_flux_mean",
        "map_flux_std",
    ])?;

    let mut total = 0usize;
    for result in reader.records() {
        let record = result?;
        let time_str = record.get(0).unwrap_or("");

        // Parse ISO timestamp: 1979-02-01T00:00:35.504Z
        let year: u16 = time_str.get(0..4).unwrap_or("0").parse().unwrap_or(0);
        let month: u8 = time_str.get(5..7).unwrap_or("0").parse().unwrap_or(0);
        let day: u8 = time_str.get(8..10).unwrap_or("0").parse().unwrap_or(0);
        let hour: u8 = time_str.get(11..13).unwrap_or("0").parse().unwrap_or(0);

        if year == 0 {
            continue;
        }

        // Compute day-of-year
        let doy = day_of_year(year, month, day);

        // Extract B-field components (1-indexed from HAPI, 0-indexed after Time column)
        let bmag: f64 = record
            .get(1 + cli.col_bmag)
            .unwrap_or("NaN")
            .parse()
            .unwrap_or(f64::NAN);
        let br: f64 = record
            .get(1 + cli.col_br)
            .unwrap_or("NaN")
            .parse()
            .unwrap_or(f64::NAN);
        let bt: f64 = record
            .get(1 + cli.col_bt)
            .unwrap_or("NaN")
            .parse()
            .unwrap_or(f64::NAN);
        let bn: f64 = record
            .get(1 + cli.col_bn)
            .unwrap_or("NaN")
            .parse()
            .unwrap_or(f64::NAN);

        let r_au = if cli.col_dist >= 0 {
            record
                .get(1 + cli.col_dist as usize)
                .unwrap_or("NaN")
                .parse()
                .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };

        // Skip fill values
        if !bmag.is_finite() || bmag.abs() > 998.0 {
            continue;
        }
        if !br.is_finite() || br.abs() > 998.0 {
            continue;
        }

        let wn = format!("{}_{:04}_{:03}_{:02}", cli.spacecraft, year, doy, hour);

        writer.write_record([
            &wn,
            &cli.spacecraft,
            "hapi_mag",
            &year.to_string(),
            &doy.to_string(),
            &hour.to_string(),
            &format!("{:.3}", r_au),
            "0.0",
            "0.0",
            "NaN",
            "NaN",
            "NaN",
            &format!("{:.6}", br),
            &format!("{:.6}", bt),
            &format!("{:.6}", bn),
            &format!("{:.6}", bmag),
            "NaN",
            "NaN",
            "NaN",
            "NaN",
            "NaN",
        ])?;
        total += 1;
    }

    writer.flush()?;
    println!("  {} records -> {}", total, cli.out_csv.display());
    Ok(())
}

fn day_of_year(year: u16, month: u8, day: u8) -> u16 {
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days_in_month = [
        0,
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut doy: u16 = day as u16;
    for m in 1..month {
        doy += days_in_month[m as usize] as u16;
    }
    doy
}
