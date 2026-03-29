//! Ingest Voyager PWS spectrum analyzer CDF files into a multi-channel CSV.
//!
//! Reads daily CDF files from SPDF, extracts the 16-channel electric field
//! spectral power density at ~61-sec cadence, and writes a CSV suitable for
//! multi-instrument ISM embedding (MAG 4-ch + PWS 16-ch = 20-ch).
//!
//! The PWS spectrum analyzer measures wave electric field intensity at 16
//! logarithmically spaced frequency channels from 10 Hz to 56.2 kHz.
//! In the ISM, the dominant signal is plasma oscillations at the local
//! plasma frequency, giving electron density via f_pe = 9 * sqrt(n_e).

use anyhow::{Context, Result};
use clap::Parser;
use data_core::cdf_support::{cdf_list_zvariables, cdf_type_to_f64, cdf_type_to_ydh, cdf_variable_rows, read_cdf_file};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "voyager-pws-ingest")]
struct Cli {
    /// Spacecraft: "v1" or "v2"
    #[arg(long)]
    spacecraft: String,

    /// Directory containing daily PWS CDF files
    #[arg(long)]
    cdf_dir: PathBuf,

    /// Output CSV path
    #[arg(long)]
    out_csv: PathBuf,

    /// Minimum year (inclusive)
    #[arg(long, default_value_t = 2012)]
    year_min: u16,

    /// Maximum year (inclusive)
    #[arg(long, default_value_t = 2026)]
    year_max: u16,

    /// List variables and exit
    #[arg(long)]
    list_vars: bool,

    /// Hourly averaging: collapse 61-sec cadence to 1-hour means
    /// for alignment with MAG 48-sec data
    #[arg(long)]
    hourly: bool,
}

/// Number of PWS spectrum analyzer frequency channels.
const PWS_CHANNELS: usize = 16;

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mission = match cli.spacecraft.as_str() {
        "v1" => "Voyager 1",
        "v2" => "Voyager 2",
        _ => anyhow::bail!("spacecraft must be 'v1' or 'v2'"),
    };

    println!("=== Voyager PWS Ingest: {} ===", mission);
    println!("  dir: {}", cli.cdf_dir.display());

    let mut cdf_paths: Vec<PathBuf> = std::fs::read_dir(&cli.cdf_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "cdf"))
        .collect();
    cdf_paths.sort();
    println!("  Found {} CDF files", cdf_paths.len());

    // --list-vars mode
    if cli.list_vars {
        if let Some(path) = cdf_paths.first() {
            let cdf = read_cdf_file(path)?;
            let vars = cdf_list_zvariables(&cdf);
            println!("  Variables: {:?}", vars);

            // Try to read frequency labels
            if let Ok(freq_rows) = cdf_variable_rows(&cdf, "frequency") {
                let freqs: Vec<f64> = freq_rows.iter()
                    .flat_map(|r| r.iter().filter_map(cdf_type_to_f64))
                    .collect();
                println!("  Frequency channels ({}):", freqs.len());
                for (i, f) in freqs.iter().enumerate() {
                    println!("    [{:>2}] {:.1} Hz", i, f);
                }
            }
        }
        return Ok(());
    }

    println!("  years: {}-{}, hourly={}", cli.year_min, cli.year_max, cli.hourly);

    // Build CSV header
    let mut writer = csv::WriterBuilder::new()
        .from_path(&cli.out_csv)
        .with_context(|| format!("create {}", cli.out_csv.display()))?;

    let mut header = vec![
        "year".to_string(), "doy".to_string(), "hour".to_string(),
    ];
    for ch in 0..PWS_CHANNELS {
        header.push(format!("pws_ch{:02}", ch));
    }
    writer.write_record(&header)?;

    let mut total = 0usize;

    for path in &cdf_paths {
        let fname = path.file_name().unwrap().to_string_lossy();
        let cdf = match read_cdf_file(path) {
            Ok(c) => c,
            Err(e) => { eprintln!("  SKIP {}: {}", fname, e); continue; }
        };

        let epoch_rows = match cdf_variable_rows(&cdf, "epoch") {
            Ok(r) => r,
            Err(_) => { continue; }
        };
        let ef_rows = match cdf_variable_rows(&cdf, "electric_field") {
            Ok(r) => r,
            Err(_) => { continue; }
        };

        let n_epochs = epoch_rows.len();
        // electric_field is stored as flat f32 values: n_epochs * PWS_CHANNELS total
        // Our CVVR fallback reads each f32 as a separate row.
        // Reshape: group every PWS_CHANNELS consecutive ef values into one spectrum.
        let n_ef = ef_rows.len();
        let n_spectra = n_ef / PWS_CHANNELS;

        if n_spectra == 0 || n_epochs == 0 { continue; }

        if cli.hourly {
            // Hourly averaging: group by (year, doy, hour), take mean of each channel
            use std::collections::BTreeMap;
            let mut hourly_acc: BTreeMap<(u16, u16, u8), (Vec<f64>, usize)> = BTreeMap::new();

            for i in 0..n_spectra.min(n_epochs) {
                let Some(epoch_val) = epoch_rows[i].first() else { continue };
                let Some((year, doy, hour)) = cdf_type_to_ydh(epoch_val) else { continue };
                if year < cli.year_min || year > cli.year_max { continue; }

                let base = i * PWS_CHANNELS;
                let mut spectrum = [0.0f64; PWS_CHANNELS];
                let mut valid = true;
                for ch in 0..PWS_CHANNELS {
                    if base + ch < n_ef {
                        let v = ef_rows[base + ch].first()
                            .and_then(cdf_type_to_f64)
                            .unwrap_or(f64::NAN);
                        if !v.is_finite() || v < 0.0 { valid = false; break; }
                        spectrum[ch] = v;
                    } else { valid = false; break; }
                }
                if !valid { continue; }

                let entry = hourly_acc.entry((year, doy, hour)).or_insert_with(|| (vec![0.0; PWS_CHANNELS], 0));
                for ch in 0..PWS_CHANNELS {
                    entry.0[ch] += spectrum[ch];
                }
                entry.1 += 1;
            }

            for ((year, doy, hour), (sums, count)) in &hourly_acc {
                let mut record = vec![
                    year.to_string(), doy.to_string(), hour.to_string(),
                ];
                for ch in 0..PWS_CHANNELS {
                    record.push(format!("{:.6e}", sums[ch] / *count as f64));
                }
                writer.write_record(&record)?;
                total += 1;
            }
        } else {
            // Full cadence output
            for i in 0..n_spectra.min(n_epochs) {
                let Some(epoch_val) = epoch_rows[i].first() else { continue };
                let Some((year, doy, hour)) = cdf_type_to_ydh(epoch_val) else { continue };
                if year < cli.year_min || year > cli.year_max { continue; }

                let base = i * PWS_CHANNELS;
                let mut record = vec![
                    year.to_string(), doy.to_string(), hour.to_string(),
                ];
                let mut valid = true;
                for ch in 0..PWS_CHANNELS {
                    if base + ch < n_ef {
                        let v = ef_rows[base + ch].first()
                            .and_then(cdf_type_to_f64)
                            .unwrap_or(f64::NAN);
                        if !v.is_finite() { valid = false; break; }
                        record.push(format!("{:.6e}", v));
                    } else { valid = false; break; }
                }
                if !valid { continue; }
                writer.write_record(&record)?;
                total += 1;
            }
        }
    }

    writer.flush()?;
    println!("  Total: {} records -> {}", total, cli.out_csv.display());
    Ok(())
}
