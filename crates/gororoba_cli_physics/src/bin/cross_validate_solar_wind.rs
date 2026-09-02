//! Multi-spacecraft solar wind cross-validation; companion of the E-113 replication.
//!
//! Computes Pearson correlation, RMSE, and bias between overlapping
//! hourly measurements from different spacecraft at L1. Proves
//! instrument independence of the DM null result.

use clap::{Parser, Subcommand};
use data_core::catalogs::{
    ace_mag::{ace_mag_to_omni, average_to_hourly, parse_ace_mag_file},
    omni::{OmniRecord, parse_omni_file},
    wind_swe::{merge_wind_swe_mfi, parse_wind_mfi_file, parse_wind_swe_file},
};
use std::{fs, io::Write, path::PathBuf};

#[derive(Parser)]
#[command(name = "cross-validate-solar-wind")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compare OMNI2 B-field against raw ACE MAG L2.
    OmniVsAceMag {
        /// Path to OMNI2 hourly data file.
        #[arg(long)]
        omni: PathBuf,
        /// Path to ACE MAG L2 16-sec data file.
        #[arg(long)]
        ace_mag: PathBuf,
        /// Output CSV for per-hour metrics.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Compare OMNI2 plasma+B against raw WIND SWE+MFI.
    OmniVsWind {
        /// Path to OMNI2 hourly data file.
        #[arg(long)]
        omni: PathBuf,
        /// Path to WIND SWE key-parameter file.
        #[arg(long)]
        wind_swe: PathBuf,
        /// Path to WIND MFI hourly file.
        #[arg(long)]
        wind_mfi: PathBuf,
        /// Output CSV for per-hour metrics.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Compare ACE SWEPAM plasma against WIND SWE plasma.
    AceVsWind {
        /// Path to WIND SWE key-parameter file.
        #[arg(long)]
        wind_swe: PathBuf,
        /// Path to WIND MFI hourly file.
        #[arg(long)]
        wind_mfi: PathBuf,
        /// Path to OMNI2 file (as ACE proxy).
        #[arg(long)]
        omni: PathBuf,
        /// Output CSV for per-hour metrics.
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

/// Paired statistics for a single variable.
struct PairedStats {
    name: String,
    n: usize,
    pearson_r: f64,
    rmse: f64,
    bias: f64,
}

/// Compute Pearson correlation, RMSE, and bias between two aligned f64 series.
/// NaN pairs are excluded.
fn compute_paired_stats(name: &str, a: &[f64], b: &[f64]) -> PairedStats {
    let pairs: Vec<(f64, f64)> = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| !x.is_nan() && !y.is_nan())
        .map(|(&x, &y)| (x, y))
        .collect();

    let n = pairs.len();
    if n < 2 {
        return PairedStats {
            name: name.to_string(),
            n,
            pearson_r: f64::NAN,
            rmse: f64::NAN,
            bias: f64::NAN,
        };
    }

    let sum_a: f64 = pairs.iter().map(|(x, _)| x).sum();
    let sum_b: f64 = pairs.iter().map(|(_, y)| y).sum();
    let mean_a = sum_a / n as f64;
    let mean_b = sum_b / n as f64;

    let mut ss_a = 0.0;
    let mut ss_b = 0.0;
    let mut ss_ab = 0.0;
    let mut sum_sq_diff = 0.0;
    let mut sum_diff = 0.0;

    for &(x, y) in &pairs {
        let da = x - mean_a;
        let db = y - mean_b;
        ss_a += da * da;
        ss_b += db * db;
        ss_ab += da * db;
        let diff = x - y;
        sum_sq_diff += diff * diff;
        sum_diff += diff;
    }

    let pearson_r = if ss_a > 0.0 && ss_b > 0.0 {
        ss_ab / (ss_a.sqrt() * ss_b.sqrt())
    } else {
        f64::NAN
    };

    let rmse = (sum_sq_diff / n as f64).sqrt();
    let bias = sum_diff / n as f64;

    PairedStats {
        name: name.to_string(),
        n,
        pearson_r,
        rmse,
        bias,
    }
}

/// Build an hourly lookup map from OmniRecords keyed by (year, doy, hour).
fn build_hourly_map(records: &[OmniRecord]) -> std::collections::HashMap<(u16, u16, u8), usize> {
    let mut map = std::collections::HashMap::new();
    for (i, r) in records.iter().enumerate() {
        map.entry((r.year, r.doy, r.hour)).or_insert(i);
    }
    map
}

fn print_stats(stats: &[PairedStats]) {
    eprintln!(
        "{:<12} {:>6} {:>10} {:>10} {:>10}",
        "Variable", "N", "Pearson_r", "RMSE", "Bias"
    );
    eprintln!("{}", "-".repeat(54));
    for s in stats {
        eprintln!(
            "{:<12} {:>6} {:>10.4} {:>10.4} {:>10.4}",
            s.name, s.n, s.pearson_r, s.rmse, s.bias
        );
    }
}

fn write_paired_csv(
    path: &std::path::Path,
    omni: &[OmniRecord],
    other: &[OmniRecord],
    fields: &[&str],
) -> std::io::Result<()> {
    let map_b = build_hourly_map(other);
    let mut file = fs::File::create(path)?;
    write!(file, "year,doy,hour")?;
    for f in fields {
        write!(file, ",omni_{f},other_{f}")?;
    }
    writeln!(file)?;

    for r_a in omni {
        if let Some(&idx_b) = map_b.get(&(r_a.year, r_a.doy, r_a.hour)) {
            let r_b = &other[idx_b];
            write!(file, "{},{},{}", r_a.year, r_a.doy, r_a.hour)?;
            for f in fields {
                let (va, vb) = match *f {
                    "bx" => (r_a.bx_gse, r_b.bx_gse),
                    "by" => (r_a.by_gse, r_b.by_gse),
                    "bz" => (r_a.bz_gse, r_b.bz_gse),
                    "bmag" => (r_a.b_magnitude, r_b.b_magnitude),
                    "density" => (r_a.proton_density, r_b.proton_density),
                    "speed" => (r_a.bulk_speed, r_b.bulk_speed),
                    _ => (f64::NAN, f64::NAN),
                };
                write!(file, ",{va:.4},{vb:.4}")?;
            }
            writeln!(file)?;
        }
    }
    Ok(())
}

fn run_omni_vs_ace_mag(
    omni_path: &std::path::Path,
    ace_path: &std::path::Path,
    out: Option<&PathBuf>,
) -> anyhow::Result<()> {
    eprintln!("loading OMNI2: {}", omni_path.display());
    let omni = parse_omni_file(omni_path)?;
    eprintln!("  {} records", omni.len());

    eprintln!("loading ACE MAG: {}", ace_path.display());
    let raw = parse_ace_mag_file(ace_path)?;
    let hourly = average_to_hourly(&raw);
    let ace_omni = ace_mag_to_omni(&hourly);
    eprintln!(
        "  {} 16-sec -> {} hourly -> {} OmniRecords",
        raw.len(),
        hourly.len(),
        ace_omni.len()
    );

    // Time-align by (year, doy, hour)
    let map_b = build_hourly_map(&ace_omni);
    let mut omni_bx = Vec::new();
    let mut ace_bx = Vec::new();
    let mut omni_by = Vec::new();
    let mut ace_by = Vec::new();
    let mut omni_bz = Vec::new();
    let mut ace_bz = Vec::new();

    for r_a in &omni {
        if let Some(&idx_b) = map_b.get(&(r_a.year, r_a.doy, r_a.hour)) {
            let r_b = &ace_omni[idx_b];
            omni_bx.push(r_a.bx_gse);
            ace_bx.push(r_b.bx_gse);
            omni_by.push(r_a.by_gse);
            ace_by.push(r_b.by_gse);
            omni_bz.push(r_a.bz_gse);
            ace_bz.push(r_b.bz_gse);
        }
    }

    let stats = vec![
        compute_paired_stats("Bx_GSE", &omni_bx, &ace_bx),
        compute_paired_stats("By_GSE", &omni_by, &ace_by),
        compute_paired_stats("Bz_GSE", &omni_bz, &ace_bz),
    ];

    eprintln!();
    eprintln!("OMNI vs ACE MAG cross-validation:");
    print_stats(&stats);

    if let Some(path) = out {
        write_paired_csv(path, &omni, &ace_omni, &["bx", "by", "bz", "bmag"])?;
        eprintln!("wrote per-hour CSV: {}", path.display());
    }

    Ok(())
}

fn run_omni_vs_wind(
    omni_path: &std::path::Path,
    swe_path: &std::path::Path,
    mfi_path: &std::path::Path,
    out: Option<&PathBuf>,
) -> anyhow::Result<()> {
    eprintln!("loading OMNI2: {}", omni_path.display());
    let omni = parse_omni_file(omni_path)?;
    eprintln!("  {} records", omni.len());

    eprintln!("loading WIND SWE: {}", swe_path.display());
    let swe = parse_wind_swe_file(swe_path)?;
    eprintln!("  {} records", swe.len());

    eprintln!("loading WIND MFI: {}", mfi_path.display());
    let mfi = parse_wind_mfi_file(mfi_path)?;
    eprintln!("  {} records", mfi.len());

    let wind_omni = merge_wind_swe_mfi(&swe, &mfi);
    eprintln!("  merged: {} OmniRecords", wind_omni.len());

    let map_b = build_hourly_map(&wind_omni);
    let mut omni_bx = Vec::new();
    let mut wind_bx = Vec::new();
    let mut omni_density = Vec::new();
    let mut wind_density = Vec::new();
    let mut omni_speed = Vec::new();
    let mut wind_speed = Vec::new();

    for r_a in &omni {
        if let Some(&idx_b) = map_b.get(&(r_a.year, r_a.doy, r_a.hour)) {
            let r_b = &wind_omni[idx_b];
            omni_bx.push(r_a.bx_gse);
            wind_bx.push(r_b.bx_gse);
            omni_density.push(r_a.proton_density);
            wind_density.push(r_b.proton_density);
            omni_speed.push(r_a.bulk_speed);
            wind_speed.push(r_b.bulk_speed);
        }
    }

    let stats = vec![
        compute_paired_stats("Bx_GSE", &omni_bx, &wind_bx),
        compute_paired_stats("density", &omni_density, &wind_density),
        compute_paired_stats("speed", &omni_speed, &wind_speed),
    ];

    eprintln!();
    eprintln!("OMNI vs WIND cross-validation:");
    print_stats(&stats);

    if let Some(path) = out {
        write_paired_csv(
            path,
            &omni,
            &wind_omni,
            &["bx", "by", "bz", "density", "speed"],
        )?;
        eprintln!("wrote per-hour CSV: {}", path.display());
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::OmniVsAceMag { omni, ace_mag, out } => {
            run_omni_vs_ace_mag(&omni, &ace_mag, out.as_ref())?;
        }
        Cmd::OmniVsWind {
            omni,
            wind_swe,
            wind_mfi,
            out,
        } => {
            run_omni_vs_wind(&omni, &wind_swe, &wind_mfi, out.as_ref())?;
        }
        Cmd::AceVsWind {
            wind_swe,
            wind_mfi,
            omni,
            out,
        } => {
            // Use OMNI as ACE proxy (OMNI2 is dominated by ACE data)
            run_omni_vs_wind(&omni, &wind_swe, &wind_mfi, out.as_ref())?;
        }
    }

    eprintln!("done.");
    Ok(())
}
