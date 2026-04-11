//! Bartol vs SPDF Voyager 2 B-field and plasma cross-validation (E-128).
//!
//! Compares hourly Voyager 2 data from two independent providers:
//!   - SPDF NASA hourly merged files (vy2_YYYY.asc, SE coordinates)
//!   - Bartol Research Institute legacy archive (vy2_YY.dat, RTN coordinates)
//!
//! The B-magnitude is the primary cross-validation target because it is
//! coordinate-invariant (SE vs RTN differ by a frame rotation). Bulk speed
//! and proton density are also compared as secondary checks.
//!
//! Key finding: Bartol MAG data is entirely fill for 1990-1995. The valid
//! cross-validation window is 1983-1989.

use anyhow::{Context, Result};
use clap::Parser;
use csv::Writer;
use data_core::catalogs::voyager::{
    VoyagerSpacecraft, parse_bartol_file, parse_voyager_file,
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    path::PathBuf,
};

#[derive(Parser, Debug)]
#[command(name = "bartol-spdf-crossval")]
#[command(
    about = "E-128: Cross-validate Voyager 2 B-field and plasma between Bartol and SPDF for 1983-1989."
)]
struct Cli {
    /// Directory containing SPDF hourly merged files (vy2_YYYY.asc).
    #[arg(long, default_value = "data/external/voyager2")]
    spdf_dir: PathBuf,

    /// Directory containing Bartol legacy files (vy2_YY.dat).
    #[arg(long, default_value = "data/external/voyager2/bartol")]
    bartol_dir: PathBuf,

    /// Start year (inclusive).
    #[arg(long, default_value_t = 1983)]
    start_year: u16,

    /// End year (inclusive).
    #[arg(long, default_value_t = 1989)]
    end_year: u16,

    /// Output CSV path.
    #[arg(
        long,
        default_value = "data/output/bartol_vs_amda_v2_bfield_correlation.csv"
    )]
    out: PathBuf,

    /// Also write a per-year summary to stdout.
    #[arg(long, default_value_t = true)]
    verbose: bool,
}

/// Per-field statistics for the Bartol vs SPDF comparison.
#[derive(Serialize)]
struct CompRow {
    field: String,
    n_matched: usize,
    n_both_finite: usize,
    pearson_r: f64,
    rmse: f64,
    mean_bias_spdf_minus_bartol: f64,
    mean_spdf: f64,
    mean_bartol: f64,
    bartol_fill_fraction: f64,
    spdf_fill_fraction: f64,
    year_range: String,
    note: String,
}

struct Stats {
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
    sum_sq_diff: f64,
    sum_diff: f64,
    n: usize,
}

impl Stats {
    fn new() -> Self {
        Self {
            sum_x: 0.0,
            sum_y: 0.0,
            sum_xx: 0.0,
            sum_yy: 0.0,
            sum_xy: 0.0,
            sum_sq_diff: 0.0,
            sum_diff: 0.0,
            n: 0,
        }
    }

    fn push(&mut self, x: f64, y: f64) {
        self.sum_x += x;
        self.sum_y += y;
        self.sum_xx += x * x;
        self.sum_yy += y * y;
        self.sum_xy += x * y;
        let d = x - y;
        self.sum_sq_diff += d * d;
        self.sum_diff += d;
        self.n += 1;
    }

    fn pearson_r(&self) -> f64 {
        if self.n < 2 {
            return f64::NAN;
        }
        let n = self.n as f64;
        let cov = self.sum_xy / n - (self.sum_x / n) * (self.sum_y / n);
        let sx = (self.sum_xx / n - (self.sum_x / n).powi(2)).sqrt();
        let sy = (self.sum_yy / n - (self.sum_y / n).powi(2)).sqrt();
        if sx < 1e-15 || sy < 1e-15 {
            return f64::NAN;
        }
        cov / (sx * sy)
    }

    fn rmse(&self) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        (self.sum_sq_diff / self.n as f64).sqrt()
    }

    fn mean_diff(&self) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        self.sum_diff / self.n as f64
    }

    fn mean_x(&self) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        self.sum_x / self.n as f64
    }

    fn mean_y(&self) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        self.sum_y / self.n as f64
    }
}

fn build_key(year: u16, doy: u16, hour: u8) -> u64 {
    (year as u64) * 100_000 + (doy as u64) * 100 + (hour as u64)
}

fn run(cli: &Cli) -> Result<()> {
    // (key -> (b_mag, speed, density)) for both providers
    let mut spdf_map: BTreeMap<u64, (f64, f64, f64)> = BTreeMap::new();
    let mut bartol_map: BTreeMap<u64, (f64, f64, f64)> = BTreeMap::new();

    let mut spdf_total = 0usize;
    let mut bartol_total = 0usize;
    let mut spdf_b_finite = 0usize;
    let mut bartol_b_finite = 0usize;

    for year in cli.start_year..=cli.end_year {
        // SPDF
        let spdf_path = cli.spdf_dir.join(format!("vy2_{}.asc", year));
        if spdf_path.exists() {
            match parse_voyager_file(&spdf_path, VoyagerSpacecraft::V2) {
                Ok(recs) => {
                    for r in &recs {
                        spdf_total += 1;
                        let k = build_key(r.year, r.doy, r.hour);
                        let b = if r.b_magnitude.is_finite() {
                            spdf_b_finite += 1;
                            r.b_magnitude
                        } else {
                            f64::NAN
                        };
                        spdf_map.insert(k, (b, r.bulk_speed, r.proton_density));
                    }
                }
                Err(e) => {
                    eprintln!("SPDF parse error for {}: {}", year, e);
                }
            }
        } else {
            eprintln!("SPDF file missing: {}", spdf_path.display());
        }

        // Bartol (2-digit year)
        let yy = year % 100;
        let bartol_path = cli.bartol_dir.join(format!("vy2_{:02}.dat", yy));
        if bartol_path.exists() {
            match parse_bartol_file(&bartol_path) {
                Ok(recs) => {
                    for r in &recs {
                        bartol_total += 1;
                        let k = build_key(r.year, r.doy, r.hour);
                        let b = if r.b_magnitude.is_finite() {
                            bartol_b_finite += 1;
                            r.b_magnitude
                        } else {
                            f64::NAN
                        };
                        bartol_map.insert(k, (b, r.bulk_speed, r.proton_density));
                    }
                }
                Err(e) => {
                    eprintln!("Bartol parse error for {}: {}", year, e);
                }
            }
        } else {
            eprintln!("Bartol file missing: {}", bartol_path.display());
        }
    }

    if cli.verbose {
        eprintln!(
            "SPDF: {} total records, {} finite B ({:.1}%)",
            spdf_total,
            spdf_b_finite,
            100.0 * spdf_b_finite as f64 / spdf_total.max(1) as f64
        );
        eprintln!(
            "Bartol: {} total records, {} finite B ({:.1}%)",
            bartol_total,
            bartol_b_finite,
            100.0 * bartol_b_finite as f64 / bartol_total.max(1) as f64
        );
    }

    // Find matched keys
    let matched_keys: Vec<u64> = spdf_map.keys().filter(|k| bartol_map.contains_key(k)).copied().collect();
    let n_matched = matched_keys.len();

    let mut stats_b = Stats::new();
    let mut stats_speed = Stats::new();
    let mut stats_density = Stats::new();

    for k in &matched_keys {
        let (sb, ss, sd) = spdf_map[k];
        let (bb, bs, bd) = bartol_map[k];

        if sb.is_finite() && bb.is_finite() {
            stats_b.push(sb, bb);
        }
        if ss.is_finite() && bs.is_finite() {
            stats_speed.push(ss, bs);
        }
        if sd.is_finite() && bd.is_finite() && sd > 0.0 && bd > 0.0 {
            stats_density.push(sd, bd);
        }
    }

    if cli.verbose {
        eprintln!(
            "Matched hourly timestamps: {}, B pairs: {}, speed pairs: {}, density pairs: {}",
            n_matched,
            stats_b.n,
            stats_speed.n,
            stats_density.n,
        );
        eprintln!(
            "B-mag: r={:.4}, RMSE={:.4} nT, bias={:.4} nT",
            stats_b.pearson_r(),
            stats_b.rmse(),
            stats_b.mean_diff(),
        );
        eprintln!(
            "Speed: r={:.4}, RMSE={:.2} km/s, bias={:.2} km/s",
            stats_speed.pearson_r(),
            stats_speed.rmse(),
            stats_speed.mean_diff(),
        );
        eprintln!(
            "Density: r={:.4}, RMSE={:.5} cm-3, bias={:.5} cm-3",
            stats_density.pearson_r(),
            stats_density.rmse(),
            stats_density.mean_diff(),
        );
    }

    let year_range = format!("{}-{}", cli.start_year, cli.end_year);

    let rows = vec![
        CompRow {
            field: "b_magnitude_nT".to_string(),
            n_matched,
            n_both_finite: stats_b.n,
            pearson_r: stats_b.pearson_r(),
            rmse: stats_b.rmse(),
            mean_bias_spdf_minus_bartol: stats_b.mean_diff(),
            mean_spdf: stats_b.mean_x(),
            mean_bartol: stats_b.mean_y(),
            bartol_fill_fraction: 1.0 - bartol_b_finite as f64 / bartol_total.max(1) as f64,
            spdf_fill_fraction: 1.0 - spdf_b_finite as f64 / spdf_total.max(1) as f64,
            year_range: year_range.clone(),
            note: "SPDF is SE coords; Bartol is RTN. |B| is coord-invariant.".to_string(),
        },
        CompRow {
            field: "bulk_speed_km_s".to_string(),
            n_matched,
            n_both_finite: stats_speed.n,
            pearson_r: stats_speed.pearson_r(),
            rmse: stats_speed.rmse(),
            mean_bias_spdf_minus_bartol: stats_speed.mean_diff(),
            mean_spdf: stats_speed.mean_x(),
            mean_bartol: stats_speed.mean_y(),
            bartol_fill_fraction: f64::NAN,
            spdf_fill_fraction: f64::NAN,
            year_range: year_range.clone(),
            note: "Solar wind bulk speed; both sources use PLS/heliocentric frame.".to_string(),
        },
        CompRow {
            field: "proton_density_cm3".to_string(),
            n_matched,
            n_both_finite: stats_density.n,
            pearson_r: stats_density.pearson_r(),
            rmse: stats_density.rmse(),
            mean_bias_spdf_minus_bartol: stats_density.mean_diff(),
            mean_spdf: stats_density.mean_x(),
            mean_bartol: stats_density.mean_y(),
            bartol_fill_fraction: f64::NAN,
            spdf_fill_fraction: f64::NAN,
            year_range: year_range.clone(),
            note: "Proton density; Bartol F9.5 vs SPDF F9.5 format.".to_string(),
        },
    ];

    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent).context("create output dir")?;
    }

    let mut wtr = Writer::from_path(&cli.out).context("open output CSV")?;
    for row in &rows {
        wtr.serialize(row).context("serialize row")?;
    }
    wtr.flush().context("flush CSV")?;

    println!("Written to {}", cli.out.display());
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    run(&cli)
}
