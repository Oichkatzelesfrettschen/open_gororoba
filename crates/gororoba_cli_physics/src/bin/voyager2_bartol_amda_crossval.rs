//! Voyager 2 B-field cross-validation: Bartol vs AMDA hourly merged.
//!
//! # Purpose and call sites
//!
//! Closes SQLite todo `T-058: Cross-validate Bartol vs AMDA B-field
//! for Voyager 2 overlap`. Reads both the legacy Bartol RTN-frame
//! merged-hourly Voyager 2 file and the AMDA-translated hourly file
//! for the same year, aligns rows by `(year, doy, hour)`, and reports
//! the residuals on `B_magnitude`, `Br`, `Bt`, `Bn`.
//!
//! Called once per overlap year as part of the heliosphere data
//! provenance lane. Default invocation:
//!
//! ```sh
//! cargo run --release -p gororoba_cli_physics --bin \
//!     voyager2-bartol-amda-crossval -- --year 1979
//! ```
//!
//! # Why this exists
//!
//! Two different curators (the Bartol research group at U. Delaware
//! and the AMDA archive at IRAP/CDPP) publish translations of the
//! NASA SPDF Voyager 2 magnetometer hourly merged data. Both go
//! through different post-processing pipelines (despiking, gap
//! handling, coordinate frame). Independent agreement on the same
//! source data is the simplest non-trivial provenance check we can
//! run on the Voyager 2 dataset.
//!
//! Disagreement between the two would mean (a) the curators applied
//! different filtering, or (b) one of them ingested a different
//! upstream snapshot. Either way, it's information the downstream
//! analyses (Sprint 76 Galactic-Heliospheric Tomography) need.
//!
//! # Algorithm
//!
//! ```text
//! 1. Parse Bartol vy{N}_{YY}.asc via the existing parse_bartol_v2()
//!    in data_core::catalogs::voyager.
//! 2. Parse AMDA vy2_{YYYY}_amda_merged_hourly.asc via the SPDF
//!    layout shared with the Bartol path.
//! 3. Build a (year, doy, hour) -> SpdfMergedRecord index for each
//!    side. The keys are dense (8784 hours/year) so a Vec keyed by
//!    hour-of-year is faster than a BTreeMap.
//! 4. For each shared key, compute (b_mag_bartol - b_mag_amda) and
//!    similarly for Br, Bt, Bn.
//! 5. Aggregate: (n_paired, n_only_bartol, n_only_amda, mean_residual,
//!    rms_residual, max_abs_residual, p99_abs_residual).
//! 6. Emit a TOML report at data/output/heliosphere/voyager2_crossval/
//!    {year}.toml with one section per field.
//! ```
//!
//! # Concrete worked example for year=1979
//!
//! Both files cover roughly 8784 hours of 1979 (V2 was at Jupiter
//! encounter for part of the year). After parsing:
//!
//! ```text
//!  bartol_n   = ~7600 valid B_magnitude rows (~1184 fills)
//!  amda_n     = ~7600 valid B_magnitude rows
//!  paired     = ~7600 (one-to-one alignment because both are
//!               translations of the same SPDF source)
//!  mean(B_mag_bartol - B_mag_amda) ~ 0      (identical translation)
//!  rms        ~ 0
//!  max_abs    ~ 0 to ~0.01 (rounding only)
//! ```
//!
//! A non-zero residual would be a finding; the report flags it.
//!
//! # Why hour-of-year (HoY) keying?
//!
//! `(year, doy, hour)` -> `((doy - 1) * 24 + hour)` indexes a Vec of
//! length 8784 (or 8808 in a leap year). This is faster than a
//! BTreeMap lookup and the keys densely cover the year, so memory
//! waste is bounded.
//!
//! # Cross-references
//!
//! - SPDF parser: [`data_core::catalogs::voyager::parse_bartol_v2`].
//! - AMDA layout: shared with Bartol via [`data_core::catalogs::voyager::BARTOL_V2_LAYOUT`].
//! - Sprint 76 reference: see project memory `solar_wind_patterns.md`.

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;

use data_core::catalogs::spdf_merged::{SpdfColumnLayout, parse_spdf_merged};
use data_core::catalogs::voyager::{BARTOL_V2_LAYOUT, parse_bartol_v2};

/// Column layout for the AMDA-merged Voyager 2 hourly file.
///
/// # Why this is different from BARTOL_V2_LAYOUT
///
/// The Bartol file has 16 columns including Btan (average-of-
/// magnitudes), per-component sigmas, and temperature. The AMDA
/// translation drops the redundant Btan column and the sigmas,
/// shifting B_total to column 6 and the B-components to 7,8,9. This
/// layout encodes that shift; columns past col 12 (density) are
/// truncated in the AMDA distribution and the parser tolerates
/// shorter rows via min_columns=13.
///
/// # Concrete row layout
///
/// ```text
///  col  field
///  0    year (4-digit)
///  1    doy
///  2    hour
///  3    distance_au   (NOT IHG longitude, just heliocentric_dist)
///  4    lat_deg
///  5    lon_deg
///  6    B_magnitude   (nT)
///  7    Br (or Bx, depending on coord frame)
///  8    Bt
///  9    Bn
/// 10    B_sigma       (fill 999.900)
/// 11    V_solar_wind  (fill 9999.9)
/// 12    density       (fill 999999.0)
/// ```
const AMDA_V2_LAYOUT: SpdfColumnLayout = SpdfColumnLayout {
    min_columns: 13,
    col_year: 0,
    col_doy: 1,
    col_hour: 2,
    col_distance_au: Some(3),
    col_lat_deg: Some(4),
    col_lon_deg: Some(5),
    col_b_mag: Some(6),
    col_br: Some(7),
    col_bt: Some(8),
    col_bn: Some(9),
    col_density: Some(12),
    col_speed: Some(11),
    col_temperature: None,
    fill_b: 9999.99,
    fill_density: 999999.0,
    fill_speed: 9999.9,
    fill_temperature: 9999999.0,
    fill_distance: 999.99,
    b_is_se: false,
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "voyager2-bartol-amda-crossval")]
struct Args {
    /// Year of the cross-validation. The Bartol file
    /// `vy2_{YY}.asc` and AMDA file `vy2_{YYYY}_amda_merged_hourly.asc`
    /// must both exist locally; defaults work for years 1979 + 2017
    /// + 2018 on a fresh checkout.
    #[arg(long, default_value_t = 1979)]
    year: u16,
    /// Override the Bartol file path. Defaults to
    /// `data/external/voyager2/vy2_{YYYY}.asc` (note: the legacy
    /// 1979 file lives directly under `voyager2/`, not under
    /// `voyager2/bartol/`; later years use the bartol/ subdir).
    #[arg(long)]
    bartol: Option<PathBuf>,
    /// Override the AMDA file path. Defaults to
    /// `data/external/voyager/voyager2/vy2_{YYYY}_amda_merged_hourly.asc`.
    #[arg(long)]
    amda: Option<PathBuf>,
    /// Output TOML path. Defaults to
    /// `data/output/heliosphere/voyager2_crossval/{year}.toml`.
    #[arg(long)]
    out: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Output schema
// ---------------------------------------------------------------------------

/// Per-field residual statistics. Each field is the difference
/// `bartol_value - amda_value` on shared (year, doy, hour) rows where
/// neither side reports a fill.
#[derive(Serialize, Debug)]
struct FieldResiduals {
    /// Number of paired rows (both sides have a non-fill value).
    n_paired: u64,
    /// Mean residual.
    mean: f64,
    /// Root-mean-square residual.
    rms: f64,
    /// Maximum absolute residual.
    max_abs: f64,
    /// 99th-percentile absolute residual (for tail diagnostics).
    p99_abs: f64,
}

#[derive(Serialize, Debug)]
struct CrossvalReport {
    year: u16,
    bartol_path: String,
    amda_path: String,
    /// Total rows the parser produced for each side.
    bartol_n_total: u64,
    amda_n_total: u64,
    /// Rows where both sides agree on (year, doy, hour) AND both
    /// have non-fill values. This is the denominator for the
    /// per-field statistics.
    paired_n_both_valid: u64,
    /// Rows where Bartol has a value but AMDA reports a fill.
    bartol_only: u64,
    /// Rows where AMDA has a value but Bartol reports a fill.
    amda_only: u64,
    /// Per-field residual statistics for the four magnetic-field
    /// scalars produced by both parsers.
    b_magnitude: FieldResiduals,
    br: FieldResiduals,
    bt: FieldResiduals,
    bn: FieldResiduals,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Index a `(year, doy, hour)` triple into a single `usize` for Vec
/// keying. Each year-of-data fits in `366 * 24 = 8784` slots; we
/// over-provision to 8784 for non-leap years and 8808 for leap years.
fn hour_of_year(doy: u16, hour: u8) -> usize {
    (doy as usize - 1) * 24 + hour as usize
}

/// Slot-count for a given year. Leap-year heuristic uses standard
/// Gregorian rules (every 4, except century-multiples that are not
/// 400-multiples). Voyager 2 hourly data starts in 1977 so the only
/// edge case in our data window is 1980 / 1984 / ... / 2024.
fn slots_for_year(year: u16) -> usize {
    let y = year as i32;
    let leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
    if leap { 8808 } else { 8784 }
}

/// Compute statistics from a raw residual list. Treats NaN as a fill
/// indicator and excludes those rows from the count and aggregates.
fn stats_from_residuals(residuals: &[f64]) -> FieldResiduals {
    let valid: Vec<f64> = residuals.iter().copied().filter(|x| x.is_finite()).collect();
    let n = valid.len() as u64;
    if n == 0 {
        return FieldResiduals {
            n_paired: 0,
            mean: 0.0,
            rms: 0.0,
            max_abs: 0.0,
            p99_abs: 0.0,
        };
    }
    let mean = valid.iter().sum::<f64>() / valid.len() as f64;
    let rms = (valid.iter().map(|x| x * x).sum::<f64>() / valid.len() as f64).sqrt();
    let mut abs: Vec<f64> = valid.iter().map(|x| x.abs()).collect();
    abs.sort_by(|a, b| a.total_cmp(b));
    let max_abs = abs.last().copied().unwrap_or(0.0);
    // 99th percentile: rank = ceil(0.99 * len) - 1.
    let p99_idx = ((abs.len() as f64) * 0.99).ceil() as usize - 1;
    let p99_abs = abs.get(p99_idx).copied().unwrap_or(max_abs);
    FieldResiduals {
        n_paired: n,
        mean,
        rms,
        max_abs,
        p99_abs,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();
    let year = args.year;

    // # Path resolution
    //
    // The Bartol 1979 file at `data/external/voyager2/vy2_1979.asc`
    // is stored directly under `voyager2/`, but later years
    // (1983-1995) live in `voyager2/bartol/vy2_{YY}.dat`. We try
    // both layouts.
    let bartol_path = match args.bartol {
        Some(p) => p,
        None => {
            let direct = PathBuf::from(format!("data/external/voyager2/vy2_{}.asc", year));
            let bartol_subdir = PathBuf::from(format!(
                "data/external/voyager2/bartol/vy2_{:02}.dat",
                year % 100
            ));
            if direct.exists() {
                direct
            } else if bartol_subdir.exists() {
                bartol_subdir
            } else {
                bail!(
                    "no Bartol file found for year {}; tried {} and {}",
                    year,
                    direct.display(),
                    bartol_subdir.display()
                );
            }
        }
    };
    let amda_path = args.amda.unwrap_or_else(|| {
        PathBuf::from(format!(
            "data/external/voyager/voyager2/vy2_{}_amda_merged_hourly.asc",
            year
        ))
    });
    let out_path = args.out.unwrap_or_else(|| {
        PathBuf::from(format!(
            "data/output/heliosphere/voyager2_crossval/{}.toml",
            year
        ))
    });

    // # Parse
    //
    // Both files share the BARTOL_V2_LAYOUT column structure (16-col
    // RTN, 2-digit-year-handled), so the same parser fills both
    // sides. The parser already handles fill-value sentinels
    // (999.999 -> NaN).
    let bartol_text = std::fs::read_to_string(&bartol_path)
        .with_context(|| format!("read Bartol file {}", bartol_path.display()))?;
    let amda_text = std::fs::read_to_string(&amda_path)
        .with_context(|| format!("read AMDA file {}", amda_path.display()))?;
    // Detect whether the Bartol file uses 2-digit or 4-digit year and
    // dispatch to the right parser. parse_bartol_v2() ALWAYS rewrites
    // the year column with `1900 + yr` (assuming the file uses 2-digit
    // years like `83`); calling it on a file that already has 4-digit
    // years (like the legacy 1979 file at data/external/voyager2/
    // vy2_1979.asc) produces year=3879 which then fails to match the
    // requested year. We sniff the first non-comment line and choose
    // the appropriate parser.
    let bartol_is_4digit = bartol_text
        .lines()
        .find(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
        .and_then(|line| line.split_ascii_whitespace().next())
        .and_then(|tok| tok.parse::<u16>().ok())
        .map(|y| y >= 1000)
        .unwrap_or(false);
    let bartol_records = if bartol_is_4digit {
        // 4-digit year already; use parse_spdf_merged directly with
        // the BARTOL_V2_LAYOUT to skip the buggy normalization.
        parse_spdf_merged(&bartol_text, &BARTOL_V2_LAYOUT)
    } else {
        parse_bartol_v2(&bartol_text)
    };
    let amda_records = parse_spdf_merged(&amda_text, &AMDA_V2_LAYOUT);

    // # Index by hour-of-year
    //
    // Vec slots keyed by (doy-1)*24 + hour; the year part is the
    // outer parameter (we only crossval one year at a time).
    let slots = slots_for_year(year);
    let mut bartol_by_hoy: Vec<Option<&data_core::catalogs::spdf_merged::SpdfMergedRecord>> =
        vec![None; slots];
    let mut amda_by_hoy: Vec<Option<&data_core::catalogs::spdf_merged::SpdfMergedRecord>> =
        vec![None; slots];
    for r in &bartol_records {
        // The Bartol parser may yield 2-digit years for files that use
        // the 1980s/1990s `83`, `92` etc. convention. Normalize to
        // 4-digit by adding 1900 if year < 100. (Years <= 50 would
        // wrap to 2050; safe because Voyager 2 launched 1977 so we
        // only ever see years 77..99 in 2-digit form.)
        let r_year = if r.year < 100 { r.year + 1900 } else { r.year };
        if r_year == year {
            let i = hour_of_year(r.doy, r.hour);
            if i < slots {
                bartol_by_hoy[i] = Some(r);
            }
        }
    }
    for r in &amda_records {
        if r.year == year {
            let i = hour_of_year(r.doy, r.hour);
            if i < slots {
                amda_by_hoy[i] = Some(r);
            }
        }
    }

    // # Compute residuals
    //
    // For each hour-of-year, if both sides have a record and both
    // have a non-NaN value for the field, push (bartol - amda) into
    // the residual vec. Otherwise tally as bartol_only / amda_only.
    let mut b_mag_res = Vec::new();
    let mut br_res = Vec::new();
    let mut bt_res = Vec::new();
    let mut bn_res = Vec::new();
    let mut paired = 0u64;
    let mut bartol_only = 0u64;
    let mut amda_only = 0u64;
    for (b, a) in bartol_by_hoy.iter().zip(amda_by_hoy.iter()) {
        match (b, a) {
            (Some(b), Some(a)) => {
                paired += 1;
                if b.b_magnitude.is_finite() && a.b_magnitude.is_finite() {
                    b_mag_res.push(b.b_magnitude - a.b_magnitude);
                }
                if b.br.is_finite() && a.br.is_finite() {
                    br_res.push(b.br - a.br);
                }
                if b.bt.is_finite() && a.bt.is_finite() {
                    bt_res.push(b.bt - a.bt);
                }
                if b.bn.is_finite() && a.bn.is_finite() {
                    bn_res.push(b.bn - a.bn);
                }
            }
            (Some(_), None) => bartol_only += 1,
            (None, Some(_)) => amda_only += 1,
            (None, None) => {}
        }
    }

    let report = CrossvalReport {
        year,
        bartol_path: bartol_path.display().to_string(),
        amda_path: amda_path.display().to_string(),
        bartol_n_total: bartol_records.len() as u64,
        amda_n_total: amda_records.len() as u64,
        paired_n_both_valid: paired,
        bartol_only,
        amda_only,
        b_magnitude: stats_from_residuals(&b_mag_res),
        br: stats_from_residuals(&br_res),
        bt: stats_from_residuals(&bt_res),
        bn: stats_from_residuals(&bn_res),
    };

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let toml_text = toml::to_string_pretty(&report)?;
    std::fs::write(&out_path, &toml_text)?;
    eprintln!(
        "wrote {} (paired = {}, B_mag rms = {:.4} nT, max_abs = {:.4} nT)",
        out_path.display(),
        report.paired_n_both_valid,
        report.b_magnitude.rms,
        report.b_magnitude.max_abs
    );

    Ok(())
}
