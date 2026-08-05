//! Voyager 2 B-field cross-validation: Bartol vs AMDA hourly merged.
//!
//! # Purpose and call sites
//!
//! Reads a Bartol RTN-frame Voyager 2 file and either the repository's
//! translated hourly file or native AMDA HAPI data. It aligns rows by
//! `(year, doy, hour)` and reports finite-field residuals for `B_magnitude`,
//! `Br`, `Bt`, and `Bn`.
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
//! The legacy merged-file mode compares two curator translations of the
//! NASA SPDF Voyager 2 magnetometer hourly data. The native HAPI mode
//! compares the AMDA RTN source packet against the Bartol reference and
//! retains source coverage independently of the legacy translation.
//!
//! Disagreement between the two would mean (a) the curators applied
//! different filtering, or (b) one of them ingested a different
//! upstream snapshot. Either way, it's information the downstream
//! downstream heliosphere analyses need.
//!
//! # Algorithm
//!
//! ```text
//! 1. Parse Bartol vy{N}_{YY}.asc via the existing parse_bartol_v2()
//!    in data_core::catalogs::voyager.
//! 2. Parse the repository AMDA merged-hourly file with its SPDF layout, or
//!    parse native `vo2-mag-full` HAPI CSV with `--amda-hapi`.
//! 3. Build a (year, doy, hour) -> SpdfMergedRecord index for each
//!    side. The keys are dense (8784 hours/year) so a Vec keyed by
//!    hour-of-year is faster than a BTreeMap.
//! 4. For each shared key, compute (b_mag_bartol - b_mag_amda) and
//!    similarly for Br, Bt, Bn.
//! 5. Aggregate timestamp overlap and finite coverage for each field before
//!    computing residual statistics.
//! 6. Emit a versioned TOML report with one coverage and statistics section
//!    per field.
//! ```
//!
//! # Concrete worked example for year=1979
//!
//! The legacy 1979 characterization files cover roughly 8784 hours. After
//! parsing:
//!
//! ```text
//!  bartol_n   = ~7600 valid B_magnitude rows (~1184 fills)
//!  amda_n     = ~7600 valid B_magnitude rows
//!  paired     = ~7600 timestamp keys
//!  mean(B_mag_bartol - B_mag_amda) ~ 0      (legacy characterization)
//!  rms        ~ 0
//!  max_abs    ~ 0 to ~0.01 (rounding only)
//! ```
//!
//! A non-zero residual is a finding only when the relevant field has finite
//! values on both sides. A report with no finite common magnetic field is a
//! data-boundary result, not evidence of agreement.
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
//! - AMDA merged layout: `AMDA_V2_LAYOUT`; native HAPI schema: the
//!   `data_core::catalogs::voyager::Voyager2AmdaHapiRecord` contract.
//! - Source schema: AMDA `vo2-mag-full` HAPI metadata and SPASE RTN contract.

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

use data_core::catalogs::{
    spdf_merged::{SpdfColumnLayout, parse_spdf_merged},
    voyager::{
        BARTOL_V2_LAYOUT, VOYAGER2_AMDA_HAPI_DATASET, parse_bartol_v2,
        parse_voyager2_amda_hapi_detailed,
    },
};

const VOYAGER2_AMDA_HAPI_INFO_SHA256: &str =
    "293f28a07f74a23c2bccc7214a01e4db0327c3f085173fff3c527ca032fa5f9a";

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
    /// Interpret the AMDA input as native `vo2-mag-full` HAPI CSV rather than
    /// the repository's derived merged-hourly ASCII format.
    #[arg(long)]
    amda_hapi: bool,
    /// Metadata JSON captured from the native `vo2-mag-full` HAPI endpoint.
    /// Native input requires the pinned source metadata hash.
    #[arg(long)]
    amda_info: Option<PathBuf>,
    /// Output TOML path. Defaults to
    /// `data/output/heliosphere/voyager2_crossval/{year}.toml`.
    #[arg(long)]
    out: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Output schema
// ---------------------------------------------------------------------------

/// Finite per-field comparison statistics.
#[derive(Serialize, Debug)]
struct ResidualStatistics {
    mean: f64,
    rms: f64,
    max_abs: f64,
    p99_abs: f64,
}

/// Per-field coverage and residual statistics. Missing finite overlap is
/// represented by `statistics = None`, never by an all-zero statistic.
#[derive(Serialize, Debug)]
struct FieldResiduals {
    /// Number of shared timestamps where both sides have a finite value.
    n_paired: u64,
    /// Number of shared timestamps with only a finite Bartol value.
    bartol_finite_only: u64,
    /// Number of shared timestamps with only a finite AMDA value.
    amda_finite_only: u64,
    /// Number of shared timestamps where both values are non-finite.
    both_nonfinite: u64,
    /// Finite Bartol values on timestamp keys absent from AMDA.
    bartol_finite_timestamp_only: u64,
    /// Finite AMDA values on timestamp keys absent from Bartol.
    amda_finite_timestamp_only: u64,
    /// `finite_overlap` or `no_finite_overlap`.
    availability: String,
    statistics: Option<ResidualStatistics>,
}

#[derive(Default)]
struct FieldSamples {
    residuals: Vec<f64>,
    bartol_finite_only: u64,
    amda_finite_only: u64,
    both_nonfinite: u64,
    bartol_finite_timestamp_only: u64,
    amda_finite_timestamp_only: u64,
}

struct ParsedAmdaInput {
    records: Vec<data_core::catalogs::spdf_merged::SpdfMergedRecord>,
    schema_validation: String,
    data_rows: u64,
    accepted_rows: u64,
    rejected_rows: u64,
}

#[derive(Debug)]
struct ValidatedAmdaInfo {
    sha256: String,
    schema_validation: String,
}

#[derive(Serialize, Debug)]
struct CrossvalReport {
    year: u16,
    bartol_path: String,
    amda_path: String,
    amda_input_format: String,
    amda_info_path: Option<String>,
    amda_info_sha256: Option<String>,
    amda_info_schema_validation: Option<String>,
    bartol_sha256: String,
    amda_sha256: String,
    amda_schema_validation: String,
    amda_data_rows: u64,
    amda_accepted_rows: u64,
    amda_rejected_rows: u64,
    report_schema: String,
    /// Total rows the parser produced for each side.
    bartol_n_total: u64,
    amda_n_total: u64,
    /// Rows where both sides have the same (year, doy, hour) key.
    /// Per-field finite overlap is reported by each FieldResiduals::n_paired.
    paired_n_same_timestamp: u64,
    /// Disposition that prevents zero residuals from being read as agreement
    /// when no finite reference field exists.
    comparison_disposition: String,
    /// Timestamp keys present only in the Bartol input.
    bartol_only_timestamp: u64,
    /// Timestamp keys present only in the AMDA input.
    amda_only_timestamp: u64,
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

/// Slot-count for a given year. Leap-year heuristic uses standard
/// Gregorian rules (every 4, except century-multiples that are not
/// 400-multiples). Voyager 2 hourly data starts in 1977 so the only
/// edge case in our data window is 1980 / 1984 / ... / 2024.
fn slots_for_year(year: u16) -> usize {
    let y = year as i32;
    let leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
    if leap { 8808 } else { 8784 }
}

fn normalized_year(year: u16) -> u16 {
    if year < 100 { year + 1900 } else { year }
}

/// Validate and index a `(year, doy, hour)` triple for Vec keying.
fn hour_of_year(year: u16, doy: u16, hour: u8) -> Result<usize> {
    let max_doy = if slots_for_year(year) == 8808 {
        366
    } else {
        365
    };
    if doy == 0 || doy > max_doy || hour >= 24 {
        bail!("invalid Voyager timestamp key: year={year}, doy={doy}, hour={hour}");
    }
    Ok((doy as usize - 1) * 24 + hour as usize)
}

fn collect_field(samples: &mut FieldSamples, bartol: f64, amda: f64) -> Result<()> {
    match (bartol.is_finite(), amda.is_finite()) {
        (true, true) => {
            let residual = bartol - amda;
            if !residual.is_finite() {
                bail!("non-finite Voyager field residual");
            }
            samples.residuals.push(residual);
        }
        (true, false) => samples.bartol_finite_only += 1,
        (false, true) => samples.amda_finite_only += 1,
        (false, false) => samples.both_nonfinite += 1,
    }
    Ok(())
}

fn collect_timestamp_only(samples: &mut FieldSamples, value: f64, bartol_source: bool) {
    if !value.is_finite() {
        return;
    }
    if bartol_source {
        samples.bartol_finite_timestamp_only += 1;
    } else {
        samples.amda_finite_timestamp_only += 1;
    }
}

/// Compute statistics from finite residuals and retain field-level coverage.
fn stats_from_samples(samples: FieldSamples) -> Result<FieldResiduals> {
    let n = samples.residuals.len() as u64;
    let statistics = if n == 0 {
        None
    } else {
        let sum = samples
            .residuals
            .iter()
            .try_fold(0.0, |sum, value| {
                let next = sum + value;
                next.is_finite().then_some(next)
            })
            .ok_or_else(|| anyhow::anyhow!("Voyager residual sum overflowed"))?;
        let sum_squares = samples
            .residuals
            .iter()
            .try_fold(0.0, |sum, value| {
                let next = sum + value * value;
                next.is_finite().then_some(next)
            })
            .ok_or_else(|| anyhow::anyhow!("Voyager residual square sum overflowed"))?;
        let mean = sum / n as f64;
        let rms = (sum_squares / n as f64).sqrt();
        if !mean.is_finite() || !rms.is_finite() {
            bail!("Voyager residual statistics are non-finite");
        }
        let mut abs: Vec<f64> = samples.residuals.iter().map(|x| x.abs()).collect();
        abs.sort_by(|a, b| a.total_cmp(b));
        let max_abs = abs.last().copied().unwrap_or(0.0);
        let p99_idx = ((abs.len() as f64) * 0.99).ceil() as usize - 1;
        let p99_abs = abs.get(p99_idx).copied().unwrap_or(max_abs);
        Some(ResidualStatistics {
            mean,
            rms,
            max_abs,
            p99_abs,
        })
    };
    let availability = if n == 0 {
        "no_finite_overlap"
    } else {
        "finite_overlap"
    };
    Ok(FieldResiduals {
        n_paired: n,
        bartol_finite_only: samples.bartol_finite_only,
        amda_finite_only: samples.amda_finite_only,
        both_nonfinite: samples.both_nonfinite,
        bartol_finite_timestamp_only: samples.bartol_finite_timestamp_only,
        amda_finite_timestamp_only: samples.amda_finite_timestamp_only,
        availability: availability.to_string(),
        statistics,
    })
}

fn insert_record<'a>(
    slots: &mut [Option<&'a data_core::catalogs::spdf_merged::SpdfMergedRecord>],
    record: &'a data_core::catalogs::spdf_merged::SpdfMergedRecord,
    year: u16,
    source: &str,
) -> Result<()> {
    let index = hour_of_year(year, record.doy, record.hour)?;
    if index >= slots.len() {
        return Ok(());
    }
    if slots[index].is_some() {
        bail!(
            "duplicate {source} timestamp key at year={year}, doy={}, hour={}",
            record.doy,
            record.hour
        );
    }
    slots[index] = Some(record);
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn validate_amda_info_document(metadata: &Value) -> Result<()> {
    if metadata.get("HAPI").and_then(Value::as_str) != Some("2.0") {
        bail!("AMDA metadata does not declare HAPI version 2.0");
    }
    let expected_resource_id =
        format!("spase://CNES/NumericalData/CDPP-AMDA/Voyager2/MAG/{VOYAGER2_AMDA_HAPI_DATASET}");
    if metadata.get("resourceID").and_then(Value::as_str) != Some(expected_resource_id.as_str()) {
        bail!("AMDA metadata resourceID is not {VOYAGER2_AMDA_HAPI_DATASET}");
    }
    let parameters = metadata
        .get("parameters")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("AMDA metadata has no parameter array"))?;
    if parameters.len() != 5 {
        bail!(
            "AMDA metadata parameter count is {}, expected 5",
            parameters.len()
        );
    }

    let expected_names = [
        "Time",
        "vo2_b_full",
        "vo2_bmag_full",
        "vo2_b_full_phi",
        "vo2_b_full_theta",
    ];
    for (parameter, expected_name) in parameters.iter().zip(expected_names) {
        if parameter.get("name").and_then(Value::as_str) != Some(expected_name) {
            bail!("AMDA metadata parameter name does not match {expected_name}");
        }
    }

    let time = &parameters[0];
    if time.get("type").and_then(Value::as_str) != Some("isotime")
        || time.get("units").and_then(Value::as_str) != Some("UTC")
        || !time.get("fill").is_some_and(Value::is_null)
    {
        bail!("AMDA metadata Time parameter does not match the pinned schema");
    }

    let vector = &parameters[1];
    let vector_size_ok = vector
        .get("size")
        .and_then(Value::as_array)
        .is_some_and(|size| size.len() == 1 && size[0].as_u64() == Some(3));
    if vector.get("type").and_then(Value::as_str) != Some("double")
        || vector.get("units").and_then(Value::as_str) != Some("nT")
        || !vector_size_ok
        || vector.get("fill").and_then(Value::as_str) != Some("-1e31")
    {
        bail!("AMDA metadata vector parameter does not match the pinned schema");
    }

    for parameter in &parameters[2..] {
        if parameter.get("type").and_then(Value::as_str) != Some("double")
            || parameter.get("fill").and_then(Value::as_str) != Some("-1e31")
        {
            bail!("AMDA metadata scalar parameter does not match the pinned schema");
        }
    }
    if parameters[2].get("units").and_then(Value::as_str) != Some("nT")
        || parameters[3].get("units").and_then(Value::as_str) != Some("degrees")
        || parameters[4].get("units").and_then(Value::as_str) != Some("degrees")
    {
        bail!("AMDA metadata units do not match the pinned schema");
    }
    Ok(())
}

fn validate_amda_info_bytes(contents: &[u8], expected_sha256: &str) -> Result<ValidatedAmdaInfo> {
    let digest = sha256_hex(contents);
    if digest != expected_sha256 {
        bail!(
            "AMDA metadata hash mismatch: expected {}, got {}",
            expected_sha256,
            digest
        );
    }
    let metadata: Value =
        serde_json::from_slice(contents).context("parse AMDA metadata as JSON")?;
    validate_amda_info_document(&metadata)?;
    Ok(ValidatedAmdaInfo {
        sha256: digest,
        schema_validation: "validated_vo2_mag_full_resource_and_parameter_schema".to_string(),
    })
}

fn validate_amda_info(path: &Path) -> Result<ValidatedAmdaInfo> {
    let contents =
        std::fs::read(path).with_context(|| format!("read AMDA metadata {}", path.display()))?;
    validate_amda_info_bytes(&contents, VOYAGER2_AMDA_HAPI_INFO_SHA256)
}

fn parse_amda_input(amda_text: &str, amda_hapi: bool) -> Result<ParsedAmdaInput> {
    if !amda_hapi {
        return Ok(ParsedAmdaInput {
            records: parse_spdf_merged(amda_text, &AMDA_V2_LAYOUT),
            schema_validation: "repository_amda_merged_hourly_layout".to_string(),
            data_rows: 0,
            accepted_rows: 0,
            rejected_rows: 0,
        });
    }

    let parsed = parse_voyager2_amda_hapi_detailed(amda_text);
    if parsed.header_present && !parsed.header_matches {
        bail!("AMDA HAPI header does not match the vo2-mag-full schema");
    }
    if parsed.rejected_rows != 0 {
        bail!(
            "AMDA HAPI input contains {} rejected rows out of {} data rows",
            parsed.rejected_rows,
            parsed.data_rows
        );
    }
    if parsed.data_rows == 0 || parsed.accepted_rows == 0 {
        bail!("AMDA HAPI input contains no accepted data rows");
    }
    let data_rows = parsed.data_rows;
    let accepted_rows = parsed.accepted_rows;
    let rejected_rows = parsed.rejected_rows;
    let schema_validation = if parsed.header_present {
        "validated_vo2_mag_full_header_exact_7_columns"
    } else {
        "validated_vo2_mag_full_dataset_id_and_pinned_metadata_exact_7_columns_without_header"
    };
    let records = parsed
        .records
        .into_iter()
        .map(|record| record.into_spdf_merged())
        .collect();
    Ok(ParsedAmdaInput {
        records,
        schema_validation: schema_validation.to_string(),
        data_rows,
        accepted_rows,
        rejected_rows,
    })
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
    let amda_hapi = args.amda_hapi;
    let amda_info_path = match (amda_hapi, args.amda_info) {
        (true, Some(path)) => Some(path),
        (true, None) => bail!("--amda-info is required with --amda-hapi"),
        (false, Some(_)) => bail!("--amda-info requires --amda-hapi"),
        (false, None) => None,
    };
    let validated_amda_info = amda_info_path
        .as_deref()
        .map(validate_amda_info)
        .transpose()?;
    let amda_path = args.amda.unwrap_or_else(|| {
        if amda_hapi {
            PathBuf::from(format!(
                "data/external/voyager/voyager2/vy2_{}_amda_hapi.csv",
                year
            ))
        } else {
            PathBuf::from(format!(
                "data/external/voyager/voyager2/vy2_{}_amda_merged_hourly.asc",
                year
            ))
        }
    });
    let out_path = args.out.unwrap_or_else(|| {
        PathBuf::from(format!(
            "data/output/heliosphere/voyager2_crossval/{}.toml",
            year
        ))
    });

    // # Parse
    //
    // The Bartol side uses the archived merged-record layout. The AMDA side
    // uses either the repository merged layout or the native HAPI RTN schema.
    // Each parser maps source fill values to NaN before alignment.
    let bartol_text = std::fs::read_to_string(&bartol_path)
        .with_context(|| format!("read Bartol file {}", bartol_path.display()))?;
    let amda_text = std::fs::read_to_string(&amda_path)
        .with_context(|| format!("read AMDA file {}", amda_path.display()))?;
    let bartol_sha256 = sha256_hex(bartol_text.as_bytes());
    let amda_sha256 = sha256_hex(amda_text.as_bytes());
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
    let parsed_amda = parse_amda_input(&amda_text, amda_hapi)?;
    let amda_records = parsed_amda.records;
    let amda_schema_validation = parsed_amda.schema_validation;
    let amda_data_rows = parsed_amda.data_rows;
    let amda_accepted_rows = parsed_amda.accepted_rows;
    let amda_rejected_rows = parsed_amda.rejected_rows;
    let bartol_n_total = bartol_records
        .iter()
        .filter(|record| normalized_year(record.year) == year)
        .count() as u64;
    let amda_n_total = amda_records
        .iter()
        .filter(|record| normalized_year(record.year) == year)
        .count() as u64;

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
        let r_year = normalized_year(r.year);
        if r_year == year {
            insert_record(&mut bartol_by_hoy, r, year, "Bartol")?;
        }
    }
    for r in &amda_records {
        if r.year == year {
            insert_record(&mut amda_by_hoy, r, year, "AMDA")?;
        }
    }

    // # Compute residuals
    //
    // For each shared hour, record finite coverage per field before adding
    // the Bartol-minus-AMDA residual. Keep timestamp-only gaps separate from
    // field-level source and reference coverage.
    let mut b_mag_samples = FieldSamples::default();
    let mut br_samples = FieldSamples::default();
    let mut bt_samples = FieldSamples::default();
    let mut bn_samples = FieldSamples::default();
    let mut paired = 0u64;
    let mut bartol_only_timestamp = 0u64;
    let mut amda_only_timestamp = 0u64;
    for (b, a) in bartol_by_hoy.iter().zip(amda_by_hoy.iter()) {
        match (b, a) {
            (Some(b), Some(a)) => {
                paired += 1;
                collect_field(&mut b_mag_samples, b.b_magnitude, a.b_magnitude)?;
                collect_field(&mut br_samples, b.br, a.br)?;
                collect_field(&mut bt_samples, b.bt, a.bt)?;
                collect_field(&mut bn_samples, b.bn, a.bn)?;
            }
            (Some(b), None) => {
                bartol_only_timestamp += 1;
                collect_timestamp_only(&mut b_mag_samples, b.b_magnitude, true);
                collect_timestamp_only(&mut br_samples, b.br, true);
                collect_timestamp_only(&mut bt_samples, b.bt, true);
                collect_timestamp_only(&mut bn_samples, b.bn, true);
            }
            (None, Some(a)) => {
                amda_only_timestamp += 1;
                collect_timestamp_only(&mut b_mag_samples, a.b_magnitude, false);
                collect_timestamp_only(&mut br_samples, a.br, false);
                collect_timestamp_only(&mut bt_samples, a.bt, false);
                collect_timestamp_only(&mut bn_samples, a.bn, false);
            }
            (None, None) => {}
        }
    }

    let report = CrossvalReport {
        year,
        bartol_path: bartol_path.display().to_string(),
        amda_path: amda_path.display().to_string(),
        amda_input_format: if amda_hapi {
            "native_amda_vo2_mag_full_hapi_csv".to_string()
        } else {
            "repository_amda_merged_hourly_ascii".to_string()
        },
        amda_info_path: amda_info_path.map(|path| path.display().to_string()),
        amda_info_sha256: validated_amda_info.as_ref().map(|info| info.sha256.clone()),
        amda_info_schema_validation: validated_amda_info
            .as_ref()
            .map(|info| info.schema_validation.clone()),
        bartol_sha256,
        amda_sha256,
        amda_schema_validation,
        amda_data_rows,
        amda_accepted_rows,
        amda_rejected_rows,
        report_schema: "voyager2_bartol_amda_crossval.v2".to_string(),
        bartol_n_total,
        amda_n_total,
        paired_n_same_timestamp: paired,
        comparison_disposition: if b_mag_samples.residuals.is_empty()
            && br_samples.residuals.is_empty()
            && bt_samples.residuals.is_empty()
            && bn_samples.residuals.is_empty()
        {
            "no_finite_common_b_field".to_string()
        } else {
            "finite_component_comparison".to_string()
        },
        bartol_only_timestamp,
        amda_only_timestamp,
        b_magnitude: stats_from_samples(b_mag_samples)?,
        br: stats_from_samples(br_samples)?,
        bt: stats_from_samples(bt_samples)?,
        bn: stats_from_samples(bn_samples)?,
    };

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let toml_text = toml::to_string_pretty(&report)?;
    std::fs::write(&out_path, &toml_text)?;
    eprintln!(
        "wrote {} (same_timestamp = {}, disposition = {}, B_mag finite_overlap = {})",
        out_path.display(),
        report.paired_n_same_timestamp,
        report.comparison_disposition,
        report.b_magnitude.n_paired
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        FieldSamples, collect_field, collect_timestamp_only, hour_of_year, insert_record,
        parse_amda_input, sha256_hex, stats_from_samples, validate_amda_info_bytes,
        validate_amda_info_document,
    };

    #[test]
    fn leap_day_and_hour_boundaries_are_checked() {
        assert_eq!(hour_of_year(1992, 366, 23).expect("leap day"), 8783);
        assert!(hour_of_year(1991, 366, 0).is_err());
        assert!(hour_of_year(1992, 0, 0).is_err());
        assert!(hour_of_year(1992, 1, 24).is_err());
    }

    #[test]
    fn missing_finite_overlap_has_no_numeric_statistics() {
        let mut samples = FieldSamples::default();
        collect_field(&mut samples, f64::NAN, 1.0).expect("finite-only sample");
        collect_field(&mut samples, f64::NAN, f64::NAN).expect("empty sample");
        collect_timestamp_only(&mut samples, 2.0, true);
        collect_timestamp_only(&mut samples, 3.0, false);
        let report = stats_from_samples(samples).expect("missing-overlap report");
        assert_eq!(report.n_paired, 0);
        assert_eq!(report.amda_finite_only, 1);
        assert_eq!(report.both_nonfinite, 1);
        assert_eq!(report.bartol_finite_timestamp_only, 1);
        assert_eq!(report.amda_finite_timestamp_only, 1);
        assert_eq!(report.availability, "no_finite_overlap");
        assert!(report.statistics.is_none());
    }

    #[test]
    fn finite_overlap_retains_residual_statistics() {
        let mut samples = FieldSamples::default();
        collect_field(&mut samples, 3.0, 1.0).expect("finite sample");
        collect_field(&mut samples, 5.0, 1.0).expect("finite sample");
        let report = stats_from_samples(samples).expect("finite-overlap report");
        let statistics = report.statistics.expect("finite statistics");
        assert_eq!(report.n_paired, 2);
        assert_eq!(report.availability, "finite_overlap");
        assert!((statistics.mean - 3.0).abs() < 1.0e-12);
        assert!((statistics.rms - (10.0_f64).sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn nonfinite_residuals_are_rejected() {
        let mut samples = FieldSamples::default();
        assert!(collect_field(&mut samples, f64::MAX, -f64::MAX).is_err());
    }

    #[test]
    fn residual_statistics_reject_sum_overflow() {
        let mut samples = FieldSamples::default();
        collect_field(&mut samples, f64::MAX / 2.0, 0.0).expect("finite sample");
        collect_field(&mut samples, f64::MAX / 2.0, 0.0).expect("finite sample");
        assert!(stats_from_samples(samples).is_err());
    }

    #[test]
    fn native_input_gate_rejects_wrong_header_and_empty_data() {
        let wrong_header = "Time,wrong_r,wrong_t,wrong_n,bmag,phi,theta\n\
                             1990-01-03T18:00:00.000Z,1,2,3,9,10,11\n";
        assert!(parse_amda_input(wrong_header, true).is_err());
        assert!(parse_amda_input("", true).is_err());
    }

    #[test]
    fn native_input_gate_rejects_malformed_rows() {
        let malformed = "1990-01-03T18:00:00.000Z,1,2\n";
        assert!(parse_amda_input(malformed, true).is_err());
    }

    #[test]
    fn native_metadata_gate_requires_dataset_identity_and_parameter_schema() {
        let valid = serde_json::json!({
            "HAPI": "2.0",
            "resourceID": "spase://CNES/NumericalData/CDPP-AMDA/Voyager2/MAG/vo2-mag-full",
            "parameters": [
                {"name": "Time", "type": "isotime", "units": "UTC", "fill": null},
                {"name": "vo2_b_full", "type": "double", "size": [3], "units": "nT", "fill": "-1e31"},
                {"name": "vo2_bmag_full", "type": "double", "units": "nT", "fill": "-1e31"},
                {"name": "vo2_b_full_phi", "type": "double", "units": "degrees", "fill": "-1e31"},
                {"name": "vo2_b_full_theta", "type": "double", "units": "degrees", "fill": "-1e31"}
            ]
        });
        validate_amda_info_document(&valid).expect("pinned AMDA metadata schema");

        let mut wrong_identity = valid.clone();
        wrong_identity["resourceID"] = serde_json::Value::String("wrong-dataset".to_string());
        assert!(validate_amda_info_document(&wrong_identity).is_err());

        let mut wrong_columns = valid;
        wrong_columns["parameters"][1]["name"] =
            serde_json::Value::String("wrong-field".to_string());
        assert!(validate_amda_info_document(&wrong_columns).is_err());

        let contents = serde_json::to_vec(&wrong_columns).expect("metadata JSON");
        let digest = sha256_hex(&contents);
        validate_amda_info_bytes(&contents, &digest).expect_err("wrong schema must fail");

        let valid_contents = serde_json::to_vec(&serde_json::json!({
            "HAPI": "2.0",
            "resourceID": "spase://CNES/NumericalData/CDPP-AMDA/Voyager2/MAG/vo2-mag-full",
            "parameters": [
                {"name": "Time", "type": "isotime", "units": "UTC", "fill": null},
                {"name": "vo2_b_full", "type": "double", "size": [3], "units": "nT", "fill": "-1e31"},
                {"name": "vo2_bmag_full", "type": "double", "units": "nT", "fill": "-1e31"},
                {"name": "vo2_b_full_phi", "type": "double", "units": "degrees", "fill": "-1e31"},
                {"name": "vo2_b_full_theta", "type": "double", "units": "degrees", "fill": "-1e31"}
            ]
        }))
        .expect("metadata JSON");
        let valid_digest = sha256_hex(&valid_contents);
        let validated = validate_amda_info_bytes(&valid_contents, &valid_digest)
            .expect("valid metadata hash and schema");
        assert_eq!(validated.sha256, valid_digest);
        assert!(validate_amda_info_bytes(&valid_contents, "wrong-hash").is_err());
    }

    #[test]
    fn duplicate_timestamp_slots_are_rejected() {
        let first = data_core::catalogs::spdf_merged::SpdfMergedRecord {
            year: 1990,
            doy: 1,
            hour: 0,
            distance_au: f64::NAN,
            lat_deg: f64::NAN,
            lon_deg: f64::NAN,
            b_magnitude: f64::NAN,
            br: f64::NAN,
            bt: f64::NAN,
            bn: f64::NAN,
            proton_density: f64::NAN,
            bulk_speed: f64::NAN,
            proton_temperature: f64::NAN,
        };
        let second = first.clone();
        let mut slots: Vec<Option<&data_core::catalogs::spdf_merged::SpdfMergedRecord>> =
            vec![None; 8784];
        insert_record(&mut slots, &first, 1990, "Bartol").expect("first timestamp");
        assert!(insert_record(&mut slots, &second, 1990, "Bartol").is_err());
    }
}
