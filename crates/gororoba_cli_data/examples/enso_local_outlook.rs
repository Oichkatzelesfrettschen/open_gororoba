//! Build a provenance-rich ENSO outlook for one local forecast target.
//!
//! The official CPC and NWS values remain operational authority.  The optional
//! Cayley-Dickson calculation measures curvature across successive forecast
//! issue vectors; it is never promoted into a rainfall forecast by this tool.

use anyhow::{Context, Result, ensure};
use chrono::{DateTime, FixedOffset, NaiveDate};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

const INPUT_SCHEMA_VERSION: u32 = 2;
const REPORT_SCHEMA_VERSION: u32 = 1;
const CD_DIM: usize = 16;
const REQUIRED_SEASONS: [&str; 4] = ["SON", "OND", "DJF", "JFM"];

#[derive(Parser, Debug)]
#[command(
    name = "enso-local-outlook",
    about = "Build a local ENSO outlook plus a non-operational 16D forecast-state diagnostic"
)]
struct Cli {
    /// Curated TOML snapshot of official ENSO, seasonal, and local forecast data.
    #[arg(long)]
    input: PathBuf,

    /// Output TOML report.
    #[arg(long)]
    out: PathBuf,

    /// Internal Cayley-Dickson arithmetic precision.
    #[arg(long, default_value = "f64", value_parser = ["f32", "f64"])]
    precision: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Snapshot {
    schema_version: u32,
    instrument_id: String,
    as_of: String,
    location: Location,
    source: Vec<SourceRecord>,
    current_weather: CurrentWeather,
    issue: Vec<ForecastIssue>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Location {
    postal_code: String,
    label: String,
    latitude_deg: f64,
    longitude_deg: f64,
    climate_division_id: u16,
    climate_division_name: String,
    normals_station_id: String,
    normals_period: String,
    annual_normal_precip_inches: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SourceRecord {
    id: String,
    authority: String,
    product: String,
    issued_on: String,
    retrieved_on: String,
    url: String,
    notes: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CurrentWeather {
    source_id: String,
    valid_from: String,
    valid_through: String,
    summary: String,
    high_f_min: f64,
    high_f_max: f64,
    low_f_min: f64,
    low_f_max: f64,
    dry_through: String,
    weekend_precip_probability_max_pct: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ForecastIssue {
    issued_on: String,
    seasonal_outlook_issued_on: String,
    next_enso_update: String,
    next_seasonal_update: String,
    very_strong_probability_relation: String,
    very_strong_probability_pct: f64,
    ond_roni_ge_2_5_probability_pct: f64,
    enso_source_id: String,
    seasonal_source_id: String,
    lead: Vec<LeadSeason>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct LeadSeason {
    season: String,
    el_nino_probability_pct: f64,
    roni_p05_c: f64,
    roni_median_c: f64,
    roni_p95_c: f64,
    local_precip_category: String,
    local_precip_probability_floor_pct: f64,
    local_precip_probability_ceiling_pct: Option<f64>,
    probability_basis: String,
    normal_precip_inches: f64,
    local_interpretation: String,
    el_nino_source_id: String,
    roni_source_id: String,
    local_precip_source_id: String,
    normals_source_id: String,
}

#[derive(Debug, Serialize)]
struct Report {
    report_schema_version: u32,
    source_snapshot_sha256: String,
    snapshot: Snapshot,
    derived: DerivedOutlook,
}

#[derive(Debug, Serialize)]
struct DerivedOutlook {
    official_enso: EnsoSummary,
    seasonal_outlook: Vec<SeasonSummary>,
    cd_diagnostic: CdDiagnostic,
    operational_summary: Vec<String>,
    caveat: Vec<String>,
}

#[derive(Debug, Serialize)]
struct EnsoSummary {
    enso_issue: String,
    seasonal_issue: String,
    very_strong_probability_relation: String,
    very_strong_probability_pct: f64,
    ond_roni_ge_2_5_probability_pct: f64,
    peak_roni_season: String,
    peak_roni_median_c: f64,
    peak_roni_p05_c: f64,
    peak_roni_p95_c: f64,
    next_enso_update: String,
    next_seasonal_update: String,
}

#[derive(Debug, Serialize)]
struct SeasonSummary {
    season: String,
    el_nino_probability_pct: f64,
    roni_p05_c: f64,
    roni_median_c: f64,
    roni_p95_c: f64,
    normal_precip_inches: f64,
    local_precip_category: String,
    local_precip_probability_floor_pct: f64,
    local_precip_probability_ceiling_pct: Option<f64>,
    local_wet_signal: String,
    probability_basis: String,
    local_interpretation: String,
}

#[derive(Debug, Serialize)]
struct CdDiagnostic {
    status: String,
    diagnostic_id: String,
    dimension: usize,
    issue_count_available: usize,
    issue_count_required: usize,
    lift_depth_from_reals: u8,
    season_order: Vec<String>,
    feature_order_within_season: Vec<String>,
    associator_norms: Vec<f64>,
    latest_associator_norm: Option<f64>,
    mean_associator_norm: Option<f64>,
    mean_l2_issue_step: Option<f64>,
    calibrated_for_target: bool,
    used_for_operational_forecast: bool,
    interpretation: String,
    next_decisive_test: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure_output_is_not_input(&cli.input, &cli.out)?;
    let raw = fs::read_to_string(&cli.input)
        .with_context(|| format!("read {}", cli.input.display()))?;
    let snapshot: Snapshot =
        toml::from_str(&raw).with_context(|| format!("parse {}", cli.input.display()))?;
    validate_snapshot(&snapshot)?;

    let report = build_report(snapshot, sha256_hex(raw.as_bytes()), &cli.precision)?;
    let text = toml::to_string_pretty(&report).context("serialize ENSO local outlook")?;
    if let Some(parent) = cli.out.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    ensure_output_is_not_input(&cli.input, &cli.out)?;
    fs::write(&cli.out, text).with_context(|| format!("write {}", cli.out.display()))?;
    println!("{}", cli.out.display());
    Ok(())
}

/// Resolve both paths through the filesystem so a symlink or a relative spelling
/// cannot smuggle the curated snapshot in as the report target.
/// Writing the report over the input destroys the source artifact and leaves the
/// next run unable to deserialize it as `Snapshot`, so the aliasing is rejected
/// before the read and again before the write.
fn ensure_output_is_not_input(input: &Path, out: &Path) -> Result<()> {
    let resolved_input = resolve_existing(input);
    let resolved_out = match out.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        Some(parent) => {
            let file = out
                .file_name()
                .with_context(|| format!("output {} has no file name", out.display()))?;
            resolve_existing(parent).join(file)
        }
        None => resolve_existing(out),
    };
    ensure!(
        resolved_input != resolved_out,
        "output {} aliases input {}; choose a distinct report path",
        out.display(),
        input.display()
    );
    Ok(())
}

/// Canonicalize when the path exists and fall back to the literal path otherwise,
/// so a not-yet-created report directory still yields a comparable value.
fn resolve_existing(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn validate_snapshot(snapshot: &Snapshot) -> Result<()> {
    ensure!(
        snapshot.schema_version == INPUT_SCHEMA_VERSION,
        "unsupported schema_version {}; expected {INPUT_SCHEMA_VERSION}",
        snapshot.schema_version
    );
    ensure!(!snapshot.instrument_id.trim().is_empty(), "instrument_id is empty");
    ensure!((-90.0..=90.0).contains(&snapshot.location.latitude_deg), "invalid latitude");
    ensure!((-180.0..=180.0).contains(&snapshot.location.longitude_deg), "invalid longitude");
    ensure!(
        snapshot.location.annual_normal_precip_inches.is_finite()
            && snapshot.location.annual_normal_precip_inches > 0.0,
        "annual normal precipitation must be positive and finite"
    );
    ensure!(!snapshot.source.is_empty(), "source list is empty");
    ensure!(!snapshot.issue.is_empty(), "forecast issue list is empty");

    let mut source_ids = BTreeSet::new();
    for source in &snapshot.source {
        ensure!(!source.id.trim().is_empty(), "source id is empty");
        ensure!(!source.url.trim().is_empty(), "source {} has no URL", source.id);
        ensure!(source_ids.insert(source.id.as_str()), "duplicate source id {}", source.id);
    }
    ensure!(
        source_ids.contains(snapshot.current_weather.source_id.as_str()),
        "current weather source {} is absent from source records",
        snapshot.current_weather.source_id
    );
    // TOML parses `inf` and `-inf` as floats, so `-inf <= inf` satisfies an
    // ordering check alone and the operational summary then prints infinite
    // temperatures. Require finiteness first, then the ordering.
    for (name, value) in [
        ("high_f_min", snapshot.current_weather.high_f_min),
        ("high_f_max", snapshot.current_weather.high_f_max),
        ("low_f_min", snapshot.current_weather.low_f_min),
        ("low_f_max", snapshot.current_weather.low_f_max),
    ] {
        ensure!(value.is_finite(), "current weather {name} must be finite");
    }
    ensure!(
        snapshot.current_weather.high_f_min <= snapshot.current_weather.high_f_max,
        "current weather high range is reversed"
    );
    ensure!(
        snapshot.current_weather.low_f_min <= snapshot.current_weather.low_f_max,
        "current weather low range is reversed"
    );

    // The near-term line quotes the NWS product, so the snapshot instant must lie
    // inside that product's own validity interval or the report presents a stale
    // short-range forecast as current.
    let as_of = parse_rfc3339("as_of", &snapshot.as_of)?;
    let valid_from = parse_rfc3339("current_weather.valid_from", &snapshot.current_weather.valid_from)?;
    let valid_through =
        parse_rfc3339("current_weather.valid_through", &snapshot.current_weather.valid_through)?;
    ensure!(
        valid_from <= valid_through,
        "current weather validity window is reversed"
    );
    ensure!(
        as_of >= valid_from && as_of <= valid_through,
        "snapshot as_of {} lies outside the current-weather validity window {} .. {}",
        snapshot.as_of,
        snapshot.current_weather.valid_from,
        snapshot.current_weather.valid_through
    );
    validate_probability(
        "weekend_precip_probability_max_pct",
        snapshot.current_weather.weekend_precip_probability_max_pct,
    )?;

    let mut issue_dates = Vec::with_capacity(snapshot.issue.len());
    for issue in &snapshot.issue {
        let issued_on = parse_issue_date(&issue.issued_on)?;
        ensure!(
            !issue_dates.contains(&issued_on),
            "duplicate forecast issue date {}",
            issue.issued_on
        );
        issue_dates.push(issued_on);
        for (field, id) in [
            ("enso_source_id", &issue.enso_source_id),
            ("seasonal_source_id", &issue.seasonal_source_id),
        ] {
            ensure!(
                source_ids.contains(id.as_str()),
                "issue {} {field} {id} is absent from source records",
                issue.issued_on
            );
        }
        validate_probability("very_strong_probability_pct", issue.very_strong_probability_pct)?;
        validate_probability(
            "ond_roni_ge_2_5_probability_pct",
            issue.ond_roni_ge_2_5_probability_pct,
        )?;
        ensure!(!issue.lead.is_empty(), "forecast issue {} has no leads", issue.issued_on);
        let mut seasons = BTreeSet::new();
        for lead in &issue.lead {
            ensure!(seasons.insert(lead.season.as_str()), "duplicate season {}", lead.season);
            validate_probability("el_nino_probability_pct", lead.el_nino_probability_pct)?;
            validate_probability(
                "local_precip_probability_floor_pct",
                lead.local_precip_probability_floor_pct,
            )?;
            if let Some(ceiling) = lead.local_precip_probability_ceiling_pct {
                validate_probability("local_precip_probability_ceiling_pct", ceiling)?;
                ensure!(
                    lead.local_precip_probability_floor_pct <= ceiling,
                    "{} precipitation probability range is reversed",
                    lead.season
                );
            }
            ensure!(
                lead.roni_p05_c.is_finite()
                    && lead.roni_median_c.is_finite()
                    && lead.roni_p95_c.is_finite(),
                "{} RONI values must be finite",
                lead.season
            );
            ensure!(
                lead.roni_p05_c <= lead.roni_median_c
                    && lead.roni_median_c <= lead.roni_p95_c,
                "{} RONI quantiles are not ordered",
                lead.season
            );
            ensure!(
                lead.normal_precip_inches.is_finite() && lead.normal_precip_inches >= 0.0,
                "{} normal precipitation is invalid",
                lead.season
            );
            // Every emitted value names the CPC record that asserts it, so the
            // report cannot present unbound numbers under `official_enso`.
            for (field, id) in [
                ("el_nino_source_id", &lead.el_nino_source_id),
                ("roni_source_id", &lead.roni_source_id),
                ("local_precip_source_id", &lead.local_precip_source_id),
                ("normals_source_id", &lead.normals_source_id),
            ] {
                ensure!(
                    source_ids.contains(id.as_str()),
                    "{} {field} {id} is absent from source records",
                    lead.season
                );
            }
        }
    }
    Ok(())
}

/// Parse an offset-aware timestamp; the snapshot carries local Pacific offsets,
/// so a naive parse would compare instants that are seven hours apart.
fn parse_rfc3339(name: &str, value: &str) -> Result<DateTime<FixedOffset>> {
    DateTime::parse_from_rfc3339(value)
        .with_context(|| format!("{name} is not an RFC 3339 timestamp: {value}"))
}

/// Forecast issues are dated calendar days in the CPC monthly cadence.
fn parse_issue_date(value: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .with_context(|| format!("forecast issue date is not YYYY-MM-DD: {value}"))
}

/// Count the leading run of parseable, unique, strictly increasing issue dates.
/// Duplicated ingestion or arbitrary date strings therefore cannot inflate the
/// record count into a diagnostic over "successive monthly issues".
fn chronological_issue_count(issues: &[ForecastIssue]) -> usize {
    let mut previous: Option<NaiveDate> = None;
    let mut count = 0;
    for issue in issues {
        let Ok(date) = parse_issue_date(&issue.issued_on) else {
            break;
        };
        if previous.is_some_and(|earlier| date <= earlier) {
            break;
        }
        previous = Some(date);
        count += 1;
    }
    count
}

fn validate_probability(name: &str, value: f64) -> Result<()> {
    ensure!(
        value.is_finite() && (0.0..=100.0).contains(&value),
        "{name} must be finite and inside [0, 100]"
    );
    Ok(())
}

fn build_report(
    mut snapshot: Snapshot,
    source_snapshot_sha256: String,
    precision: &str,
) -> Result<Report> {
    snapshot.issue.sort_by_key(|issue| issue.issued_on.clone());
    let latest = snapshot.issue.last().context("missing latest issue")?;
    let by_season = latest
        .lead
        .iter()
        .map(|lead| (lead.season.as_str(), lead))
        .collect::<BTreeMap<_, _>>();
    for season in REQUIRED_SEASONS {
        ensure!(by_season.contains_key(season), "latest issue is missing {season}");
    }
    let peak = latest
        .lead
        .iter()
        .max_by(|left, right| left.roni_median_c.total_cmp(&right.roni_median_c))
        .context("latest issue has no lead seasons")?;

    let official_enso = EnsoSummary {
        enso_issue: latest.issued_on.clone(),
        seasonal_issue: latest.seasonal_outlook_issued_on.clone(),
        very_strong_probability_relation: latest.very_strong_probability_relation.clone(),
        very_strong_probability_pct: latest.very_strong_probability_pct,
        ond_roni_ge_2_5_probability_pct: latest.ond_roni_ge_2_5_probability_pct,
        peak_roni_season: peak.season.clone(),
        peak_roni_median_c: peak.roni_median_c,
        peak_roni_p05_c: peak.roni_p05_c,
        peak_roni_p95_c: peak.roni_p95_c,
        next_enso_update: latest.next_enso_update.clone(),
        next_seasonal_update: latest.next_seasonal_update.clone(),
    };
    let seasonal_outlook = latest
        .lead
        .iter()
        .map(|lead| SeasonSummary {
            season: lead.season.clone(),
            el_nino_probability_pct: lead.el_nino_probability_pct,
            roni_p05_c: lead.roni_p05_c,
            roni_median_c: lead.roni_median_c,
            roni_p95_c: lead.roni_p95_c,
            normal_precip_inches: lead.normal_precip_inches,
            local_precip_category: lead.local_precip_category.clone(),
            local_precip_probability_floor_pct: lead.local_precip_probability_floor_pct,
            local_precip_probability_ceiling_pct: lead.local_precip_probability_ceiling_pct,
            local_wet_signal: classify_wet_signal(lead).to_string(),
            probability_basis: lead.probability_basis.clone(),
            local_interpretation: lead.local_interpretation.clone(),
        })
        .collect();
    let cd_diagnostic = compute_cd_diagnostic(&snapshot.issue, precision)?;

    let derived = DerivedOutlook {
        official_enso,
        seasonal_outlook,
        cd_diagnostic,
        operational_summary: operational_summary(latest, &snapshot.location, &snapshot.current_weather),
        caveat: vec![
            "CPC category probabilities are chances of seasonal terciles, not deterministic ZIP-code rainfall totals.".to_string(),
            "The 91103 probability bounds are conservative regional readings from CPC text and maps, not digitized point probabilities.".to_string(),
            "A very strong El Nino raises wet-season odds but does not guarantee a wet Pasadena winter.".to_string(),
            "The Cayley-Dickson diagnostic remains excluded from operational forecasting until archived hindcasts beat matched controls.".to_string(),
        ],
    };

    Ok(Report {
        report_schema_version: REPORT_SCHEMA_VERSION,
        source_snapshot_sha256,
        snapshot,
        derived,
    })
}

/// The probability bound is a chance of the category CPC actually named, so the
/// polarity comes from `local_precip_category` and the bound only sets strength.
/// Reading a 50 percent floor as an above-normal signal under a `below` category
/// reports the opposite of the official outlook.
/// Build the advice from the latest issue's own leads and the snapshot location.
/// Fixed strings would prescribe a fall wet tilt and a December-April above-normal
/// window for any refreshed snapshot, including a below-normal or equal-chances
/// outlook and any other accepted location.
fn operational_summary(
    latest: &ForecastIssue,
    location: &Location,
    weather: &CurrentWeather,
) -> Vec<String> {
    let mut lines = vec![format!(
        "Near term for {}: {} Highs {:.0}-{:.0} F and lows {:.0}-{:.0} F. ENSO does not replace the NWS short-range forecast.",
        location.label,
        weather.summary,
        weather.high_f_min,
        weather.high_f_max,
        weather.low_f_min,
        weather.low_f_max
    )];
    let by_season = latest
        .lead
        .iter()
        .map(|lead| (lead.season.as_str(), lead))
        .collect::<BTreeMap<_, _>>();
    let mut ordered = Vec::new();
    for season in REQUIRED_SEASONS {
        if let Some(lead) = by_season.get(season) {
            ordered.push(*lead);
        }
    }
    for lead in &ordered {
        let bound = match lead.local_precip_probability_ceiling_pct {
            Some(ceiling) => format!(
                "{:.0}-{:.0} percent",
                lead.local_precip_probability_floor_pct, ceiling
            ),
            None => format!(
                "at least {:.0} percent",
                lead.local_precip_probability_floor_pct
            ),
        };
        lines.push(format!(
            "{}: official category {} at {} ({}); {:.1} in normal precipitation.",
            lead.season,
            lead.local_precip_category,
            bound,
            classify_wet_signal(lead),
            lead.normal_precip_inches
        ));
    }
    if let Some(strongest) = ordered.iter().max_by(|left, right| {
        signal_rank(left)
            .cmp(&signal_rank(right))
            .then_with(|| left.local_precip_probability_floor_pct.total_cmp(&right.local_precip_probability_floor_pct))
    }) {
        lines.push(format!(
            "Principal seasonal signal for {}: {} at {}. The CPC category, not this ordering, is the operational authority.",
            location.postal_code,
            strongest.season,
            classify_wet_signal(strongest)
        ));
    }
    lines
}

/// Rank a lead by how far its bound departs from equal chances, in either
/// direction, so the strongest departure is selected without assuming polarity.
fn signal_rank(lead: &LeadSeason) -> u8 {
    match classify_wet_signal(lead) {
        "elevated_above_normal" | "elevated_below_normal" => 3,
        "moderate_above_normal" | "moderate_below_normal" => 2,
        "slight_above_normal" | "slight_below_normal" => 1,
        _ => 0,
    }
}

fn classify_wet_signal(lead: &LeadSeason) -> &'static str {
    let strength = if lead.local_precip_probability_floor_pct >= 50.0 {
        Strength::Elevated
    } else if lead
        .local_precip_probability_ceiling_pct
        .is_some_and(|ceiling| ceiling >= 50.0)
    {
        Strength::Moderate
    } else {
        Strength::Slight
    };
    match lead.local_precip_category.trim().to_ascii_lowercase().as_str() {
        "above" => match strength {
            Strength::Elevated => "elevated_above_normal",
            Strength::Moderate => "moderate_above_normal",
            Strength::Slight => "slight_above_normal",
        },
        "below" => match strength {
            Strength::Elevated => "elevated_below_normal",
            Strength::Moderate => "moderate_below_normal",
            Strength::Slight => "slight_below_normal",
        },
        "equal" | "equal_chances" | "ec" => "equal_chances",
        _ => "unspecified_category",
    }
}

enum Strength {
    Elevated,
    Moderate,
    Slight,
}

fn compute_cd_diagnostic(issues: &[ForecastIssue], precision: &str) -> Result<CdDiagnostic> {
    let season_order = REQUIRED_SEASONS
        .iter()
        .map(|season| (*season).to_string())
        .collect();
    let feature_order_within_season = vec![
        "RONI median / 3.5".to_string(),
        "RONI 90% interval width / 3.5".to_string(),
        "El Nino probability / 100".to_string(),
        "local above-normal precipitation probability floor / 100".to_string(),
    ];

    let usable = chronological_issue_count(issues);
    if usable < 3 {
        return Ok(CdDiagnostic {
            status: "insufficient_issue_history".to_string(),
            diagnostic_id: "CD-ENSO-ISSUE-S16-v1".to_string(),
            dimension: CD_DIM,
            issue_count_available: usable,
            issue_count_required: 3,
            lift_depth_from_reals: 4,
            season_order,
            feature_order_within_season,
            associator_norms: Vec::new(),
            latest_associator_norm: None,
            mean_associator_norm: None,
            mean_l2_issue_step: None,
            calibrated_for_target: false,
            used_for_operational_forecast: false,
            interpretation: "Three complete monthly issues with distinct, parseable, chronologically increasing dates are required to form one associator over successive forecast-state vectors.".to_string(),
            next_decisive_test: "Archive at least ten years of CPC issues, hindcast the local target, and compare against matched L2, angular, cumulative-change, and scrambled-sign controls.".to_string(),
        });
    }

    let vectors = issues
        .iter()
        .map(build_issue_vector)
        .collect::<Result<Vec<_>>>()?;
    let associator_norms =
        cd_kernel::batch_sliding_associator_norms_dispatch(&vectors, CD_DIM, precision);
    let l2_steps = vectors
        .windows(2)
        .map(|pair| l2_distance(&pair[0], &pair[1]))
        .collect::<Vec<_>>();

    Ok(CdDiagnostic {
        status: "computed_unvalidated".to_string(),
        diagnostic_id: "CD-ENSO-ISSUE-S16-v1".to_string(),
        dimension: CD_DIM,
        issue_count_available: usable,
        issue_count_required: 3,
        lift_depth_from_reals: 4,
        season_order,
        feature_order_within_season,
        latest_associator_norm: associator_norms.last().copied(),
        mean_associator_norm: mean(&associator_norms),
        mean_l2_issue_step: mean(&l2_steps),
        associator_norms,
        calibrated_for_target: false,
        used_for_operational_forecast: false,
        interpretation: "The norm measures non-associative curvature across three successive issue-state vectors. It does not itself predict rainfall.".to_string(),
        next_decisive_test: "Require held-out incremental skill beyond matched issue-change controls before operational use.".to_string(),
    })
}

fn build_issue_vector(issue: &ForecastIssue) -> Result<Vec<f64>> {
    let by_season = issue
        .lead
        .iter()
        .map(|lead| (lead.season.as_str(), lead))
        .collect::<BTreeMap<_, _>>();
    let mut vector = Vec::with_capacity(CD_DIM);
    for season in REQUIRED_SEASONS {
        let lead = by_season
            .get(season)
            .with_context(|| format!("issue {} is missing {season}", issue.issued_on))?;
        vector.push(lead.roni_median_c / 3.5);
        vector.push((lead.roni_p95_c - lead.roni_p05_c) / 3.5);
        vector.push(lead.el_nino_probability_pct / 100.0);
        vector.push(lead.local_precip_probability_floor_pct / 100.0);
    }
    ensure!(vector.len() == CD_DIM, "internal issue vector has wrong size");
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    ensure!(
        norm.is_finite() && norm > f64::EPSILON,
        "issue {} produced a degenerate vector",
        issue.issued_on
    );
    for value in &mut vector {
        *value /= norm;
    }
    Ok(vector)
}

fn l2_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lead(season: &str, floor: f64, ceiling: Option<f64>) -> LeadSeason {
        LeadSeason {
            season: season.to_string(),
            el_nino_probability_pct: 100.0,
            roni_p05_c: 1.0,
            roni_median_c: 2.0,
            roni_p95_c: 3.0,
            local_precip_category: "above".to_string(),
            local_precip_probability_floor_pct: floor,
            local_precip_probability_ceiling_pct: ceiling,
            probability_basis: "test".to_string(),
            normal_precip_inches: 1.0,
            local_interpretation: "test".to_string(),
            el_nino_source_id: "SRC-ENSO".to_string(),
            roni_source_id: "SRC-ENSO".to_string(),
            local_precip_source_id: "SRC-ENSO".to_string(),
            normals_source_id: "SRC-ENSO".to_string(),
        }
    }

    fn issue(date: &str) -> ForecastIssue {
        ForecastIssue {
            issued_on: date.to_string(),
            seasonal_outlook_issued_on: date.to_string(),
            next_enso_update: date.to_string(),
            next_seasonal_update: date.to_string(),
            very_strong_probability_relation: "greater_than".to_string(),
            very_strong_probability_pct: 90.0,
            ond_roni_ge_2_5_probability_pct: 69.0,
            enso_source_id: "SRC-ENSO".to_string(),
            seasonal_source_id: "SRC-ENSO".to_string(),
            lead: vec![
                lead("SON", 33.0, Some(40.0)),
                lead("OND", 33.0, Some(50.0)),
                lead("DJF", 50.0, None),
                lead("JFM", 50.0, None),
            ],
        }
    }

    fn location() -> Location {
        Location {
            postal_code: "91103".to_string(),
            label: "test locality".to_string(),
            latitude_deg: 34.0,
            longitude_deg: -118.0,
            climate_division_id: 6,
            climate_division_name: "test division".to_string(),
            normals_station_id: "USC00046719".to_string(),
            normals_period: "1991-2020".to_string(),
            annual_normal_precip_inches: 20.1,
        }
    }

    fn weather() -> CurrentWeather {
        CurrentWeather {
            source_id: "SRC-ENSO".to_string(),
            valid_from: "2026-09-01T00:00:00-07:00".to_string(),
            valid_through: "2026-09-07T18:00:00-07:00".to_string(),
            summary: "Sunny.".to_string(),
            high_f_min: 84.0,
            high_f_max: 90.0,
            low_f_min: 56.0,
            low_f_max: 66.0,
            dry_through: "2026-09-04T23:59:59-07:00".to_string(),
            weekend_precip_probability_max_pct: 30.0,
        }
    }

    fn snapshot() -> Snapshot {
        Snapshot {
            schema_version: INPUT_SCHEMA_VERSION,
            instrument_id: "TEST".to_string(),
            as_of: "2026-09-01T15:57:00-07:00".to_string(),
            location: location(),
            source: vec![SourceRecord {
                id: "SRC-ENSO".to_string(),
                authority: "NOAA Climate Prediction Center".to_string(),
                product: "test product".to_string(),
                issued_on: "2026-08-13".to_string(),
                retrieved_on: "2026-09-01".to_string(),
                url: "https://example.invalid/".to_string(),
                notes: String::new(),
            }],
            current_weather: weather(),
            issue: vec![issue("2026-08-13")],
        }
    }

    #[test]
    fn wet_signal_uses_conservative_bounds() {
        assert_eq!(
            classify_wet_signal(&lead("SON", 33.0, Some(40.0))),
            "slight_above_normal"
        );
        assert_eq!(
            classify_wet_signal(&lead("OND", 33.0, Some(50.0))),
            "moderate_above_normal"
        );
        assert_eq!(
            classify_wet_signal(&lead("DJF", 50.0, None)),
            "elevated_above_normal"
        );
    }

    #[test]
    fn wet_signal_follows_the_official_category_polarity() {
        let mut below = lead("DJF", 60.0, None);
        below.local_precip_category = "below".to_string();
        assert_eq!(classify_wet_signal(&below), "elevated_below_normal");

        let mut equal = lead("DJF", 50.0, None);
        equal.local_precip_category = "equal_chances".to_string();
        assert_eq!(classify_wet_signal(&equal), "equal_chances");

        let mut unknown = lead("DJF", 55.0, None);
        unknown.local_precip_category = "unclassified".to_string();
        assert_eq!(classify_wet_signal(&unknown), "unspecified_category");
    }

    #[test]
    fn operational_summary_tracks_the_latest_leads_and_location() {
        let mut dry = issue("2026-08-13");
        for entry in &mut dry.lead {
            entry.local_precip_category = "below".to_string();
            entry.local_precip_probability_floor_pct = 55.0;
            entry.local_precip_probability_ceiling_pct = None;
        }
        let mut place = location();
        place.postal_code = "97201".to_string();
        place.label = "Portland, Oregon".to_string();
        let lines = operational_summary(&dry, &place, &weather());

        let joined = lines.join("\n");
        assert!(joined.contains("Portland, Oregon"));
        assert!(joined.contains("97201"));
        assert!(joined.contains("elevated_below_normal"));
        assert!(!joined.contains("wet tilt"));
        assert!(!joined.contains("December-April"));
    }

    #[test]
    fn issue_vector_is_a_unit_sedenion() {
        let vector = build_issue_vector(&issue("2026-08-13")).unwrap();
        assert_eq!(vector.len(), CD_DIM);
        let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn one_issue_cannot_emit_an_associator() {
        let diagnostic = compute_cd_diagnostic(&[issue("2026-08-13")], "f64").unwrap();
        assert_eq!(diagnostic.status, "insufficient_issue_history");
        assert!(diagnostic.associator_norms.is_empty());
        assert!(!diagnostic.used_for_operational_forecast);
    }

    #[test]
    fn three_issues_compute_only_an_unvalidated_diagnostic() {
        let diagnostic = compute_cd_diagnostic(
            &[
                issue("2026-06-11"),
                issue("2026-07-09"),
                issue("2026-08-13"),
            ],
            "f64",
        )
        .unwrap();
        assert_eq!(diagnostic.status, "computed_unvalidated");
        assert_eq!(diagnostic.associator_norms.len(), 1);
        assert!(!diagnostic.calibrated_for_target);
        assert!(!diagnostic.used_for_operational_forecast);
    }

    #[test]
    fn duplicate_or_unparseable_issue_dates_block_the_diagnostic() {
        let duplicated = compute_cd_diagnostic(
            &[
                issue("2026-08-13"),
                issue("2026-08-13"),
                issue("2026-08-13"),
            ],
            "f64",
        )
        .unwrap();
        assert_eq!(duplicated.status, "insufficient_issue_history");
        assert_eq!(duplicated.issue_count_available, 1);

        let unparseable = compute_cd_diagnostic(
            &[issue("august"), issue("later"), issue("latest")],
            "f64",
        )
        .unwrap();
        assert_eq!(unparseable.status, "insufficient_issue_history");
        assert_eq!(unparseable.issue_count_available, 0);

        let mut duplicate_snapshot = snapshot();
        duplicate_snapshot.issue = vec![issue("2026-08-13"), issue("2026-08-13")];
        assert!(validate_snapshot(&duplicate_snapshot).is_err());
    }

    #[test]
    fn unbound_source_ids_are_rejected() {
        assert!(validate_snapshot(&snapshot()).is_ok());

        let mut unbound_lead = snapshot();
        unbound_lead.issue[0].lead[0].roni_source_id = "SRC-ABSENT".to_string();
        assert!(validate_snapshot(&unbound_lead).is_err());

        let mut unbound_issue = snapshot();
        unbound_issue.issue[0].seasonal_source_id = "SRC-ABSENT".to_string();
        assert!(validate_snapshot(&unbound_issue).is_err());
    }

    #[test]
    fn as_of_outside_the_weather_window_is_rejected() {
        let mut expired = snapshot();
        expired.as_of = "2026-09-30T00:00:00-07:00".to_string();
        assert!(validate_snapshot(&expired).is_err());

        let mut early = snapshot();
        early.as_of = "2026-08-01T00:00:00-07:00".to_string();
        assert!(validate_snapshot(&early).is_err());

        let mut unparseable = snapshot();
        unparseable.as_of = "2026-09-01".to_string();
        assert!(validate_snapshot(&unparseable).is_err());
    }

    #[test]
    fn nonfinite_temperature_bounds_are_rejected() {
        let mut infinite = snapshot();
        infinite.current_weather.high_f_min = f64::NEG_INFINITY;
        infinite.current_weather.high_f_max = f64::INFINITY;
        assert!(validate_snapshot(&infinite).is_err());

        let mut nan = snapshot();
        nan.current_weather.low_f_min = f64::NAN;
        assert!(validate_snapshot(&nan).is_err());
    }

    #[test]
    fn output_may_not_alias_the_input() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("snapshot.toml");
        fs::write(&input, "schema_version = 2\n").unwrap();

        assert!(ensure_output_is_not_input(&input, &input).is_err());
        assert!(
            ensure_output_is_not_input(&input, &dir.path().join(".").join("snapshot.toml"))
                .is_err()
        );
        assert!(ensure_output_is_not_input(&input, &dir.path().join("report.toml")).is_ok());
    }
}
