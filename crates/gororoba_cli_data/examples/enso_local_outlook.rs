//! Build a provenance-rich ENSO outlook for one local forecast target.
//!
//! The official CPC and NWS values remain operational authority.  The optional
//! Cayley-Dickson calculation measures curvature across successive forecast
//! issue vectors; it is never promoted into a rainfall forecast by this tool.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

const INPUT_SCHEMA_VERSION: u32 = 1;
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
    fs::write(&cli.out, text).with_context(|| format!("write {}", cli.out.display()))?;
    println!("{}", cli.out.display());
    Ok(())
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
    ensure!(
        snapshot.current_weather.high_f_min <= snapshot.current_weather.high_f_max,
        "current weather high range is reversed"
    );
    ensure!(
        snapshot.current_weather.low_f_min <= snapshot.current_weather.low_f_max,
        "current weather low range is reversed"
    );
    validate_probability(
        "weekend_precip_probability_max_pct",
        snapshot.current_weather.weekend_precip_probability_max_pct,
    )?;

    for issue in &snapshot.issue {
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
        }
    }
    Ok(())
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
        operational_summary: vec![
            format!(
                "Near term: {} Highs {:.0}-{:.0} F and lows {:.0}-{:.0} F. ENSO does not replace the NWS short-range forecast.",
                snapshot.current_weather.summary,
                snapshot.current_weather.high_f_min,
                snapshot.current_weather.high_f_max,
                snapshot.current_weather.low_f_min,
                snapshot.current_weather.low_f_max
            ),
            "Fall: retain a slight-to-moderate wet tilt, with substantial delayed-onset uncertainty.".to_string(),
            "Core and late winter: treat December-April as the principal above-normal precipitation window; January-March carries the broadest coastal California signal.".to_string(),
        ],
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

fn classify_wet_signal(lead: &LeadSeason) -> &'static str {
    if lead.local_precip_probability_floor_pct >= 50.0 {
        "elevated_above_normal"
    } else if lead
        .local_precip_probability_ceiling_pct
        .is_some_and(|ceiling| ceiling >= 50.0)
    {
        "moderate_above_normal"
    } else if lead.local_precip_category.eq_ignore_ascii_case("above") {
        "slight_above_normal"
    } else {
        "equal_or_unspecified"
    }
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

    if issues.len() < 3 {
        return Ok(CdDiagnostic {
            status: "insufficient_issue_history".to_string(),
            diagnostic_id: "CD-ENSO-ISSUE-S16-v1".to_string(),
            dimension: CD_DIM,
            issue_count_available: issues.len(),
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
            interpretation: "Three complete monthly issues are required to form one associator over successive forecast-state vectors.".to_string(),
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
        issue_count_available: issues.len(),
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
    format!("{:x}", Sha256::digest(bytes))
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
            lead: vec![
                lead("SON", 33.0, Some(40.0)),
                lead("OND", 33.0, Some(50.0)),
                lead("DJF", 50.0, None),
                lead("JFM", 50.0, None),
            ],
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
}
