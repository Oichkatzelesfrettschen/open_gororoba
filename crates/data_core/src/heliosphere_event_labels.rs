//! Official heliosphere event-label ingestion and windowing helpers.
//!
//! This module stages bounded official label families from CCMC/NASA DONKI and
//! the official CCMC CME Scoreboard so heliosphere evaluation can anchor itself
//! to physically constrained windows instead of repo-internal heuristics.
//!
//! Official sources:
//! - <https://api.nasa.gov/>
//! - <https://ccmc.gsfc.nasa.gov/donki/>
//! - <https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME>
//! - <https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CMEAnalysis>
//! - <https://kauai.ccmc.gsfc.nasa.gov/CMEscoreboard/>
//!
//! Current official label families:
//! - DONKI `IPS` (interplanetary shock onset)
//! - DONKI `GST` (geomagnetic storm onset)
//! - DONKI `SEP` (solar energetic particle onset)
//! - DONKI `FLR` (solar flare peak)
//! - DONKI `CME` (official CME onset/observation records)
//! - DONKI `WSA-ENLIL` impact targets embedded in `CME` records
//! - CCMC CME Scoreboard observed Earth/L1 arrival windows
//! - CCMC CME Scoreboard forecast residual rows for Earth/L1 validation

use crate::fetcher::{FetchError, validate_not_html};
use chrono::{DateTime, Datelike, Duration, NaiveDate, NaiveDateTime, Utc};
use regex::Regex;
use reqwest::blocking::Client;
use serde::{Deserialize, Deserializer, Serialize};
use std::{env, fs, path::Path, thread::sleep, time::Duration as StdDuration};

const DONKI_API_ROOT: &str = "https://api.nasa.gov/DONKI";
const DONKI_CCMC_ROOT: &str = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get";

/// Official source family for heliosphere event labels.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HeliosphereEventSource {
    DonkiIps,
    DonkiGst,
    DonkiSep,
    DonkiFlr,
    DonkiCme,
    DonkiEnlilImpact,
    CcmcScoreboardArrival,
    CcmcScoreboardForecastResidual,
}

/// Physical event family represented by an official label.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HeliosphereEventKind {
    InterplanetaryShock,
    GeomagneticStorm,
    SolarEnergeticParticle,
    SolarFlare,
    CoronalMassEjection,
    PredictedCmeImpact,
    ObservedCmeArrival,
}

/// One official event label with mission/location applicability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeliosphereEventLabel {
    pub source: HeliosphereEventSource,
    pub kind: HeliosphereEventKind,
    pub label_id: String,
    pub location: String,
    pub event_time_utc: String,
    pub linked_activity_ids: Vec<String>,
    pub linked_cme_ids: Vec<String>,
    pub instrument_names: Vec<String>,
    pub mission_targets: Vec<String>,
    pub note: Option<String>,
}

/// A prediction/evaluation window derived from an official label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeliosphereEventWindow {
    pub label_id: String,
    pub source: HeliosphereEventSource,
    pub kind: HeliosphereEventKind,
    pub mission: String,
    pub event_time_utc: String,
    pub window_start_utc: String,
    pub window_end_utc: String,
}

/// Optional forecast residual placeholder for later CCMC-style comparisons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResidual {
    pub source: HeliosphereEventSource,
    pub label_id: String,
    pub mission: String,
    pub forecast_method: String,
    pub predicted_arrival_utc: String,
    pub actual_arrival_utc: String,
    pub model_completion_utc: Option<String>,
    pub residual_hours: f64,
    pub note: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DonkiLinkedEvent {
    #[serde(rename = "activityID")]
    activity_id: String,
}

#[derive(Debug, Deserialize)]
struct DonkiInstrument {
    #[serde(rename = "displayName")]
    display_name: String,
}

#[derive(Debug, Deserialize)]
struct DonkiIpsRecord {
    #[serde(rename = "activityID")]
    activity_id: String,
    #[serde(default, deserialize_with = "string_or_empty")]
    location: String,
    #[serde(rename = "eventTime")]
    event_time: String,
    #[serde(default, deserialize_with = "null_vec_default")]
    instruments: Vec<DonkiInstrument>,
    #[serde(
        default,
        rename = "linkedEvents",
        deserialize_with = "null_vec_default"
    )]
    linked_events: Vec<DonkiLinkedEvent>,
}

#[derive(Debug, Deserialize)]
struct DonkiGstRecord {
    #[serde(rename = "gstID")]
    gst_id: String,
    #[serde(rename = "startTime")]
    start_time: String,
    #[serde(
        default,
        rename = "linkedEvents",
        deserialize_with = "null_vec_default"
    )]
    linked_events: Vec<DonkiLinkedEvent>,
}

#[derive(Debug, Deserialize)]
struct DonkiSepRecord {
    #[serde(rename = "sepID")]
    sep_id: String,
    #[serde(rename = "eventTime")]
    event_time: String,
    #[serde(default, deserialize_with = "null_vec_default")]
    instruments: Vec<DonkiInstrument>,
    #[serde(
        default,
        rename = "linkedEvents",
        deserialize_with = "null_vec_default"
    )]
    linked_events: Vec<DonkiLinkedEvent>,
}

#[derive(Debug, Deserialize)]
struct DonkiFlrRecord {
    #[serde(rename = "flrID")]
    flr_id: String,
    #[serde(rename = "peakTime")]
    peak_time: String,
    #[serde(default, deserialize_with = "string_or_empty")]
    note: String,
    #[serde(default, deserialize_with = "null_vec_default")]
    instruments: Vec<DonkiInstrument>,
    #[serde(
        default,
        rename = "linkedEvents",
        deserialize_with = "null_vec_default"
    )]
    linked_events: Vec<DonkiLinkedEvent>,
}

#[derive(Debug, Deserialize)]
struct DonkiImpact {
    #[serde(default, deserialize_with = "string_or_empty")]
    location: String,
    #[serde(default, rename = "arrivalTime", deserialize_with = "string_or_empty")]
    arrival_time: String,
    #[serde(default)]
    #[serde(rename = "isGlancingBlow")]
    is_glancing_blow: bool,
    #[serde(default)]
    #[serde(rename = "isMinorImpact")]
    is_minor_impact: bool,
}

#[derive(Debug, Deserialize)]
struct DonkiEnlilResult {
    #[serde(
        default,
        rename = "modelCompletionTime",
        deserialize_with = "string_or_empty"
    )]
    model_completion_time: String,
    #[serde(
        default,
        rename = "estimatedShockArrivalTime",
        deserialize_with = "string_or_empty"
    )]
    estimated_shock_arrival_time: String,
    #[serde(default, rename = "impactList", deserialize_with = "null_vec_default")]
    impact_list: Vec<DonkiImpact>,
}

#[derive(Debug, Deserialize)]
struct DonkiCmeAnalysis {
    #[serde(default)]
    #[serde(rename = "isMostAccurate")]
    is_most_accurate: bool,
    #[serde(default, rename = "enlilList", deserialize_with = "null_vec_default")]
    enlil_list: Vec<DonkiEnlilResult>,
}

#[derive(Debug, Deserialize)]
struct DonkiCmeRecord {
    #[serde(rename = "activityID")]
    activity_id: String,
    #[serde(rename = "startTime")]
    start_time: String,
    #[serde(
        default,
        rename = "sourceLocation",
        deserialize_with = "string_or_empty"
    )]
    source_location: String,
    #[serde(default, deserialize_with = "string_or_empty")]
    note: String,
    #[serde(default, deserialize_with = "null_vec_default")]
    instruments: Vec<DonkiInstrument>,
    #[serde(default, rename = "cmeAnalyses", deserialize_with = "null_vec_default")]
    cme_analyses: Vec<DonkiCmeAnalysis>,
    #[serde(
        default,
        rename = "linkedEvents",
        deserialize_with = "null_vec_default"
    )]
    linked_events: Vec<DonkiLinkedEvent>,
}

#[derive(Debug, Clone)]
struct ScoreboardPredictionRow {
    predicted_arrival_utc: String,
    method: String,
    residual_hours: Option<f64>,
}

#[derive(Debug, Clone)]
struct ScoreboardCmeBlock {
    cme_id: String,
    actual_arrival_utc: Option<String>,
    detected_at_earth: bool,
    predictions: Vec<ScoreboardPredictionRow>,
}

fn null_vec_default<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<Vec<T>>::deserialize(deserializer).map(Option::unwrap_or_default)
}

fn string_or_empty<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer).map(Option::unwrap_or_default)
}

/// Fetch bounded official DONKI event labels into a cache root and parse them.
pub fn fetch_donki_event_labels(
    start_date: NaiveDate,
    end_date: NaiveDate,
    cache_root: &Path,
) -> Result<Vec<HeliosphereEventLabel>, FetchError> {
    let mut labels = Vec::new();
    labels.extend(fetch_donki_ips(start_date, end_date, cache_root)?);
    labels.extend(fetch_donki_gst(start_date, end_date, cache_root)?);
    labels.extend(fetch_donki_sep(start_date, end_date, cache_root)?);
    if let Ok(flares) = fetch_donki_flr(start_date, end_date, cache_root) {
        labels.extend(flares);
    }
    if let Ok(cme_labels) = fetch_donki_cme_labels(start_date, end_date, cache_root) {
        labels.extend(cme_labels);
    }
    if let Ok(scoreboard_labels) =
        fetch_ccmc_scoreboard_arrival_labels(start_date, end_date, cache_root)
    {
        labels.extend(scoreboard_labels);
    }
    labels.sort_by(|a, b| {
        (a.event_time_utc.as_str(), a.label_id.as_str())
            .cmp(&(b.event_time_utc.as_str(), b.label_id.as_str()))
    });
    labels.dedup_by(|a, b| a.label_id == b.label_id && a.event_time_utc == b.event_time_utc);
    Ok(labels)
}

/// Fetch official forecast residuals from the CCMC CME Scoreboard.
pub fn fetch_official_forecast_residuals(
    start_date: NaiveDate,
    end_date: NaiveDate,
    cache_root: &Path,
) -> Result<Vec<ForecastResidual>, FetchError> {
    let mut residuals = Vec::new();
    for year in start_date.year()..=end_date.year() {
        let page = fetch_ccmc_scoreboard_year(year, cache_root)?;
        for block in parse_scoreboard_blocks(&page) {
            let Some(actual_arrival_text) = block.actual_arrival_utc.as_deref() else {
                continue;
            };
            let Some(actual_arrival) = parse_donki_time(actual_arrival_text) else {
                continue;
            };
            if actual_arrival.date_naive() < start_date || actual_arrival.date_naive() > end_date {
                continue;
            }
            for mission in mission_targets_from_location("Earth") {
                for prediction in &block.predictions {
                    let Some(residual_hours) = prediction.residual_hours else {
                        continue;
                    };
                    residuals.push(ForecastResidual {
                        source: HeliosphereEventSource::CcmcScoreboardForecastResidual,
                        label_id: block.cme_id.clone(),
                        mission: mission.clone(),
                        forecast_method: prediction.method.clone(),
                        predicted_arrival_utc: prediction.predicted_arrival_utc.clone(),
                        actual_arrival_utc: actual_arrival.to_rfc3339(),
                        model_completion_utc: None,
                        residual_hours,
                        note: if block.detected_at_earth {
                            None
                        } else {
                            Some("scoreboard row belonged to a no-detection block".to_string())
                        },
                    });
                }
            }
        }
    }
    residuals.sort_by(|a, b| {
        (
            a.actual_arrival_utc.as_str(),
            a.label_id.as_str(),
            a.mission.as_str(),
            a.forecast_method.as_str(),
        )
            .cmp(&(
                b.actual_arrival_utc.as_str(),
                b.label_id.as_str(),
                b.mission.as_str(),
                b.forecast_method.as_str(),
            ))
    });
    Ok(residuals)
}

/// Convert official labels into mission-specific prediction windows.
pub fn labels_to_prediction_windows(
    labels: &[HeliosphereEventLabel],
    mission: &str,
    horizon_hours: i64,
) -> Vec<HeliosphereEventWindow> {
    let mission_key = normalize_mission_name(mission);
    let horizon = Duration::hours(horizon_hours.max(1));
    labels
        .iter()
        .filter(|label| {
            label
                .mission_targets
                .iter()
                .any(|target| normalize_mission_name(target) == mission_key)
        })
        .filter_map(|label| {
            let event_time = parse_donki_time(&label.event_time_utc)?;
            Some(HeliosphereEventWindow {
                label_id: label.label_id.clone(),
                source: label.source,
                kind: label.kind,
                mission: mission.to_string(),
                event_time_utc: event_time.to_rfc3339(),
                window_start_utc: (event_time - horizon).to_rfc3339(),
                window_end_utc: event_time.to_rfc3339(),
            })
        })
        .collect()
}

fn fetch_donki_ips(
    start_date: NaiveDate,
    end_date: NaiveDate,
    cache_root: &Path,
) -> Result<Vec<HeliosphereEventLabel>, FetchError> {
    let body = fetch_donki_json("IPS", start_date, end_date, cache_root)?;
    let rows: Vec<DonkiIpsRecord> = serde_json::from_str(&body)
        .map_err(|err| FetchError::Validation(format!("invalid DONKI IPS JSON: {err}")))?;
    Ok(rows
        .into_iter()
        .map(|row| HeliosphereEventLabel {
            source: HeliosphereEventSource::DonkiIps,
            kind: HeliosphereEventKind::InterplanetaryShock,
            label_id: row.activity_id,
            location: row.location.clone(),
            event_time_utc: row.event_time,
            linked_activity_ids: row
                .linked_events
                .iter()
                .map(|event| event.activity_id.clone())
                .collect(),
            linked_cme_ids: linked_cme_ids(&row.linked_events),
            instrument_names: row
                .instruments
                .iter()
                .map(|instrument| instrument.display_name.clone())
                .collect(),
            mission_targets: mission_targets_from_location(&row.location),
            note: None,
        })
        .collect())
}

fn fetch_donki_gst(
    start_date: NaiveDate,
    end_date: NaiveDate,
    cache_root: &Path,
) -> Result<Vec<HeliosphereEventLabel>, FetchError> {
    let body = fetch_donki_json("GST", start_date, end_date, cache_root)?;
    let rows: Vec<DonkiGstRecord> = serde_json::from_str(&body)
        .map_err(|err| FetchError::Validation(format!("invalid DONKI GST JSON: {err}")))?;
    Ok(rows
        .into_iter()
        .map(|row| HeliosphereEventLabel {
            source: HeliosphereEventSource::DonkiGst,
            kind: HeliosphereEventKind::GeomagneticStorm,
            label_id: row.gst_id,
            location: "Earth".to_string(),
            event_time_utc: row.start_time,
            linked_activity_ids: row
                .linked_events
                .iter()
                .map(|event| event.activity_id.clone())
                .collect(),
            linked_cme_ids: linked_cme_ids(&row.linked_events),
            instrument_names: Vec::new(),
            mission_targets: mission_targets_from_location("Earth"),
            note: None,
        })
        .collect())
}

fn fetch_donki_sep(
    start_date: NaiveDate,
    end_date: NaiveDate,
    cache_root: &Path,
) -> Result<Vec<HeliosphereEventLabel>, FetchError> {
    let body = fetch_donki_json("SEP", start_date, end_date, cache_root)?;
    let rows: Vec<DonkiSepRecord> = serde_json::from_str(&body)
        .map_err(|err| FetchError::Validation(format!("invalid DONKI SEP JSON: {err}")))?;
    Ok(rows
        .into_iter()
        .map(|row| {
            let location = infer_sep_location(&row.instruments);
            HeliosphereEventLabel {
                source: HeliosphereEventSource::DonkiSep,
                kind: HeliosphereEventKind::SolarEnergeticParticle,
                label_id: row.sep_id,
                location: location.clone(),
                event_time_utc: row.event_time,
                linked_activity_ids: row
                    .linked_events
                    .iter()
                    .map(|event| event.activity_id.clone())
                    .collect(),
                linked_cme_ids: linked_cme_ids(&row.linked_events),
                instrument_names: row
                    .instruments
                    .iter()
                    .map(|instrument| instrument.display_name.clone())
                    .collect(),
                mission_targets: mission_targets_from_location(&location),
                note: None,
            }
        })
        .collect())
}

fn fetch_donki_flr(
    start_date: NaiveDate,
    end_date: NaiveDate,
    cache_root: &Path,
) -> Result<Vec<HeliosphereEventLabel>, FetchError> {
    let body = fetch_donki_json("FLR", start_date, end_date, cache_root)?;
    let rows: Vec<DonkiFlrRecord> = serde_json::from_str(&body)
        .map_err(|err| FetchError::Validation(format!("invalid DONKI FLR JSON: {err}")))?;
    Ok(rows
        .into_iter()
        .map(|row| HeliosphereEventLabel {
            source: HeliosphereEventSource::DonkiFlr,
            kind: HeliosphereEventKind::SolarFlare,
            label_id: row.flr_id,
            location: "Sun".to_string(),
            event_time_utc: row.peak_time,
            linked_activity_ids: row
                .linked_events
                .iter()
                .map(|event| event.activity_id.clone())
                .collect(),
            linked_cme_ids: linked_cme_ids(&row.linked_events),
            instrument_names: row
                .instruments
                .iter()
                .map(|instrument| instrument.display_name.clone())
                .collect(),
            mission_targets: vec!["SOHO".to_string()],
            note: if row.note.trim().is_empty() {
                None
            } else {
                Some(row.note)
            },
        })
        .collect())
}

fn fetch_donki_cme_labels(
    start_date: NaiveDate,
    end_date: NaiveDate,
    cache_root: &Path,
) -> Result<Vec<HeliosphereEventLabel>, FetchError> {
    let body = fetch_donki_json("CME", start_date, end_date, cache_root)?;
    let rows: Vec<DonkiCmeRecord> = serde_json::from_str(&body)
        .map_err(|err| FetchError::Validation(format!("invalid DONKI CME JSON: {err}")))?;
    let mut labels = Vec::new();
    for row in rows {
        let linked_activity_ids = row
            .linked_events
            .iter()
            .map(|event| event.activity_id.clone())
            .collect::<Vec<_>>();
        let instrument_names = row
            .instruments
            .iter()
            .map(|instrument| instrument.display_name.clone())
            .collect::<Vec<_>>();
        let linked_cme_ids = vec![row.activity_id.clone()];
        let source_note = trimmed_option(&row.note);
        let mission_targets = mission_targets_from_instruments(&row.instruments);
        if !mission_targets.is_empty() {
            labels.push(HeliosphereEventLabel {
                source: HeliosphereEventSource::DonkiCme,
                kind: HeliosphereEventKind::CoronalMassEjection,
                label_id: row.activity_id.clone(),
                location: if row.source_location.trim().is_empty() {
                    "Sun".to_string()
                } else {
                    row.source_location.clone()
                },
                event_time_utc: row.start_time.clone(),
                linked_activity_ids: linked_activity_ids.clone(),
                linked_cme_ids: linked_cme_ids.clone(),
                instrument_names: instrument_names.clone(),
                mission_targets,
                note: source_note.clone(),
            });
        }
        for analysis in row
            .cme_analyses
            .into_iter()
            .filter(|analysis| analysis.is_most_accurate || !analysis.enlil_list.is_empty())
        {
            for enlil in analysis.enlil_list {
                for impact in enlil.impact_list {
                    let targets = mission_targets_from_location(&impact.location);
                    if targets.is_empty() || impact.arrival_time.trim().is_empty() {
                        continue;
                    }
                    let mut note_parts = Vec::new();
                    if !enlil.model_completion_time.trim().is_empty() {
                        note_parts.push(format!(
                            "model_completion_utc={}",
                            enlil.model_completion_time.trim()
                        ));
                    }
                    if !enlil.estimated_shock_arrival_time.trim().is_empty() {
                        note_parts.push(format!(
                            "estimated_shock_arrival_utc={}",
                            enlil.estimated_shock_arrival_time.trim()
                        ));
                    }
                    if impact.is_glancing_blow {
                        note_parts.push("glancing_blow=true".to_string());
                    }
                    if impact.is_minor_impact {
                        note_parts.push("minor_impact=true".to_string());
                    }
                    if let Some(note) = &source_note {
                        note_parts.push(note.clone());
                    }
                    labels.push(HeliosphereEventLabel {
                        source: HeliosphereEventSource::DonkiEnlilImpact,
                        kind: HeliosphereEventKind::PredictedCmeImpact,
                        label_id: format!(
                            "{}::{}::{}",
                            row.activity_id,
                            normalize_mission_name(&impact.location),
                            impact.arrival_time.trim()
                        ),
                        location: impact.location.clone(),
                        event_time_utc: impact.arrival_time.clone(),
                        linked_activity_ids: linked_activity_ids.clone(),
                        linked_cme_ids: linked_cme_ids.clone(),
                        instrument_names: instrument_names.clone(),
                        mission_targets: targets,
                        note: if note_parts.is_empty() {
                            None
                        } else {
                            Some(note_parts.join("; "))
                        },
                    });
                }
            }
        }
    }
    Ok(labels)
}

fn fetch_ccmc_scoreboard_arrival_labels(
    start_date: NaiveDate,
    end_date: NaiveDate,
    cache_root: &Path,
) -> Result<Vec<HeliosphereEventLabel>, FetchError> {
    let mut labels = Vec::new();
    for year in start_date.year()..=end_date.year() {
        let page = fetch_ccmc_scoreboard_year(year, cache_root)?;
        for block in parse_scoreboard_blocks(&page) {
            let Some(actual_arrival_text) = block.actual_arrival_utc.as_deref() else {
                continue;
            };
            let Some(actual_arrival) = parse_donki_time(actual_arrival_text) else {
                continue;
            };
            if actual_arrival.date_naive() < start_date || actual_arrival.date_naive() > end_date {
                continue;
            }
            labels.push(HeliosphereEventLabel {
                source: HeliosphereEventSource::CcmcScoreboardArrival,
                kind: HeliosphereEventKind::ObservedCmeArrival,
                label_id: block.cme_id.clone(),
                location: "Earth".to_string(),
                event_time_utc: actual_arrival.to_rfc3339(),
                linked_activity_ids: Vec::new(),
                linked_cme_ids: vec![block.cme_id.clone()],
                instrument_names: Vec::new(),
                mission_targets: mission_targets_from_location("Earth"),
                note: Some("official CCMC CME Scoreboard observed arrival".to_string()),
            });
        }
    }
    Ok(labels)
}

fn fetch_donki_json(
    endpoint: &str,
    start_date: NaiveDate,
    end_date: NaiveDate,
    cache_root: &Path,
) -> Result<String, FetchError> {
    let api_key = env::var("NASA_API_KEY").unwrap_or_else(|_| "DEMO_KEY".to_string());
    let nasa_url = format!(
        "{DONKI_API_ROOT}/{endpoint}?startDate={}&endDate={}&api_key={api_key}",
        start_date.format("%Y-%m-%d"),
        end_date.format("%Y-%m-%d")
    );
    let ccmc_url = format!(
        "{DONKI_CCMC_ROOT}/{endpoint}?startDate={}&endDate={}",
        start_date.format("%Y-%m-%d"),
        end_date.format("%Y-%m-%d")
    );
    let cache_path = cache_root.join("space_weather").join("donki").join(format!(
        "{}_{}_{}.json",
        endpoint.to_ascii_lowercase(),
        start_date.format("%Y%m%d"),
        end_date.format("%Y%m%d")
    ));
    if cache_path.exists() {
        return fs::read_to_string(&cache_path).map_err(FetchError::Io);
    }
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let client = Client::builder()
        .timeout(StdDuration::from_secs(60))
        .build()
        .map_err(|source| FetchError::HttpError {
            url: nasa_url.clone(),
            source: Box::new(source),
        })?;
    if let Ok(body) = fetch_donki_json_from_url(&client, &nasa_url) {
        validate_not_html(body.as_bytes())?;
        fs::write(&cache_path, &body)?;
        return Ok(body);
    }
    let body = fetch_donki_json_from_url(&client, &ccmc_url)?;
    validate_not_html(body.as_bytes())?;
    fs::write(&cache_path, &body)?;
    Ok(body)
}

fn fetch_ccmc_scoreboard_year(year: i32, cache_root: &Path) -> Result<String, FetchError> {
    let current_year = Utc::now().year();
    let url = if year == current_year {
        "https://kauai.ccmc.gsfc.nasa.gov/CMEscoreboard/".to_string()
    } else {
        format!("https://kauai.ccmc.gsfc.nasa.gov/CMEscoreboard/PreviousPredictions/{year}")
    };
    let cache_path = cache_root
        .join("space_weather")
        .join("ccmc_scoreboard")
        .join(format!("scoreboard_{year}.html"));
    if cache_path.exists() {
        return fs::read_to_string(&cache_path).map_err(FetchError::Io);
    }
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let client = Client::builder()
        .timeout(StdDuration::from_secs(60))
        .build()
        .map_err(|source| FetchError::HttpError {
            url: url.clone(),
            source: Box::new(source),
        })?;
    let body = fetch_text_from_url(&client, &url)?;
    fs::write(&cache_path, &body)?;
    Ok(body)
}

fn fetch_text_from_url(client: &Client, url: &str) -> Result<String, FetchError> {
    let mut last_status = None;
    let mut body = None;
    for attempt in 0..4 {
        let response = client
            .get(url)
            .send()
            .map_err(|source| FetchError::HttpError {
                url: url.to_string(),
                source: Box::new(source),
            })?;
        let status = response.status();
        if status.is_success() {
            body = Some(response.text().map_err(|source| FetchError::HttpError {
                url: url.to_string(),
                source: Box::new(source),
            })?);
            break;
        }
        last_status = Some(status.as_u16());
        if (status.as_u16() == 429 || status.is_server_error()) && attempt < 3 {
            sleep(StdDuration::from_secs(2_u64.pow(attempt)));
            continue;
        }
        return Err(FetchError::HttpStatus {
            url: url.to_string(),
            status: status.as_u16(),
        });
    }
    body.ok_or_else(|| FetchError::HttpStatus {
        url: url.to_string(),
        status: last_status.unwrap_or(599),
    })
}

fn fetch_donki_json_from_url(client: &Client, url: &str) -> Result<String, FetchError> {
    let mut last_status = None;
    let mut body = None;
    for attempt in 0..4 {
        let response = client
            .get(url)
            .send()
            .map_err(|source| FetchError::HttpError {
                url: url.to_string(),
                source: Box::new(source),
            })?;
        let status = response.status();
        if status.is_success() {
            body = Some(response.text().map_err(|source| FetchError::HttpError {
                url: url.to_string(),
                source: Box::new(source),
            })?);
            break;
        }
        last_status = Some(status.as_u16());
        if (status.as_u16() == 429 || status.is_server_error()) && attempt < 3 {
            sleep(StdDuration::from_secs(2_u64.pow(attempt)));
            continue;
        }
        return Err(FetchError::HttpStatus {
            url: url.to_string(),
            status: status.as_u16(),
        });
    }
    body.ok_or_else(|| FetchError::HttpStatus {
        url: url.to_string(),
        status: last_status.unwrap_or(599),
    })
}

fn linked_cme_ids(events: &[DonkiLinkedEvent]) -> Vec<String> {
    events
        .iter()
        .map(|event| event.activity_id.clone())
        .filter(|value| value.contains("-CME-"))
        .collect()
}

fn mission_targets_from_instruments(instruments: &[DonkiInstrument]) -> Vec<String> {
    let mut targets = Vec::new();
    for instrument in instruments {
        targets.extend(mission_targets_from_location(&instrument.display_name));
    }
    targets.sort();
    targets.dedup();
    targets
}

fn infer_sep_location(instruments: &[DonkiInstrument]) -> String {
    let names = instruments
        .iter()
        .map(|instrument| instrument.display_name.to_ascii_uppercase())
        .collect::<Vec<_>>();
    if names.iter().any(|name| name.contains("STEREO A")) {
        return "STEREO A".to_string();
    }
    if names.iter().any(|name| name.contains("SOHO")) {
        return "SOHO".to_string();
    }
    "Earth".to_string()
}

fn mission_targets_from_location(location: &str) -> Vec<String> {
    match normalize_mission_name(location).as_str() {
        "earth" => vec![
            "OMNI".to_string(),
            "ACE".to_string(),
            "WIND".to_string(),
            "SOHO".to_string(),
        ],
        "stereo-a" => vec!["STEREO-A".to_string()],
        "soho" => vec!["SOHO".to_string()],
        "solar-orbiter" => vec!["Solar Orbiter".to_string()],
        "bepicolombo" => vec!["BepiColombo".to_string()],
        "psp" | "parker-solar-probe" => vec!["PSP".to_string()],
        "juno" => vec!["Juno".to_string()],
        "new-horizons" => vec!["New Horizons".to_string()],
        "juice" => vec!["JUICE".to_string()],
        "osiris-apex" => vec!["OSIRIS-APEX".to_string()],
        _ => Vec::new(),
    }
}

fn normalize_mission_name(name: &str) -> String {
    name.trim()
        .to_ascii_lowercase()
        .replace(['(', ')', ':', '/'], " ")
        .replace([' ', '_'], "-")
        .replace("--", "-")
}

fn parse_donki_time(value: &str) -> Option<DateTime<Utc>> {
    let text = value.trim();
    if let Ok(date) = DateTime::parse_from_rfc3339(text) {
        return Some(date.with_timezone(&Utc));
    }
    if let Ok(date) = NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%MZ") {
        return Some(DateTime::<Utc>::from_naive_utc_and_offset(date, Utc));
    }
    if let Ok(date) = NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%SZ") {
        return Some(DateTime::<Utc>::from_naive_utc_and_offset(date, Utc));
    }
    None
}

fn parse_scoreboard_blocks(html: &str) -> Vec<ScoreboardCmeBlock> {
    let cme_re = Regex::new(
        r#"<a href="https://kauai\.ccmc\.gsfc\.nasa\.gov/DONKI/view/CME/[^"]*"[^>]*><b>CME:\s*([0-9T:\-]+-CME-[0-9]+)</b></a>"#,
    )
    .expect("valid cme block regex");
    let row_re = Regex::new(r#"(?s)<tr>(?P<row>.*?)</tr>"#).expect("valid scoreboard row regex");
    let td_re =
        Regex::new(r#"(?s)<td[^>]*>(?P<cell>.*?)</td>"#).expect("valid scoreboard cell regex");
    let actual_re = Regex::new(r#"Actual Shock Arrival Time:\s*([0-9T:\-]+Z)"#)
        .expect("valid actual arrival regex");
    let mut blocks = Vec::new();
    let captures = cme_re
        .captures_iter(html)
        .filter_map(|capture| {
            let whole = capture.get(0)?;
            let cme_id = capture.get(1)?.as_str().to_string();
            Some((whole.start(), whole.end(), cme_id))
        })
        .collect::<Vec<_>>();
    for (idx, (_start, body_start, cme_id)) in captures.iter().enumerate() {
        let body_end = captures
            .get(idx + 1)
            .map(|(next_start, _, _)| *next_start)
            .unwrap_or(html.len());
        let body = &html[*body_start..body_end];
        let actual_arrival_utc = actual_re
            .captures(body)
            .and_then(|caps| caps.get(1).map(|value| value.as_str().to_string()));
        let detected_at_earth = !body.contains("This CME was not detected at Earth!");
        let mut predictions = Vec::new();
        for row_caps in row_re.captures_iter(body) {
            let row_html = row_caps["row"].to_string();
            let cells = td_re
                .captures_iter(&row_html)
                .filter_map(|caps| {
                    caps.get(1)
                        .map(|value| clean_scoreboard_cell(value.as_str()))
                })
                .collect::<Vec<_>>();
            if cells.len() < 7 {
                continue;
            }
            let predicted_arrival_utc = cells[0].trim().to_string();
            if parse_donki_time(&predicted_arrival_utc).is_none() {
                continue;
            }
            let residual_hours = parse_scoreboard_residual(&cells[1]);
            let method = cells
                .get(6)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "unknown".to_string());
            predictions.push(ScoreboardPredictionRow {
                predicted_arrival_utc,
                method,
                residual_hours,
            });
        }
        blocks.push(ScoreboardCmeBlock {
            cme_id: cme_id.clone(),
            actual_arrival_utc,
            detected_at_earth,
            predictions,
        });
    }
    blocks
}

fn clean_scoreboard_cell(cell_html: &str) -> String {
    let tag_re = Regex::new(r"(?s)<[^>]+>").expect("valid tag regex");
    let whitespace_re = Regex::new(r"\s+").expect("valid whitespace regex");
    let without_tags = tag_re.replace_all(cell_html, " ");
    whitespace_re
        .replace_all(&without_tags, " ")
        .trim()
        .to_string()
}

fn parse_scoreboard_residual(value: &str) -> Option<f64> {
    let text = value.trim();
    if text == "----" || text == "---" || text.is_empty() {
        return None;
    }
    text.parse::<f64>().ok()
}

fn trimmed_option(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ips_json() {
        let body = r#"[{
            "activityID":"2024-03-24T14:10:00-IPS-001",
            "location":"Earth",
            "eventTime":"2024-03-24T14:10Z",
            "instruments":[{"displayName":"ACE: SWEPAM"},{"displayName":"ACE: MAG"}],
            "linkedEvents":[{"activityID":"2024-03-23T01:25:00-CME-001"}]
        }]"#;
        let rows: Vec<DonkiIpsRecord> = serde_json::from_str(body).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].activity_id, "2024-03-24T14:10:00-IPS-001");
        assert_eq!(rows[0].location, "Earth");
    }

    #[test]
    fn parse_sep_location_from_instrument() {
        let instruments = vec![DonkiInstrument {
            display_name: "STEREO A: IMPACT 13-100 MeV".to_string(),
        }];
        assert_eq!(infer_sep_location(&instruments), "STEREO A");
    }

    #[test]
    fn prediction_windows_match_mission_targets() {
        let labels = vec![HeliosphereEventLabel {
            source: HeliosphereEventSource::DonkiIps,
            kind: HeliosphereEventKind::InterplanetaryShock,
            label_id: "L1".to_string(),
            location: "Earth".to_string(),
            event_time_utc: "2024-03-24T14:10Z".to_string(),
            linked_activity_ids: Vec::new(),
            linked_cme_ids: Vec::new(),
            instrument_names: Vec::new(),
            mission_targets: mission_targets_from_location("Earth"),
            note: None,
        }];
        let windows = labels_to_prediction_windows(&labels, "ACE", 6);
        assert_eq!(windows.len(), 1);
        assert!(windows[0].window_start_utc < windows[0].event_time_utc);
    }

    #[test]
    fn parse_scoreboard_block_extracts_arrival_and_residuals() {
        let html = r#"
<a href="https://kauai.ccmc.gsfc.nasa.gov/DONKI/view/CME/43719/-1" target="_blank"><b>CME: 2026-01-01T19:36:00-CME-001</b></a>
<td>Actual Shock Arrival Time: 2026-01-04T20:41Z</td>
<tr>
<td>2026-01-04T15:17Z</td>
<td align="right">-5.40</td>
<td align="right">----</td>
<td>2026-01-01T22:31Z</td>
<td>70.17</td>
<td>Max Kp Range: 4.0 - 6.0<br></td>
<td>WSA-ENLIL + Cone (NASA M2M)</td>
<td>Melissa Kane (M2M SWAO)</td>
<td><a href="/CMEscoreboard/prediction/detail/156 "> Detail</a></td>
</tr>
"#;
        let blocks = parse_scoreboard_blocks(html);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].cme_id, "2026-01-01T19:36:00-CME-001");
        assert_eq!(
            blocks[0].actual_arrival_utc.as_deref(),
            Some("2026-01-04T20:41Z")
        );
        assert_eq!(blocks[0].predictions.len(), 1);
        assert_eq!(blocks[0].predictions[0].residual_hours, Some(-5.40));
        assert_eq!(
            blocks[0].predictions[0].method,
            "WSA-ENLIL + Cone (NASA M2M)"
        );
    }

    #[test]
    fn donki_cme_target_mapping_covers_modern_missions() {
        assert_eq!(
            mission_targets_from_location("Solar Orbiter"),
            vec!["Solar Orbiter".to_string()]
        );
        assert_eq!(
            mission_targets_from_location("BepiColombo"),
            vec!["BepiColombo".to_string()]
        );
        assert_eq!(
            mission_targets_from_location("STEREO A"),
            vec!["STEREO-A".to_string()]
        );
    }
}
