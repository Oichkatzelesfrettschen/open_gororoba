//! Official heliosphere event-label ingestion and windowing helpers.
//!
//! This module stages bounded official event labels from NASA DONKI so the
//! heliosphere analysis lanes can test physically anchored hypotheses instead
//! of relying only on internal heuristics.
//!
//! Official sources:
//! - <https://api.nasa.gov/>
//! - <https://api.nasa.gov/assets/html/authentication.html>
//! - <https://ccmc.gsfc.nasa.gov/donki/>
//!
//! The first implementation uses DONKI event families that are directly usable
//! as onset labels for near-Earth and STEREO-A heliosphere windows:
//! - `IPS` (interplanetary shock)
//! - `GST` (geomagnetic storm onset)
//! - `SEP` (solar energetic particle onset)
//! - `FLR` (flare peak time, kept as an official solar-origin label family)

use crate::fetcher::{FetchError, validate_not_html};
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, Utc};
use reqwest::blocking::Client;
use serde::{Deserialize, Deserializer, Serialize};
use std::{
    env, fs,
    path::Path,
    thread::sleep,
    time::Duration as StdDuration,
};

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
}

/// Physical event family represented by an official label.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HeliosphereEventKind {
    InterplanetaryShock,
    GeomagneticStorm,
    SolarEnergeticParticle,
    SolarFlare,
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
    pub label_id: String,
    pub mission: String,
    pub residual_hours: f64,
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
    #[serde(default, rename = "linkedEvents", deserialize_with = "null_vec_default")]
    linked_events: Vec<DonkiLinkedEvent>,
}

#[derive(Debug, Deserialize)]
struct DonkiGstRecord {
    #[serde(rename = "gstID")]
    gst_id: String,
    #[serde(rename = "startTime")]
    start_time: String,
    #[serde(default, rename = "linkedEvents", deserialize_with = "null_vec_default")]
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
    #[serde(default, rename = "linkedEvents", deserialize_with = "null_vec_default")]
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
    #[serde(default, rename = "linkedEvents", deserialize_with = "null_vec_default")]
    linked_events: Vec<DonkiLinkedEvent>,
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
    labels.sort_by(|a, b| {
        (a.event_time_utc.as_str(), a.label_id.as_str())
            .cmp(&(b.event_time_utc.as_str(), b.label_id.as_str()))
    });
    Ok(labels)
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
        _ => Vec::new(),
    }
}

fn normalize_mission_name(name: &str) -> String {
    name.trim()
        .to_ascii_lowercase()
        .replace([' ', '_'], "-")
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
}
