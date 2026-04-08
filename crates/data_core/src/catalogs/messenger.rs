//! MESSENGER MAG data provider for Mercury boundary analysis.
//!
//! MESSENGER orbited Mercury from 2011-2015, crossing the bow shock and
//! magnetopause on every orbit (~8h period). Mercury has an intrinsic
//! dipole but is much smaller than Earth's, making it strongly solar-wind-
//! coupled. The Zenodo curated crossing list provides 4054 MP+BS events.
//!
//! Data: RTN coordinates via CDAWeb HAPI (MESSENGER_MAG_RTN@0).
//! Reference: Anderson et al. (2007), Space Sci. Rev. 131, 417

use crate::{
    fetcher::{DailyHapiFetchRequest, DatasetProvider, FetchConfig, FetchError, fetch_daily_hapi_csv_range},
    parse::parse_hapi_spacephysics_f64_or_nan,
};
use chrono::{DateTime, Datelike, Timelike, Utc};
use csv::ReaderBuilder;
use std::{collections::BTreeMap, path::PathBuf};

const MESSENGER_MAG_HAPI_DATASET: &str = "MESSENGER_MAG_RTN@0";

/// Minute-resolution MESSENGER MAG record in RTN coordinates.
#[derive(Debug, Clone)]
pub struct MessengerMagMinuteRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub minute: u8,
    pub elapsed_hours: f64,
    pub br: f64,
    pub bt: f64,
    pub bn: f64,
    pub b_magnitude: f64,
}

/// Parse MESSENGER MAG HAPI CSV into minute-averaged records.
pub fn parse_messenger_mag_hapi_csv_minutes(content: &str) -> Vec<MessengerMagMinuteRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());

    #[derive(Default)]
    struct MinuteAcc {
        br: f64,
        bt: f64,
        bn: f64,
        count: usize,
    }

    let mut buckets: BTreeMap<(u16, u16, u8, u8), MinuteAcc> = BTreeMap::new();

    for record in reader.records().flatten() {
        let Some(time_str) = record.get(0) else {
            continue;
        };
        let Ok(dt) = DateTime::parse_from_rfc3339(time_str) else {
            continue;
        };
        let utc = dt.with_timezone(&Utc);

        // Columns: Time, radialDistance, lat, az, B_radial_q, B_tangential_q, B_normal_q,
        //          B_radial, B_tangential, B_normal, Quality_Flag, Epoch_cadence
        // Use quality-filtered columns (indices 4,5,6) or all-quality (7,8,9)
        let br = parse_hapi_spacephysics_f64_or_nan(record.get(7).unwrap_or(""));
        let bt = parse_hapi_spacephysics_f64_or_nan(record.get(8).unwrap_or(""));
        let bn = parse_hapi_spacephysics_f64_or_nan(record.get(9).unwrap_or(""));
        if !br.is_finite() || !bt.is_finite() || !bn.is_finite() {
            continue;
        }

        let key = (
            utc.year() as u16,
            utc.ordinal() as u16,
            utc.hour() as u8,
            utc.minute() as u8,
        );
        let acc = buckets.entry(key).or_default();
        acc.br += br;
        acc.bt += bt;
        acc.bn += bn;
        acc.count += 1;
    }

    let keys: Vec<_> = buckets.keys().copied().collect();
    let first = keys.first().copied();

    keys.into_iter()
        .filter_map(|key| {
            let acc = &buckets[&key];
            if acc.count == 0 {
                return None;
            }
            let n = acc.count as f64;
            let (year, doy, hour, minute) = key;
            let br = acc.br / n;
            let bt = acc.bt / n;
            let bn = acc.bn / n;

            let elapsed = match first {
                Some((fy, fd, fh, fm)) => {
                    (year as f64 - fy as f64) * 365.25 * 24.0
                        + (doy as f64 - fd as f64) * 24.0
                        + (hour as f64 - fh as f64)
                        + (minute as f64 - fm as f64) / 60.0
                }
                None => 0.0,
            };

            Some(MessengerMagMinuteRecord {
                year,
                doy,
                hour,
                minute,
                elapsed_hours: elapsed,
                br,
                bt,
                bn,
                b_magnitude: (br * br + bt * bt + bn * bn).sqrt(),
            })
        })
        .collect()
}

/// MESSENGER MAG provider. Daily HAPI chunks.
pub struct MessengerMagProvider {
    pub year: u16,
    pub doy_start: u16,
    pub doy_end: u16,
}

impl DatasetProvider for MessengerMagProvider {
    fn name(&self) -> &str {
        "MESSENGER MAG RTN"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        fetch_daily_hapi_csv_range(
            config,
            &DailyHapiFetchRequest {
                subdir: "messenger",
                file_prefix: "messenger_mag",
                log_label: "MESSENGER MAG",
                dataset_id: MESSENGER_MAG_HAPI_DATASET,
                year: self.year,
                doy_start: self.doy_start,
                doy_end: self.doy_end,
                parameters: None,
            },
        )
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("messenger").exists()
    }
}
