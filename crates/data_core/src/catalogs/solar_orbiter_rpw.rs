//! Solar Orbiter RPW BIA spacecraft-potential parser via direct CDAWeb CDFs.
//!
//! This follow-on product complements the merged hourly and SWA/MAG lanes with
//! higher-cadence electrostatic/plasma-environment support from the official
//! public RPW BIA spacecraft-potential product family.
//!
//! Fetch logic lives in `solar_orbiter_rpw_fetch`.

use crate::parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh};
use csv::ReaderBuilder;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct SolarOrbiterRpwRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub scpot: f64,
    pub psp: f64,
}

#[derive(Default)]
pub(crate) struct ScalarAccumulator {
    pub scpot_sum: f64,
    pub scpot_count: usize,
    pub psp_sum: f64,
    pub psp_count: usize,
}

pub fn parse_solar_orbiter_rpw_hapi_csv(content: &str) -> Vec<SolarOrbiterRpwRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let headers = match reader.headers() {
        Ok(headers) => headers.clone(),
        Err(_) => return Vec::new(),
    };
    let scpot_col = headers.iter().position(|value| value == "SCPOT");
    let psp_col = headers.iter().position(|value| value == "PSP");
    let (Some(scpot_col), Some(psp_col)) = (scpot_col, psp_col) else {
        return Vec::new();
    };
    let mut hourly: BTreeMap<(u16, u16, u8), ScalarAccumulator> = BTreeMap::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let scpot = parse_hapi_spacephysics_f64_or_nan(record.get(scpot_col).unwrap_or(""));
        let psp = parse_hapi_spacephysics_f64_or_nan(record.get(psp_col).unwrap_or(""));
        let entry = hourly.entry((year, doy, hour)).or_default();
        if scpot.is_finite() {
            entry.scpot_sum += scpot;
            entry.scpot_count += 1;
        }
        if psp.is_finite() {
            entry.psp_sum += psp;
            entry.psp_count += 1;
        }
    }
    hourly
        .into_iter()
        .filter_map(|((year, doy, hour), acc)| {
            let scpot = if acc.scpot_count > 0 {
                acc.scpot_sum / acc.scpot_count as f64
            } else {
                f64::NAN
            };
            let psp = if acc.psp_count > 0 {
                acc.psp_sum / acc.psp_count as f64
            } else {
                f64::NAN
            };
            if !scpot.is_finite() && !psp.is_finite() {
                return None;
            }
            Some(SolarOrbiterRpwRecord {
                year,
                doy,
                hour,
                scpot,
                psp,
            })
        })
        .collect()
}
