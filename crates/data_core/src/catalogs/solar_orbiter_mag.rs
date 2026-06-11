//! Solar Orbiter MAG RTN 1-minute parser via CDAWeb HAPI.
//!
//! This promotes a mission-native magnetic-field follow-on beyond the merged
//! hourly support lane so the feature-cube path can compare Solar Orbiter's
//! native MAG product against the merged feed.

use crate::{
    fetcher::FetchError,
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use csv::ReaderBuilder;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct SolarOrbiterMagRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub br: f64,
    pub bt: f64,
    pub bn: f64,
    pub b_magnitude: f64,
}

#[cfg(feature = "fetch")]
pub use super::solar_orbiter_mag_fetch::SolarOrbiterMagProvider;

#[derive(Default)]
struct MagAccumulator {
    br_sum: f64,
    bt_sum: f64,
    bn_sum: f64,
    count: usize,
}

pub fn parse_solar_orbiter_mag_hapi_csv(content: &str) -> Vec<SolarOrbiterMagRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let headers = match reader.headers() {
        Ok(headers) => headers.clone(),
        Err(_) => return Vec::new(),
    };
    let br_col = headers.iter().position(|value| value == "B_RTN_0");
    let bt_col = headers.iter().position(|value| value == "B_RTN_1");
    let bn_col = headers.iter().position(|value| value == "B_RTN_2");
    let (Some(br_col), Some(bt_col), Some(bn_col)) = (br_col, bt_col, bn_col) else {
        return Vec::new();
    };

    let mut hourly: BTreeMap<(u16, u16, u8), MagAccumulator> = BTreeMap::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let br = parse_hapi_spacephysics_f64_or_nan(record.get(br_col).unwrap_or(""));
        let bt = parse_hapi_spacephysics_f64_or_nan(record.get(bt_col).unwrap_or(""));
        let bn = parse_hapi_spacephysics_f64_or_nan(record.get(bn_col).unwrap_or(""));
        if !br.is_finite() || !bt.is_finite() || !bn.is_finite() {
            continue;
        }
        let entry = hourly.entry((year, doy, hour)).or_default();
        entry.br_sum += br;
        entry.bt_sum += bt;
        entry.bn_sum += bn;
        entry.count += 1;
    }

    hourly
        .into_iter()
        .map(|((year, doy, hour), acc)| {
            let count = acc.count.max(1) as f64;
            let br = acc.br_sum / count;
            let bt = acc.bt_sum / count;
            let bn = acc.bn_sum / count;
            SolarOrbiterMagRecord {
                year,
                doy,
                hour,
                br,
                bt,
                bn,
                b_magnitude: (br * br + bt * bt + bn * bn).sqrt(),
            }
        })
        .collect()
}

pub fn parse_solar_orbiter_mag_file(
    path: &std::path::Path,
) -> Result<Vec<SolarOrbiterMagRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
    Ok(parse_solar_orbiter_mag_hapi_csv(&content))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_solar_orbiter_mag_hapi_csv() {
        let csv = "Time,B_RTN_0,B_RTN_1,B_RTN_2\n\
2020-06-16T00:00:29Z,1.0,-2.0,3.0\n\
2020-06-16T00:01:29Z,3.0,-4.0,5.0\n";
        let rows = parse_solar_orbiter_mag_hapi_csv(csv);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.year, 2020);
        assert_eq!(row.hour, 0);
        assert!((row.br - 2.0).abs() < 1.0e-9);
        assert!(row.b_magnitude > 5.0);
    }
}
