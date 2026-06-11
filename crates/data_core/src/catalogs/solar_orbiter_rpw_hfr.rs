//! Solar Orbiter RPW HFR survey-flux parser via official CDAWeb products.
//!
//! Public official sources:
//!   <https://cdaweb.gsfc.nasa.gov/hapi/info?id=SOLO_L3_RPW-HFR-SURV-FLUX>
//!   <https://cdaweb.gsfc.nasa.gov/pub/data/solar-orbiter/rpw/science/l3/hfr-surv-flux/>
//!
//! The executed Rust lane stages the authoritative daily CDFs for provenance
//! and materializes parser-friendly daily HAPI CSV mirrors for feature-cube use.
//!
//! Fetch logic lives in `solar_orbiter_rpw_hfr_fetch`.

use crate::parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh};
use std::collections::BTreeMap;

const AU_KM: f64 = 149_597_870.7;

#[derive(Debug, Clone)]
pub struct SolarOrbiterRpwHfrRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub r_au: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub spectral_mean: f64,
    pub spectral_peak: f64,
    pub band_count: usize,
}

#[cfg(feature = "fetch")]
pub use super::solar_orbiter_rpw_hfr_fetch::{
    SolarOrbiterRpwHfrProvider, parse_solar_orbiter_rpw_hfr_file,
};

#[derive(Default)]
pub(crate) struct HfrAccumulator {
    pub r_sum: f64,
    pub lat_sum: f64,
    pub lon_sum: f64,
    pub pos_count: usize,
    pub spectral_sum: f64,
    pub spectral_count: usize,
    pub spectral_peak: f64,
    pub band_count: usize,
}

pub(crate) fn km_xyz_to_spherical_au(x_km: f64, y_km: f64, z_km: f64) -> (f64, f64, f64) {
    if !x_km.is_finite() || !y_km.is_finite() || !z_km.is_finite() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let r_km = (x_km * x_km + y_km * y_km + z_km * z_km).sqrt();
    if r_km == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let r_au = r_km / AU_KM;
    let lat_deg = (z_km / r_km).asin().to_degrees();
    let lon_deg = y_km.atan2(x_km).to_degrees();
    (r_au, lat_deg, lon_deg)
}

pub fn parse_solar_orbiter_rpw_hfr_csv(content: &str) -> Vec<SolarOrbiterRpwHfrRecord> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let headers = match reader.headers() {
        Ok(headers) => headers.clone(),
        Err(_) => return Vec::new(),
    };
    let spectral_cols = headers
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| value.starts_with("PSD_FLUX_DB_").then_some(idx))
        .collect::<Vec<_>>();
    let x_col = headers.iter().position(|value| value == "SC_POS_HCI_0");
    let y_col = headers.iter().position(|value| value == "SC_POS_HCI_1");
    let z_col = headers.iter().position(|value| value == "SC_POS_HCI_2");
    let (Some(x_col), Some(y_col), Some(z_col)) = (x_col, y_col, z_col) else {
        return Vec::new();
    };
    if spectral_cols.is_empty() {
        return Vec::new();
    }
    let mut hourly: BTreeMap<(u16, u16, u8), HfrAccumulator> = BTreeMap::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let x = parse_hapi_spacephysics_f64_or_nan(record.get(x_col).unwrap_or(""));
        let y = parse_hapi_spacephysics_f64_or_nan(record.get(y_col).unwrap_or(""));
        let z = parse_hapi_spacephysics_f64_or_nan(record.get(z_col).unwrap_or(""));
        let bands = spectral_cols
            .iter()
            .filter_map(|idx| record.get(*idx))
            .map(parse_hapi_spacephysics_f64_or_nan)
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if bands.is_empty() && !(x.is_finite() && y.is_finite() && z.is_finite()) {
            continue;
        }
        let entry = hourly.entry((year, doy, hour)).or_default();
        if x.is_finite() && y.is_finite() && z.is_finite() {
            let (r_au, lat_deg, lon_deg) = km_xyz_to_spherical_au(x, y, z);
            if r_au.is_finite() && lat_deg.is_finite() && lon_deg.is_finite() {
                entry.r_sum += r_au;
                entry.lat_sum += lat_deg;
                entry.lon_sum += lon_deg;
                entry.pos_count += 1;
            }
        }
        entry.band_count = entry.band_count.max(bands.len());
        for band in bands {
            entry.spectral_sum += band;
            entry.spectral_count += 1;
            entry.spectral_peak = entry.spectral_peak.max(band);
        }
    }
    hourly
        .into_iter()
        .filter_map(|((year, doy, hour), acc)| {
            if acc.spectral_count == 0 && acc.pos_count == 0 {
                return None;
            }
            Some(SolarOrbiterRpwHfrRecord {
                year,
                doy,
                hour,
                r_au: if acc.pos_count > 0 {
                    acc.r_sum / acc.pos_count as f64
                } else {
                    f64::NAN
                },
                lat_deg: if acc.pos_count > 0 {
                    acc.lat_sum / acc.pos_count as f64
                } else {
                    f64::NAN
                },
                lon_deg: if acc.pos_count > 0 {
                    acc.lon_sum / acc.pos_count as f64
                } else {
                    f64::NAN
                },
                spectral_mean: if acc.spectral_count > 0 {
                    acc.spectral_sum / acc.spectral_count as f64
                } else {
                    f64::NAN
                },
                spectral_peak: if acc.spectral_peak > 0.0 || acc.spectral_peak.is_finite() {
                    acc.spectral_peak
                } else {
                    f64::NAN
                },
                band_count: acc.band_count,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_solar_orbiter_rpw_hfr_csv() {
        let csv = "Time,PSD_FLUX_DB_0,PSD_FLUX_DB_1,PSD_FLUX_DB_2,SC_POS_HCI_0,SC_POS_HCI_1,SC_POS_HCI_2\n\
2020-06-15T00:00:49.038180000Z,1.76,1.05,0.82,2.49e+07,7.25e+07,-8.95e+06\n\
2020-06-15T00:01:42.033410000Z,0.83,0.91,0.55,2.49e+07,7.25e+07,-8.95e+06\n";
        let rows = parse_solar_orbiter_rpw_hfr_csv(csv);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.band_count, 3);
        assert!(row.spectral_mean.is_finite());
        assert!((row.spectral_peak - 1.76).abs() < 1.0e-9);
        assert!(row.r_au.is_finite());
    }
}
