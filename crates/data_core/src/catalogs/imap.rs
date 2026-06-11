//! IMAP public-product staging providers.
//!
//! Verified public official surfaces:
//!   - IMAP helio1hr position support:
//!     <https://cdaweb.gsfc.nasa.gov/pub/data/imap/helio1hr/>
//!   - IMAP-Hi L2 ENA h90 product family:
//!     <https://cdaweb.gsfc.nasa.gov/pub/data/imap/hi/l2/h90-ena-h-sf-nsp-full-4deg-3mo/>
//!   - IMAP I-ALiRT archive CDF lane:
//!     <https://cdaweb.gsfc.nasa.gov/pub/data/imap/ialirt/l1/realtime/>
//!   - IMAP I-ALiRT public live JSON API:
//!     <https://ialirt.imap-mission.com/space-weather>
//!     <https://ialirt.imap-mission.com/ialirt-archive-query>
//!
//! Fetch logic lives in `imap_fetch`.

#[cfg(feature = "fetch")]
use chrono::{DateTime, Datelike, TimeZone, Timelike, Utc};
#[cfg(feature = "fetch")]
use serde_json::Value;
#[cfg(feature = "fetch")]
use std::collections::BTreeMap;

#[cfg(feature = "fetch")]
pub const AU_KM: f64 = 149_597_870.7;

#[derive(Debug, Clone)]
pub struct ImapHelio1hrRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub r_au: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
}

#[cfg(feature = "fetch")]
pub use super::imap_fetch::{
    ImapHelio1hrProvider, ImapHiL2H90Provider, ImapIalirtRealtimeProvider,
    parse_imap_helio1hr_file, parse_imap_hi_h90_file, parse_imap_ialirt_file,
    parse_imap_ialirt_live_day,
};

#[derive(Debug, Clone)]
pub struct ImapHiH90Summary {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub map_flux_mean: f64,
    pub map_flux_std: f64,
    pub pixel_count: usize,
    pub energy_bin_count: usize,
}

#[derive(Debug, Clone)]
pub struct ImapIalirtRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub r_au: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub pseudo_density: f64,
    pub pseudo_speed: f64,
    pub pseudo_temperature: f64,
    pub br: f64,
    pub bt: f64,
    pub bn: f64,
    pub b_magnitude: f64,
    pub spectral_mean: f64,
    pub spectral_peak: f64,
}

#[cfg(feature = "fetch")]
#[derive(Default)]
pub(crate) struct IalirtAccumulator {
    pub r_sum: f64,
    pub lat_sum: f64,
    pub lon_sum: f64,
    pub pos_count: usize,
    pub density_sum: f64,
    pub density_count: usize,
    pub speed_sum: f64,
    pub speed_count: usize,
    pub temp_sum: f64,
    pub temp_count: usize,
    pub br_sum: f64,
    pub bt_sum: f64,
    pub bn_sum: f64,
    pub bmag_sum: f64,
    pub mag_count: usize,
    pub spectral_mean_sum: f64,
    pub spectral_mean_count: usize,
    pub spectral_peak_sum: f64,
    pub spectral_peak_count: usize,
}

#[cfg(any(feature = "fetch", test))]
pub(crate) fn sanitize_numeric(value: f64) -> f64 {
    if !value.is_finite() || value.abs() >= 1.0e30 {
        f64::NAN
    } else {
        value
    }
}

#[cfg(any(feature = "fetch", test))]
pub(crate) fn mean(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.iter().sum::<f64>() / finite.len() as f64
}

#[cfg(any(feature = "fetch", test))]
pub(crate) fn stddev(values: &[f64], mean: f64) -> f64 {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.len() < 2 || !mean.is_finite() {
        return 0.0;
    }
    let var = finite
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / finite.len() as f64;
    var.sqrt()
}

#[cfg(feature = "fetch")]
pub(crate) fn json_f64(value: &Value, key: &str) -> f64 {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map_or(f64::NAN, sanitize_numeric)
}

#[cfg(feature = "fetch")]
pub(crate) fn json_vec_f64(value: &Value, key: &str) -> Vec<f64> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_f64)
                .map(sanitize_numeric)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

#[cfg(feature = "fetch")]
pub(crate) fn iso_time_to_ydh(value: &str) -> Option<(u16, u16, u8)> {
    use chrono::NaiveDateTime;
    let dt = if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
        dt.with_timezone(&Utc)
    } else {
        let naive = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S").ok()?;
        Utc.from_utc_datetime(&naive)
    };
    Some((dt.year() as u16, dt.ordinal() as u16, dt.hour() as u8))
}

#[cfg(feature = "fetch")]
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

#[cfg(feature = "fetch")]
pub(crate) fn rows_from_accumulator(
    hourly: BTreeMap<(u16, u16, u8), IalirtAccumulator>,
) -> Vec<ImapIalirtRecord> {
    hourly
        .into_iter()
        .map(|((year, doy, hour), acc)| ImapIalirtRecord {
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
            pseudo_density: if acc.density_count > 0 {
                acc.density_sum / acc.density_count as f64
            } else {
                f64::NAN
            },
            pseudo_speed: if acc.speed_count > 0 {
                acc.speed_sum / acc.speed_count as f64
            } else {
                f64::NAN
            },
            pseudo_temperature: if acc.temp_count > 0 {
                acc.temp_sum / acc.temp_count as f64
            } else {
                f64::NAN
            },
            br: if acc.mag_count > 0 {
                acc.br_sum / acc.mag_count as f64
            } else {
                f64::NAN
            },
            bt: if acc.mag_count > 0 {
                acc.bt_sum / acc.mag_count as f64
            } else {
                f64::NAN
            },
            bn: if acc.mag_count > 0 {
                acc.bn_sum / acc.mag_count as f64
            } else {
                f64::NAN
            },
            b_magnitude: if acc.mag_count > 0 {
                acc.bmag_sum / acc.mag_count as f64
            } else {
                f64::NAN
            },
            spectral_mean: if acc.spectral_mean_count > 0 {
                acc.spectral_mean_sum / acc.spectral_mean_count as f64
            } else {
                f64::NAN
            },
            spectral_peak: if acc.spectral_peak_count > 0 {
                acc.spectral_peak_sum / acc.spectral_peak_count as f64
            } else {
                f64::NAN
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imap_summary_stats() {
        let values = vec![1.0, 2.0, 3.0];
        let mean_value = mean(&values);
        assert!((mean_value - 2.0).abs() < 1.0e-12);
        assert!(stddev(&values, mean_value) > 0.8);
    }

    #[test]
    fn test_sanitize_numeric() {
        assert!(sanitize_numeric(-1.0e31).is_nan());
        assert!((sanitize_numeric(3.5) - 3.5).abs() < 1.0e-12);
    }
}
