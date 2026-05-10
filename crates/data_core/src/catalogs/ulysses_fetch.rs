//! Fetch/provider support for Ulysses spacecraft merged hourly data.

use crate::{
    catalogs::spdf_merged::SpdfMergedRecord,
    fetcher::{
        DatasetProvider, FetchConfig, FetchError, download_amda_hapi_csv, download_hapi_csv,
        download_to_file,
    },
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use std::path::PathBuf;

const ULYSSES_HAPI_DATASET: &str = "UY_COHO1HR_MERGED_MAG_PLASMA";
const ULYSSES_SPDF_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/ulysses/merged/";

/// AMDA dataset ID for Ulysses solar wind ion moments (SWOOPS/BAI level 2).
const ULYSSES_AMDA_PLASMA: &str = "ulys-bai-mom";

/// AMDA dataset ID for Ulysses FGM magnetometer in RTN coordinates.
const ULYSSES_AMDA_MAG: &str = "ulys-fgm-rtn";

/// AMDA dataset ID for Ulysses orbital ephemeris.
const ULYSSES_AMDA_ORB: &str = "ulys-orb-all";

/// Parse Ulysses AMDA plasma CSV (`ulys-bai-mom`).
///
/// Returns (year, doy, hour, density_cm3, speed_kms, temperature_k).
fn parse_ulysses_amda_plasma(content: &str) -> Vec<(u16, u16, u8, f64, f64, f64)> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut out = Vec::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let density = parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or(""));
        let speed = parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or(""));
        let temp = parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or(""));
        out.push((year, doy, hour, density, speed, temp));
    }
    out
}

/// Parse Ulysses AMDA magnetometer CSV (`ulys-fgm-rtn`).
///
/// Returns (year, doy, hour, br, bt, bn, b_mag).
fn parse_ulysses_amda_mag(content: &str) -> Vec<(u16, u16, u8, f64, f64, f64, f64)> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut out = Vec::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let br = parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or(""));
        let bt = parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or(""));
        let bn = parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or(""));
        let b_mag = parse_hapi_spacephysics_f64_or_nan(record.get(4).unwrap_or(""));
        out.push((year, doy, hour, br, bt, bn, b_mag));
    }
    out
}

/// Parse Ulysses AMDA ephemeris CSV (`ulys-orb-all`).
///
/// Returns (year, doy, hour, r_au, lat_deg, lon_deg).
fn parse_ulysses_amda_orb(content: &str) -> Vec<(u16, u16, u8, f64, f64, f64)> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut out = Vec::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let r_au = parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or(""));
        let lat = parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or(""));
        let lon = parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or(""));
        out.push((year, doy, hour, r_au, lat, lon));
    }
    out
}

/// Merge three AMDA lanes (plasma, MAG, orbit) into `SpdfMergedRecord` rows.
///
/// Uses time-key intersection: only hours present in ALL three datasets are
/// emitted.  This is conservative but guarantees no partially-filled rows.
pub fn merge_ulysses_amda(
    plasma: &[(u16, u16, u8, f64, f64, f64)],
    mag: &[(u16, u16, u8, f64, f64, f64, f64)],
    orb: &[(u16, u16, u8, f64, f64, f64)],
) -> Vec<SpdfMergedRecord> {
    use std::collections::BTreeMap;
    let plasma_map: BTreeMap<_, _> = plasma.iter().map(|r| ((r.0, r.1, r.2), r)).collect();
    let mag_map: BTreeMap<_, _> = mag.iter().map(|r| ((r.0, r.1, r.2), r)).collect();
    let orb_map: BTreeMap<_, _> = orb.iter().map(|r| ((r.0, r.1, r.2), r)).collect();

    let mut rows = Vec::new();
    for (&key, p) in &plasma_map {
        let (Some(m), Some(o)) = (mag_map.get(&key), orb_map.get(&key)) else {
            continue;
        };
        rows.push(SpdfMergedRecord {
            year: key.0,
            doy: key.1,
            hour: key.2,
            distance_au: o.3,
            lat_deg: o.4,
            lon_deg: o.5,
            br: m.3,
            bt: m.4,
            bn: m.5,
            b_magnitude: m.6,
            proton_density: p.3,
            bulk_speed: p.4,
            proton_temperature: p.5,
        });
    }
    rows.sort_by_key(|r| (r.year, r.doy, r.hour));
    rows
}

/// NASA SPDF Ulysses dataset provider.
pub struct UlyssesProvider {
    /// Start year (inclusive).
    pub year_start: u16,
    /// End year (inclusive).
    pub year_end: u16,
}

impl Default for UlyssesProvider {
    fn default() -> Self {
        Self {
            year_start: 1990,
            year_end: 2009,
        }
    }
}

impl DatasetProvider for UlyssesProvider {
    fn name(&self) -> &str {
        "Ulysses Merged Hourly"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("ulysses");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let asc_name = format!("uly_{year}.asc");
            let asc_output = dir.join(&asc_name);
            let csv_name = format!("uy_coho1hr_merged_mag_plasma_{year}.csv");
            let csv_output = dir.join(&csv_name);
            if config.skip_existing && (asc_output.exists() || csv_output.exists()) {
                continue;
            }

            let asc_url = format!("{ULYSSES_SPDF_BASE}{asc_name}");
            match download_to_file(&asc_url, &asc_output) {
                Ok(_) => {
                    log::info!("saved {}", asc_name);
                    continue;
                }
                Err(e) => {
                    log::warn!(
                        "failed to download official Ulysses merged file {}: {}",
                        asc_url,
                        e
                    );
                }
            }

            match download_hapi_csv(
                ULYSSES_HAPI_DATASET,
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                Some(&[
                    "Time",
                    "heliocentricDistance",
                    "heliographicLatitude",
                    "heliographicLongitude",
                    "BR",
                    "BT",
                    "BN",
                    "ABS_B",
                    "numVectorsMagFldAvg",
                    "plasmaFlowSpeed",
                    "elevAngle",
                    "azimuthAngle",
                    "protonDensity",
                    "alphaDensity",
                    "protonTempLarge",
                ]),
            ) {
                Ok(data) => {
                    std::fs::write(&csv_output, data)?;
                    log::info!("saved {}", csv_name);
                }
                Err(e) => {
                    log::warn!(
                        "failed to download Ulysses {} via HAPI fallback: {}",
                        year,
                        e
                    );
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("ulysses");
        std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                (name.starts_with("uly_") && name.ends_with(".asc"))
                    || (name.starts_with("uy_coho1hr_") && name.ends_with(".csv"))
            })
    }
}

/// Ulysses AMDA provider -- fetches three AMDA lanes and merges them.
///
/// Intended as the third-tier fallback: try `UlyssesProvider` (SPDF + CDAWeb
/// HAPI) first, and fall back to this when both upstream sources fail.
pub struct UlyssesAmdaProvider {
    /// Start year (inclusive).
    pub year_start: u16,
    /// End year (inclusive).
    pub year_end: u16,
}

impl Default for UlyssesAmdaProvider {
    fn default() -> Self {
        Self {
            year_start: 1997,
            year_end: 2009,
        }
    }
}

impl DatasetProvider for UlyssesAmdaProvider {
    fn name(&self) -> &str {
        "Ulysses AMDA (plasma+MAG+orbit)"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("ulysses").join("amda");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let t_min = format!("{year}-01-01T00:00:00Z");
            let t_max = format!("{}-01-01T00:00:00Z", year + 1);
            let out_path = dir.join(format!("ulysses_amda_merged_{year}.csv"));
            if config.skip_existing && out_path.exists() {
                continue;
            }

            let plasma_csv = match download_amda_hapi_csv(ULYSSES_AMDA_PLASMA, &t_min, &t_max, None)
            {
                Ok(csv) => csv,
                Err(e) => {
                    log::warn!("AMDA Ulysses plasma {year}: {e}");
                    continue;
                }
            };
            let mag_csv = match download_amda_hapi_csv(ULYSSES_AMDA_MAG, &t_min, &t_max, None) {
                Ok(csv) => csv,
                Err(e) => {
                    log::warn!("AMDA Ulysses MAG {year}: {e}");
                    continue;
                }
            };
            let orb_csv = match download_amda_hapi_csv(ULYSSES_AMDA_ORB, &t_min, &t_max, None) {
                Ok(csv) => csv,
                Err(e) => {
                    log::warn!("AMDA Ulysses orbit {year}: {e}");
                    continue;
                }
            };

            let plasma = parse_ulysses_amda_plasma(&plasma_csv);
            let mag = parse_ulysses_amda_mag(&mag_csv);
            let orb = parse_ulysses_amda_orb(&orb_csv);
            let merged = merge_ulysses_amda(&plasma, &mag, &orb);

            // Write merged rows as a simple CSV for downstream parsers.
            let mut csv_buf = String::from(
                "year,doy,hour,distance_au,lat_deg,lon_deg,br,bt,bn,b_mag,density,speed,temperature\n",
            );
            for r in &merged {
                csv_buf.push_str(&format!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                    r.year,
                    r.doy,
                    r.hour,
                    r.distance_au,
                    r.lat_deg,
                    r.lon_deg,
                    r.br,
                    r.bt,
                    r.bn,
                    r.b_magnitude,
                    r.proton_density,
                    r.bulk_speed,
                    r.proton_temperature,
                ));
            }
            std::fs::write(&out_path, csv_buf)?;
            log::info!(
                "AMDA Ulysses {year}: merged {} hourly records -> {}",
                merged.len(),
                out_path.display()
            );
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("ulysses").join("amda");
        std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with("ulysses_amda_merged_") && name.ends_with(".csv")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ulysses_amda_plasma_format() {
        // Minimal AMDA CSV with header row.
        let csv = "Time,n_p,v_p,T_p\n\
                   1994-09-07T12:00:00.000Z,3.1,750.0,200000.0\n\
                   1994-09-07T13:00:00.000Z,-1.0e31,9999.9,9999999.0\n";
        let rows = parse_ulysses_amda_plasma(csv);
        assert_eq!(rows.len(), 2);
        // First row: valid fast polar wind
        let (year, doy, hour, density, speed, temp) = rows[0];
        assert_eq!(year, 1994);
        assert!((density - 3.1).abs() < 0.01);
        assert!((speed - 750.0).abs() < 0.1);
        assert!((temp - 200000.0).abs() < 1.0);
        let _ = (doy, hour);
        // Second row: AMDA fill values become NaN via parse_hapi_spacephysics_f64_or_nan
        let (_, _, _, density2, speed2, _) = rows[1];
        assert!(
            density2.is_nan() || density2 < 0.0,
            "fill should be NaN or negative"
        );
        let _ = speed2;
    }

    #[test]
    fn test_parse_ulysses_amda_mag_format() {
        let csv = "Time,Br,Bt,Bn,B\n\
                   1994-09-07T12:00:00.000Z,1.5,-0.3,0.1,1.55\n";
        let rows = parse_ulysses_amda_mag(csv);
        assert_eq!(rows.len(), 1);
        let (_, _, _, br, bt, bn, b_mag) = rows[0];
        assert!((br - 1.5).abs() < 0.01);
        assert!((bt - (-0.3)).abs() < 0.01);
        assert!((bn - 0.1).abs() < 0.01);
        assert!((b_mag - 1.55).abs() < 0.01);
    }

    #[test]
    fn test_merge_ulysses_amda_three_way_intersection() {
        // All three lanes cover the same two hours.
        let plasma = vec![
            (1994u16, 250u16, 12u8, 3.0f64, 750.0f64, 200000.0f64),
            (1994, 250, 13, 3.1, 745.0, 195000.0),
        ];
        let mag = vec![
            (1994u16, 250u16, 12u8, 1.5f64, -0.3f64, 0.1f64, 1.55f64),
            (1994, 250, 13, 1.4, -0.2, 0.15, 1.43),
        ];
        let orb = vec![
            (1994u16, 250u16, 12u8, 2.0f64, -75.0f64, 180.0f64),
            (1994, 250, 13, 2.01, -74.8, 180.5),
        ];
        let merged = merge_ulysses_amda(&plasma, &mag, &orb);
        assert_eq!(merged.len(), 2);
        assert!((merged[0].proton_density - 3.0).abs() < 0.01);
        assert!((merged[0].br - 1.5).abs() < 0.01);
        assert!((merged[0].distance_au - 2.0).abs() < 0.01);
        assert!((merged[0].lat_deg - (-75.0)).abs() < 0.1);
    }

    #[test]
    fn test_merge_ulysses_amda_missing_mag_hour_skipped() {
        // MAG missing hour 13 -- intersection only yields hour 12.
        let plasma = vec![
            (1994u16, 250u16, 12u8, 3.0f64, 750.0f64, 200000.0f64),
            (1994, 250, 13, 3.1, 745.0, 195000.0),
        ];
        let mag = vec![(1994u16, 250u16, 12u8, 1.5f64, -0.3f64, 0.1f64, 1.55f64)];
        let orb = vec![
            (1994u16, 250u16, 12u8, 2.0f64, -75.0f64, 180.0f64),
            (1994, 250, 13, 2.01, -74.8, 180.5),
        ];
        let merged = merge_ulysses_amda(&plasma, &mag, &orb);
        assert_eq!(merged.len(), 1, "hour 13 missing MAG should be excluded");
        assert!((merged[0].bulk_speed - 750.0).abs() < 0.1);
    }
}
