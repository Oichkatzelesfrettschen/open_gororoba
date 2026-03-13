//! THINGS catalog parser (Walter et al. 2008, J/AJ/136/2563).
//!
//! Parses the VizieR fixed-width tables for The HI Nearby Galaxy Survey:
//! - table1.dat: Galaxy properties (positions, distances, morphology) -- 34 galaxies
//! - table4.dat: HI spectra (velocity channel, flux density) -- 2688 rows
//!
//! NOTE: table4.dat contains integrated HI spectra (velocity vs flux), NOT
//! tilted-ring rotation curves. The actual rotation curves (de Blok et al. 2008)
//! are separate data products hosted at MPIA.
//!
//! Reference: Walter, Brinks, de Blok et al., AJ 136 (2008) 2563.
//!            "THINGS: The HI Nearby Galaxy Survey"

use std::path::Path;

/// Galaxy properties from THINGS table1.dat.
///
/// Fixed-width format (115 bytes/line), 34 galaxies. Byte ranges from ReadMe:
///   Name(1-8), AName(10-24), RA(26-35), Dec(37-45), Dist(51-54),
///   logD25(62-65), Bmag(68-72), BMAG(74-79), i(81-82), PA(84-86),
///   Metal(93-96), SFR(102-106), TT(112-113).
#[derive(Debug, Clone)]
pub struct ThingsGalaxy {
    /// Primary galaxy name (e.g. "NGC 628").
    pub name: String,
    /// Alternative names (e.g. "M 74, UGC 1149").
    pub alt_names: String,
    /// Right ascension J2000 in decimal hours.
    pub ra_hours: f64,
    /// Declination J2000 in decimal degrees.
    pub dec_deg: f64,
    /// Distance in Mpc.
    pub distance_mpc: f64,
    /// log10 of optical diameter D25 in 0.1 arcmin.
    pub log_d25: f64,
    /// Apparent blue magnitude (corrected).
    pub b_mag: f64,
    /// Absolute B magnitude.
    pub abs_b_mag: f64,
    /// Inclination in degrees.
    pub inclination_deg: f64,
    /// Position angle in degrees.
    pub pa_deg: f64,
    /// Metallicity 12+log(O/H), NaN if unavailable.
    pub metallicity: f64,
    /// Star formation rate in Msun/yr, NaN if unavailable.
    pub sfr_msun_yr: f64,
    /// Morphological type T (de Vaucouleurs numerical type).
    pub morph_type: i32,
}

/// A single velocity channel in a THINGS HI spectrum.
#[derive(Debug, Clone)]
pub struct ThingsHiChannel {
    /// Velocity in km/s.
    pub velocity_km_s: f64,
    /// HI flux density in Jy (0 = blanked/undetected).
    pub flux_jy: f64,
}

/// Integrated HI spectrum for one THINGS galaxy.
#[derive(Debug, Clone)]
pub struct ThingsHiSpectrum {
    /// Galaxy name (matches ThingsGalaxy.name).
    pub name: String,
    /// Velocity channels sorted by velocity.
    pub channels: Vec<ThingsHiChannel>,
}

impl ThingsHiSpectrum {
    /// Total HI flux integral (sum of flux * channel width) in Jy km/s.
    ///
    /// Assumes uniform channel spacing. Returns 0 if fewer than 2 channels.
    pub fn integrated_flux_jy_km_s(&self) -> f64 {
        if self.channels.len() < 2 {
            return 0.0;
        }
        let dv = (self.channels[1].velocity_km_s - self.channels[0].velocity_km_s).abs();
        self.channels.iter().map(|ch| ch.flux_jy * dv).sum()
    }

    /// Systemic velocity estimate: flux-weighted mean velocity in km/s.
    pub fn systemic_velocity_km_s(&self) -> f64 {
        let total_flux: f64 = self.channels.iter().map(|ch| ch.flux_jy).sum();
        if total_flux <= 0.0 {
            return f64::NAN;
        }
        let weighted: f64 = self
            .channels
            .iter()
            .map(|ch| ch.velocity_km_s * ch.flux_jy)
            .sum();
        weighted / total_flux
    }

    /// W50 linewidth: velocity width at 50% of peak flux, in km/s.
    ///
    /// Returns NaN if spectrum has no positive flux.
    pub fn w50_km_s(&self) -> f64 {
        let peak = self
            .channels
            .iter()
            .map(|ch| ch.flux_jy)
            .fold(0.0_f64, f64::max);
        if peak <= 0.0 {
            return f64::NAN;
        }
        let half_peak = 0.5 * peak;
        let above: Vec<f64> = self
            .channels
            .iter()
            .filter(|ch| ch.flux_jy >= half_peak)
            .map(|ch| ch.velocity_km_s)
            .collect();
        if above.is_empty() {
            return f64::NAN;
        }
        let v_min = above.iter().cloned().fold(f64::INFINITY, f64::min);
        let v_max = above.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (v_max - v_min).abs()
    }
}

/// Parse THINGS table1.dat (galaxy properties).
///
/// Fixed-width format, 115 bytes/line. Byte ranges per CDS ReadMe.
/// Skips comment lines (starting with #) and lines shorter than 80 bytes.
pub fn parse_things_galaxies(path: &Path) -> Result<Vec<ThingsGalaxy>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("read error: {e}"))?;

    let mut galaxies = Vec::new();

    for line in content.lines() {
        if line.starts_with('#') || line.starts_with('-') || line.len() < 80 {
            continue;
        }

        let get = |start: usize, end: usize| -> &str {
            line.get(start..end.min(line.len())).unwrap_or("").trim()
        };

        let pf = |start: usize, end: usize| -> f64 {
            get(start, end)
                .trim_end_matches('*')
                .parse()
                .unwrap_or(f64::NAN)
        };

        let pi = |start: usize, end: usize| -> i32 { get(start, end).parse().unwrap_or(0) };

        // Name: bytes 1-8 (0-indexed: 0..8)
        let name = get(0, 8).to_string();
        if name.is_empty() {
            continue;
        }

        // Alt names: bytes 10-24 (0-indexed: 9..24)
        let alt_names = get(9, 24).to_string();

        // RA: bytes 26-35 -> h(26-27) m(29-30) s(32-35)
        let ra_h = pf(25, 27);
        let ra_m = pf(28, 30);
        let ra_s = pf(31, 35);
        let ra_hours = ra_h + ra_m / 60.0 + ra_s / 3600.0;

        // Dec: bytes 37-45 -> sign(37) d(38-39) m(41-42) s(44-45)
        let dec_sign = if get(36, 37) == "-" { -1.0 } else { 1.0 };
        let dec_d = pf(37, 39);
        let dec_m = pf(40, 42);
        let dec_s = pf(43, 45);
        let dec_deg = dec_sign * (dec_d + dec_m / 60.0 + dec_s / 3600.0);

        let distance_mpc = pf(50, 54);
        let log_d25 = pf(61, 66);
        let b_mag = pf(67, 72);
        let abs_b_mag = pf(73, 79);
        let inclination_deg = pf(80, 82);
        let pa_deg = pf(83, 86);

        // Metallicity: bytes 93-96, "---" means unavailable
        let metal_str = get(92, 96);
        let metallicity = if metal_str == "---" || metal_str == "-" {
            f64::NAN
        } else {
            metal_str.parse().unwrap_or(f64::NAN)
        };

        // SFR: bytes 102-106, "---" means unavailable
        let sfr_str = get(101, 106);
        let sfr_msun_yr = if sfr_str == "---" || sfr_str == "-" {
            f64::NAN
        } else {
            sfr_str.parse().unwrap_or(f64::NAN)
        };

        // Morphological type: bytes 112-113
        let morph_type = pi(111, 113);

        if !distance_mpc.is_finite() || distance_mpc <= 0.0 {
            continue;
        }

        galaxies.push(ThingsGalaxy {
            name,
            alt_names,
            ra_hours,
            dec_deg,
            distance_mpc,
            log_d25,
            b_mag,
            abs_b_mag,
            inclination_deg,
            pa_deg,
            metallicity,
            sfr_msun_yr,
            morph_type,
        });
    }

    if galaxies.is_empty() {
        return Err(format!("no valid galaxies in {}", path.display()));
    }

    Ok(galaxies)
}

/// Parse THINGS table4.dat (HI spectra).
///
/// Fixed-width format, 32 bytes/line. Byte ranges per CDS ReadMe:
///   Name(1-8), Vel(13-21), SHI(23-32).
///
/// Returns one `ThingsHiSpectrum` per galaxy, channels sorted by velocity.
pub fn parse_things_hi_spectra(path: &Path) -> Result<Vec<ThingsHiSpectrum>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("read error: {e}"))?;

    let mut spectra: Vec<ThingsHiSpectrum> = Vec::new();
    let mut current_name = String::new();
    let mut current_channels: Vec<ThingsHiChannel> = Vec::new();

    for line in content.lines() {
        if line.starts_with('#') || line.starts_with('-') || line.len() < 20 {
            continue;
        }

        // Name: bytes 1-8 (0-indexed: 0..8)
        let name = line.get(0..8).unwrap_or("").trim().to_string();
        if name.is_empty() {
            continue;
        }

        // Velocity: bytes 13-21 (0-indexed: 12..21)
        let vel: f64 = line
            .get(12..21)
            .unwrap_or("")
            .trim()
            .parse()
            .unwrap_or(f64::NAN);

        // Flux: bytes 23-32 (0-indexed: 22..32)
        let flux: f64 = line
            .get(22..32)
            .unwrap_or("0")
            .trim()
            .parse()
            .unwrap_or(0.0);

        if !vel.is_finite() {
            continue;
        }

        if name != current_name {
            if !current_channels.is_empty() {
                current_channels
                    .sort_by(|a, b| a.velocity_km_s.partial_cmp(&b.velocity_km_s).unwrap());
                spectra.push(ThingsHiSpectrum {
                    name: current_name.clone(),
                    channels: std::mem::take(&mut current_channels),
                });
            }
            current_name = name;
        }

        current_channels.push(ThingsHiChannel {
            velocity_km_s: vel,
            flux_jy: flux,
        });
    }

    // Flush last galaxy
    if !current_channels.is_empty() {
        current_channels.sort_by(|a, b| a.velocity_km_s.partial_cmp(&b.velocity_km_s).unwrap());
        spectra.push(ThingsHiSpectrum {
            name: current_name,
            channels: current_channels,
        });
    }

    if spectra.is_empty() {
        return Err(format!("no valid spectra in {}", path.display()));
    }

    Ok(spectra)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(content: &str, suffix: &str) -> tempfile::TempPath {
        let mut f = tempfile::Builder::new().suffix(suffix).tempfile().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f.into_temp_path()
    }

    #[test]
    fn test_parse_things_galaxies() {
        // Reproduce actual table1.dat format (115 chars/line)
        let line1 = "NGC 628  M 74, UGC 1149  01 36 41.8 +15 47 00 NED  7.3 K04   1.99   9.35 -19.97  7  20 T08  8.33 M06 1.21  L06  5";
        let line2 = "NGC 925  UGC 1913        02 27 16.5 +33 34 44 T08  9.2 F01   2.03   9.77 -20.04 66 287 dB08 8.24 M06 1.09  L06  7";
        let content = format!("{line1}\n{line2}\n");
        let path = write_temp(&content, ".dat");
        let galaxies = parse_things_galaxies(Path::new(&path)).unwrap();
        assert_eq!(galaxies.len(), 2);
        assert_eq!(galaxies[0].name, "NGC 628");
        assert!(galaxies[0].alt_names.contains("M 74"));
        assert!((galaxies[0].distance_mpc - 7.3).abs() < 0.1);
        assert_eq!(galaxies[0].morph_type, 5);
        assert_eq!(galaxies[1].name, "NGC 925");
        assert!((galaxies[1].distance_mpc - 9.2).abs() < 0.1);
    }

    #[test]
    fn test_parse_things_galaxies_ra_dec() {
        // NGC 628: RA = 01h 36m 41.8s, Dec = +15d 47' 00"
        let line = "NGC 628  M 74, UGC 1149  01 36 41.8 +15 47 00 NED  7.3 K04   1.99   9.35 -19.97  7  20 T08  8.33 M06 1.21  L06  5";
        let path = write_temp(line, ".dat");
        let galaxies = parse_things_galaxies(Path::new(&path)).unwrap();
        let g = &galaxies[0];
        // RA = 1 + 36/60 + 41.8/3600 = 1.61161 hours
        assert!((g.ra_hours - 1.61161).abs() < 0.001);
        // Dec = +15 + 47/60 + 0/3600 = 15.7833 deg
        assert!((g.dec_deg - 15.7833).abs() < 0.01);
    }

    #[test]
    fn test_parse_things_galaxies_negative_dec() {
        // NGC 3521: Dec = -00 02 09
        let line = "NGC 3521                 11 05 48.6 -00 02 09 T08 10.7 vflow 1.92   9.21 -20.94 73 340 dB08 8.36 M06 3.34  L06  4";
        let path = write_temp(line, ".dat");
        let galaxies = parse_things_galaxies(Path::new(&path)).unwrap();
        assert!(galaxies[0].dec_deg < 0.0);
    }

    #[test]
    fn test_parse_things_galaxies_missing_sfr() {
        // NGC 4449 has SFR = "---" (unavailable)
        let line = "NGC 4449 UGC 7592        12 28 11.9 +44 05 40 NED  4.2 K04   1.67   8.98 -19.14 60 230 H98  8.32 S89 ---       10 W";
        let path = write_temp(line, ".dat");
        let galaxies = parse_things_galaxies(Path::new(&path)).unwrap();
        assert!(galaxies[0].sfr_msun_yr.is_nan());
    }

    #[test]
    fn test_parse_things_hi_spectra() {
        let content = "\
DDO 154       447.3   0.
DDO 154       444.8   0.
DDO 154       439.6   5.6403E-03
DDO 154       437.0   1.9650E-02
NGC 628       657.0   1.2345E-01
NGC 628       654.5   2.3456E-01
";
        let path = write_temp(content, ".dat");
        let spectra = parse_things_hi_spectra(Path::new(&path)).unwrap();
        assert_eq!(spectra.len(), 2);
        assert_eq!(spectra[0].name, "DDO 154");
        assert_eq!(spectra[0].channels.len(), 4);
        assert_eq!(spectra[1].name, "NGC 628");
        assert_eq!(spectra[1].channels.len(), 2);
    }

    #[test]
    fn test_hi_spectrum_sorted_by_velocity() {
        let content = "\
DDO 154       447.3   0.
DDO 154       444.8   0.
DDO 154       439.6   5.6403E-03
";
        let path = write_temp(content, ".dat");
        let spectra = parse_things_hi_spectra(Path::new(&path)).unwrap();
        for w in spectra[0].channels.windows(2) {
            assert!(w[0].velocity_km_s < w[1].velocity_km_s);
        }
    }

    #[test]
    fn test_hi_integrated_flux() {
        // Uniform channels at dv = 2.5 km/s, 3 channels with flux 1.0 Jy each
        let spectrum = ThingsHiSpectrum {
            name: "test".to_string(),
            channels: vec![
                ThingsHiChannel {
                    velocity_km_s: 100.0,
                    flux_jy: 1.0,
                },
                ThingsHiChannel {
                    velocity_km_s: 102.5,
                    flux_jy: 1.0,
                },
                ThingsHiChannel {
                    velocity_km_s: 105.0,
                    flux_jy: 1.0,
                },
            ],
        };
        // Integral = 3 * 1.0 * 2.5 = 7.5 Jy km/s
        assert!((spectrum.integrated_flux_jy_km_s() - 7.5).abs() < 1e-10);
    }

    #[test]
    fn test_hi_systemic_velocity() {
        // Symmetric spectrum -> systemic = center velocity
        let spectrum = ThingsHiSpectrum {
            name: "test".to_string(),
            channels: vec![
                ThingsHiChannel {
                    velocity_km_s: 100.0,
                    flux_jy: 1.0,
                },
                ThingsHiChannel {
                    velocity_km_s: 110.0,
                    flux_jy: 2.0,
                },
                ThingsHiChannel {
                    velocity_km_s: 120.0,
                    flux_jy: 1.0,
                },
            ],
        };
        // Weighted mean = (100*1 + 110*2 + 120*1) / (1+2+1) = 440/4 = 110
        assert!((spectrum.systemic_velocity_km_s() - 110.0).abs() < 1e-10);
    }

    #[test]
    fn test_hi_w50() {
        // Box spectrum with peak=2.0 at center, flanks at 0.5
        let spectrum = ThingsHiSpectrum {
            name: "test".to_string(),
            channels: vec![
                ThingsHiChannel {
                    velocity_km_s: 100.0,
                    flux_jy: 0.5,
                },
                ThingsHiChannel {
                    velocity_km_s: 110.0,
                    flux_jy: 2.0,
                },
                ThingsHiChannel {
                    velocity_km_s: 120.0,
                    flux_jy: 2.0,
                },
                ThingsHiChannel {
                    velocity_km_s: 130.0,
                    flux_jy: 2.0,
                },
                ThingsHiChannel {
                    velocity_km_s: 140.0,
                    flux_jy: 0.5,
                },
            ],
        };
        // Half-peak = 1.0. Channels >= 1.0: 110, 120, 130. W50 = 130-110 = 20
        assert!((spectrum.w50_km_s() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_table4() {
        let content = "# header only\n";
        let path = write_temp(content, ".dat");
        let result = parse_things_hi_spectra(Path::new(&path));
        assert!(result.is_err());
    }

    #[test]
    fn test_things_galaxy_count() {
        // THINGS has exactly 34 galaxies
        let data_path = Path::new("data/external/things/table1.dat");
        if data_path.exists() {
            let galaxies = parse_things_galaxies(data_path).unwrap();
            assert_eq!(galaxies.len(), 34, "THINGS should have exactly 34 galaxies");
        }
    }

    #[test]
    fn test_things_spectra_count() {
        // THINGS table4.dat has 2688 spectral lines across 34 galaxies
        let data_path = Path::new("data/external/things/table4.dat");
        if data_path.exists() {
            let spectra = parse_things_hi_spectra(data_path).unwrap();
            assert_eq!(spectra.len(), 34, "THINGS should have 34 galaxy spectra");
            let total: usize = spectra.iter().map(|s| s.channels.len()).sum();
            assert_eq!(total, 2688, "THINGS should have 2688 total channels");
        }
    }
}
