//! MaNGA DR17 rotation curve parser.
//!
//! Parses rotation curves extracted from MaNGA IFU MAPS files via the
//! pseudo-slit method. The master CSV
//! format is SPARC-compatible: name, r_kpc, v_obs_km_s, v_err_km_s.
//!
//! Also parses the DAPall selection catalog CSV for galaxy metadata
//! (stellar mass, Sersic index, redshift, etc.) used for NFW priors
//! via the stellar-mass-to-halo-mass (SMHM) relation.
//!
//! Pipeline:
//!   1. Governed MaNGA selection tables -> dapall_selection.csv
//!   2. Rust-native `manga-maps-extractor` -> manga_rotcurves_all.csv
//!   3. This parser -> Rust NormalizedResiduals -> harmonic stacking
//!
//! Reference: Westfall et al., AJ 158 (2019) 231 (MaNGA DAP).
//!            Bundy et al., ApJ 798 (2015) 7 (MaNGA survey design).

use std::path::Path;

/// A single radial bin in a MaNGA pseudo-slit rotation curve.
#[derive(Debug, Clone)]
pub struct MangaRotationPoint {
    /// Galactocentric radius in kpc.
    pub r_kpc: f64,
    /// Observed circular velocity in km/s (deprojected by sin(i)).
    pub v_obs_km_s: f64,
    /// Velocity uncertainty in km/s.
    pub v_err_km_s: f64,
    /// True if this bin is within 1 PSF FWHM of the galaxy center.
    /// Beam smearing biases the velocity gradient in this region.
    pub psf_flag: bool,
}

/// A MaNGA galaxy rotation curve.
#[derive(Debug, Clone)]
pub struct MangaGalaxy {
    /// Plate-IFU identifier (e.g. "8485-1901").
    pub plateifu: String,
    /// Rotation curve data points sorted by radius.
    pub points: Vec<MangaRotationPoint>,
}

/// Galaxy metadata from DAPall selection catalog.
#[derive(Debug, Clone)]
pub struct MangaDapallEntry {
    /// Plate-IFU identifier.
    pub plateifu: String,
    /// MaNGA ID.
    pub mangaid: String,
    /// Spectroscopic redshift.
    pub z: f64,
    /// Axis ratio b/a (for inclination: i = arccos(b/a)).
    pub ba: f64,
    /// Position angle in degrees.
    pub pa_deg: f64,
    /// Effective radius in arcsec.
    pub r_eff_arcsec: f64,
    /// Sersic index.
    pub sersic_n: f64,
    /// Absolute r-band magnitude.
    pub abs_mag_r: f64,
    /// log10 stellar mass in solar masses.
    pub log_stellar_mass: f64,
    /// Stellar velocity dispersion within 1 R_e in km/s.
    pub stellar_sigma_km_s: f64,
    /// H-alpha equivalent width within 1 R_e in angstroms.
    pub ha_ew: f64,
}

impl MangaDapallEntry {
    /// Inclination in degrees from axis ratio.
    pub fn inclination_deg(&self) -> f64 {
        if self.ba > 0.0 && self.ba <= 1.0 {
            (self.ba.acos()).to_degrees()
        } else {
            f64::NAN
        }
    }

    /// Halo mass estimate from SMHM relation (Moster et al. 2013).
    ///
    /// log(M_halo) from log(M_star) using the parameterized relation:
    ///   M_star/M_halo = 2*N * [(M_halo/M1)^-beta + (M_halo/M1)^gamma]^-1
    ///
    /// Inverted numerically for the z=0 calibration.
    pub fn estimated_log_m200(&self) -> f64 {
        if !self.log_stellar_mass.is_finite() || self.log_stellar_mass < 7.0 {
            return f64::NAN;
        }
        // Moster+2013 z=0 parameters
        let log_m1 = 11.59;
        let n = 0.0351;
        let beta = 1.376;
        let gamma = 0.608;

        // Simple iterative inversion: scan log(M_halo) from 10 to 15
        let m_star_target = 10.0_f64.powf(self.log_stellar_mass);
        let mut best_log_mh = 12.0;
        let mut best_diff = f64::INFINITY;

        for i in 0..500 {
            let log_mh = 10.0 + i as f64 * 0.01;
            let mh = 10.0_f64.powf(log_mh);
            let m1 = 10.0_f64.powf(log_m1);
            let ratio = mh / m1;
            let m_star_pred = 2.0 * n * mh / (ratio.powf(-beta) + ratio.powf(gamma));
            let diff = (m_star_pred - m_star_target).abs();
            if diff < best_diff {
                best_diff = diff;
                best_log_mh = log_mh;
            }
        }
        best_log_mh
    }
}

/// Parse MaNGA rotation curves from the master CSV.
///
/// Format: name,r_kpc,v_obs_km_s,v_err_km_s
/// Rows for one galaxy are contiguous, grouped by name.
pub fn parse_manga_rotcurves(path: &Path) -> Result<Vec<MangaGalaxy>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("read error: {e}"))?;

    let mut galaxies: Vec<MangaGalaxy> = Vec::new();
    let mut current_name = String::new();
    let mut current_points: Vec<MangaRotationPoint> = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("name") || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 4 {
            continue;
        }

        let name = fields[0].trim().to_string();
        let r: f64 = fields[1].trim().parse().unwrap_or(f64::NAN);
        let v: f64 = fields[2].trim().parse().unwrap_or(f64::NAN);
        let e: f64 = fields[3].trim().parse().unwrap_or(f64::NAN);

        if !r.is_finite() || !v.is_finite() {
            continue;
        }

        if name != current_name {
            if !current_points.is_empty() {
                galaxies.push(MangaGalaxy {
                    plateifu: current_name.clone(),
                    points: std::mem::take(&mut current_points),
                });
            }
            current_name = name;
        }

        let psf: bool = fields
            .get(4)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(false);
        current_points.push(MangaRotationPoint {
            r_kpc: r,
            v_obs_km_s: v,
            v_err_km_s: if e.is_finite() && e > 0.0 { e } else { 10.0 },
            psf_flag: psf,
        });
    }

    // Flush last galaxy
    if !current_points.is_empty() {
        galaxies.push(MangaGalaxy {
            plateifu: current_name,
            points: current_points,
        });
    }

    if galaxies.is_empty() {
        return Err(format!("no valid rotation curves in {}", path.display()));
    }

    Ok(galaxies)
}

/// Parse DAPall selection catalog CSV.
///
/// Columns: plateifu, mangaid, plate, ifudesign, z, nsa_elpetro_ba,
///          nsa_elpetro_phi, nsa_elpetro_th50_r, nsa_sersic_n, ...
pub fn parse_manga_dapall_csv(path: &Path) -> Result<Vec<MangaDapallEntry>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("read error: {e}"))?;

    let mut entries = Vec::new();
    let mut header_map: Option<std::collections::HashMap<String, usize>> = None;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();

        if header_map.is_none() {
            // First non-comment line is header
            let map: std::collections::HashMap<String, usize> = fields
                .iter()
                .enumerate()
                .map(|(i, &f)| (f.trim().to_lowercase(), i))
                .collect();
            header_map = Some(map);
            continue;
        }

        let hm = header_map.as_ref().unwrap();
        let get_f64 = |key: &str| -> f64 {
            hm.get(key)
                .and_then(|&i| fields.get(i))
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(f64::NAN)
        };
        let get_str = |key: &str| -> String {
            hm.get(key)
                .and_then(|&i| fields.get(i))
                .map(|s| s.trim().to_string())
                .unwrap_or_default()
        };

        let z = get_f64("z");
        if !z.is_finite() || z <= 0.0 {
            continue;
        }

        entries.push(MangaDapallEntry {
            plateifu: get_str("plateifu"),
            mangaid: get_str("mangaid"),
            z,
            ba: get_f64("nsa_elpetro_ba"),
            pa_deg: get_f64("nsa_elpetro_phi"),
            r_eff_arcsec: get_f64("nsa_elpetro_th50_r"),
            sersic_n: get_f64("nsa_sersic_n"),
            abs_mag_r: get_f64("nsa_elpetro_absmag_r"),
            log_stellar_mass: get_f64("nsa_elpetro_mass"),
            stellar_sigma_km_s: get_f64("stellar_sigma_1re"),
            ha_ew: get_f64("ha_gew_1re"),
        });
    }

    if entries.is_empty() {
        return Err(format!("no valid entries in {}", path.display()));
    }

    Ok(entries)
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
    fn test_parse_manga_rotcurves() {
        let content = "\
name,r_kpc,v_obs_km_s,v_err_km_s
8485-1901,0.50,45.2,8.3
8485-1901,1.00,89.5,6.1
8485-1901,1.50,120.3,5.0
8485-1901,2.00,145.8,4.5
8485-1901,2.50,158.2,4.2
8485-1901,3.00,165.1,4.0
7443-1902,0.40,38.5,9.1
7443-1902,0.80,72.3,7.0
7443-1902,1.20,105.6,5.8
7443-1902,1.60,128.4,5.2
7443-1902,2.00,140.1,4.8
";
        let path = write_temp(content, ".csv");
        let galaxies = parse_manga_rotcurves(Path::new(&path)).unwrap();
        assert_eq!(galaxies.len(), 2);
        assert_eq!(galaxies[0].plateifu, "8485-1901");
        assert_eq!(galaxies[0].points.len(), 6);
        assert_eq!(galaxies[1].plateifu, "7443-1902");
        assert_eq!(galaxies[1].points.len(), 5);
        assert!((galaxies[0].points[0].r_kpc - 0.50).abs() < 1e-10);
    }

    #[test]
    fn test_parse_manga_dapall_csv() {
        let content = "\
plateifu,mangaid,plate,ifudesign,z,nsa_elpetro_ba,nsa_elpetro_phi,nsa_elpetro_th50_r,nsa_sersic_n,nsa_sersic_ba,nsa_elpetro_absmag_r,nsa_elpetro_mass,stellar_sigma_1re,ha_gsb_1re,ha_gew_1re,ha_z,ha_gvel_lo,ha_gvel_hi,ha_gsigma_1re,emline_rchi2_1re,dapqual
8485-1901,1-209232,8485,1901,0.0407,0.75,120.5,8.3,1.2,0.72,-20.5,10.2,85.3,1.5e-16,15.3,0.0408,-80.0,90.0,45.0,1.2,0
7443-1902,1-167076,7443,1902,0.0291,0.65,45.2,6.1,0.9,0.63,-19.8,9.8,72.1,2.1e-16,22.1,0.0292,-60.0,70.0,35.0,0.9,0
";
        let path = write_temp(content, ".csv");
        let entries = parse_manga_dapall_csv(Path::new(&path)).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].plateifu, "8485-1901");
        assert!((entries[0].z - 0.0407).abs() < 1e-4);
        assert!((entries[0].ba - 0.75).abs() < 1e-4);
        assert!((entries[0].sersic_n - 1.2).abs() < 1e-4);
        assert!((entries[0].log_stellar_mass - 10.2).abs() < 0.1);
    }

    #[test]
    fn test_smhm_relation() {
        let entry = MangaDapallEntry {
            plateifu: "test".to_string(),
            mangaid: "test".to_string(),
            z: 0.03,
            ba: 0.7,
            pa_deg: 0.0,
            r_eff_arcsec: 5.0,
            sersic_n: 1.0,
            abs_mag_r: -20.0,
            log_stellar_mass: 10.5,
            stellar_sigma_km_s: 100.0,
            ha_ew: 20.0,
        };
        let log_m200 = entry.estimated_log_m200();
        // For log(M*) ~ 10.5, expect log(M_halo) ~ 12-13
        assert!(
            (11.5..13.5).contains(&log_m200),
            "SMHM: log(M*) = 10.5 -> log(M_halo) = {log_m200:.2}, expected ~12-13"
        );
    }

    #[test]
    fn test_inclination_from_ba() {
        let entry = MangaDapallEntry {
            plateifu: "test".to_string(),
            mangaid: "test".to_string(),
            z: 0.03,
            ba: 0.5, // cos(60) = 0.5 -> i = 60 deg
            pa_deg: 0.0,
            r_eff_arcsec: 5.0,
            sersic_n: 1.0,
            abs_mag_r: -20.0,
            log_stellar_mass: 10.0,
            stellar_sigma_km_s: 80.0,
            ha_ew: 15.0,
        };
        assert!((entry.inclination_deg() - 60.0).abs() < 0.1);
    }

    #[test]
    fn test_empty_rotcurves() {
        let content = "name,r_kpc,v_obs_km_s,v_err_km_s\n";
        let path = write_temp(content, ".csv");
        assert!(parse_manga_rotcurves(Path::new(&path)).is_err());
    }
}
