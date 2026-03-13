//! HI 21cm velocity cube data structures and CSV rotation curve parser.
//!
//! Defines typed structs for HI FITS cube metadata and the extracted rotation
//! curve products. All FITS I/O (moment-1 map computation, elliptical ring
//! averaging) lives in the binary:
//!   gororoba_cli_physics/src/bin/hi_cube_rotcurve_extractor.rs
//!
//! This split mirrors the manga.rs / manga_maps_extractor.rs pattern: data_core
//! owns CSV I/O and struct definitions; the binary owns FITS I/O and physics.
//!
//! Output CSV from the extractor:
//!   name, r_kpc, v_obs_km_s, v_err_km_s, psf_flag
//!
//! This format is compatible with parse_manga_rotcurves() (which reads the first
//! four columns), and with harmonic-halo-stacking input via harmonic_halo_stacking_manga.
//!
//! Reference: E-193, C-1375
//! THINGS: Walter et al. (2008), AJ 136, 2563
//! de Blok et al. (2008), AJ 136, 2648 (THINGS rotation curves)
//! LITTLE THINGS: Hunter et al. (2012), AJ 144, 134

use std::path::Path;

/// Metadata required to extract a rotation curve from a 3D HI FITS velocity cube.
///
/// Supplied via CSV, one row per galaxy. The extractor binary reads this catalog
/// before opening each cube.
#[derive(Debug, Clone)]
pub struct HiCubeMetadata {
    /// Galaxy name (e.g. "NGC 3198"). Used to locate the FITS cube on disk.
    pub name: String,
    /// Right ascension J2000 in decimal degrees.
    pub ra_deg: f64,
    /// Declination J2000 in decimal degrees.
    pub dec_deg: f64,
    /// Distance in Mpc (for arcsec -> kpc conversion).
    pub distance_mpc: f64,
    /// Disk inclination in degrees (0 = face-on, 90 = edge-on).
    ///
    /// Galaxies with i < 20 or i > 80 deg are excluded by the extractor
    /// to avoid projection noise and severe beam smearing respectively.
    pub inclination_deg: f64,
    /// Major-axis position angle in degrees (N through E, kinematic PA).
    pub pa_deg: f64,
    /// Synthesized beam FWHM (major axis) in arcsec.
    ///
    /// Sets the ring width (1 beam FWHM) and the inner PSF-smearing
    /// exclusion radius for the psf_flag column.
    pub beam_fwhm_arcsec: f64,
    /// Velocity channel width in km/s.
    pub channel_width_km_s: f64,
    /// Systemic heliocentric velocity (optical) in km/s.
    pub v_sys_km_s: f64,
}

/// A single radial bin in an extracted HI rotation curve.
#[derive(Debug, Clone)]
pub struct HiRotationPoint {
    /// Galactocentric radius in kpc.
    pub r_kpc: f64,
    /// Circular velocity in km/s (deprojected from ring-averaged moment-1).
    pub v_circ_km_s: f64,
    /// Velocity uncertainty in km/s (ring scatter / sqrt(N_pixels)).
    pub v_err_km_s: f64,
    /// True if the ring radius is within 1 synthesized beam FWHM of the
    /// galaxy center (beam-smearing-affected inner region).
    pub psf_flag: bool,
}

/// An extracted HI 21cm rotation curve for one galaxy.
#[derive(Debug, Clone)]
pub struct HiRotationCurve {
    /// Galaxy name (matches HiCubeMetadata.name).
    pub name: String,
    /// Distance in Mpc (propagated from metadata for downstream use).
    pub distance_mpc: f64,
    /// Rotation curve points sorted by ascending radius.
    pub points: Vec<HiRotationPoint>,
}

/// Parse HI cube metadata catalog CSV.
///
/// Expected columns (header row required, case-insensitive):
///   name, ra_deg, dec_deg, distance_mpc, inclination_deg, pa_deg,
///   beam_fwhm_arcsec, channel_width_km_s, v_sys_km_s
///
/// Rows with missing or non-positive distance are skipped.
pub fn parse_hi_cube_metadata(path: &Path) -> Result<Vec<HiCubeMetadata>, String> {
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
            let map = fields
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

        let name = get_str("name");
        if name.is_empty() {
            continue;
        }
        let distance_mpc = get_f64("distance_mpc");
        if !distance_mpc.is_finite() || distance_mpc <= 0.0 {
            continue;
        }

        entries.push(HiCubeMetadata {
            name,
            ra_deg: get_f64("ra_deg"),
            dec_deg: get_f64("dec_deg"),
            distance_mpc,
            inclination_deg: get_f64("inclination_deg"),
            pa_deg: get_f64("pa_deg"),
            beam_fwhm_arcsec: get_f64("beam_fwhm_arcsec"),
            channel_width_km_s: get_f64("channel_width_km_s"),
            v_sys_km_s: get_f64("v_sys_km_s"),
        });
    }

    if entries.is_empty() {
        return Err(format!("no valid entries in {}", path.display()));
    }
    Ok(entries)
}

/// Parse extracted HI rotation curve CSV.
///
/// Format produced by hi-cube-rotcurve-extractor:
///   name, r_kpc, v_obs_km_s, v_err_km_s[, psf_flag]
///
/// The psf_flag column is optional; if absent, all points default to false.
/// Rows for one galaxy must be contiguous (grouped by name).
pub fn parse_hi_rotcurves(path: &Path) -> Result<Vec<HiRotationCurve>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("read error: {e}"))?;
    let mut galaxies: Vec<HiRotationCurve> = Vec::new();
    let mut current_name = String::new();
    let mut current_dist = 0.0_f64;
    let mut current_points: Vec<HiRotationPoint> = Vec::new();

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
        let psf: bool = fields
            .get(4)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(false);
        // Optional 6th column: distance_mpc (populated when grouping galaxies)
        let dist: f64 = fields
            .get(5)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        if !r.is_finite() || !v.is_finite() {
            continue;
        }

        if name != current_name {
            if !current_points.is_empty() {
                galaxies.push(HiRotationCurve {
                    name: current_name.clone(),
                    distance_mpc: current_dist,
                    points: std::mem::take(&mut current_points),
                });
            }
            current_name = name;
            current_dist = dist;
        }

        current_points.push(HiRotationPoint {
            r_kpc: r,
            v_circ_km_s: v,
            v_err_km_s: if e.is_finite() && e > 0.0 { e } else { 5.0 },
            psf_flag: psf,
        });
    }

    if !current_points.is_empty() {
        galaxies.push(HiRotationCurve {
            name: current_name,
            distance_mpc: current_dist,
            points: current_points,
        });
    }

    if galaxies.is_empty() {
        return Err(format!("no valid rotation curves in {}", path.display()));
    }
    Ok(galaxies)
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
    fn test_parse_hi_cube_metadata() {
        let content = "\
name,ra_deg,dec_deg,distance_mpc,inclination_deg,pa_deg,beam_fwhm_arcsec,channel_width_km_s,v_sys_km_s
NGC 3198,154.978,45.549,13.8,72.0,215.0,8.1,2.6,660.0
NGC 2403,114.214,65.600,3.2,63.0,124.0,11.2,5.2,131.0
";
        let path = write_temp(content, ".csv");
        let metas = parse_hi_cube_metadata(Path::new(&path)).unwrap();
        assert_eq!(metas.len(), 2);
        assert_eq!(metas[0].name, "NGC 3198");
        assert!((metas[0].distance_mpc - 13.8).abs() < 1e-6);
        assert!((metas[0].inclination_deg - 72.0).abs() < 1e-6);
        assert!((metas[0].beam_fwhm_arcsec - 8.1).abs() < 1e-6);
        assert!((metas[0].v_sys_km_s - 660.0).abs() < 1e-6);
        assert_eq!(metas[1].name, "NGC 2403");
        assert!((metas[1].distance_mpc - 3.2).abs() < 1e-6);
    }

    #[test]
    fn test_parse_hi_cube_metadata_skips_bad_distance() {
        let content = "\
name,ra_deg,dec_deg,distance_mpc,inclination_deg,pa_deg,beam_fwhm_arcsec,channel_width_km_s,v_sys_km_s
good_gal,10.0,20.0,5.0,45.0,90.0,6.0,2.6,500.0
bad_gal,10.0,20.0,-1.0,45.0,90.0,6.0,2.6,500.0
";
        let path = write_temp(content, ".csv");
        let metas = parse_hi_cube_metadata(Path::new(&path)).unwrap();
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].name, "good_gal");
    }

    #[test]
    fn test_parse_hi_rotcurves_with_psf_flag() {
        let content = "\
name,r_kpc,v_obs_km_s,v_err_km_s,psf_flag
NGC 3198,0.5,80.0,5.0,true
NGC 3198,1.0,120.0,4.0,true
NGC 3198,5.0,150.0,3.0,false
NGC 3198,10.0,145.0,3.5,false
";
        let path = write_temp(content, ".csv");
        let curves = parse_hi_rotcurves(Path::new(&path)).unwrap();
        assert_eq!(curves.len(), 1);
        assert_eq!(curves[0].name, "NGC 3198");
        assert_eq!(curves[0].points.len(), 4);
        assert!(curves[0].points[0].psf_flag);
        assert!(curves[0].points[1].psf_flag);
        assert!(!curves[0].points[2].psf_flag);
    }

    #[test]
    fn test_parse_hi_rotcurves_no_psf_column() {
        // Backward compatibility: no psf_flag column -> all false
        let content = "\
name,r_kpc,v_obs_km_s,v_err_km_s
NGC 2403,1.0,80.0,5.0
NGC 2403,3.0,100.0,4.0
";
        let path = write_temp(content, ".csv");
        let curves = parse_hi_rotcurves(Path::new(&path)).unwrap();
        assert_eq!(curves[0].points.len(), 2);
        assert!(!curves[0].points[0].psf_flag);
    }

    #[test]
    fn test_parse_hi_cube_metadata_empty() {
        let content = "name,ra_deg,dec_deg,distance_mpc,inclination_deg,pa_deg,beam_fwhm_arcsec,channel_width_km_s,v_sys_km_s\n";
        let path = write_temp(content, ".csv");
        assert!(parse_hi_cube_metadata(Path::new(&path)).is_err());
    }
}
