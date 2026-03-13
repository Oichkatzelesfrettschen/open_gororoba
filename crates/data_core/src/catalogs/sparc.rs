//! SPARC rotation curve database parser (Lelli, McGaugh & Schombert 2016).
//!
//! Parses rotation curves from two sources:
//! - Individual Rotmod_LTG `.dat` files (from astroweb.case.edu tarball)
//! - VizieR table2.dat (J/AJ/152/157, all 175 galaxies in one file)
//!
//! Also parses Li et al. (2020) NFW halo fits from VizieR J/ApJS/247/31.
//!
//! Reference: Lelli, McGaugh & Schombert, AJ 152 (2016) 157.
//!            Li, Lelli, McGaugh & Schombert, ApJS 247 (2020) 31.

use std::path::Path;

/// A single radial bin in a SPARC rotation curve.
#[derive(Debug, Clone)]
pub struct SparcRotationPoint {
    /// Galactocentric radius in kpc.
    pub r_kpc: f64,
    /// Observed circular velocity in km/s.
    pub v_obs_km_s: f64,
    /// Uncertainty on v_obs in km/s.
    pub v_err_km_s: f64,
    /// Gas contribution to rotation velocity in km/s.
    pub v_gas_km_s: f64,
    /// Disk contribution to rotation velocity in km/s.
    pub v_disk_km_s: f64,
    /// Bulge contribution to rotation velocity in km/s.
    pub v_bul_km_s: f64,
    /// Disk surface brightness in L_sun/pc^2.
    pub sb_disk: f64,
    /// Bulge surface brightness in L_sun/pc^2.
    pub sb_bul: f64,
}

/// A complete SPARC galaxy rotation curve.
#[derive(Debug, Clone)]
pub struct SparcGalaxy {
    /// Galaxy name (from filename or table column).
    pub name: String,
    /// Rotation curve data points sorted by radius.
    pub points: Vec<SparcRotationPoint>,
}

/// Parse a single SPARC Rotmod `.dat` file (original tarball format).
///
/// Format: whitespace-separated columns, comment lines start with `#` or `!`.
/// Columns: Rad Vobs errV Vgas Vdisk Vbul SBdisk SBbul
pub fn parse_sparc_rotmod(path: &Path) -> Result<SparcGalaxy, String> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let content = std::fs::read_to_string(path).map_err(|e| format!("read error: {e}"))?;

    let mut points = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('!') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 8 {
            continue;
        }
        let parse = |i: usize| -> f64 { fields[i].parse().unwrap_or(f64::NAN) };
        let pt = SparcRotationPoint {
            r_kpc: parse(0),
            v_obs_km_s: parse(1),
            v_err_km_s: parse(2),
            v_gas_km_s: parse(3),
            v_disk_km_s: parse(4),
            v_bul_km_s: parse(5),
            sb_disk: parse(6),
            sb_bul: parse(7),
        };
        if pt.r_kpc.is_finite() && pt.v_obs_km_s.is_finite() {
            points.push(pt);
        }
    }

    if points.is_empty() {
        return Err(format!("no valid data points in {}", path.display()));
    }

    Ok(SparcGalaxy { name, points })
}

/// Load all SPARC rotation curves from a Rotmod_LTG directory.
pub fn load_sparc_rotmod_dir(dir: &Path) -> Result<Vec<SparcGalaxy>, String> {
    if !dir.is_dir() {
        return Err(format!("{} is not a directory", dir.display()));
    }

    let mut galaxies = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("readdir error: {e}"))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "dat"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        match parse_sparc_rotmod(&entry.path()) {
            Ok(galaxy) => galaxies.push(galaxy),
            Err(e) => {
                log::warn!("Skipping {}: {e}", entry.path().display());
            }
        }
    }

    Ok(galaxies)
}

/// Parse VizieR table2.dat (J/AJ/152/157) containing all 175 SPARC galaxies.
///
/// Each line has: Name(col0) Distance(col1) Rad(col2) Vobs(col3) errV(col4)
///                Vgas(col5) Vdisk(col6) Vbul(col7) SBdisk(col8) SBbul(col9)
///
/// All rows for one galaxy are contiguous, grouped by galaxy name.
pub fn parse_sparc_vizier_table2(path: &Path) -> Result<Vec<SparcGalaxy>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("read error: {e}"))?;

    let mut galaxies: Vec<SparcGalaxy> = Vec::new();
    let mut current_name = String::new();
    let mut current_points: Vec<SparcRotationPoint> = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('-') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        // Need at least: name distance rad vobs errv vgas vdisk vbul sbdisk (sbbul optional)
        if fields.len() < 9 {
            continue;
        }

        let name = fields[0].to_string();
        let pf = |i: usize| -> f64 { fields[i].parse().unwrap_or(f64::NAN) };

        // fields[1] = distance, fields[2] = rad, etc.
        let pt = SparcRotationPoint {
            r_kpc: pf(2),
            v_obs_km_s: pf(3),
            v_err_km_s: pf(4),
            v_gas_km_s: pf(5),
            v_disk_km_s: pf(6),
            v_bul_km_s: pf(7),
            sb_disk: pf(8),
            sb_bul: if fields.len() > 9 { pf(9) } else { 0.0 },
        };

        if !pt.r_kpc.is_finite() || !pt.v_obs_km_s.is_finite() {
            continue;
        }

        if name != current_name {
            if !current_points.is_empty() {
                galaxies.push(SparcGalaxy {
                    name: current_name.clone(),
                    points: std::mem::take(&mut current_points),
                });
            }
            current_name = name;
        }
        current_points.push(pt);
    }

    // Flush last galaxy
    if !current_points.is_empty() {
        galaxies.push(SparcGalaxy {
            name: current_name,
            points: current_points,
        });
    }

    if galaxies.is_empty() {
        return Err(format!("no valid galaxies in {}", path.display()));
    }

    Ok(galaxies)
}

/// NFW fit parameters for a SPARC galaxy (Li et al. 2020).
#[derive(Debug, Clone)]
pub struct SparcNfwFit {
    /// Galaxy name.
    pub name: String,
    /// Dark matter halo model used (e.g. "NFW-Flat", "NFW-LCDM").
    pub model: String,
    /// Distance in Mpc.
    pub distance_mpc: f64,
    /// Inclination in degrees.
    pub inclination_deg: f64,
    /// Disk mass-to-light ratio (Y_disk).
    pub y_disk: f64,
    /// Bulge mass-to-light ratio (Y_bulge).
    pub y_bulge: f64,
    /// NFW scale radius in kpc.
    pub r_s_kpc: f64,
    /// log10 of volume density rho_s in Msun/pc^3.
    pub log_rho_s: f64,
    /// Virial velocity V200 in km/s.
    pub v200_km_s: f64,
    /// Concentration parameter c200.
    pub c200: f64,
    /// log10 of halo mass M200 in Msun.
    pub log_m200: f64,
    /// Reduced chi-squared of the fit.
    pub chi2_red: f64,
}

impl SparcNfwFit {
    /// Characteristic density rho_s in Msun/kpc^3.
    ///
    /// Converts from log10(Msun/pc^3) to linear Msun/kpc^3.
    pub fn rho_s_solar_per_kpc3(&self) -> f64 {
        10.0_f64.powf(self.log_rho_s) * 1e9 // pc^3 -> kpc^3
    }

    /// Halo mass M200 in solar masses.
    pub fn m200_solar(&self) -> f64 {
        10.0_f64.powf(self.log_m200)
    }
}

/// Parse Li et al. (2020) NFW fit table (VizieR J/ApJS/247/31 table1.dat).
///
/// Fixed-width format: 152 chars/line, 2100 records (12 models x 175 galaxies).
/// Only extracts rows matching the specified model (default: "NFW-Flat").
///
/// Column byte ranges (1-indexed per ReadMe):
///   Name: 1-11, Model: 13-24, Y_disk: 26-29, Y_bulge: 36-39,
///   Dist: 46-51, Inc: 59-62, V200: 69-74, C200: 83-88,
///   r_s: 97-102, log(rho_s): 113-117, log(M200): 126-130, chi2: 148-152
pub fn parse_sparc_nfw_fits(path: &Path) -> Result<Vec<SparcNfwFit>, String> {
    parse_sparc_nfw_fits_model(path, "NFW-Flat")
}

/// Parse NFW fits for a specific halo model.
pub fn parse_sparc_nfw_fits_model(
    path: &Path,
    model_filter: &str,
) -> Result<Vec<SparcNfwFit>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("read error: {e}"))?;

    let mut fits = Vec::new();

    for line in content.lines() {
        if line.len() < 130 {
            continue;
        }

        // Extract fixed-width fields (0-indexed byte slices)
        let name = line.get(0..11).unwrap_or("").trim().to_string();
        let model = line.get(12..24).unwrap_or("").trim().to_string();

        if model != model_filter {
            continue;
        }

        let pf = |start: usize, end: usize| -> f64 {
            line.get(start..end)
                .unwrap_or("")
                .trim()
                .parse()
                .unwrap_or(f64::NAN)
        };

        let fit = SparcNfwFit {
            name,
            model,
            y_disk: pf(25, 29),
            y_bulge: pf(35, 39),
            distance_mpc: pf(45, 51),
            inclination_deg: pf(58, 62),
            v200_km_s: pf(68, 74),
            c200: pf(82, 88),
            r_s_kpc: pf(96, 102),
            log_rho_s: pf(112, 117),
            log_m200: pf(125, 130),
            chi2_red: pf(147, 152),
        };

        if fit.r_s_kpc.is_finite() && fit.r_s_kpc > 0.0 && fit.c200.is_finite() {
            fits.push(fit);
        }
    }

    if fits.is_empty() {
        return Err(format!(
            "no valid {model_filter} fits in {}",
            path.display()
        ));
    }

    Ok(fits)
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
    fn test_parse_sparc_rotmod() {
        let content = "\
# NGC2403 rotation curve
# Rad Vobs errV Vgas Vdisk Vbul SBdisk SBbul
 0.25  30.5  3.2  15.1  20.3   0.0  120.5   0.0
 0.50  55.2  2.8  20.4  35.1   0.0   95.3   0.0
 1.00  85.0  3.5  25.6  60.2   0.0   72.1   0.0
 2.00 110.3  4.1  28.3  80.5   0.0   45.6   0.0
";
        let path = write_temp(content, ".dat");
        let galaxy = parse_sparc_rotmod(Path::new(&path)).unwrap();
        assert_eq!(galaxy.points.len(), 4);
        assert!((galaxy.points[0].r_kpc - 0.25).abs() < 1e-10);
        assert!((galaxy.points[0].v_obs_km_s - 30.5).abs() < 1e-10);
        assert!((galaxy.points[3].r_kpc - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_sparc_rotmod_skips_bad_lines() {
        let content = "\
 0.25  30.5  3.2  15.1  20.3   0.0  120.5   0.0
bad line
 0.50  55.2  2.8  20.4  35.1   0.0   95.3   0.0
";
        let path = write_temp(content, ".dat");
        let galaxy = parse_sparc_rotmod(Path::new(&path)).unwrap();
        assert_eq!(galaxy.points.len(), 2);
    }

    #[test]
    fn test_parse_vizier_table2() {
        let content = "\
CamB          3.36   0.16   1.99  1.50   1.86   3.75   0.00   30.32     0.00
CamB          3.36   0.41   4.84  1.50   4.24   9.47   0.00   23.77     0.00
D512-2       15.2    0.96  22.90  2.71   4.08  14.85   0.00   16.45     0.00
D512-2       15.2    1.92  36.30  2.71   6.20  19.10   0.00    4.35     0.00
";
        let path = write_temp(content, ".dat");
        let galaxies = parse_sparc_vizier_table2(Path::new(&path)).unwrap();
        assert_eq!(galaxies.len(), 2);
        assert_eq!(galaxies[0].name, "CamB");
        assert_eq!(galaxies[0].points.len(), 2);
        assert_eq!(galaxies[1].name, "D512-2");
        assert_eq!(galaxies[1].points.len(), 2);
        assert!((galaxies[0].points[0].r_kpc - 0.16).abs() < 1e-10);
        assert!((galaxies[1].points[0].v_obs_km_s - 22.90).abs() < 1e-10);
    }

    #[test]
    fn test_parse_nfw_fits_li2020_format() {
        // Reproduce actual Li et al. 2020 fixed-width format (152 chars)
        // Byte positions: Name(1-11) Model(13-24) Ydisk(26-29) eYdisk(31-34)
        //   Ybul(36-39) eYbul(41-44) Dist(46-51) eDist(53-57) Inc(59-62) eInc(64-67)
        //   V200(69-74) eV200(76-81) C200(83-88) eC200(90-95) rs(97-102) ers(104-111)
        //   log_rhos(113-117) elog_rhos(119-124) logM200(126-130) elogM200(132-136)
        //   alpha(138-141) ealpha(143-146) chi2(148-152)
        let line = "CamB        NFW-Flat     0.37 0.08 0.00 0.00   3.18  0.26 59.2  5.2  38.00  12.93   1.01   0.72  51.67    40.93 -4.29   1.19 10.24  0.44 0.00 0.00  8.61";
        let content = format!(
            "{line}\nCamB        Burkert-Flat 0.33 0.07 0.00 0.00   3.07  0.26 58.1  5.2 423.46 130.51   9.47   1.35  61.26    20.80 -2.27   0.60 13.38  0.40 0.00 0.00  3.01\n"
        );
        let path = write_temp(&content, ".dat");
        let fits = parse_sparc_nfw_fits(Path::new(&path)).unwrap();
        // Should only get the NFW-Flat row
        assert_eq!(fits.len(), 1);
        assert_eq!(fits[0].name, "CamB");
        assert_eq!(fits[0].model, "NFW-Flat");
        assert!((fits[0].v200_km_s - 38.00).abs() < 0.1);
        assert!((fits[0].c200 - 1.01).abs() < 0.1);
        assert!((fits[0].r_s_kpc - 51.67).abs() < 0.1);
        assert!((fits[0].log_rho_s - (-4.29)).abs() < 0.1);
        assert!((fits[0].log_m200 - 10.24).abs() < 0.1);
    }

    #[test]
    fn test_nfw_fit_rho_s_conversion() {
        let fit = SparcNfwFit {
            name: "test".to_string(),
            model: "NFW-Flat".to_string(),
            distance_mpc: 3.0,
            inclination_deg: 60.0,
            y_disk: 0.5,
            y_bulge: 0.0,
            r_s_kpc: 10.0,
            log_rho_s: -2.0, // 0.01 Msun/pc^3
            v200_km_s: 100.0,
            c200: 10.0,
            log_m200: 11.5,
            chi2_red: 1.0,
        };
        // 0.01 Msun/pc^3 * 1e9 pc^3/kpc^3 = 1e7 Msun/kpc^3
        assert!((fit.rho_s_solar_per_kpc3() - 1e7).abs() < 1e3);
    }

    #[test]
    fn test_parse_sparc_nfw_empty() {
        let content = "# header only\n";
        let path = write_temp(content, ".dat");
        let result = parse_sparc_nfw_fits(Path::new(&path));
        assert!(result.is_err());
    }

    #[test]
    fn test_radii_monotonic() {
        let content = "\
 0.25  30.5  3.2  15.1  20.3   0.0  120.5   0.0
 0.50  55.2  2.8  20.4  35.1   0.0   95.3   0.0
 1.00  85.0  3.5  25.6  60.2   0.0   72.1   0.0
";
        let path = write_temp(content, ".dat");
        let galaxy = parse_sparc_rotmod(Path::new(&path)).unwrap();
        for w in galaxy.points.windows(2) {
            assert!(
                w[0].r_kpc < w[1].r_kpc,
                "radii should increase: {} < {}",
                w[0].r_kpc,
                w[1].r_kpc
            );
        }
    }
}
