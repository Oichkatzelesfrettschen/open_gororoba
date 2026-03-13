//! HI 21cm FITS cube rotation curve extractor (E-193).
//!
//! Reads 3D or 4D (single-Stokes) HI FITS velocity cubes (THINGS, HALOGAS,
//! LITTLE THINGS), computes moment-1 velocity maps with sigma-clip noise
//! rejection, and extracts tilted-ring rotation curves via elliptical ring
//! averaging.
//!
//! Output CSV (name,r_kpc,v_obs_km_s,v_err_km_s,psf_flag) is compatible with
//! harmonic-halo-stacking-manga and parse_manga_rotcurves().
//!
//! Algorithm:
//!   1. Read cube data and WCS velocity axis from FITS header.
//!   2. Estimate global RMS from first 10% (emission-free) channels.
//!   3. Moment-1 map: v_mom1(y,x) = sum_ch(T*v)/sum_ch(T), T > sigma_clip*rms.
//!   4. Ring averaging: v_circ = (v_obs - v_sys) / (sin(i)*cos(theta)),
//!      excluding |cos(theta)| < 0.5 (minor-axis noise).
//!   5. psf_flag = r_kpc < beam_fwhm_kpc (inner beam-smearing region).
//!
//! Usage:
//!   hi-cube-rotcurve-extractor \
//!     --cube-dir data/external/things/ \
//!     --meta-csv data/external/things/things_metadata.csv \
//!     --out-csv data/external/things/things_rotcurves.csv
//!
//! Reference: E-193, C-1375
//! THINGS: Walter et al. (2008), AJ 136, 2563
//! de Blok et al. (2008), AJ 136, 2648 (THINGS rotation curves)
//! LITTLE THINGS: Hunter et al. (2012), AJ 144, 134

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use data_core::catalogs::hi_cube::{HiCubeMetadata, parse_hi_cube_metadata};
use fitsio::{FitsFile, hdu::HduInfo};
use std::path::{Path, PathBuf};

/// HI 21cm rest frequency in Hz.
const HI_REST_FREQ_HZ: f64 = 1_420_405_751.77;
/// Speed of light in km/s.
const C_KM_S: f64 = 299_792.458;
/// arcsec -> radians conversion.
const ARCSEC_TO_RAD: f64 = std::f64::consts::PI / (180.0 * 3_600.0);

// ---- CLI ----

#[derive(Parser)]
#[command(name = "hi-cube-rotcurve-extractor")]
#[command(about = "Extract HI 21cm rotation curves from FITS velocity cubes (E-193)")]
struct Cli {
    /// Directory containing FITS cube files (located by galaxy-name substring match).
    #[arg(long, default_value = "data/external/things")]
    cube_dir: PathBuf,

    /// Galaxy metadata CSV: name,ra_deg,dec_deg,distance_mpc,inclination_deg,pa_deg,
    ///   beam_fwhm_arcsec,channel_width_km_s,v_sys_km_s.
    #[arg(long, default_value = "data/external/things/things_metadata.csv")]
    meta_csv: PathBuf,

    /// Output rotation curves CSV.
    #[arg(long, default_value = "data/external/things/things_rotcurves.csv")]
    out_csv: PathBuf,

    /// Sigma-clipping threshold for moment-1 mask (T > sigma_clip * rms).
    #[arg(long, default_value_t = 3.0)]
    sigma_clip: f64,

    /// Maximum ring radius in arcsec (0 = auto: half the cube extent).
    #[arg(long, default_value_t = 0.0)]
    max_r_arcsec: f64,

    /// Minimum unmasked pixels per ring for inclusion.
    #[arg(long, default_value_t = 5)]
    min_ring_pixels: usize,
}

// ---- FITS cube in memory ----

/// HI velocity cube loaded into RAM.
struct HiCube {
    /// Flattened [n_ch, ny, nx] data in C row-major order (channel outermost).
    data: Vec<f32>,
    n_ch: usize,
    ny: usize,
    nx: usize,
    /// Heliocentric velocity at each channel in km/s.
    v_km_s: Vec<f64>,
    /// Pixel angular scale in arcsec/pixel (from CDELT2, assumed square).
    cdelt_arcsec: f64,
}

// ---- FITS loading ----

/// Load a HI FITS velocity cube from disk.
///
/// Handles PRIMARY HDU shapes:
///   - 3D [n_ch, ny, nx]         (standard THINGS format)
///   - 4D [1, n_ch, ny, nx]      (single-Stokes wrapper)
///
/// Velocity axis is reconstructed from WCS keywords CRPIX3, CRVAL3, CDELT3.
/// CTYPE3 = FREQ is converted to radio velocity using the HI 21cm rest frequency.
/// CUNIT3 = m/s is converted to km/s.
fn load_hi_cube(fits_path: &Path) -> Result<HiCube> {
    let path_str = fits_path.to_str().context("non-UTF8 path")?;
    let mut fptr =
        FitsFile::open(path_str).map_err(|e| anyhow!("fitsio open {}: {}", path_str, e))?;

    // Access primary HDU (index 0).
    let hdu = fptr
        .hdu(0usize)
        .map_err(|e| anyhow!("primary HDU: {}", e))?;

    // Determine 3D shape from FITS NAXIS layout.
    // cfitsio reports shape as [NAXIS_N, ..., NAXIS_2, NAXIS_1].
    let (n_ch, ny, nx) = match &hdu.info {
        HduInfo::ImageInfo { shape, .. } if shape.len() == 3 => (shape[0], shape[1], shape[2]),
        HduInfo::ImageInfo { shape, .. } if shape.len() == 4 => {
            // [Stokes=1, n_ch, ny, nx]: single-Stokes wrapper
            (shape[1], shape[2], shape[3])
        }
        HduInfo::ImageInfo { shape, .. } => {
            return Err(anyhow!(
                "unsupported cube shape {:?} (expected 3D or 4D)",
                shape
            ));
        }
        other => return Err(anyhow!("primary HDU is not an image: {:?}", other)),
    };

    // Read WCS velocity axis keywords; fall back to sensible defaults.
    let crpix3: f64 = hdu.read_key::<f64>(&mut fptr, "CRPIX3").unwrap_or(1.0);
    let crval3: f64 = hdu.read_key::<f64>(&mut fptr, "CRVAL3").unwrap_or(0.0);
    let cdelt3: f64 = hdu.read_key::<f64>(&mut fptr, "CDELT3").unwrap_or(1.0);
    let ctype3: String = hdu
        .read_key::<String>(&mut fptr, "CTYPE3")
        .unwrap_or_else(|_| "VRAD".to_string());
    let cunit3: String = hdu
        .read_key::<String>(&mut fptr, "CUNIT3")
        .unwrap_or_else(|_| "km/s".to_string());

    // Pixel angular scale from CDELT2 (degrees/pixel).
    let cdelt2_deg: f64 = hdu
        .read_key::<f64>(&mut fptr, "CDELT2")
        .unwrap_or(1.0 / 3_600.0);
    let cdelt_arcsec = cdelt2_deg.abs() * 3_600.0;

    // Build per-channel velocity in km/s using FITS WCS linear formula:
    //   v[i] = CRVAL3 + (i + 1 - CRPIX3) * CDELT3   (1-indexed channel i+1)
    let mut v_km_s: Vec<f64> = (0..n_ch)
        .map(|i| crval3 + (i as f64 + 1.0 - crpix3) * cdelt3)
        .collect();

    if ctype3.trim().starts_with("FREQ") {
        // Frequency axis: convert Hz -> radio velocity km/s.
        // v_radio = c * (1 - f/f_rest)
        for v in &mut v_km_s {
            *v = C_KM_S * (1.0 - *v / HI_REST_FREQ_HZ);
        }
    } else if cunit3.trim().eq_ignore_ascii_case("m/s") {
        // VRAD / VOPT in m/s: convert to km/s.
        for v in &mut v_km_s {
            *v /= 1_000.0;
        }
    }

    // Read cube data. For 4D cubes, read only the first Stokes plane (n_ch*ny*nx pixels).
    let n_total = n_ch * ny * nx;
    let data: Vec<f32> = hdu
        .read_section(&mut fptr, 0, n_total)
        .map_err(|e| anyhow!("read_section primary HDU: {}", e))?;

    Ok(HiCube {
        data,
        n_ch,
        ny,
        nx,
        v_km_s,
        cdelt_arcsec,
    })
}

// ---- Moment-1 computation ----

/// Estimate global RMS noise from the first 10% of velocity channels.
///
/// For THINGS / HALOGAS cubes the systemic velocity is well away from v=0,
/// so the first fraction of channels is typically emission-free and gives a
/// reliable noise estimate without requiring per-pixel masking.
fn estimate_rms(cube: &HiCube) -> f64 {
    let n_noise_ch = (cube.n_ch / 10).clamp(1, cube.n_ch);
    let n_pix = cube.ny * cube.nx;
    let mut sum_sq = 0.0_f64;
    let mut count = 0_usize;
    for ch in 0..n_noise_ch {
        for pix in 0..n_pix {
            let v = cube.data[ch * n_pix + pix];
            if v.is_finite() {
                sum_sq += (v as f64) * (v as f64);
                count += 1;
            }
        }
    }
    if count == 0 {
        return 1.0;
    }
    (sum_sq / count as f64).sqrt()
}

/// Compute the flux-weighted mean velocity (moment-1) map with sigma-clip masking.
///
/// Returns a flat [ny * nx] array. Pixels with no unmasked emission are NaN.
fn moment1_map(cube: &HiCube, sigma_clip: f64) -> Vec<f64> {
    let rms = estimate_rms(cube);
    let threshold = sigma_clip * rms;
    let n_pix = cube.ny * cube.nx;

    let mut sum_tv = vec![0.0_f64; n_pix];
    let mut sum_t = vec![0.0_f64; n_pix];

    for ch in 0..cube.n_ch {
        let v = cube.v_km_s[ch];
        let base = ch * n_pix;
        for pix in 0..n_pix {
            let t = cube.data[base + pix] as f64;
            if t.is_finite() && t > threshold {
                sum_tv[pix] += t * v;
                sum_t[pix] += t;
            }
        }
    }

    sum_tv
        .iter()
        .zip(sum_t.iter())
        .map(|(&tv, &t)| if t > 0.0 { tv / t } else { f64::NAN })
        .collect()
}

// ---- Tilted-ring extraction ----

/// Extract rotation curve points from a moment-1 map using tilted-ring averaging.
///
/// For each pixel:
///   1. Project to galaxy frame (PA rotation, inclination deprojection).
///   2. Compute azimuthal angle theta in the disk plane.
///   3. Skip minor-axis pixels (|cos(theta)| < 0.5) to reduce projection noise.
///   4. Derive v_circ = (v_obs - v_sys) / (sin(i) * cos(theta)).
///   5. Average within rings of width = beam_fwhm_arcsec.
///
/// Returns Vec of (r_kpc, v_circ_km_s, v_err_km_s, psf_flag) sorted by radius.
/// Rings with fewer than `min_ring_pixels` unmasked pixels are dropped.
fn extract_rings(
    mom1: &[f64],
    meta: &HiCubeMetadata,
    cube: &HiCube,
    max_r_arcsec: f64,
    min_ring_pixels: usize,
) -> Vec<(f64, f64, f64, bool)> {
    let incl_deg = meta.inclination_deg;
    if !(20.0..=80.0).contains(&incl_deg) {
        // Face-on (i < 20 deg): v_los -> 0, projection noise dominates.
        // Edge-on (i > 80 deg): severe beam smearing along the line of sight.
        return vec![];
    }

    let sin_i = incl_deg.to_radians().sin();
    let cos_i = incl_deg.to_radians().cos();
    let pa_rad = meta.pa_deg.to_radians();
    let (sin_pa, cos_pa) = pa_rad.sin_cos();

    let ny = cube.ny as i32;
    let nx = cube.nx as i32;
    // Image center in pixel coordinates.
    let cx = (nx as f64 - 1.0) / 2.0;
    let cy = (ny as f64 - 1.0) / 2.0;

    let cdelt = cube.cdelt_arcsec;
    // kpc per arcsec at the galaxy distance.
    let kpc_per_arcsec = meta.distance_mpc * 1_000.0 * ARCSEC_TO_RAD;

    // Ring width = synthesized beam FWHM (at least 2 pixels).
    let ring_width_arcsec = meta.beam_fwhm_arcsec.max(cdelt * 2.0);
    let psf_radius_kpc = meta.beam_fwhm_arcsec * kpc_per_arcsec;

    let effective_max = if max_r_arcsec > 0.0 {
        max_r_arcsec
    } else {
        (nx.min(ny) as f64 / 2.0 - 1.0) * cdelt
    };
    let n_rings = ((effective_max / ring_width_arcsec) as usize).max(1);

    // Accumulate v_circ samples per ring.
    let mut ring_samples: Vec<Vec<f64>> = vec![Vec::new(); n_rings];

    for iy in 0..ny {
        for ix in 0..nx {
            let idx = (iy * nx + ix) as usize;
            let v_obs = mom1[idx];
            if !v_obs.is_finite() {
                continue;
            }

            // Offset from center in arcsec (East = +dx, North = +dy).
            let dx = (ix as f64 - cx) * cdelt;
            let dy = (iy as f64 - cy) * cdelt;

            // Rotate to galaxy major-axis frame.
            // xi: along major axis; eta: along minor axis.
            let xi = -dx * sin_pa + dy * cos_pa;
            let eta = -dx * cos_pa - dy * sin_pa;

            // Deproject eta to galaxy plane using inclination.
            let eta_disk = eta / cos_i;
            let r_arcsec = (xi * xi + eta_disk * eta_disk).sqrt();

            if r_arcsec >= effective_max || r_arcsec < cdelt * 0.5 {
                continue;
            }

            // Azimuthal angle in the disk plane.
            let theta = eta_disk.atan2(xi);
            let cos_theta = theta.cos();
            // Exclude minor axis: projection factor -> 0, velocity noise -> infinity.
            if cos_theta.abs() < 0.5 {
                continue;
            }

            // Deproject line-of-sight velocity to circular velocity.
            let v_circ = (v_obs - meta.v_sys_km_s) / (sin_i * cos_theta);

            let ring_idx = (r_arcsec / ring_width_arcsec) as usize;
            if ring_idx < n_rings {
                ring_samples[ring_idx].push(v_circ);
            }
        }
    }

    // Reduce each ring to (r_kpc, mean_v_circ, err, psf_flag).
    let mut result = Vec::new();
    for (i, samples) in ring_samples.iter().enumerate() {
        if samples.len() < min_ring_pixels {
            continue;
        }
        let n = samples.len() as f64;
        let mean = samples.iter().copied().sum::<f64>() / n;
        let var = samples
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f64>()
            / n;
        // Standard error of the mean; floor at 1 km/s to avoid division artifacts.
        let err = (var.sqrt() / n.sqrt()).max(1.0);

        // Ring center radius.
        let r_arcsec = (i as f64 + 0.5) * ring_width_arcsec;
        let r_kpc = r_arcsec * kpc_per_arcsec;
        let psf_flag = r_kpc < psf_radius_kpc;

        result.push((r_kpc, mean, err, psf_flag));
    }
    result
}

// ---- File discovery ----

/// Find a FITS cube file for a galaxy by case-insensitive name substring match.
///
/// Searches `dir` for files ending in `.fits` or `.fits.gz` whose name
/// contains `needle` (spaces replaced with underscores for matching).
fn find_fits_file(dir: &Path, needle: &str) -> Option<PathBuf> {
    let needle_lower = needle.to_lowercase().replace(' ', "_");
    let entries = std::fs::read_dir(dir).ok()?;
    let mut candidates: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let name = p
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_lowercase();
            (name.ends_with(".fits") || name.ends_with(".fits.gz"))
                && name.contains(needle_lower.as_str())
        })
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

// ---- Entry point ----

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    eprintln!("Loading metadata from {}...", cli.meta_csv.display());
    let metas = parse_hi_cube_metadata(&cli.meta_csv).map_err(|e| anyhow::anyhow!(e))?;
    eprintln!("  {} galaxies in metadata catalog", metas.len());

    let mut wtr = csv::Writer::from_path(&cli.out_csv)?;
    wtr.write_record(["name", "r_kpc", "v_obs_km_s", "v_err_km_s", "psf_flag"])?;

    let mut n_ok = 0_usize;
    let mut n_skip = 0_usize;

    for meta in &metas {
        let cube_path = match find_fits_file(&cli.cube_dir, &meta.name) {
            Some(p) => p,
            None => {
                eprintln!(
                    "  [skip] {}: no FITS file found in {}",
                    meta.name,
                    cli.cube_dir.display()
                );
                n_skip += 1;
                continue;
            }
        };

        eprintln!("Processing {} ({})...", meta.name, cube_path.display());
        let cube = match load_hi_cube(&cube_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  [skip] {}: load error: {}", meta.name, e);
                n_skip += 1;
                continue;
            }
        };

        let mom1 = moment1_map(&cube, cli.sigma_clip);
        let rings = extract_rings(&mom1, meta, &cube, cli.max_r_arcsec, cli.min_ring_pixels);

        if rings.is_empty() {
            eprintln!("  [skip] {}: no valid rings extracted", meta.name);
            n_skip += 1;
            continue;
        }

        for (r_kpc, v_circ, v_err, psf_flag) in &rings {
            wtr.write_record([
                meta.name.as_str(),
                &format!("{r_kpc:.6}"),
                &format!("{v_circ:.4}"),
                &format!("{v_err:.4}"),
                if *psf_flag { "true" } else { "false" },
            ])?;
        }

        eprintln!(
            "  {} rings (r = {:.1} to {:.1} kpc)",
            rings.len(),
            rings.first().map(|r| r.0).unwrap_or(0.0),
            rings.last().map(|r| r.0).unwrap_or(0.0),
        );
        n_ok += 1;
    }

    wtr.flush()?;
    eprintln!(
        "\nDone: {} extracted, {} skipped. Output: {}",
        n_ok,
        n_skip,
        cli.out_csv.display()
    );
    Ok(())
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use data_core::catalogs::hi_cube::HiCubeMetadata;

    fn make_cube(n_ch: usize, ny: usize, nx: usize, fill: f32, v0: f64) -> HiCube {
        HiCube {
            data: vec![fill; n_ch * ny * nx],
            n_ch,
            ny,
            nx,
            v_km_s: (0..n_ch).map(|i| v0 + i as f64 * 10.0).collect(),
            cdelt_arcsec: 5.0,
        }
    }

    #[test]
    fn test_moment1_single_channel_uniform() {
        // 1 channel at 100 km/s, all pixels above threshold.
        let cube = make_cube(1, 3, 3, 10.0, 100.0);
        let mom1 = moment1_map(&cube, 0.0); // sigma_clip=0: accept all
        assert_eq!(mom1.len(), 9);
        for v in &mom1 {
            assert!((*v - 100.0).abs() < 1e-6, "expected 100 km/s, got {v}");
        }
    }

    #[test]
    fn test_moment1_all_clipped() {
        // Very high sigma_clip clips all emission -> all NaN.
        let cube = make_cube(1, 3, 3, 0.001, 50.0);
        let mom1 = moment1_map(&cube, 1_000.0);
        assert!(mom1.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_find_fits_file_no_match() {
        let dir = std::env::temp_dir();
        assert!(find_fits_file(&dir, "nonexistent_galaxy_xyz12345").is_none());
    }

    #[test]
    fn test_extract_rings_face_on_returns_empty() {
        // Inclination < 20 deg: face-on, should return empty.
        let meta = HiCubeMetadata {
            name: "test".to_string(),
            ra_deg: 0.0,
            dec_deg: 0.0,
            distance_mpc: 10.0,
            inclination_deg: 5.0,
            pa_deg: 0.0,
            beam_fwhm_arcsec: 10.0,
            channel_width_km_s: 2.6,
            v_sys_km_s: 500.0,
        };
        let cube = make_cube(1, 5, 5, 5.0, 500.0);
        let mom1 = vec![500.0_f64; 25];
        let rings = extract_rings(&mom1, &meta, &cube, 0.0, 1);
        assert!(rings.is_empty(), "face-on galaxy should yield no rings");
    }

    #[test]
    fn test_extract_rings_edge_on_returns_empty() {
        // Inclination > 80 deg: edge-on, excluded.
        let meta = HiCubeMetadata {
            name: "test".to_string(),
            ra_deg: 0.0,
            dec_deg: 0.0,
            distance_mpc: 10.0,
            inclination_deg: 85.0,
            pa_deg: 0.0,
            beam_fwhm_arcsec: 10.0,
            channel_width_km_s: 2.6,
            v_sys_km_s: 500.0,
        };
        let cube = make_cube(1, 5, 5, 5.0, 500.0);
        let mom1 = vec![500.0_f64; 25];
        let rings = extract_rings(&mom1, &meta, &cube, 0.0, 1);
        assert!(rings.is_empty(), "edge-on galaxy should yield no rings");
    }
}
