//! Pseudo-slit rotation curve de-projection via Tikhonov inversion.
//!
//! Reads MaNGA- or HI-format rotation curves (name, r_kpc, v_obs_km_s, v_err_km_s)
//! and a metadata CSV with inclination and PSF FWHM, then applies the Gaussian PSF
//! projection model and Tikhonov-regularized inversion from cosmology_core::slit_projection
//! to recover the intrinsic circular velocity v_circ.
//!
//! Metadata CSV formats supported:
//!   DAPall style: plateifu, nsa_elpetro_ba, psf_fwhm_arcsec, distance_mpc[, z]
//!   HiCube style: name (or object_name), inclination_deg, beam_fwhm_arcsec, distance_mpc
//!
//! The format is detected automatically from the header.
//!
//! Reference: E-194 (slit de-projection)

use clap::Parser;
use cosmology_core::slit_projection::{compute_projection_matrix, deproject_rotation_curve};
use std::{collections::HashMap, path::PathBuf};

/// arcsec-to-kpc conversion factor: 1 arcsec at 1 Mpc = pi/(180*3600) Mpc = 4.8481e-6 Mpc = 4.8481e-3 kpc.
const ARCSEC_TO_KPC_PER_MPC: f64 = 4.848_136_811e-3;

#[derive(Parser)]
#[command(name = "slit-deproject")]
#[command(about = "De-project pseudo-slit rotation curves via Tikhonov-regularized PSF inversion")]
struct Cli {
    /// Input rotation curves CSV (name, r_kpc, v_obs_km_s, v_err_km_s).
    #[arg(long)]
    rotcurves: PathBuf,

    /// Metadata CSV (inclination, PSF FWHM, distance).
    #[arg(long)]
    meta_csv: PathBuf,

    /// Output CSV path (name, r_kpc, v_circ_km_s, v_circ_err_km_s).
    #[arg(long)]
    out_csv: PathBuf,

    /// Tikhonov regularization parameter.
    #[arg(long, default_value_t = 0.01)]
    lambda: f64,

    /// Minimum radial bins required per galaxy.
    #[arg(long, default_value_t = 6)]
    min_points: usize,

    /// Slit half-width in arcsec (converted to kpc using galaxy distance).
    #[arg(long, default_value_t = 1.0)]
    slit_half_width: f64,
}

/// Per-galaxy metadata extracted from the meta CSV.
struct GalMeta {
    /// Disk inclination in degrees.
    incl_deg: f64,
    /// PSF FWHM in arcsec.
    psf_fwhm_arcsec: f64,
    /// Distance in Mpc.
    distance_mpc: f64,
}

/// Parse the metadata CSV and return a name -> GalMeta map.
/// Auto-detects DAPall vs HiCube format from header.
fn parse_meta_csv(path: &std::path::Path) -> anyhow::Result<HashMap<String, GalMeta>> {
    let content = std::fs::read_to_string(path)?;
    let mut map = HashMap::new();
    let mut header: Option<HashMap<String, usize>> = None;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(|f| f.trim()).collect();

        if header.is_none() {
            let hdr: HashMap<String, usize> = fields
                .iter()
                .enumerate()
                .map(|(i, &f)| (f.to_lowercase(), i))
                .collect();
            header = Some(hdr);
            continue;
        }

        let hdr = header.as_ref().unwrap();
        let get =
            |key: &str| -> Option<&str> { hdr.get(key).and_then(|&i| fields.get(i)).copied() };
        let get_f64 =
            |key: &str| -> f64 { get(key).and_then(|s| s.parse().ok()).unwrap_or(f64::NAN) };

        // Detect name column: plateifu (DAPall) or name/object_name (HiCube)
        let name = get("plateifu")
            .or_else(|| get("name"))
            .or_else(|| get("object_name"))
            .map(|s| s.to_string())
            .unwrap_or_default();
        if name.is_empty() {
            continue;
        }

        // Inclination: nsa_elpetro_ba -> arccos(ba) [DAPall], or inclination_deg [HiCube]
        let incl_deg = if hdr.contains_key("inclination_deg") {
            get_f64("inclination_deg")
        } else if hdr.contains_key("nsa_elpetro_ba") {
            let ba = get_f64("nsa_elpetro_ba");
            if ba.is_finite() && ba > 0.0 && ba <= 1.0 {
                ba.acos().to_degrees()
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };
        if !(10.0_f64..=85.0).contains(&incl_deg) {
            continue;
        }

        // PSF FWHM: psf_fwhm_arcsec or beam_fwhm_arcsec
        let psf_fwhm_arcsec = get_f64("psf_fwhm_arcsec").max(get_f64("beam_fwhm_arcsec"));
        if !psf_fwhm_arcsec.is_finite() || psf_fwhm_arcsec <= 0.0 {
            continue;
        }

        // Distance: distance_mpc or compute from redshift z
        let distance_mpc = {
            let d = get_f64("distance_mpc");
            if d.is_finite() && d > 0.0 {
                d
            } else {
                // Try computing from z: d_A = cz/H0 in Mpc (simple for z < 0.1)
                let z = get_f64("z").max(get_f64("nsa_z"));
                if z.is_finite() && z > 0.0 {
                    z * 299792.458 / 70.0 // cz/H0 in Mpc
                } else {
                    f64::NAN
                }
            }
        };
        if !distance_mpc.is_finite() || distance_mpc <= 0.0 {
            continue;
        }

        map.insert(
            name,
            GalMeta {
                incl_deg,
                psf_fwhm_arcsec,
                distance_mpc,
            },
        );
    }

    Ok(map)
}

type RotcurveMap = HashMap<String, Vec<(f64, f64, f64)>>;

/// Parse a generic 4-column rotation curve CSV (name, r_kpc, v_obs_km_s, v_err_km_s).
/// Returns a map from galaxy name to Vec<(r_kpc, v_obs, v_err)>.
fn parse_rotcurves(path: &std::path::Path) -> anyhow::Result<RotcurveMap> {
    let content = std::fs::read_to_string(path)?;
    let mut map: RotcurveMap = HashMap::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("name") {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(|f| f.trim()).collect();
        if fields.len() < 4 {
            continue;
        }
        let name = fields[0].to_string();
        let r: f64 = fields[1].parse().unwrap_or(f64::NAN);
        let v: f64 = fields[2].parse().unwrap_or(f64::NAN);
        let e: f64 = fields[3].parse().unwrap_or(f64::NAN);
        if !r.is_finite() || !v.is_finite() || !e.is_finite() || r <= 0.0 {
            continue;
        }
        map.entry(name).or_default().push((r, v, e));
    }

    // Sort each galaxy's points by ascending r_kpc
    for pts in map.values_mut() {
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    Ok(map)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    eprintln!(
        "Loading rotation curves from {}...",
        cli.rotcurves.display()
    );
    let rotcurves = parse_rotcurves(&cli.rotcurves)?;
    eprintln!("  Loaded {} galaxies", rotcurves.len());

    eprintln!("Loading metadata from {}...", cli.meta_csv.display());
    let meta_map = parse_meta_csv(&cli.meta_csv)?;
    eprintln!("  Loaded {} metadata entries", meta_map.len());

    let mut wtr = csv::Writer::from_path(&cli.out_csv)?;
    wtr.write_record(["name", "r_kpc", "v_circ_km_s", "v_circ_err_km_s"])?;

    let mut n_processed = 0_usize;
    let mut n_skipped_meta = 0_usize;
    let mut n_skipped_pts = 0_usize;

    // Process each galaxy
    let mut names: Vec<&String> = rotcurves.keys().collect();
    names.sort();

    for name in names {
        let pts = &rotcurves[name];

        let meta = match meta_map.get(name.as_str()) {
            Some(m) => m,
            None => {
                n_skipped_meta += 1;
                continue;
            }
        };

        if pts.len() < cli.min_points {
            n_skipped_pts += 1;
            continue;
        }

        // Convert PSF FWHM arcsec -> kpc
        let psf_fwhm_kpc = meta.psf_fwhm_arcsec * meta.distance_mpc * ARCSEC_TO_KPC_PER_MPC;
        let slit_hw_kpc = cli.slit_half_width * meta.distance_mpc * ARCSEC_TO_KPC_PER_MPC;

        let r_kpc: Vec<f64> = pts.iter().map(|&(r, _, _)| r).collect();
        let v_obs: Vec<f64> = pts.iter().map(|&(_, v, _)| v).collect();
        let sigma_obs: Vec<f64> = pts.iter().map(|&(_, _, e)| e.max(0.1)).collect();

        let m = compute_projection_matrix(&r_kpc, psf_fwhm_kpc, meta.incl_deg, slit_hw_kpc);
        let (v_circ, v_circ_err) = deproject_rotation_curve(&v_obs, &sigma_obs, &m, cli.lambda);

        for (i, &r) in r_kpc.iter().enumerate() {
            wtr.write_record([
                name.as_str(),
                &format!("{r:.4}"),
                &format!("{:.4}", v_circ[i]),
                &format!("{:.4}", v_circ_err[i]),
            ])?;
        }

        n_processed += 1;
    }

    wtr.flush()?;

    eprintln!(
        "De-projected {} galaxies (skipped: {} no metadata, {} too few points)",
        n_processed, n_skipped_meta, n_skipped_pts
    );
    eprintln!("Output written to {}", cli.out_csv.display());

    Ok(())
}
