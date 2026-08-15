//! HI 21cm rotation curve stacking with eigenmode analysis.
//!
//! Loads HI rotation curves from the hi-cube-rotcurve-extractor output,
//! normalizes to x = r/r_s using SMHM-derived NFW parameters, and performs:
//! 1. Traditional DFT stacking (via stack_residuals) -> stack_csv
//! 2. PCA eigenmode stacking (via eigenmode_stack) -> eigen_csv
//!
//! For THINGS/LITTLE-THINGS galaxies lacking stellar mass data, falls back
//! to a fiducial M_star = 1e11 Msun.
//!
//! Reference: E-195 (HI eigenmode stacking)

use clap::Args;
use cosmology_core::{
    CdDimensionParams, EigenmodeStackResult, NormalizedPoint, NormalizedResiduals, StackingConfig,
    detection_threshold, eigenmode_stack,
    nfw_utils::{nfw_enclosed_mass_from_params, nfw_params_from_mass},
    stack_residuals,
};
use data_core::catalogs::{
    hi_cube::{HiCubeMetadata, HiRotationCurve, parse_hi_cube_metadata, parse_hi_rotcurves},
    manga::moster2013_log_m200,
};
use std::{collections::HashMap, path::PathBuf};

/// Gravitational constant in kpc^3 Msun^{-1} (km/s)^2 units.
const G_KPC_KMS2: f64 = 4.302e-6;

/// Fiducial stellar mass when no measurement is available.
const FIDUCIAL_M_STAR_MSUN: f64 = 1.0e11;

#[derive(Args)]
pub struct Cli {
    /// HI rotation curves CSV (from hi-cube-rotcurve-extractor).
    #[arg(long, default_value = "data/external/things/things_rotcurves.csv")]
    rotcurves: PathBuf,

    /// HiCubeMetadata CSV (name, ra_deg, dec_deg, distance_mpc, inclination_deg, ...).
    #[arg(long, default_value = "data/external/things/things_metadata.csv")]
    meta_csv: PathBuf,

    /// Cayley-Dickson dimension.
    #[arg(long, default_value_t = 16)]
    cd_dim: usize,

    /// Minimum x = r/r_s for analysis grid.
    #[arg(long, default_value_t = 0.5)]
    x_min: f64,

    /// Maximum x = r/r_s for analysis grid.
    #[arg(long, default_value_t = 10.0)]
    x_max: f64,

    /// Number of grid points.
    #[arg(long, default_value_t = 500)]
    n_grid: usize,

    /// Top K eigenmodes to analyze.
    #[arg(long, default_value_t = 20)]
    n_modes: usize,

    /// Minimum galaxies per x-bin.
    #[arg(long, default_value_t = 2)]
    min_per_bin: usize,

    /// Minimum rotation curve points per galaxy.
    #[arg(long, default_value_t = 8)]
    min_points: usize,

    /// Output CSV for traditional DFT stacked profile.
    #[arg(long, default_value = "hi_stack.csv")]
    stack_csv: PathBuf,

    /// Output CSV for eigenmode analysis.
    #[arg(long, default_value = "hi_eigen.csv")]
    eigen_csv: PathBuf,
}

/// Try to read stellar masses from a sidecar CSV (column: name, log_m_star or stellar_mass_msun).
/// Returns a map from galaxy name to log10(M_star / Msun).
fn try_load_stellar_masses(meta_path: &std::path::Path) -> HashMap<String, f64> {
    let content = match std::fs::read_to_string(meta_path) {
        Ok(c) => c,
        Err(_) => return HashMap::new(),
    };
    let mut map = HashMap::new();
    let mut hdr_map: Option<HashMap<String, usize>> = None;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(|f| f.trim()).collect();
        if hdr_map.is_none() {
            let h: HashMap<String, usize> = fields
                .iter()
                .enumerate()
                .map(|(i, &f)| (f.to_lowercase(), i))
                .collect();
            hdr_map = Some(h);
            continue;
        }
        let h = hdr_map.as_ref().unwrap();
        let get = |key: &str| -> Option<f64> {
            h.get(key)
                .and_then(|&i| fields.get(i))
                .and_then(|s| s.parse().ok())
        };
        let name = h
            .get("name")
            .and_then(|&i| fields.get(i))
            .map(|s| s.to_string())
            .unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        // Accept log_m_star or convert linear stellar_mass_msun
        let log_ms = get("log_m_star")
            .or_else(|| get("stellar_mass_msun").map(|m| m.log10()))
            .unwrap_or(f64::NAN);
        if log_ms.is_finite() && log_ms > 7.0 {
            map.insert(name, log_ms);
        }
    }
    map
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    eprintln!(
        "Loading HI rotation curves from {}...",
        cli.rotcurves.display()
    );
    let rotcurves = parse_hi_rotcurves(&cli.rotcurves).map_err(|e| anyhow::anyhow!(e))?;
    eprintln!("  Loaded {} galaxies", rotcurves.len());

    eprintln!("Loading HI metadata from {}...", cli.meta_csv.display());
    let metadata = parse_hi_cube_metadata(&cli.meta_csv).map_err(|e| anyhow::anyhow!(e))?;
    let meta_map: HashMap<&str, &HiCubeMetadata> =
        metadata.iter().map(|m| (m.name.as_str(), m)).collect();
    eprintln!("  Loaded {} metadata entries", metadata.len());

    // Try to load stellar masses from the same meta CSV
    let stellar_masses = try_load_stellar_masses(&cli.meta_csv);
    eprintln!(
        "  Stellar masses found for {}/{} galaxies (fiducial {:.0e} Msun for rest)",
        stellar_masses.len(),
        metadata.len(),
        FIDUCIAL_M_STAR_MSUN
    );

    // Build a map for quick rotation curve lookup by name
    let rotcurve_map: HashMap<&str, &HiRotationCurve> =
        rotcurves.iter().map(|rc| (rc.name.as_str(), rc)).collect();

    // Normalize each galaxy
    let mut normalized: Vec<NormalizedResiduals> = Vec::new();
    let skipped_no_meta = 0_usize;
    let mut skipped_smhm = 0_usize;
    let mut skipped_few_pts = 0_usize;

    for (name, meta) in &meta_map {
        let Some(rc) = rotcurve_map.get(name) else {
            continue;
        };

        // Determine log10(M_halo) via SMHM
        let log_ms = stellar_masses
            .get(*name)
            .copied()
            .unwrap_or_else(|| FIDUCIAL_M_STAR_MSUN.log10());
        let log_m200 = moster2013_log_m200(log_ms);
        if !log_m200.is_finite() || log_m200 < 10.0 {
            skipped_smhm += 1;
            continue;
        }
        let m200 = 10.0_f64.powf(log_m200);

        // Approximate redshift from distance: z ~ d_H/c (low-z approximation)
        let z = (meta.distance_mpc * 70.0 / 299792.458).max(0.0);
        let nfw = nfw_params_from_mass(m200, z);
        if nfw.r_s_kpc <= 0.0 {
            skipped_smhm += 1;
            continue;
        }

        // Normalize rotation curve
        let mut points: Vec<NormalizedPoint> = Vec::new();
        for pt in &rc.points {
            if pt.psf_flag {
                continue; // exclude beam-smearing-affected inner bins
            }
            let x = pt.r_kpc / nfw.r_s_kpc;
            if !(0.1..=20.0).contains(&x) {
                continue;
            }
            let m_enc = nfw_enclosed_mass_from_params(pt.r_kpc, &nfw);
            let v_nfw = (G_KPC_KMS2 * m_enc / pt.r_kpc).max(0.0).sqrt();
            if v_nfw < 1.0 {
                continue;
            }
            let delta_v = (pt.v_circ_km_s - v_nfw) / v_nfw;
            let delta_v_err = pt.v_err_km_s / v_nfw;
            points.push(NormalizedPoint {
                x,
                delta_v,
                delta_v_err,
            });
        }

        if points.len() >= cli.min_points {
            normalized.push(NormalizedResiduals {
                name: name.to_string(),
                r_s_kpc: nfw.r_s_kpc,
                points,
            });
        } else {
            skipped_few_pts += 1;
        }
    }

    // Log any rotation curves with no metadata entry (informational only)
    let _ = skipped_no_meta;

    eprintln!(
        "Normalized {} galaxies (skipped: {} bad SMHM, {} too few points)",
        normalized.len(),
        skipped_smhm,
        skipped_few_pts,
    );

    if normalized.is_empty() {
        anyhow::bail!("No galaxies available for stacking");
    }

    // Build stacking config
    let cd_params = CdDimensionParams::new(cli.cd_dim);
    let config = StackingConfig {
        x_min: cli.x_min,
        x_max: cli.x_max,
        n_grid: cli.n_grid,
        min_galaxies_per_bin: cli.min_per_bin,
        inverse_variance_weighting: true,
        cd_params: cd_params.clone(),
        exclude_psf_flagged: false,
    };

    // Traditional DFT stack
    eprintln!(
        "Stacking {} galaxies (D={}, {} modes)...",
        normalized.len(),
        cd_params.cd_dim,
        cd_params.n_modes
    );
    let stack = stack_residuals(&normalized, &config);

    eprintln!(
        "  RMS={:.6}, SNR={:.3}, alpha_zd estimate={:.6}",
        stack.rms_residual, stack.detection_snr, stack.alpha_zd_estimate
    );

    let thr = detection_threshold(normalized.len(), 0.05);
    eprintln!("  Detection threshold (5% v_err): alpha_zd >= {:.6}", thr);

    // Write stack CSV
    let mut wtr_stack = csv::Writer::from_path(&cli.stack_csv)?;
    wtr_stack.write_record(["x", "delta_stack", "delta_stack_err", "n_contributing"])?;
    for i in 0..stack.x_grid.len() {
        wtr_stack.write_record(&[
            format!("{:.6}", stack.x_grid[i]),
            format!("{:.8e}", stack.delta_stack[i]),
            format!("{:.8e}", stack.delta_stack_err[i]),
            format!("{}", stack.n_contributing[i]),
        ])?;
    }
    wtr_stack.flush()?;
    eprintln!("Stack written to {}", cli.stack_csv.display());

    // PCA eigenmode stacking
    eprintln!(
        "Computing eigenmode decomposition (top {} modes)...",
        cli.n_modes
    );
    let eigen: EigenmodeStackResult = eigenmode_stack(&normalized, &config, cli.n_modes);

    eprintln!(
        "  reconstruction_snr={:.3}, n_modes={}, n_gal={}",
        eigen.reconstruction_snr, eigen.n_modes_kept, eigen.n_galaxies
    );
    if !eigen.zd_overlaps.is_empty() {
        eprintln!(
            "  Leading ZD overlaps: {:.3}, {:.3}, {:.3}",
            eigen.zd_overlaps.first().unwrap_or(&0.0),
            eigen.zd_overlaps.get(1).unwrap_or(&0.0),
            eigen.zd_overlaps.get(2).unwrap_or(&0.0),
        );
    }

    // Write eigenmode CSV
    let mut wtr_eigen = csv::Writer::from_path(&cli.eigen_csv)?;
    wtr_eigen.write_record([
        "mode_index",
        "eigenvalue",
        "zd_overlap",
        "explained_variance",
        "reconstruction_snr_mode",
    ])?;
    for k in 0..eigen.n_modes_kept {
        let ev = eigen.eigenvalues.get(k).copied().unwrap_or(0.0);
        let ov = eigen.zd_overlaps.get(k).copied().unwrap_or(0.0);
        let cv = eigen.explained_variance.get(k).copied().unwrap_or(0.0);
        let lm = if !eigen.eigenvalues.is_empty() {
            eigen.eigenvalues.iter().map(|e| e.abs()).sum::<f64>() / eigen.eigenvalues.len() as f64
        } else {
            1.0
        };
        let snr_k = ov * (ev.abs() / lm.max(1.0e-30)).sqrt();
        wtr_eigen.write_record(&[
            format!("{k}"),
            format!("{ev:.6e}"),
            format!("{ov:.6}"),
            format!("{cv:.6}"),
            format!("{snr_k:.4}"),
        ])?;
    }
    wtr_eigen.flush()?;
    eprintln!("Eigenmode result written to {}", cli.eigen_csv.display());

    Ok(())
}
