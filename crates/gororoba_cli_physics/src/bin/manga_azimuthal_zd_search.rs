//! MaNGA 2D azimuthal Fourier-mode search on IFU velocity maps.
//!
//! This is the first Rust-native scaffold for the one credible reversal lane of
//! the radial-profile MaNGA null: preserve the 2D H-alpha velocity field,
//! deproject to disk coordinates, and measure annular Fourier coefficients
//! m = 0..4 instead of collapsing immediately to a pseudo-slit rotation curve.

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use fitsio::{FitsFile, hdu::HduInfo};
use gauss_quad::GaussLegendre;
use num_complex::Complex64;
use rayon::prelude::*;
use std::{
    fs,
    io::BufReader,
    path::{Path, PathBuf},
};

const H0_KM_S_MPC: f64 = 67.36;
const OM0: f64 = 0.3153;
const C_KM_S: f64 = 299_792.458;
const MPC_IN_KPC: f64 = 1_000.0;
const SPAXEL_SIZE_ARCSEC: f64 = 0.5;
const HA_CHANNEL: usize = 23;

#[derive(Parser, Debug)]
#[command(
    name = "manga-azimuthal-zd-search",
    about = "Annular Fourier search for anisotropic structure in MaNGA 2D IFU velocity maps"
)]
struct Args {
    /// DAPall-derived galaxy selection CSV.
    #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
    selection: PathBuf,

    /// Directory containing cached MaNGA MAPS FITS.gz files.
    #[arg(long, default_value = "data/external/manga/maps_cache")]
    maps_cache: PathBuf,

    /// Output directory for summary TOML and per-ring CSV.
    #[arg(long, default_value = "data/results/e208")]
    output_dir: PathBuf,

    /// Maximum number of galaxies to process.
    #[arg(long, default_value_t = 64)]
    n_max: usize,

    /// Maximum azimuthal Fourier mode to measure.
    #[arg(long, default_value_t = 4)]
    m_max: usize,

    /// Ring width in spaxels for annular aggregation.
    #[arg(long, default_value_t = 5.0)]
    ring_width_spaxels: f64,

    /// Minimum valid pixels required for a ring to contribute.
    #[arg(long, default_value_t = 16)]
    min_ring_pixels: usize,

    /// Exclude near-minor-axis pixels with |cos(theta)| below this threshold.
    #[arg(long, default_value_t = 0.35)]
    min_cos_theta: f64,

    /// Minimum inclination allowed in degrees.
    #[arg(long, default_value_t = 30.0)]
    min_inclination_deg: f64,

    /// Maximum inclination allowed in degrees.
    #[arg(long, default_value_t = 70.0)]
    max_inclination_deg: f64,

    /// Rayon thread count (0 = auto).
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

#[derive(Debug, Clone)]
struct GalaxyRecord {
    plateifu: String,
    plate: u32,
    ifudesign: String,
    z: f64,
    ba: f64,
    pa_deg: f64,
}

#[derive(Debug)]
struct SlicedMaps {
    vel: Vec<f32>,
    ivar: Vec<f32>,
    mask: Vec<i32>,
    ny: usize,
    nx: usize,
}

#[derive(Debug, Clone)]
struct PixelSample {
    theta: f64,
    v_circ: f64,
    weight: f64,
    r_kpc: f64,
}

#[derive(Debug, Clone)]
struct RingSummary {
    ring_idx: usize,
    r_kpc: f64,
    n_pixels: usize,
    mean_v_circ_km_s: f64,
    rms_residual_km_s: f64,
    mode_abs_km_s: Vec<f64>,
    mode_rel_axisymmetric: Vec<f64>,
}

#[derive(Debug, Clone)]
struct GalaxySummary {
    plateifu: String,
    rings: Vec<RingSummary>,
}

enum GalaxyOutcome {
    Processed(GalaxySummary),
    MissingMaps,
    NoValidRings,
}

#[derive(Debug, Default)]
struct ProcessingCounters {
    requested_galaxies: usize,
    processed_galaxies: usize,
    skipped_missing_maps: usize,
    skipped_bad_inclination: usize,
    skipped_fits_error: usize,
    skipped_no_valid_rings: usize,
}

fn e_inv(z: f64) -> f64 {
    let e2 = OM0 * (1.0 + z).powi(3) + (1.0 - OM0);
    1.0 / e2.sqrt()
}

fn angular_diameter_distance_kpc(z: f64) -> f64 {
    let gl = GaussLegendre::new(32).expect("Gauss-Legendre init");
    let chi = gl.integrate(0.0, z, e_inv);
    let d_mpc = (C_KM_S / H0_KM_S_MPC) * chi / (1.0 + z);
    d_mpc * MPC_IN_KPC
}

fn load_selection_csv(path: &Path) -> Result<Vec<GalaxyRecord>> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut rdr = csv::Reader::from_reader(BufReader::new(file));
    let mut records = Vec::new();
    for result in rdr.deserialize::<std::collections::HashMap<String, String>>() {
        let row = result.context("CSV row")?;
        let plateifu = row.get("plateifu").cloned().unwrap_or_default();
        let plate = row
            .get("plate")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0_u32);
        let ifudesign = row.get("ifudesign").cloned().unwrap_or_default();
        let z = row
            .get("z")
            .and_then(|v| v.parse().ok())
            .unwrap_or(f64::NAN);
        let ba = row
            .get("nsa_elpetro_ba")
            .and_then(|v| v.parse().ok())
            .unwrap_or(f64::NAN);
        let pa_deg = row
            .get("nsa_elpetro_phi")
            .and_then(|v| v.parse().ok())
            .unwrap_or(f64::NAN);
        if plate == 0 || plateifu.is_empty() || ifudesign.is_empty() || !z.is_finite() {
            continue;
        }
        records.push(GalaxyRecord {
            plateifu,
            plate,
            ifudesign,
            z,
            ba,
            pa_deg,
        });
    }
    Ok(records)
}

fn read_maps_channel(path: &Path) -> Result<SlicedMaps> {
    let path_str = path.to_str().context("non-UTF8 path")?;
    let mut fptr =
        FitsFile::open(path_str).map_err(|e| anyhow!("fitsio open {}: {}", path_str, e))?;

    let hdu_vel = fptr
        .hdu("EMLINE_GVEL")
        .map_err(|e| anyhow!("EMLINE_GVEL: {}", e))?;
    let (_n_ch, ny, nx) = match &hdu_vel.info {
        HduInfo::ImageInfo { shape, .. } if shape.len() == 3 => (shape[0], shape[1], shape[2]),
        _ => return Err(anyhow!("unexpected EMLINE_GVEL shape")),
    };
    let n_pixels = ny * nx;
    let ch_start = HA_CHANNEL * n_pixels;
    let ch_end = ch_start + n_pixels;

    let vel: Vec<f32> = hdu_vel
        .read_section(&mut fptr, ch_start, ch_end)
        .map_err(|e| anyhow!("read_section EMLINE_GVEL: {}", e))?;
    let hdu_ivar = fptr
        .hdu("EMLINE_GVEL_IVAR")
        .map_err(|e| anyhow!("EMLINE_GVEL_IVAR: {}", e))?;
    let ivar: Vec<f32> = hdu_ivar
        .read_section(&mut fptr, ch_start, ch_end)
        .map_err(|e| anyhow!("read_section EMLINE_GVEL_IVAR: {}", e))?;
    let hdu_mask = fptr
        .hdu("EMLINE_GVEL_MASK")
        .map_err(|e| anyhow!("EMLINE_GVEL_MASK: {}", e))?;
    let mask: Vec<i32> = hdu_mask
        .read_section(&mut fptr, ch_start, ch_end)
        .map_err(|e| anyhow!("read_section EMLINE_GVEL_MASK: {}", e))?;

    Ok(SlicedMaps {
        vel,
        ivar,
        mask,
        ny,
        nx,
    })
}

fn find_maps_path(galaxy: &GalaxyRecord, maps_cache: &Path) -> Option<PathBuf> {
    let filename = format!(
        "manga-{}-{}-MAPS-SPX-MILESHC-MASTARSSP.fits.gz",
        galaxy.plate, galaxy.ifudesign
    );
    let primary = maps_cache.join(&filename);
    if primary.exists() {
        return Some(primary);
    }

    let parent = maps_cache.parent()?;
    let stem = maps_cache.file_name().and_then(|name| name.to_str())?;

    let mut candidates: Vec<PathBuf> = fs::read_dir(parent)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(stem))
        })
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .map(|dir| dir.join(&filename))
        .find(|path| path.exists())
}

fn inclination_deg_from_ba(ba: f64) -> f64 {
    if (0.0..=1.0).contains(&ba) {
        ba.acos().to_degrees()
    } else {
        f64::NAN
    }
}

fn build_ring_samples(
    galaxy: &GalaxyRecord,
    maps: &SlicedMaps,
    ring_width_spaxels: f64,
    min_cos_theta: f64,
    min_inclination_deg: f64,
    max_inclination_deg: f64,
) -> Result<Vec<Vec<PixelSample>>> {
    let incl_deg = inclination_deg_from_ba(galaxy.ba);
    if !incl_deg.is_finite() || !(min_inclination_deg..=max_inclination_deg).contains(&incl_deg) {
        return Err(anyhow!("inclination outside allowed range"));
    }

    let sin_i = incl_deg.to_radians().sin();
    let cos_i = incl_deg.to_radians().cos();
    if sin_i.abs() < 1e-6 || cos_i.abs() < 1e-6 {
        return Err(anyhow!("degenerate inclination"));
    }

    let pa_rad = galaxy.pa_deg.to_radians();
    let (sin_pa, cos_pa) = pa_rad.sin_cos();
    let cx = (maps.nx as f64 - 1.0) / 2.0;
    let cy = (maps.ny as f64 - 1.0) / 2.0;
    let max_r_spax = cx.min(cy) - 1.0;
    if max_r_spax <= 1.0 {
        return Err(anyhow!("map too small for annular decomposition"));
    }
    let n_rings = ((max_r_spax / ring_width_spaxels).floor() as usize).max(1);
    let kpc_per_spaxel =
        SPAXEL_SIZE_ARCSEC * angular_diameter_distance_kpc(galaxy.z) * std::f64::consts::PI
            / (180.0 * 3600.0);

    let mut rings = vec![Vec::new(); n_rings];
    for iy in 0..maps.ny {
        for ix in 0..maps.nx {
            let idx = iy * maps.nx + ix;
            let vel = maps.vel[idx] as f64;
            let ivar = maps.ivar[idx] as f64;
            if !vel.is_finite() || !ivar.is_finite() || ivar <= 0.0 || maps.mask[idx] != 0 {
                continue;
            }

            let dx = ix as f64 - cx;
            let dy = iy as f64 - cy;
            let xi = -dx * sin_pa + dy * cos_pa;
            let eta = -dx * cos_pa - dy * sin_pa;
            let eta_disk = eta / cos_i;
            let r_spax = (xi * xi + eta_disk * eta_disk).sqrt();
            if r_spax < 0.5 || r_spax >= max_r_spax {
                continue;
            }

            let theta = eta_disk.atan2(xi);
            let cos_theta = theta.cos();
            if cos_theta.abs() < min_cos_theta {
                continue;
            }

            let projection = sin_i * cos_theta;
            if projection.abs() < 1e-6 {
                continue;
            }

            let v_circ = vel / projection;
            let weight = ivar * projection * projection;
            if !v_circ.is_finite() || !weight.is_finite() || weight <= 0.0 {
                continue;
            }

            let ring_idx = (r_spax / ring_width_spaxels).floor() as usize;
            if ring_idx >= n_rings {
                continue;
            }
            rings[ring_idx].push(PixelSample {
                theta,
                v_circ,
                weight,
                r_kpc: r_spax * kpc_per_spaxel,
            });
        }
    }

    Ok(rings)
}

fn summarize_ring(samples: &[PixelSample], ring_idx: usize, m_max: usize) -> Option<RingSummary> {
    if samples.is_empty() {
        return None;
    }
    let sum_w: f64 = samples.iter().map(|sample| sample.weight).sum();
    if !sum_w.is_finite() || sum_w <= 0.0 {
        return None;
    }

    let mean_v_circ_km_s = samples
        .iter()
        .map(|sample| sample.weight * sample.v_circ)
        .sum::<f64>()
        / sum_w;
    let r_kpc = samples
        .iter()
        .map(|sample| sample.weight * sample.r_kpc)
        .sum::<f64>()
        / sum_w;
    let rms_residual_km_s = (samples
        .iter()
        .map(|sample| {
            let residual = sample.v_circ - mean_v_circ_km_s;
            sample.weight * residual * residual
        })
        .sum::<f64>()
        / sum_w)
        .sqrt();

    let mut mode_abs_km_s = Vec::with_capacity(m_max + 1);
    let mut mode_rel_axisymmetric = Vec::with_capacity(m_max + 1);
    let axisymmetric_scale = mean_v_circ_km_s.abs().max(1e-9);
    for mode in 0..=m_max {
        let coeff = samples
            .iter()
            .fold(Complex64::new(0.0, 0.0), |acc, sample| {
                let phase = Complex64::from_polar(1.0, -(mode as f64) * sample.theta);
                acc + phase * (sample.weight * sample.v_circ)
            })
            / sum_w;
        let amplitude = coeff.norm();
        mode_abs_km_s.push(amplitude);
        mode_rel_axisymmetric.push(amplitude / axisymmetric_scale);
    }

    Some(RingSummary {
        ring_idx,
        r_kpc,
        n_pixels: samples.len(),
        mean_v_circ_km_s,
        rms_residual_km_s,
        mode_abs_km_s,
        mode_rel_axisymmetric,
    })
}

fn process_galaxy(galaxy: &GalaxyRecord, args: &Args) -> Result<GalaxyOutcome> {
    let Some(maps_path) = find_maps_path(galaxy, &args.maps_cache) else {
        return Ok(GalaxyOutcome::MissingMaps);
    };
    let maps = read_maps_channel(&maps_path)?;
    let ring_samples = build_ring_samples(
        galaxy,
        &maps,
        args.ring_width_spaxels,
        args.min_cos_theta,
        args.min_inclination_deg,
        args.max_inclination_deg,
    )?;
    let rings: Vec<RingSummary> = ring_samples
        .iter()
        .enumerate()
        .filter_map(|(ring_idx, samples)| {
            if samples.len() < args.min_ring_pixels {
                None
            } else {
                summarize_ring(samples, ring_idx, args.m_max)
            }
        })
        .collect();
    if rings.is_empty() {
        return Ok(GalaxyOutcome::NoValidRings);
    }
    Ok(GalaxyOutcome::Processed(GalaxySummary {
        plateifu: galaxy.plateifu.clone(),
        rings,
    }))
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

fn write_ring_csv(path: &Path, summaries: &[GalaxySummary], m_max: usize) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)
        .with_context(|| format!("create ring CSV {}", path.display()))?;
    let mut header = vec![
        "plateifu".to_string(),
        "ring_idx".to_string(),
        "r_kpc".to_string(),
        "n_pixels".to_string(),
        "mean_v_circ_km_s".to_string(),
        "rms_residual_km_s".to_string(),
    ];
    for mode in 0..=m_max {
        header.push(format!("mode_{mode}_abs_km_s"));
    }
    for mode in 0..=m_max {
        header.push(format!("mode_{mode}_rel_axisymmetric"));
    }
    writer.write_record(&header)?;

    for galaxy in summaries {
        for ring in &galaxy.rings {
            let mut row = vec![
                galaxy.plateifu.clone(),
                ring.ring_idx.to_string(),
                format!("{:.6}", ring.r_kpc),
                ring.n_pixels.to_string(),
                format!("{:.6}", ring.mean_v_circ_km_s),
                format!("{:.6}", ring.rms_residual_km_s),
            ];
            row.extend(
                ring.mode_abs_km_s
                    .iter()
                    .map(|value| format!("{:.6}", value)),
            );
            row.extend(
                ring.mode_rel_axisymmetric
                    .iter()
                    .map(|value| format!("{:.6}", value)),
            );
            writer.write_record(&row)?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_summary_toml(
    path: &Path,
    summaries: &[GalaxySummary],
    counters: &ProcessingCounters,
    args: &Args,
) -> Result<()> {
    let total_rings: usize = summaries.iter().map(|galaxy| galaxy.rings.len()).sum();
    let mut out = String::new();
    out.push_str("[metadata]\n");
    out.push_str("title = \"MaNGA 2D azimuthal ZD Fourier-mode preflight\"\n");
    out.push_str("method = \"Deproject H-alpha IFU velocity maps to disk coordinates and measure annular Fourier modes m=0..m_max on v_circ(theta, r).\"\n");
    out.push_str(&format!("selection = \"{}\"\n", args.selection.display()));
    out.push_str(&format!("maps_cache = \"{}\"\n", args.maps_cache.display()));
    out.push_str(&format!("n_max = {}\n", args.n_max));
    out.push_str(&format!("m_max = {}\n", args.m_max));
    out.push_str(&format!(
        "ring_width_spaxels = {:.3}\n",
        args.ring_width_spaxels
    ));
    out.push_str(&format!("min_ring_pixels = {}\n", args.min_ring_pixels));
    out.push_str(&format!("min_cos_theta = {:.3}\n", args.min_cos_theta));
    out.push_str(&format!(
        "inclination_range_deg = [{:.1}, {:.1}]\n\n",
        args.min_inclination_deg, args.max_inclination_deg
    ));

    out.push_str("[counts]\n");
    out.push_str(&format!(
        "requested_galaxies = {}\n",
        counters.requested_galaxies
    ));
    out.push_str(&format!(
        "processed_galaxies = {}\n",
        counters.processed_galaxies
    ));
    out.push_str(&format!(
        "skipped_missing_maps = {}\n",
        counters.skipped_missing_maps
    ));
    out.push_str(&format!(
        "skipped_bad_inclination = {}\n",
        counters.skipped_bad_inclination
    ));
    out.push_str(&format!(
        "skipped_fits_error = {}\n",
        counters.skipped_fits_error
    ));
    out.push_str(&format!(
        "skipped_no_valid_rings = {}\n",
        counters.skipped_no_valid_rings
    ));
    out.push_str(&format!("total_rings = {}\n\n", total_rings));

    for mode in 0..=args.m_max {
        let mut abs_values = Vec::new();
        let mut rel_values = Vec::new();
        for galaxy in summaries {
            for ring in &galaxy.rings {
                abs_values.push(ring.mode_abs_km_s[mode]);
                rel_values.push(ring.mode_rel_axisymmetric[mode]);
            }
        }
        let abs_mean = if abs_values.is_empty() {
            f64::NAN
        } else {
            abs_values.iter().sum::<f64>() / abs_values.len() as f64
        };
        let rel_mean = if rel_values.is_empty() {
            f64::NAN
        } else {
            rel_values.iter().sum::<f64>() / rel_values.len() as f64
        };
        out.push_str(&format!("[mode_{mode}]\n"));
        out.push_str(&format!("ring_count = {}\n", abs_values.len()));
        out.push_str(&format!("mean_abs_km_s = {:.6}\n", abs_mean));
        out.push_str(&format!("median_abs_km_s = {:.6}\n", median(&abs_values)));
        out.push_str(&format!("mean_rel_axisymmetric = {:.6}\n", rel_mean));
        out.push_str(&format!(
            "median_rel_axisymmetric = {:.6}\n\n",
            median(&rel_values)
        ));
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, out)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("configure Rayon")?;
    }

    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create {}", args.output_dir.display()))?;

    let mut selection = load_selection_csv(&args.selection)?;
    selection.sort_by(|a, b| a.plateifu.cmp(&b.plateifu));
    if args.n_max > 0 && selection.len() > args.n_max {
        selection.truncate(args.n_max);
    }

    let counters_seed = ProcessingCounters {
        requested_galaxies: selection.len(),
        ..ProcessingCounters::default()
    };
    let results: Vec<Result<GalaxyOutcome>> = selection
        .par_iter()
        .map(|galaxy| process_galaxy(galaxy, &args))
        .collect();

    let mut counters = counters_seed;
    let mut summaries = Vec::new();
    for result in results {
        match result {
            Ok(GalaxyOutcome::Processed(summary)) => {
                counters.processed_galaxies += 1;
                summaries.push(summary);
            }
            Ok(GalaxyOutcome::MissingMaps) => {
                counters.skipped_missing_maps += 1;
            }
            Ok(GalaxyOutcome::NoValidRings) => {
                counters.skipped_no_valid_rings += 1;
            }
            Err(err) => {
                let text = err.to_string();
                if text.contains("inclination outside allowed range")
                    || text.contains("degenerate inclination")
                {
                    counters.skipped_bad_inclination += 1;
                } else if text.contains("read_section") || text.contains("fitsio") {
                    counters.skipped_fits_error += 1;
                } else {
                    counters.skipped_no_valid_rings += 1;
                }
            }
        }
    }

    summaries.sort_by(|a, b| a.plateifu.cmp(&b.plateifu));
    let ring_csv = args.output_dir.join("manga_azimuthal_ring_modes.csv");
    let summary_toml = args.output_dir.join("manga_azimuthal_mode_summary.toml");
    write_ring_csv(&ring_csv, &summaries, args.m_max)?;
    write_summary_toml(&summary_toml, &summaries, &counters, &args)?;
    eprintln!(
        "Processed {} galaxies into {} ring rows",
        counters.processed_galaxies,
        summaries
            .iter()
            .map(|galaxy| galaxy.rings.len())
            .sum::<usize>()
    );
    eprintln!("Ring CSV: {}", ring_csv.display());
    eprintln!("Summary: {}", summary_toml.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{PixelSample, summarize_ring};

    fn sample_ring<F: Fn(f64) -> f64>(n: usize, func: F) -> Vec<PixelSample> {
        (0..n)
            .map(|idx| {
                let theta = 2.0 * std::f64::consts::PI * idx as f64 / n as f64;
                PixelSample {
                    theta,
                    v_circ: func(theta),
                    weight: 1.0,
                    r_kpc: 5.0,
                }
            })
            .collect()
    }

    #[test]
    fn constant_ring_is_axisymmetric() {
        let ring = sample_ring(64, |_| 120.0);
        let summary = summarize_ring(&ring, 0, 4).expect("ring summary");
        assert!((summary.mode_abs_km_s[0] - 120.0).abs() < 1e-9);
        for mode in 1..=4 {
            assert!(
                summary.mode_abs_km_s[mode] < 1e-9,
                "mode {mode} should vanish for an axisymmetric ring"
            );
        }
    }

    #[test]
    fn m2_modulation_dominates_m2_coefficient() {
        let ring = sample_ring(128, |theta| 100.0 + 12.0 * (2.0 * theta).cos());
        let summary = summarize_ring(&ring, 0, 4).expect("ring summary");
        assert!(summary.mode_abs_km_s[2] > 5.0);
        assert!(summary.mode_abs_km_s[2] > summary.mode_abs_km_s[1]);
        assert!(summary.mode_abs_km_s[2] > summary.mode_abs_km_s[3]);
        assert!(summary.mode_abs_km_s[2] > summary.mode_abs_km_s[4]);
    }
}
