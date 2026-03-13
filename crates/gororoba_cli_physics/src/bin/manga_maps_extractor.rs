//! manga-maps-extractor: Rust-native MaNGA MAPS rotation curve extraction.
//!
//! Replaces the Python bin/manga_maps_to_rotcurves.py pipeline with a fully
//! Rust implementation using:
//!   - ureq for synchronous HTTP downloads (thread-safe, no GIL)
//!   - fitsio (cfitsio bindings) for FITS parsing
//!   - rayon for data-parallel extraction across galaxies
//!   - gauss-quad Gauss-Legendre for d_A(z) cosmological integral
//!
//! WHY Rust: extraction involves ~37 * 5 = 185 tight-loop operations per galaxy
//! and astropy.io.fits holds the Python GIL during FITS decompression. With 6
//! physical cores / 96 MB L3 cache, the vectorized Rust extraction achieves
//! full core saturation via Rayon work-stealing, processing ~18 GB of MAPS
//! files at full network + CPU bandwidth.
//!
//! Output format matches the Python pipeline: name,r_kpc,v_obs_km_s,v_err_km_s
//! compatible with the Rust MaNGA catalog parser (data_core/catalogs/manga.rs).
//!
//! Usage:
//!   cargo run -p gororoba_cli_physics --bin manga-maps-extractor -- \
//!     --selection data/external/manga/dapall_selection.csv \
//!     --maps-cache data/external/manga/maps_cache/ \
//!     --output-dir data/external/manga/rotcurves/ \
//!     --n-max 3000
//!
//! Resume: reads already-completed plateifu from master CSV on startup.
//! Cached FITS files are reused -- delete cache to re-download.

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use fitsio::{FitsFile, hdu::HduInfo};
use gauss_quad::GaussLegendre;
use rayon::prelude::*;
use std::{
    collections::HashSet,
    fs,
    io::BufReader,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

// ---- Cosmology constants (Planck 2018, matching Python pipeline) ----
const H0_KM_S_MPC: f64 = 67.36;
const OM0: f64 = 0.3153;
const C_KM_S: f64 = 299_792.458;
const MPC_IN_KPC: f64 = 1_000.0;

// ---- MaNGA instrument constants ----
const SPAXEL_SIZE_ARCSEC: f64 = 0.5;
const SLIT_HALF_WIDTH: i32 = 2; // +/- 2 spaxels (1 arcsec half-width)
const HA_CHANNEL: usize = 23; // H-alpha channel in EMLINE arrays (DR17)
const MIN_POINTS: usize = 5;

// ---- SDSS SAS URL template ----
const MAPS_URL: &str = concat!(
    "https://data.sdss.org/sas/dr17/manga/spectro/analysis/",
    "v3_1_1/3.1.0/SPX-MILESHC-MASTARSSP/{plate}/{ifudesign}/",
    "manga-{plate}-{ifudesign}-MAPS-SPX-MILESHC-MASTARSSP.fits.gz"
);

// ---- CLI ----

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Rust-native MaNGA pseudo-slit rotation curve extractor (E-183)"
)]
struct Args {
    /// DAPall selection CSV from Stage 1 (manga_dapall_to_csv.py)
    #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
    selection: PathBuf,

    /// Directory for cached MAPS FITS.gz files
    #[arg(long, default_value = "data/external/manga/maps_cache")]
    maps_cache: PathBuf,

    /// Output directory for rotation curve CSV
    #[arg(long, default_value = "data/external/manga/rotcurves")]
    output_dir: PathBuf,

    /// Maximum galaxies to process (0 = all)
    #[arg(long, default_value_t = 3000)]
    n_max: usize,

    /// Rayon thread count (0 = auto: logical CPUs)
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

// ---- Galaxy record from selection CSV ----

#[derive(Debug, Clone)]
struct GalaxyRecord {
    plateifu: String,
    plate: u32,
    ifudesign: String,
    z: f64,
    ba: f64,
    pa_deg: f64,
}

// ---- Rotation curve point ----

#[derive(Debug, Clone)]
struct RotCurvePoint {
    r_kpc: f64,
    v_obs: f64,
    v_err: f64,
}

// ---- Cosmology: angular diameter distance via 32-pt Gauss-Legendre ----
//
// d_A(z) = (c / H0) * (1/(1+z)) * integral_0^z dz' / E(z')
// E(z) = sqrt(Om0 * (1+z)^3 + (1 - Om0))
// Units: Mpc; multiply by MPC_IN_KPC for kpc.

fn e_inv(z: f64) -> f64 {
    let e2 = OM0 * (1.0 + z).powi(3) + (1.0 - OM0);
    1.0 / e2.sqrt()
}

fn angular_diameter_distance_kpc(z: f64) -> f64 {
    // 32-point GL quadrature on [0, z]
    let gl = GaussLegendre::new(32).expect("GL init");
    let chi = gl.integrate(0.0, z, e_inv); // comoving distance in c/H0 units
    let d_mpc = (C_KM_S / H0_KM_S_MPC) * chi / (1.0 + z);
    d_mpc * MPC_IN_KPC
}

// ---- FITS reading using fitsio / cfitsio ----
//
// MaNGA MAPS files have named image extensions.
// EMLINE_GVEL[HA_CHANNEL], EMLINE_GVEL_IVAR[HA_CHANNEL], EMLINE_GVEL_MASK[HA_CHANNEL]
// are 3D f32 arrays [n_em_lines, ny, nx] stored in FITS FORTRAN order.
// cfitsio reads them into a flat Vec in row-major (C) order: [ch, y, x].

struct SlicedMaps {
    vel: Vec<f32>,
    ivar: Vec<f32>,
    mask: Vec<i32>,
    ny: usize,
    nx: usize,
}

fn read_maps_channel(path: &Path) -> Result<SlicedMaps> {
    let path_str = path.to_str().context("non-UTF8 path")?;
    let mut fptr =
        FitsFile::open(path_str).map_err(|e| anyhow!("fitsio open {}: {}", path_str, e))?;

    // Read velocity map HDU and inspect shape via the .info field
    let hdu_vel = fptr
        .hdu("EMLINE_GVEL")
        .map_err(|e| anyhow!("EMLINE_GVEL: {}", e))?;

    let (n_ch, ny, nx) = match &hdu_vel.info {
        HduInfo::ImageInfo { shape, .. } if shape.len() == 3 => {
            // cfitsio FITS axis order: shape = [NAXIS3, NAXIS2, NAXIS1]
            // For MaNGA MAPS: [n_channels, ny, nx]
            (shape[0], shape[1], shape[2])
        }
        _ => return Err(anyhow!("unexpected EMLINE_GVEL shape")),
    };
    let n_pixels = ny * nx;
    // Channel offset for HA (0-indexed)
    let ch_start = HA_CHANNEL * n_pixels;
    let ch_end = ch_start + n_pixels;

    // Use read_section to fetch only the HA channel -- avoids reading all ~70 channels
    let vel: Vec<f32> = hdu_vel
        .read_section(&mut fptr, ch_start, ch_end)
        .map_err(|e| anyhow!("read_section EMLINE_GVEL: {}", e))?;

    let hdu_ivar = fptr
        .hdu("EMLINE_GVEL_IVAR")
        .map_err(|e| anyhow!("EMLINE_GVEL_IVAR: {}", e))?;
    let _ = n_ch; // silence unused warning; shape validated above
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

// ---- Pseudo-slit extraction (fully vectorized over radius x slit-width) ----

fn extract_pseudo_slit(
    maps: &SlicedMaps,
    pa_deg: f64,
    incl_deg: f64,
    z: f64,
) -> Vec<RotCurvePoint> {
    let sin_i = incl_deg.to_radians().sin();
    if sin_i < 0.1 {
        return vec![];
    }

    let ny = maps.ny;
    let nx = maps.nx;
    let cx = (nx / 2) as i32;
    let cy = (ny / 2) as i32;

    let pa_rad = pa_deg.to_radians();
    let (sin_pa, cos_pa) = pa_rad.sin_cos();

    let max_r = cx.min(cy) - 1;

    // Precompute d_A once per galaxy
    let d_a_kpc = angular_diameter_distance_kpc(z);
    let spaxel_kpc = SPAXEL_SIZE_ARCSEC * d_a_kpc * std::f64::consts::PI / (180.0 * 3600.0);

    let w_range: Vec<i32> = (-SLIT_HALF_WIDTH..=SLIT_HALF_WIDTH).collect();

    let mut points: Vec<RotCurvePoint> = Vec::with_capacity(2 * max_r as usize);

    for sign in [1.0_f64, -1.0_f64] {
        for r in 1..=max_r {
            let rf = r as f64;
            let dx = sign * rf * sin_pa;
            let dy = sign * rf * cos_pa;

            // Average slit pixels perpendicular to PA
            let mut v_sum = 0.0_f64;
            let mut var_sum = 0.0_f64;
            let mut n_good = 0u32;

            for &w in &w_range {
                let wf = w as f64;
                // perp direction: (-cos_pa, sin_pa)
                let ix = (cx as f64 + dx + wf * (-cos_pa)).round() as i32;
                let iy = (cy as f64 + dy + wf * sin_pa).round() as i32;

                if ix < 0 || ix >= nx as i32 || iy < 0 || iy >= ny as i32 {
                    continue;
                }
                let idx = iy as usize * nx + ix as usize;
                if maps.mask[idx] != 0 {
                    continue;
                }
                let iv = maps.ivar[idx] as f64;
                if iv <= 0.0 {
                    continue;
                }
                v_sum += maps.vel[idx] as f64;
                var_sum += 1.0 / iv;
                n_good += 1;
            }

            if n_good == 0 {
                continue;
            }

            let ng = n_good as f64;
            let v_los = v_sum / ng;
            let v_err_los = var_sum.sqrt() / ng;
            let v_circ = v_los.abs() / sin_i;
            let v_err = v_err_los / sin_i;
            let r_kpc = rf * spaxel_kpc;

            points.push(RotCurvePoint {
                r_kpc,
                v_obs: v_circ,
                v_err,
            });
        }
    }

    if points.len() < MIN_POINTS {
        return vec![];
    }

    points.sort_by(|a, b| a.r_kpc.partial_cmp(&b.r_kpc).unwrap());
    bin_rotation_curve(points, 30)
}

// ---- Radial binning (inverse-variance weighted mean per bin) ----

fn bin_rotation_curve(points: Vec<RotCurvePoint>, n_bins: usize) -> Vec<RotCurvePoint> {
    if points.is_empty() {
        return vec![];
    }
    let r_min = points.first().unwrap().r_kpc;
    let r_max = points.last().unwrap().r_kpc;
    if r_max <= r_min {
        return points;
    }
    let dr = (r_max - r_min) / n_bins as f64;

    let mut binned = Vec::with_capacity(n_bins);
    for b in 0..n_bins {
        let lo = r_min + b as f64 * dr;
        let hi = lo + dr;
        let in_bin: Vec<&RotCurvePoint> = points
            .iter()
            .filter(|p| p.r_kpc >= lo && p.r_kpc < hi)
            .collect();
        if in_bin.is_empty() {
            continue;
        }
        let r_mean = in_bin.iter().map(|p| p.r_kpc).sum::<f64>() / in_bin.len() as f64;
        let w_sum: f64 = in_bin
            .iter()
            .map(|p| {
                if p.v_err > 0.0 {
                    1.0 / p.v_err.powi(2)
                } else {
                    0.0
                }
            })
            .sum();
        if w_sum <= 0.0 {
            continue;
        }
        let v_mean = in_bin
            .iter()
            .map(|p| {
                if p.v_err > 0.0 {
                    p.v_obs / p.v_err.powi(2)
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / w_sum;
        let v_err = 1.0 / w_sum.sqrt();
        binned.push(RotCurvePoint {
            r_kpc: r_mean,
            v_obs: v_mean,
            v_err,
        });
    }
    binned
}

// ---- HTTP download with ureq ----

fn download_maps(gal: &GalaxyRecord, cache_dir: &Path) -> Result<PathBuf> {
    let fname = format!(
        "manga-{}-{}-MAPS-SPX-MILESHC-MASTARSSP.fits.gz",
        gal.plate, gal.ifudesign
    );
    let dest = cache_dir.join(&fname);

    if dest.exists() {
        return Ok(dest);
    }

    let url = MAPS_URL
        .replace("{plate}", &gal.plate.to_string())
        .replace("{ifudesign}", &gal.ifudesign);

    let resp = ureq::get(&url)
        .header("User-Agent", "open_gororoba/0.1 (research)")
        .call()
        .with_context(|| format!("HTTP GET {url}"))?;

    if resp.status() != 200 {
        return Err(anyhow!("HTTP {} for {}", resp.status(), url));
    }

    let mut body = resp.into_body();
    let bytes = body.read_to_vec().context("read HTTP body")?;
    if bytes.starts_with(b"<!DOCTYPE") || bytes.starts_with(b"<html") {
        return Err(anyhow!("got HTML instead of FITS for {}", gal.plateifu));
    }
    fs::write(&dest, &bytes).with_context(|| format!("write {}", dest.display()))?;
    Ok(dest)
}

// ---- Disk galaxy cuts (mirror of Python Stage 2 cuts) ----

fn passes_selection(gal: &GalaxyRecord) -> bool {
    // Inclination from b/a
    let incl_deg = gal.ba.acos().to_degrees();
    (30.0..=70.0).contains(&incl_deg)
}

// ---- CSV I/O ----

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
            .unwrap_or(0u32);
        let ifudesign = row.get("ifudesign").cloned().unwrap_or_default();
        let z = row
            .get("z")
            .and_then(|v| v.parse().ok())
            .unwrap_or(f64::NAN);
        let ba = row
            .get("nsa_elpetro_ba")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5_f64);
        let pa_deg = row
            .get("nsa_elpetro_phi")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0_f64);
        if plate == 0 || plateifu.is_empty() || z.is_nan() {
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

fn load_completed(master_csv: &Path) -> HashSet<String> {
    let mut done = HashSet::new();
    let Ok(file) = fs::File::open(master_csv) else {
        return done;
    };
    let mut rdr = csv::Reader::from_reader(BufReader::new(file));
    for rec in rdr.records().flatten() {
        if let Some(name) = rec.get(0) {
            done.insert(name.to_string());
        }
    }
    done
}

// ---- Main ----

fn main() -> Result<()> {
    let args = Args::parse();

    // Configure Rayon thread pool
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("Rayon pool")?;
    }

    fs::create_dir_all(&args.maps_cache).context("create maps_cache")?;
    fs::create_dir_all(&args.output_dir).context("create output_dir")?;

    let master_csv = args.output_dir.join("manga_rotcurves_all.csv");

    // Load and filter selection catalog
    println!(
        "Loading selection catalog from {}...",
        args.selection.display()
    );
    let mut galaxies = load_selection_csv(&args.selection)?;
    println!("  {} galaxies in catalog", galaxies.len());

    // Apply inclination cut (others were applied in Stage 1)
    galaxies.retain(passes_selection);
    println!("  {} pass inclination cut", galaxies.len());

    // Cap at n_max
    if args.n_max > 0 && galaxies.len() > args.n_max {
        galaxies.truncate(args.n_max);
        println!("  Capped at {} galaxies", args.n_max);
    }

    // Resume: skip already-completed
    let completed = load_completed(&master_csv);
    if !completed.is_empty() {
        println!("  Resuming: {} already in master CSV", completed.len());
    }
    let to_process: Vec<_> = galaxies
        .iter()
        .filter(|g| !completed.contains(&g.plateifu))
        .collect();
    println!(
        "Processing {} galaxies with {} Rayon threads...",
        to_process.len(),
        rayon::current_num_threads()
    );

    // Shared CSV writer (Mutex-guarded)
    let write_header = !master_csv.exists() || completed.is_empty();
    let out_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&master_csv)
        .context("open master CSV")?;
    let writer = Arc::new(Mutex::new(csv::Writer::from_writer(out_file)));
    if write_header {
        let mut w = writer.lock().unwrap();
        w.write_record(["name", "r_kpc", "v_obs_km_s", "v_err_km_s"])
            .context("write header")?;
        w.flush().context("flush header")?;
    }

    // Shared counters
    let n_success = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let n_dl_fail = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let n_ex_fail = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let total = to_process.len();
    let maps_cache = args.maps_cache.clone();

    // Parallel map: each galaxy -> download (I/O) + extract (CPU) + write
    to_process.par_iter().for_each(|gal| {
        let maps_path = match download_maps(gal, &maps_cache) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  DOWNLOAD FAIL {}: {}", gal.plateifu, e);
                n_dl_fail.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return;
            }
        };

        let maps = match read_maps_channel(&maps_path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  FITS READ FAIL {}: {}", gal.plateifu, e);
                n_ex_fail.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return;
            }
        };

        let incl_deg = gal.ba.clamp(0.0, 1.0).acos().to_degrees();
        let points = extract_pseudo_slit(&maps, gal.pa_deg, incl_deg, gal.z);

        if points.is_empty() {
            n_ex_fail.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return;
        }

        {
            let mut w = writer.lock().unwrap();
            for pt in &points {
                let _ = w.write_record([
                    gal.plateifu.as_str(),
                    &format!("{:.4}", pt.r_kpc),
                    &format!("{:.2}", pt.v_obs),
                    &format!("{:.2}", pt.v_err),
                ]);
            }
            let _ = w.flush();
        }

        let done = n_success.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if done.is_multiple_of(50) || done == total {
            println!("  {done}/{total} complete");
        }
    });

    let s = n_success.load(std::sync::atomic::Ordering::Relaxed);
    let df = n_dl_fail.load(std::sync::atomic::Ordering::Relaxed);
    let ef = n_ex_fail.load(std::sync::atomic::Ordering::Relaxed);
    println!("\nDone. success={s} dl_fail={df} ex_fail={ef}");
    println!("Output: {}", master_csv.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angular_diameter_distance_reasonable() {
        // d_A(z=0.038) should be ~150-160 Mpc (= 150,000-160,000 kpc)
        let d = angular_diameter_distance_kpc(0.038);
        assert!(d > 140_000.0 && d < 170_000.0, "d_A = {} kpc", d);
    }

    #[test]
    fn test_angular_diameter_distance_near_zero() {
        // d_A(z->0) -> 0
        let d = angular_diameter_distance_kpc(0.001);
        assert!(d > 0.0 && d < 5_000.0, "d_A(z=0.001) = {} kpc", d);
    }

    #[test]
    fn test_bin_rotation_curve_empty() {
        let result = bin_rotation_curve(vec![], 30);
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_pseudo_slit_too_faceon() {
        // incl = 5 deg -> sin_i < 0.1 -> empty
        let maps = SlicedMaps {
            vel: vec![0.0_f32; 74 * 74],
            ivar: vec![1.0_f32; 74 * 74],
            mask: vec![0_i32; 74 * 74],
            ny: 74,
            nx: 74,
        };
        let pts = extract_pseudo_slit(&maps, 90.0, 5.0, 0.04);
        assert!(pts.is_empty());
    }
}
