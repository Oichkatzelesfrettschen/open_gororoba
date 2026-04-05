//! SDO/HMI SHARP FITS IMAGE parser: read vector magnetogram cutouts.
//!
//! Reads HMI SHARP CEA (cylindrical equal-area) FITS images containing
//! the three components of the photospheric magnetic field:
//!   - Bp: toroidal (East-West) component
//!   - Br: radial (line-of-sight proxy in CEA) component
//!   - Bt: poloidal (North-South) component
//!
//! Each FITS file contains a single 2D image HDU. The binary computes
//! spatial statistics and builds a per-pixel CD embedding for magnetic
//! complexity analysis.
//!
//! Usage:
//!   sdo-fits-parse --fits-dir data/external/sdo_aia/sharp_fits/ --harpnum 7115
//!   sdo-fits-parse --fits-file hmi.sharp_cea_720s.7115.Bp.fits --single

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "sdo-fits-parse")]
struct Cli {
    /// Directory containing SHARP FITS cutout images
    #[arg(long)]
    fits_dir: Option<PathBuf>,

    /// Single FITS file to parse
    #[arg(long)]
    fits_file: Option<PathBuf>,

    /// HARPNUM filter (only process files matching this HARP)
    #[arg(long)]
    harpnum: Option<u32>,

    /// Output JSON with per-image statistics
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/sdo_fits_stats.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct FitsImageStats {
    filename: String,
    component: String,
    naxis1: usize,
    naxis2: usize,
    n_pixels: usize,
    n_finite: usize,
    mean: f64,
    std_dev: f64,
    min_val: f64,
    max_val: f64,
    abs_mean: f64,
    rms: f64,
    unsigned_flux_proxy: f64,
    skewness: f64,
    kurtosis: f64,
}

#[derive(Debug, Serialize)]
struct VectorMagnetogramStats {
    harpnum: Option<u32>,
    timestamp: String,
    bp_stats: Option<FitsImageStats>,
    br_stats: Option<FitsImageStats>,
    bt_stats: Option<FitsImageStats>,
    total_unsigned_flux: f64,
    mean_shear_proxy: f64,
    complexity_index: f64,
}

#[derive(Debug, Serialize)]
struct ParseResult {
    n_files: usize,
    n_images: usize,
    image_stats: Vec<FitsImageStats>,
    magnetograms: Vec<VectorMagnetogramStats>,
}

fn parse_fits_image(path: &std::path::Path) -> Result<(Vec<f64>, usize, usize)> {
    use fitsio::FitsFile;

    let mut fptr =
        FitsFile::open(path).with_context(|| format!("Opening FITS {}", path.display()))?;

    // Try primary HDU first, then HDU 1
    let hdu = fptr.primary_hdu().with_context(|| "Reading primary HDU")?;

    let naxis1: usize = hdu.read_key::<i64>(&mut fptr, "NAXIS1").unwrap_or(0) as usize;
    let naxis2: usize = hdu.read_key::<i64>(&mut fptr, "NAXIS2").unwrap_or(0) as usize;

    if naxis1 == 0 || naxis2 == 0 {
        anyhow::bail!("Image dimensions are 0x0");
    }

    // Read image data as f64
    let data: Vec<f64> = hdu
        .read_image(&mut fptr)
        .with_context(|| "Reading image data")?;

    Ok((data, naxis1, naxis2))
}

fn compute_stats(
    data: &[f64],
    naxis1: usize,
    naxis2: usize,
    filename: &str,
    component: &str,
) -> FitsImageStats {
    let finite: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
    let n_finite = finite.len();

    if n_finite == 0 {
        return FitsImageStats {
            filename: filename.to_string(),
            component: component.to_string(),
            naxis1,
            naxis2,
            n_pixels: data.len(),
            n_finite: 0,
            mean: 0.0,
            std_dev: 0.0,
            min_val: 0.0,
            max_val: 0.0,
            abs_mean: 0.0,
            rms: 0.0,
            unsigned_flux_proxy: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
        };
    }

    let n = n_finite as f64;
    let mean = finite.iter().sum::<f64>() / n;
    let variance = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let min_val = finite.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let abs_mean = finite.iter().map(|v| v.abs()).sum::<f64>() / n;
    let rms = (finite.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    let unsigned_flux_proxy = finite.iter().map(|v| v.abs()).sum::<f64>();

    // Higher moments
    let (skewness, kurtosis) = if std_dev > 1e-12 {
        let m3 = finite
            .iter()
            .map(|v| ((v - mean) / std_dev).powi(3))
            .sum::<f64>()
            / n;
        let m4 = finite
            .iter()
            .map(|v| ((v - mean) / std_dev).powi(4))
            .sum::<f64>()
            / n;
        (m3, m4 - 3.0) // excess kurtosis
    } else {
        (0.0, 0.0)
    };

    FitsImageStats {
        filename: filename.to_string(),
        component: component.to_string(),
        naxis1,
        naxis2,
        n_pixels: data.len(),
        n_finite,
        mean,
        std_dev,
        min_val,
        max_val,
        abs_mean,
        rms,
        unsigned_flux_proxy,
        skewness,
        kurtosis,
    }
}

fn detect_component(filename: &str) -> String {
    let lower = filename.to_lowercase();
    if lower.contains(".bp.") || lower.contains("_bp_") || lower.ends_with("_bp.fits") {
        "Bp".to_string()
    } else if lower.contains(".br.") || lower.contains("_br_") || lower.ends_with("_br.fits") {
        "Br".to_string()
    } else if lower.contains(".bt.") || lower.contains("_bt_") || lower.ends_with("_bt.fits") {
        "Bt".to_string()
    } else if lower.contains("magnetogram") || lower.contains("_b_") {
        "B_total".to_string()
    } else {
        "unknown".to_string()
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== SDO/HMI SHARP FITS Parser ===");

    let mut files: Vec<PathBuf> = Vec::new();

    if let Some(fits_file) = &cli.fits_file {
        files.push(fits_file.clone());
    }

    if let Some(fits_dir) = &cli.fits_dir {
        let entries = std::fs::read_dir(fits_dir)
            .with_context(|| format!("Reading {}", fits_dir.display()))?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("fits") {
                // Filter by HARPNUM if specified
                if let Some(harp) = cli.harpnum {
                    let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if !fname.contains(&format!(".{}.", harp))
                        && !fname.contains(&format!("_{}_", harp))
                    {
                        continue;
                    }
                }
                files.push(path);
            }
        }
    }

    files.sort();
    println!("  Found {} FITS files", files.len());

    let mut all_stats = Vec::new();

    for path in &files {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        let component = detect_component(filename);
        print!("  {} ({})... ", filename, component);

        match parse_fits_image(path) {
            Ok((data, naxis1, naxis2)) => {
                let stats = compute_stats(&data, naxis1, naxis2, filename, &component);
                println!(
                    "{}x{}, mean={:.1} G, rms={:.1} G, unsigned_flux={:.2e}",
                    naxis1, naxis2, stats.mean, stats.rms, stats.unsigned_flux_proxy
                );
                all_stats.push(stats);
            }
            Err(e) => {
                println!("FAILED: {}", e);
            }
        }
    }

    // Group into vector magnetograms (Bp + Br + Bt sets)
    // For now, just output individual statistics
    println!("\n=== Summary ===");
    println!("  {} images parsed successfully", all_stats.len());

    if !all_stats.is_empty() {
        println!("\n  Per-component statistics:");
        for comp in ["Bp", "Br", "Bt", "B_total", "unknown"] {
            let comp_stats: Vec<&FitsImageStats> =
                all_stats.iter().filter(|s| s.component == comp).collect();
            if !comp_stats.is_empty() {
                let mean_rms: f64 =
                    comp_stats.iter().map(|s| s.rms).sum::<f64>() / comp_stats.len() as f64;
                let mean_kurt: f64 =
                    comp_stats.iter().map(|s| s.kurtosis).sum::<f64>() / comp_stats.len() as f64;
                println!(
                    "    {}: {} images, mean_rms={:.1} G, mean_kurtosis={:.2}",
                    comp,
                    comp_stats.len(),
                    mean_rms,
                    mean_kurt
                );
            }
        }
    }

    let result = ParseResult {
        n_files: files.len(),
        n_images: all_stats.len(),
        image_stats: all_stats,
        magnetograms: Vec::new(), // TODO: group by timestamp into vector sets
    };

    let json = serde_json::to_string_pretty(&result)?;
    if let Some(parent) = cli.out_json.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out_json, &json)?;
    println!("\nResults -> {}", cli.out_json.display());

    Ok(())
}
