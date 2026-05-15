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

#[derive(Debug, Clone, Serialize)]
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

/// Extract the SHARP CEA timestamp and HARPNUM from a filename.
///
/// SHARP filename convention (JSOC):
///   `hmi.sharp_cea_720s.{HARPNUM}.{YYYYMMDD_HHMMSS_TAI}.{COMPONENT}.fits`
///
/// e.g. `hmi.sharp_cea_720s.7115.20170714_103600_TAI.Bp.fits` yields
/// HARPNUM=7115 and timestamp="20170714_103600_TAI". Either field is
/// `None` if the filename does not match the SHARP convention.
fn extract_sharp_timestamp_and_harp(filename: &str) -> (Option<String>, Option<u32>) {
    // The TAI marker is the most reliable anchor; split around it.
    let tai_idx = filename.find("_TAI");
    let timestamp = tai_idx.and_then(|idx| {
        // Walk backward 15 chars from "_TAI" to grab "YYYYMMDD_HHMMSS".
        if idx < 15 {
            return None;
        }
        let ts_start = idx - 15;
        let ts = &filename[ts_start..idx + 4];
        // Validate: 8 digits, underscore, 6 digits, "_TAI"
        let bytes = ts.as_bytes();
        if bytes.len() == 19
            && bytes[..8].iter().all(|b| b.is_ascii_digit())
            && bytes[8] == b'_'
            && bytes[9..15].iter().all(|b| b.is_ascii_digit())
            && &bytes[15..] == b"_TAI"
        {
            Some(ts.to_string())
        } else {
            None
        }
    });

    // HARPNUM lives between the second and third dot from the start of
    // the SHARP filename (sharp_cea_720s.{HARPNUM}.{TIMESTAMP}...).
    let mut parts = filename.split('.');
    let harpnum = parts
        .find(|p| *p == "sharp_cea_720s")
        .and_then(|_| parts.next())
        .and_then(|h| h.parse::<u32>().ok());

    (timestamp, harpnum)
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

    // Per-component summary (groups individual stats by component label).
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

    // Group into vector magnetograms (Bp + Br + Bt sets sharing the
    // same SHARP T_REC timestamp and HARPNUM). The grouping key is
    // (HARPNUM, timestamp); each group with at least one of the three
    // components becomes a VectorMagnetogramStats row. Total unsigned
    // flux is the sum of per-component unsigned_flux_proxy across the
    // available components; mean_shear_proxy and complexity_index use
    // the Br kurtosis as a smoothness proxy when Br is present.
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<(Option<u32>, Option<String>), Vec<FitsImageStats>> = BTreeMap::new();
    for s in &all_stats {
        let (ts, harp) = extract_sharp_timestamp_and_harp(&s.filename);
        groups.entry((harp, ts)).or_default().push(s.clone());
    }
    let mut magnetograms: Vec<VectorMagnetogramStats> = Vec::with_capacity(groups.len());
    for ((harp, ts), comps) in groups {
        let bp = comps.iter().find(|s| s.component == "Bp").cloned();
        let br = comps.iter().find(|s| s.component == "Br").cloned();
        let bt = comps.iter().find(|s| s.component == "Bt").cloned();
        let total_unsigned_flux = bp.as_ref().map(|s| s.unsigned_flux_proxy).unwrap_or(0.0)
            + br.as_ref().map(|s| s.unsigned_flux_proxy).unwrap_or(0.0)
            + bt.as_ref().map(|s| s.unsigned_flux_proxy).unwrap_or(0.0);
        let mean_shear_proxy = match (bp.as_ref(), bt.as_ref()) {
            (Some(p), Some(t)) => (p.rms - t.rms).abs(),
            _ => 0.0,
        };
        let complexity_index = br.as_ref().map(|s| s.kurtosis).unwrap_or(0.0);
        magnetograms.push(VectorMagnetogramStats {
            harpnum: harp,
            timestamp: ts.unwrap_or_else(|| "unknown".to_string()),
            bp_stats: bp,
            br_stats: br,
            bt_stats: bt,
            total_unsigned_flux,
            mean_shear_proxy,
            complexity_index,
        });
    }
    println!(
        "  {} vector magnetograms grouped (harpnum + SHARP timestamp)",
        magnetograms.len()
    );

    let result = ParseResult {
        n_files: files.len(),
        n_images: all_stats.len(),
        image_stats: all_stats.clone(),
        magnetograms,
    };

    let json = serde_json::to_string_pretty(&result)?;
    if let Some(parent) = cli.out_json.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out_json, &json)?;
    println!("\nResults -> {}", cli.out_json.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharp_timestamp_and_harp_canonical_filename() {
        let f = "hmi.sharp_cea_720s.7115.20170714_103600_TAI.Bp.fits";
        let (ts, harp) = extract_sharp_timestamp_and_harp(f);
        assert_eq!(ts.as_deref(), Some("20170714_103600_TAI"));
        assert_eq!(harp, Some(7115));
    }

    #[test]
    fn sharp_timestamp_handles_each_component_label() {
        for comp in ["Bp", "Br", "Bt"] {
            let f = format!("hmi.sharp_cea_720s.42.20200101_000000_TAI.{}.fits", comp);
            let (ts, harp) = extract_sharp_timestamp_and_harp(&f);
            assert_eq!(ts.as_deref(), Some("20200101_000000_TAI"));
            assert_eq!(harp, Some(42));
        }
    }

    #[test]
    fn sharp_timestamp_returns_none_for_non_sharp_filename() {
        let (ts, harp) = extract_sharp_timestamp_and_harp("aia_lev1_193a_2025_03_28.fits");
        assert_eq!(ts, None);
        assert_eq!(harp, None);
    }

    #[test]
    fn sharp_timestamp_returns_none_on_malformed_tai_block() {
        // _TAI present but digits before it are wrong shape.
        let (ts, _) = extract_sharp_timestamp_and_harp("foo.bar_TAI.Bp.fits");
        assert_eq!(ts, None);
    }
}
