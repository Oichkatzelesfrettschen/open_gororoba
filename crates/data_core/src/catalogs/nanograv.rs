//! NANOGrav 15-year Gravitational Wave Background dataset.
//!
//! NANOGrav (North American Nanohertz Observatory for Gravitational Waves)
//! detected a gravitational wave background in their 15-year dataset.
//! This module provides the free spectrum (per-frequency bin) strain estimates.
//!
//! The KDE data (Zenodo 10344086) stores full posterior probability densities
//! for 30 frequency bins via kernel density estimation. We extract point
//! estimates (median + 90% credible interval) in pure Rust by reading the
//! numpy arrays from the ZIP archive.
//!
//! Source: Zenodo, https://doi.org/10.5281/zenodo.10344086
//! Reference: Agazie et al. (2023), ApJL 951, L8
//! KDE method: Lamb, Taylor & van Haasteren (2023), PhysRevD 108, 103019

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_file, validate_not_html};
use std::fs;
use std::io::Read as IoRead;
use std::path::{Path, PathBuf};

/// A single frequency bin from the NANOGrav free spectrum.
#[derive(Debug, Clone)]
pub struct FreeSpectrumPoint {
    /// Frequency (Hz), typically in nHz range.
    pub frequency: f64,
    /// Median log10 of characteristic strain power (log10(rho)).
    pub log10_rho: f64,
    /// Lower 5th percentile of log10(rho).
    pub log10_rho_lo: f64,
    /// Upper 95th percentile of log10(rho).
    pub log10_rho_hi: f64,
}

/// NANOGrav 15yr HD-correlated free spectrum, 30 frequency bins.
///
/// Extracted from Zenodo record 10344086 (KDE representation v1.1.0),
/// HD-correlated analysis (`30f_fs{hd}_ceffyl`).
/// Median and 5th/95th percentile CDF values from the KDE posterior.
///
/// Frequencies are f_k = k / T_obs where T_obs ~ 16.03 years.
pub mod bestfit {
    use super::FreeSpectrumPoint;

    /// Number of frequency bins in the published free spectrum.
    pub const N_BINS: usize = 30;

    /// HD-correlated free spectrum point estimates.
    /// Columns: frequency (Hz), median log10(rho), 5th pctl, 95th pctl.
    pub const HD_FREE_SPECTRUM: [FreeSpectrumPoint; N_BINS] = [
        FreeSpectrumPoint { frequency: 1.9768264576e-09, log10_rho: -3.824356, log10_rho_lo: -12.239962, log10_rho_hi: -1.192436 },
        FreeSpectrumPoint { frequency: 3.9536529151e-09, log10_rho: -8.314518, log10_rho_lo: -14.419953, log10_rho_hi: -1.563951 },
        FreeSpectrumPoint { frequency: 5.9304793727e-09, log10_rho: -5.686763, log10_rho_lo: -14.198086, log10_rho_hi: -1.378676 },
        FreeSpectrumPoint { frequency: 7.9073058303e-09, log10_rho: -4.324256, log10_rho_lo: -12.283386, log10_rho_hi: -1.242426 },
        FreeSpectrumPoint { frequency: 9.8841322878e-09, log10_rho: -4.191523, log10_rho_lo: -10.567086, log10_rho_hi: -1.229152 },
        FreeSpectrumPoint { frequency: 1.1860958745e-08, log10_rho: -4.139625, log10_rho_lo: -9.835739,  log10_rho_hi: -1.223963 },
        FreeSpectrumPoint { frequency: 1.3837785203e-08, log10_rho: -4.119372, log10_rho_lo: -9.763380,  log10_rho_hi: -1.221937 },
        FreeSpectrumPoint { frequency: 1.5814611661e-08, log10_rho: -4.455705, log10_rho_lo: -12.469490, log10_rho_hi: -1.255570 },
        FreeSpectrumPoint { frequency: 1.7791438118e-08, log10_rho: -4.284785, log10_rho_lo: -9.058763,  log10_rho_hi: -1.238479 },
        FreeSpectrumPoint { frequency: 1.9768264576e-08, log10_rho: -4.197701, log10_rho_lo: -9.572466,  log10_rho_hi: -1.229770 },
        FreeSpectrumPoint { frequency: 2.1745091033e-08, log10_rho: -4.256292, log10_rho_lo: -9.804122,  log10_rho_hi: -1.235629 },
        FreeSpectrumPoint { frequency: 2.3721917491e-08, log10_rho: -4.297346, log10_rho_lo: -9.526431,  log10_rho_hi: -1.239735 },
        FreeSpectrumPoint { frequency: 2.5698743948e-08, log10_rho: -4.366242, log10_rho_lo: -8.755714,  log10_rho_hi: -1.246624 },
        FreeSpectrumPoint { frequency: 2.7675570406e-08, log10_rho: -4.330105, log10_rho_lo: -8.838375,  log10_rho_hi: -1.243011 },
        FreeSpectrumPoint { frequency: 2.9652396864e-08, log10_rho: -4.253942, log10_rho_lo: -9.057492,  log10_rho_hi: -1.235394 },
        FreeSpectrumPoint { frequency: 3.1629223321e-08, log10_rho: -3.355449, log10_rho_lo: -11.156964, log10_rho_hi: -1.145545 },
        FreeSpectrumPoint { frequency: 3.3606049779e-08, log10_rho: -4.019221, log10_rho_lo: -10.332758, log10_rho_hi: -1.211922 },
        FreeSpectrumPoint { frequency: 3.5582876236e-08, log10_rho: -4.218475, log10_rho_lo: -9.979593,  log10_rho_hi: -1.231847 },
        FreeSpectrumPoint { frequency: 3.7559702694e-08, log10_rho: -4.283774, log10_rho_lo: -9.594341,  log10_rho_hi: -1.238377 },
        FreeSpectrumPoint { frequency: 3.9536529151e-08, log10_rho: -4.354106, log10_rho_lo: -9.628553,  log10_rho_hi: -1.245411 },
        FreeSpectrumPoint { frequency: 4.1513355609e-08, log10_rho: -4.358596, log10_rho_lo: -8.971711,  log10_rho_hi: -1.245860 },
        FreeSpectrumPoint { frequency: 4.3490182066e-08, log10_rho: -4.326085, log10_rho_lo: -9.048752,  log10_rho_hi: -1.242608 },
        FreeSpectrumPoint { frequency: 4.5467008524e-08, log10_rho: -4.281646, log10_rho_lo: -9.260830,  log10_rho_hi: -1.238165 },
        FreeSpectrumPoint { frequency: 4.7443834982e-08, log10_rho: -4.320090, log10_rho_lo: -9.515511,  log10_rho_hi: -1.242009 },
        FreeSpectrumPoint { frequency: 4.9420661439e-08, log10_rho: -4.288476, log10_rho_lo: -10.414934, log10_rho_hi: -1.238848 },
        FreeSpectrumPoint { frequency: 5.1397487897e-08, log10_rho: -4.339775, log10_rho_lo: -10.295456, log10_rho_hi: -1.243978 },
        FreeSpectrumPoint { frequency: 5.3374314354e-08, log10_rho: -4.454512, log10_rho_lo: -8.581017,  log10_rho_hi: -1.255451 },
        FreeSpectrumPoint { frequency: 5.5351140812e-08, log10_rho: -4.357294, log10_rho_lo: -8.794067,  log10_rho_hi: -1.245729 },
        FreeSpectrumPoint { frequency: 5.7327967269e-08, log10_rho: -4.424931, log10_rho_lo: -8.688157,  log10_rho_hi: -1.252493 },
        FreeSpectrumPoint { frequency: 5.9304793727e-08, log10_rho: -4.274741, log10_rho_lo: -9.584661,  log10_rho_hi: -1.237474 },
    ];
}

/// Parse NANOGrav free spectrum CSV (either hand-written or extracted from KDE).
///
/// Expected columns: frequency, log10_rho, log10_rho_lo, log10_rho_hi
pub fn parse_nanograv_free_spectrum(path: &Path) -> Result<Vec<FreeSpectrumPoint>, FetchError> {
    let content = fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;

    let mut points = Vec::new();
    let mut header_seen = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if !header_seen {
            header_seen = true;
            continue;
        }

        let fields: Vec<&str> = trimmed.split(',').collect();
        if fields.len() < 4 {
            continue;
        }

        let parse = |s: &str| -> f64 {
            s.trim().parse::<f64>().unwrap_or(f64::NAN)
        };

        points.push(FreeSpectrumPoint {
            frequency: parse(fields[0]),
            log10_rho: parse(fields[1]),
            log10_rho_lo: parse(fields[2]),
            log10_rho_hi: parse(fields[3]),
        });
    }

    Ok(points)
}

/// Read a 1D or multi-dimensional f64 numpy array from raw .npy bytes.
///
/// Supports only `<f8` (little-endian float64) dtype. Returns the raw f64
/// values in row-major order along with the shape.
fn read_npy_f64(data: &[u8]) -> Result<(Vec<usize>, Vec<f64>), FetchError> {
    // Validate magic number
    if data.len() < 10 || &data[..6] != b"\x93NUMPY" {
        return Err(FetchError::Validation("Not a numpy file".to_string()));
    }

    let major = data[6];
    let header_len = if major <= 1 {
        // v1: 2-byte header length at offset 8
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else {
        // v2+: 4-byte header length at offset 8
        if data.len() < 12 {
            return Err(FetchError::Validation("Truncated numpy v2 header".to_string()));
        }
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    };

    let header_offset = if major <= 1 { 10 } else { 12 };
    let data_offset = header_offset + header_len;

    if data.len() < data_offset {
        return Err(FetchError::Validation("Truncated numpy file".to_string()));
    }

    // Parse header to extract shape
    let header = std::str::from_utf8(&data[header_offset..data_offset])
        .map_err(|_| FetchError::Validation("Invalid numpy header encoding".to_string()))?;

    // Verify dtype is little-endian float64
    if !header.contains("'<f8'") {
        return Err(FetchError::Validation(
            format!("Unsupported numpy dtype (need <f8): {}", header),
        ));
    }

    // Extract shape tuple: 'shape': (30,) or 'shape': (1, 30, 10000)
    let shape = parse_npy_shape(header)?;

    let n_elements: usize = shape.iter().product();
    let expected_bytes = n_elements * 8;
    let actual_bytes = data.len() - data_offset;

    if actual_bytes < expected_bytes {
        return Err(FetchError::Validation(
            format!("Numpy data too short: {} bytes for {} f64 values", actual_bytes, n_elements),
        ));
    }

    let values: Vec<f64> = (0..n_elements)
        .map(|i| {
            let offset = data_offset + i * 8;
            f64::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
                data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
            ])
        })
        .collect();

    Ok((shape, values))
}

/// Parse the shape tuple from a numpy header string.
fn parse_npy_shape(header: &str) -> Result<Vec<usize>, FetchError> {
    let shape_start = header.find("'shape': (")
        .ok_or_else(|| FetchError::Validation("No shape in numpy header".to_string()))?;
    let after = &header[shape_start + 10..];
    let shape_end = after.find(')')
        .ok_or_else(|| FetchError::Validation("Unclosed shape tuple".to_string()))?;
    let shape_str = &after[..shape_end];

    let shape: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| {
            let t = s.trim();
            if t.is_empty() { None } else { t.parse().ok() }
        })
        .collect();

    if shape.is_empty() {
        return Err(FetchError::Validation("Empty shape tuple".to_string()));
    }

    Ok(shape)
}

/// Extract free spectrum point estimates from the KDE ZIP archive.
///
/// Reads the `30f_fs{hd}_ceffyl` analysis (HD-correlated, the main GWB
/// detection) and computes median + 5th/95th percentile from the KDE
/// posterior density at each frequency bin.
pub fn extract_free_spectrum_from_kde_zip(
    zip_path: &Path,
) -> Result<Vec<FreeSpectrumPoint>, FetchError> {
    let file = fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| FetchError::Validation(format!("ZIP open error: {}", e)))?;

    let prefix = "ceffyl_data/30f_fs{hd}_ceffyl/";

    // Read frequency array (30,)
    let freqs = read_npy_from_zip(&mut archive, &format!("{prefix}freqs.npy"))?;
    let n_freq = freqs.1.len();

    // Read grid array (10000,)
    let grid = read_npy_from_zip(&mut archive, &format!("{prefix}log10rhogrid.npy"))?;
    let n_grid = grid.1.len();

    // Read density array (1, 30, 10000)
    let density = read_npy_from_zip(&mut archive, &format!("{prefix}density.npy"))?;

    // Validate total element count matches n_freq * n_grid
    // (the density array may have shape (1, n_freq, n_grid) or (n_freq, n_grid))
    let expected_elements = n_freq * n_grid;
    if density.1.len() != expected_elements {
        return Err(FetchError::Validation(format!(
            "Density shape mismatch: {} values, expected {} ({} freqs x {} grid)",
            density.1.len(), expected_elements, n_freq, n_grid,
        )));
    }

    // Compute percentiles for each frequency bin
    let mut points = Vec::with_capacity(n_freq);
    let g = &grid.1;

    // Grid spacing (uniform grid)
    let mut dx = Vec::with_capacity(n_grid);
    for i in 0..n_grid - 1 {
        dx.push(g[i + 1] - g[i]);
    }
    dx.push(dx[n_grid - 2]);

    for i in 0..n_freq {
        let offset = i * n_grid;
        let di = &density.1[offset..offset + n_grid];

        // Normalize to proper PDF
        let area: f64 = di.iter().zip(dx.iter()).map(|(d, w)| d * w).sum();
        if area <= 0.0 {
            // Degenerate bin -- use NaN
            points.push(FreeSpectrumPoint {
                frequency: freqs.1[i],
                log10_rho: f64::NAN,
                log10_rho_lo: f64::NAN,
                log10_rho_hi: f64::NAN,
            });
            continue;
        }

        // Compute CDF
        let mut cdf = Vec::with_capacity(n_grid);
        let mut cumsum = 0.0;
        for j in 0..n_grid {
            cumsum += di[j] * dx[j] / area;
            cdf.push(cumsum);
        }
        // Normalize CDF endpoint to exactly 1.0
        let cdf_max = cdf[n_grid - 1];
        for v in &mut cdf {
            *v /= cdf_max;
        }

        // Interpolate percentiles
        let lo = interp_percentile(0.05, &cdf, g);
        let med = interp_percentile(0.50, &cdf, g);
        let hi = interp_percentile(0.95, &cdf, g);

        points.push(FreeSpectrumPoint {
            frequency: freqs.1[i],
            log10_rho: med,
            log10_rho_lo: lo,
            log10_rho_hi: hi,
        });
    }

    Ok(points)
}

/// Read a numpy file from inside a ZIP archive.
fn read_npy_from_zip(
    archive: &mut zip::ZipArchive<fs::File>,
    name: &str,
) -> Result<(Vec<usize>, Vec<f64>), FetchError> {
    let mut entry = archive.by_name(name)
        .map_err(|e| FetchError::Validation(format!("ZIP entry '{}': {}", name, e)))?;
    let mut buf = Vec::new();
    entry.read_to_end(&mut buf)?;
    read_npy_f64(&buf)
}

/// Linear interpolation to find the grid value at a given CDF percentile.
fn interp_percentile(p: f64, cdf: &[f64], grid: &[f64]) -> f64 {
    // Find first index where CDF >= p
    for i in 1..cdf.len() {
        if cdf[i] >= p {
            // Linear interpolation between grid[i-1] and grid[i]
            let t = (p - cdf[i - 1]) / (cdf[i] - cdf[i - 1]);
            return grid[i - 1] + t * (grid[i] - grid[i - 1]);
        }
    }
    // Fallback: return last grid point
    *grid.last().unwrap_or(&f64::NAN)
}

/// Write free spectrum points to CSV.
pub fn write_free_spectrum_csv(
    points: &[FreeSpectrumPoint],
    path: &Path,
) -> Result<(), FetchError> {
    let mut output = String::from("frequency,log10_rho,log10_rho_lo,log10_rho_hi\n");
    for p in points {
        output.push_str(&format!(
            "{:.10e},{:.6},{:.6},{:.6}\n",
            p.frequency, p.log10_rho, p.log10_rho_lo, p.log10_rho_hi,
        ));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, &output)?;
    Ok(())
}

/// NANOGrav 15yr KDE free spectra, Zenodo record 10344086 (CC BY 4.0).
///
/// The ZIP contains numpy arrays with full posterior KDE densities for
/// 30 frequency bins. We extract point estimates on download.
const NANOGRAV_KDE_URL: &str =
    "https://zenodo.org/api/records/10344086/files/NANOGrav15yr_KDE-FreeSpectra_v1.1.0.zip/content";

/// NANOGrav 15-year free spectrum data provider.
///
/// Downloads the KDE ZIP (5.3 MB) from Zenodo, extracts point estimates
/// (median + 90% credible interval) for each of 30 frequency bins, and
/// saves as a simple CSV.
pub struct NanoGrav15yrProvider;

impl DatasetProvider for NanoGrav15yrProvider {
    fn name(&self) -> &str { "NANOGrav 15yr Free Spectrum" }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let csv_output = config.output_dir.join("nanograv_15yr_freespectrum.csv");
        if config.skip_existing && csv_output.exists() {
            eprintln!("  {} already cached at {}", self.name(), csv_output.display());
            return Ok(csv_output);
        }

        // Download KDE ZIP
        let zip_path = config.output_dir.join("nanograv_15yr_kde.zip");
        eprintln!("  Downloading {} from Zenodo...", self.name());
        match download_to_file(NANOGRAV_KDE_URL, &zip_path) {
            Ok(bytes) => {
                let data = fs::read(&zip_path)?;
                if let Err(e) = validate_not_html(&data) {
                    eprintln!("  Validation failed: {}", e);
                    fs::remove_file(&zip_path).ok();
                    return Err(e);
                }
                eprintln!("  Saved {} bytes to {}", bytes, zip_path.display());
            }
            Err(e) => {
                eprintln!("  Download failed: {}", e);
                // Fall back to hardcoded values
                eprintln!("  Using hardcoded bestfit values instead");
                write_free_spectrum_csv(&bestfit::HD_FREE_SPECTRUM, &csv_output)?;
                eprintln!("  Wrote {} bins to {}", bestfit::N_BINS, csv_output.display());
                return Ok(csv_output);
            }
        }

        // Extract point estimates from KDE
        eprintln!("  Extracting free spectrum from KDE posterior densities...");
        match extract_free_spectrum_from_kde_zip(&zip_path) {
            Ok(points) => {
                write_free_spectrum_csv(&points, &csv_output)?;
                eprintln!("  Extracted {} frequency bins to {}", points.len(), csv_output.display());
            }
            Err(e) => {
                eprintln!("  KDE extraction failed: {}", e);
                eprintln!("  Using hardcoded bestfit values instead");
                write_free_spectrum_csv(&bestfit::HD_FREE_SPECTRUM, &csv_output)?;
                eprintln!("  Wrote {} bins to {}", bestfit::N_BINS, csv_output.display());
            }
        }

        Ok(csv_output)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("nanograv_15yr_freespectrum.csv").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bestfit_frequencies_monotonic() {
        let fs = &bestfit::HD_FREE_SPECTRUM;
        assert_eq!(fs.len(), 30);
        for i in 1..fs.len() {
            assert!(
                fs[i].frequency > fs[i - 1].frequency,
                "Frequencies must be monotonically increasing: f[{}]={} <= f[{}]={}",
                i, fs[i].frequency, i - 1, fs[i - 1].frequency,
            );
        }
    }

    #[test]
    fn test_bestfit_credible_intervals() {
        for (i, p) in bestfit::HD_FREE_SPECTRUM.iter().enumerate() {
            assert!(
                p.log10_rho_lo < p.log10_rho,
                "Bin {}: lower bound {} should be < median {}",
                i, p.log10_rho_lo, p.log10_rho,
            );
            assert!(
                p.log10_rho < p.log10_rho_hi,
                "Bin {}: median {} should be < upper bound {}",
                i, p.log10_rho, p.log10_rho_hi,
            );
        }
    }

    #[test]
    fn test_bestfit_first_bin_strongest() {
        // The lowest frequency bin should have the strongest signal
        // (most negative log10_rho means stronger strain power)
        // Actually, log10_rho is log10 of rho, where higher rho = more power.
        // The first bin has the highest (least negative) median, consistent
        // with a red-tilt spectrum.
        let first = bestfit::HD_FREE_SPECTRUM[0].log10_rho;
        let mean_rest: f64 = bestfit::HD_FREE_SPECTRUM[4..15]
            .iter()
            .map(|p| p.log10_rho)
            .sum::<f64>() / 11.0;
        assert!(
            first > mean_rest,
            "First bin ({:.2}) should be stronger than mean of mid-range ({:.2})",
            first, mean_rest,
        );
    }

    #[test]
    fn test_bestfit_frequency_range() {
        let first = bestfit::HD_FREE_SPECTRUM[0].frequency;
        let last = bestfit::HD_FREE_SPECTRUM[29].frequency;
        // NANOGrav operates in the nanohertz band
        assert!(first > 1e-10, "First freq {} should be > 0.1 nHz", first);
        assert!(first < 1e-8, "First freq {} should be < 10 nHz", first);
        assert!(last > 1e-8, "Last freq {} should be > 10 nHz", last);
        assert!(last < 1e-7, "Last freq {} should be < 100 nHz", last);
    }

    #[test]
    fn test_read_npy_f64_rejects_bad_magic() {
        let data = b"NOT_NUMPY_DATA";
        assert!(read_npy_f64(data).is_err());
    }

    #[test]
    fn test_parse_npy_shape_1d() {
        let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (30,), }";
        let shape = parse_npy_shape(header).unwrap();
        assert_eq!(shape, vec![30]);
    }

    #[test]
    fn test_parse_npy_shape_3d() {
        let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (1, 30, 10000), }";
        let shape = parse_npy_shape(header).unwrap();
        assert_eq!(shape, vec![1, 30, 10000]);
    }

    #[test]
    fn test_interp_percentile() {
        let cdf = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let grid = vec![0.0, 1.0, 2.0, 3.0, 4.0];

        let p50 = interp_percentile(0.5, &cdf, &grid);
        assert!((p50 - 2.0).abs() < 1e-10, "50th percentile should be 2.0, got {}", p50);

        let p25 = interp_percentile(0.25, &cdf, &grid);
        assert!((p25 - 1.0).abs() < 1e-10, "25th percentile should be 1.0, got {}", p25);
    }

    #[test]
    fn test_parse_nanograv_csv() {
        let csv = "frequency,log10_rho,log10_rho_lo,log10_rho_hi\n\
                   1.9768e-09,-3.824,-12.240,-1.192\n\
                   3.9537e-09,-8.315,-14.420,-1.564\n";
        let tmp = std::env::temp_dir().join("test_nanograv.csv");
        fs::write(&tmp, csv).unwrap();
        let points = parse_nanograv_free_spectrum(&tmp).unwrap();
        assert_eq!(points.len(), 2);
        assert!((points[0].frequency - 1.9768e-09).abs() < 1e-13);
        assert!((points[0].log10_rho - (-3.824)).abs() < 1e-3);
        fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_extract_kde_zip_if_available() {
        let path = Path::new("data/external/nanograv_15yr_kde.zip");
        if !path.exists() {
            eprintln!("Skipping: NANOGrav KDE ZIP not available (run fetch-datasets first)");
            return;
        }

        let points = extract_free_spectrum_from_kde_zip(path)
            .expect("Failed to extract free spectrum from KDE ZIP");
        assert_eq!(points.len(), 30, "Should extract 30 frequency bins");

        // Verify first bin matches bestfit
        let first = &points[0];
        assert!(
            (first.frequency - bestfit::HD_FREE_SPECTRUM[0].frequency).abs() < 1e-15,
            "First frequency should match bestfit",
        );
    }
}
