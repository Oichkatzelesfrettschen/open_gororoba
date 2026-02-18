//! Breakthrough Listen filterbank HDF5 reader.
//!
//! BL filterbank files from the GBT use HDF5 format with a `data` dataset
//! of shape `[n_time, n_ifs, n_chans]` and header metadata as root attributes.
//! The medium-resolution products (res_level 0002) have `nfpc=1024` fine
//! channels per coarse channel, with `nchans / nfpc = 64` coarse channels.
//!
//! This module provides zero-copy-friendly readers that load one coarse
//! channel at a time to stay within memory budget even for high-res files
//! (3+ GB each).
//!
//! Feature-gated behind `hdf5-export` (same gate as `hdf5_export.rs`).
//!
//! References:
//! - BL data format: <https://github.com/UCBerkeleySETI/blimpy>
//! - Perez et al. (2022) RNAAS 6 197
//! - hdf5-metno 0.12 API: <https://docs.rs/hdf5>

#[cfg(feature = "hdf5-export")]
use crate::fetcher::FetchError;
use std::path::{Path, PathBuf};

/// BL filterbank header parsed from HDF5 root-group attributes.
#[derive(Debug, Clone)]
pub struct FilterbankHeader {
    /// Source/target name (e.g. "DIAG_6EQUJ5_1").
    pub source_name: String,
    /// Center frequency of first channel (MHz). Typically the highest freq.
    pub fch1: f64,
    /// Channel bandwidth (MHz). Negative means descending frequency order.
    pub foff: f64,
    /// Total number of fine frequency channels.
    pub nchans: u32,
    /// Fine channels per coarse channel.
    pub nfpc: u32,
    /// Time sample interval (seconds).
    pub tsamp: f64,
    /// Start time (MJD).
    pub tstart: f64,
    /// Bits per sample (typically 32 = float32).
    pub nbits: u32,
    /// Number of IF (polarization) channels (typically 1).
    pub nifs: u32,
    /// Telescope ID (6 = GBT).
    pub telescope_id: u32,
    /// Backend ID (20 = GUPPI).
    pub machine_id: u32,
    /// Right ascension of source (hourangle).
    pub src_raj: f64,
    /// Declination of source (degrees).
    pub src_dej: f64,
}

impl FilterbankHeader {
    /// Number of coarse channels = nchans / nfpc.
    pub fn n_coarse_channels(&self) -> u32 {
        if self.nfpc == 0 {
            return 0;
        }
        self.nchans / self.nfpc
    }

    /// Channel width in Hz (absolute value).
    pub fn channel_width_hz(&self) -> f64 {
        self.foff.abs() * 1e6
    }

    /// Total bandwidth in MHz (absolute value).
    pub fn bandwidth_mhz(&self) -> f64 {
        self.foff.abs() * self.nchans as f64
    }
}

/// Handle for an opened BL filterbank HDF5 file.
#[derive(Debug)]
pub struct FilterbankFile {
    /// Parsed header attributes.
    pub header: FilterbankHeader,
    /// Number of time samples in the data array.
    pub n_time: usize,
    /// Path to the HDF5 file.
    pub path: PathBuf,
}

impl FilterbankFile {
    /// Total observation duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.n_time as f64 * self.header.tsamp
    }

    /// Drift rate resolution (Hz/s) for Doppler search.
    /// This is the minimum resolvable drift rate given the time/freq resolution.
    pub fn drift_resolution_hz_s(&self) -> f64 {
        if self.n_time <= 1 {
            return 0.0;
        }
        let total_time = (self.n_time - 1) as f64 * self.header.tsamp;
        self.header.channel_width_hz() / total_time
    }
}

/// Open a BL filterbank HDF5 file and parse header attributes.
///
/// Reads root-group attributes to populate `FilterbankHeader`, then checks
/// the `data` dataset shape to determine `n_time`.
#[cfg(feature = "hdf5-export")]
pub fn open_filterbank(path: &Path) -> Result<FilterbankFile, FetchError> {
    use hdf5::File as H5File;

    let file = H5File::open(path).map_err(|e| FetchError::Validation(format!(
        "Failed to open HDF5 file {}: {}", path.display(), e
    )))?;

    let header = read_header_from_file(&file, path)?;

    // Read data shape: [n_time, n_ifs, n_chans]
    let data = file.dataset("data").map_err(|e| FetchError::Validation(format!(
        "No 'data' dataset in {}: {}", path.display(), e
    )))?;
    let shape = data.shape();
    if shape.len() != 3 {
        return Err(FetchError::Validation(format!(
            "Expected 3D data array, got {}D in {}", shape.len(), path.display()
        )));
    }
    let n_time = shape[0];

    Ok(FilterbankFile {
        header,
        n_time,
        path: path.to_path_buf(),
    })
}

/// Read header-only (no data access) for quick metadata extraction.
#[cfg(feature = "hdf5-export")]
pub fn read_filterbank_header(path: &Path) -> Result<FilterbankHeader, FetchError> {
    use hdf5::File as H5File;

    let file = H5File::open(path).map_err(|e| FetchError::Validation(format!(
        "Failed to open HDF5 file {}: {}", path.display(), e
    )))?;

    read_header_from_file(&file, path)
}

/// Internal: extract header attributes from an open HDF5 file.
#[cfg(feature = "hdf5-export")]
fn read_header_from_file(
    file: &hdf5::File,
    path: &Path,
) -> Result<FilterbankHeader, FetchError> {
    let read_f64 = |name: &str| -> Result<f64, FetchError> {
        file.attr(name)
            .and_then(|a| a.read_scalar::<f64>())
            .map_err(|e| FetchError::Validation(format!(
                "Missing/invalid attr '{}' in {}: {}", name, path.display(), e
            )))
    };
    let read_i32 = |name: &str| -> Result<i32, FetchError> {
        file.attr(name)
            .and_then(|a| a.read_scalar::<i32>())
            .map_err(|e| FetchError::Validation(format!(
                "Missing/invalid attr '{}' in {}: {}", name, path.display(), e
            )))
    };
    let read_string = |name: &str| -> Result<String, FetchError> {
        // BL filterbank stores source_name as fixed-length ASCII string
        file.attr(name)
            .and_then(|a| a.read_scalar::<hdf5::types::VarLenUnicode>())
            .map(|s| s.to_string())
            .or_else(|_| {
                // Fallback: try reading as fixed-length byte string
                file.attr(name)
                    .and_then(|a| {
                        let raw = a.read_raw()?;
                        let bytes: Vec<u8> = raw.iter().copied().collect();
                        Ok(String::from_utf8_lossy(&bytes).trim_end_matches('\0').to_string())
                    })
            })
            .map_err(|e| FetchError::Validation(format!(
                "Missing/invalid string attr '{}' in {}: {}", name, path.display(), e
            )))
    };

    Ok(FilterbankHeader {
        source_name: read_string("source_name")?,
        fch1: read_f64("fch1")?,
        foff: read_f64("foff")?,
        nchans: read_i32("nchans")? as u32,
        nfpc: read_i32("nfpc")? as u32,
        tsamp: read_f64("tsamp")?,
        tstart: read_f64("tstart")?,
        nbits: read_i32("nbits")? as u32,
        nifs: read_i32("nifs")? as u32,
        telescope_id: read_i32("telescope_id")? as u32,
        machine_id: read_i32("machine_id")? as u32,
        src_raj: read_f64("src_raj")?,
        src_dej: read_f64("src_dej")?,
    })
}

/// Read a single coarse channel's data as a flat f32 array.
///
/// Returns `[n_time][nfpc]` row-major data (time-major ordering).
/// Coarse channel index 0 = highest frequency channels (since foff < 0).
#[cfg(feature = "hdf5-export")]
pub fn read_coarse_channel(
    fb: &FilterbankFile,
    coarse_idx: usize,
) -> Result<Vec<f32>, FetchError> {
    use hdf5::File as H5File;
    use ndarray::s;

    let n_coarse = fb.header.n_coarse_channels() as usize;
    if coarse_idx >= n_coarse {
        return Err(FetchError::Validation(format!(
            "Coarse channel {} out of range (0..{})", coarse_idx, n_coarse
        )));
    }

    let nfpc = fb.header.nfpc as usize;
    let chan_start = coarse_idx * nfpc;
    let chan_end = chan_start + nfpc;

    let file = H5File::open(&fb.path).map_err(|e| FetchError::Validation(format!(
        "Failed to reopen HDF5 file {}: {}", fb.path.display(), e
    )))?;
    let dataset = file.dataset("data").map_err(|e| FetchError::Validation(format!(
        "No 'data' dataset in {}: {}", fb.path.display(), e
    )))?;

    // Read slice [0..n_time, 0..1, chan_start..chan_end] as f32
    // The BL data is float32 (nbits=32), stored as the native type.
    let arr = dataset
        .read_slice_2d::<f32, _>(s![.., 0, chan_start..chan_end])
        .map_err(|e| FetchError::Validation(format!(
            "Failed to read coarse channel {} from {}: {}",
            coarse_idx, fb.path.display(), e
        )))?;

    Ok(arr.into_raw_vec())
}

/// Compute frequency array for a coarse channel (MHz).
///
/// Returns `nfpc` frequencies in the native ordering (descending if foff < 0).
pub fn coarse_channel_freqs(header: &FilterbankHeader, coarse_idx: usize) -> Vec<f64> {
    let nfpc = header.nfpc as usize;
    let chan_start = coarse_idx * nfpc;
    (0..nfpc)
        .map(|i| header.fch1 + (chan_start + i) as f64 * header.foff)
        .collect()
}

/// Compute time array (seconds from observation start).
pub fn time_array(tsamp: f64, n_time: usize) -> Vec<f64> {
    (0..n_time).map(|t| t as f64 * tsamp).collect()
}

/// Metadata for a single BL observation file.
#[derive(Debug, Clone)]
pub struct BlObservation {
    /// Node identifier (e.g. "blc40").
    pub node: String,
    /// MJD integer day.
    pub mjd: u64,
    /// Start seconds within the MJD day.
    pub seconds: u64,
    /// Source/target name.
    pub source: String,
    /// Observation sequence number (e.g. "0016").
    pub obs_id: String,
    /// Resolution level ("0000" = high, "0001" = mid, "0002" = medium).
    pub res_level: String,
    /// Pointing type: "ON" or "OFF".
    pub pointing_type: String,
    /// ABACAD cadence group (1 or 2).
    pub cadence: u32,
}

impl BlObservation {
    /// Construct the BL data archive filename.
    pub fn filename(&self) -> String {
        format!(
            "{}_guppi_{}_{}_{}_{}_.rawspec.{}.h5",
            self.node, self.mjd, self.seconds, self.source, self.obs_id, self.res_level
        )
    }

    /// Construct the full download URL on the BL data archive.
    pub fn download_url(&self) -> String {
        format!(
            "https://bldata.berkeley.edu/6EQUJ5/6EQUJ5_GBT/{}",
            self.filename()
        )
    }
}

/// Build the observation list for the BL 6EQUJ5 GBT dataset (node blc40).
///
/// The 6EQUJ5 campaign uses ABACAD cadence:
/// - Cadence 1: obs 0016 (ON), 0017 (OFF), 0018 (ON), 0019 (OFF), 0020 (ON), 0021 (OFF)
/// - Cadence 2: obs 0022 (ON), 0023 (OFF), 0024 (ON), 0025 (OFF), 0026 (ON), 0027 (OFF)
pub fn bl_6equj5_observations() -> Vec<BlObservation> {
    let mjd = 59720u64;
    let node = "blc40";

    // (seconds, source, obs_id, pointing_type, cadence)
    let obs_list: &[(u64, &str, &str, &str, u32)] = &[
        (37784, "DIAG_6EQUJ5_1", "0016", "ON", 1),
        (38125, "HIP94637", "0017", "OFF", 1),
        (38461, "DIAG_6EQUJ5_1", "0018", "ON", 1),
        (38800, "HIP94394", "0019", "OFF", 1),
        (39139, "DIAG_6EQUJ5_1", "0020", "ON", 1),
        (39481, "HIP94062", "0021", "OFF", 1),
        (39831, "DIAG_6EQUJ5_2", "0022", "ON", 2),
        (40168, "HIP94637", "0023", "OFF", 2),
        (40505, "DIAG_6EQUJ5_2", "0024", "ON", 2),
        (40843, "HIP94783", "0025", "OFF", 2),
        (41181, "DIAG_6EQUJ5_2", "0026", "ON", 2),
        (41521, "HIP94394", "0027", "OFF", 2),
    ];

    obs_list
        .iter()
        .map(|(secs, source, obs, pt, cad)| BlObservation {
            node: node.to_string(),
            mjd,
            seconds: *secs,
            source: source.to_string(),
            obs_id: obs.to_string(),
            res_level: "0002".to_string(),
            pointing_type: pt.to_string(),
            cadence: *cad,
        })
        .collect()
}

/// Resolve a BL observation to its local file path.
pub fn observation_file_path(obs: &BlObservation, data_dir: &Path) -> PathBuf {
    data_dir.join(obs.filename())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_list_completeness() {
        let obs = bl_6equj5_observations();
        assert_eq!(obs.len(), 12, "12 observations in ABACAD x 2 cadences");
        let on_count = obs.iter().filter(|o| o.pointing_type == "ON").count();
        let off_count = obs.iter().filter(|o| o.pointing_type == "OFF").count();
        assert_eq!(on_count, 6, "6 ON pointings");
        assert_eq!(off_count, 6, "6 OFF pointings");
    }

    #[test]
    fn test_observation_cadence_groups() {
        let obs = bl_6equj5_observations();
        let cad1: Vec<_> = obs.iter().filter(|o| o.cadence == 1).collect();
        let cad2: Vec<_> = obs.iter().filter(|o| o.cadence == 2).collect();
        assert_eq!(cad1.len(), 6, "6 observations in cadence 1");
        assert_eq!(cad2.len(), 6, "6 observations in cadence 2");
    }

    #[test]
    fn test_observation_filename_format() {
        let obs = &bl_6equj5_observations()[0];
        let fname = obs.filename();
        assert!(fname.starts_with("blc40_guppi_59720_"));
        assert!(fname.contains("DIAG_6EQUJ5_1"));
        assert!(fname.ends_with(".rawspec.0002.h5"));
    }

    #[test]
    fn test_observation_url_format() {
        let obs = &bl_6equj5_observations()[0];
        let url = obs.download_url();
        assert!(url.starts_with("https://bldata.berkeley.edu/6EQUJ5/6EQUJ5_GBT/"));
        assert!(url.ends_with(".h5"));
    }

    #[test]
    fn test_coarse_channel_freqs_descending() {
        let header = FilterbankHeader {
            source_name: "TEST".to_string(),
            fch1: 2251.46484375,
            foff: -0.00286102294921875,
            nchans: 65536,
            nfpc: 1024,
            tsamp: 1.074,
            tstart: 59720.0,
            nbits: 32,
            nifs: 1,
            telescope_id: 6,
            machine_id: 20,
            src_raj: 19.472,
            src_dej: -26.670,
        };
        let freqs = coarse_channel_freqs(&header, 0);
        assert_eq!(freqs.len(), 1024);
        // foff < 0 means descending frequency
        assert!(freqs[0] > freqs[1], "Frequency should descend with channel index");
        let diff = (freqs[1] - freqs[0]).abs();
        assert!((diff - 0.00286102294921875).abs() < 1e-12);
    }

    #[test]
    fn test_coarse_channel_count() {
        let header = FilterbankHeader {
            source_name: "TEST".to_string(),
            fch1: 2251.46484375,
            foff: -0.00286102294921875,
            nchans: 65536,
            nfpc: 1024,
            tsamp: 1.074,
            tstart: 59720.0,
            nbits: 32,
            nifs: 1,
            telescope_id: 6,
            machine_id: 20,
            src_raj: 19.472,
            src_dej: -26.670,
        };
        assert_eq!(header.n_coarse_channels(), 64);
    }

    #[test]
    fn test_channel_width_hz() {
        let header = FilterbankHeader {
            source_name: "TEST".to_string(),
            fch1: 2251.46484375,
            foff: -0.00286102294921875,
            nchans: 65536,
            nfpc: 1024,
            tsamp: 1.074,
            tstart: 59720.0,
            nbits: 32,
            nifs: 1,
            telescope_id: 6,
            machine_id: 20,
            src_raj: 19.472,
            src_dej: -26.670,
        };
        let width = header.channel_width_hz();
        assert!((width - 2861.02294921875).abs() < 0.01);
    }

    #[test]
    fn test_time_array_monotone() {
        let t = time_array(1.074, 10);
        assert_eq!(t.len(), 10);
        assert!((t[0] - 0.0).abs() < 1e-12);
        for w in t.windows(2) {
            assert!(w[1] > w[0], "Time must be monotonically increasing");
        }
    }

    #[test]
    fn test_drift_resolution_medium_res() {
        // Medium-res: foff = 2861 Hz, n_time ~ 279, tsamp = 1.074s
        // Expected: ~9.58 Hz/s
        let fb = FilterbankFile {
            header: FilterbankHeader {
                source_name: "TEST".to_string(),
                fch1: 2251.46484375,
                foff: -0.00286102294921875,
                nchans: 65536,
                nfpc: 1024,
                tsamp: 1.073741823999999,
                tstart: 59720.0,
                nbits: 32,
                nifs: 1,
                telescope_id: 6,
                machine_id: 20,
                src_raj: 19.472,
                src_dej: -26.670,
            },
            n_time: 279,
            path: PathBuf::from("test.h5"),
        };
        let dr = fb.drift_resolution_hz_s();
        // turboSETI computed 9.584659205397045 Hz/s
        assert!(
            (dr - 9.58).abs() < 0.1,
            "Expected ~9.58 Hz/s, got {}",
            dr
        );
    }

    // --- Tests that require real HDF5 files (skipped if absent) ---

    #[cfg(feature = "hdf5-export")]
    #[test]
    fn test_open_filterbank_if_available() {
        let data_dir = Path::new("data/bl_6equj5_gbt");
        let obs = bl_6equj5_observations();
        let first = observation_file_path(&obs[0], data_dir);
        if !first.exists() {
            eprintln!("Skipping: {} not available", first.display());
            return;
        }
        let fb = open_filterbank(&first).expect("Failed to open filterbank");
        assert_eq!(fb.header.telescope_id, 6, "GBT telescope_id = 6");
        assert_eq!(fb.header.machine_id, 20, "GUPPI machine_id = 20");
        assert_eq!(fb.header.nchans, 65536);
        assert_eq!(fb.header.nfpc, 1024);
        assert_eq!(fb.header.n_coarse_channels(), 64);
        assert!(fb.header.foff < 0.0, "Descending frequency order");
        assert!(fb.n_time > 0, "Must have at least one time sample");
        eprintln!(
            "Filterbank: {} time samples, {:.1}s total, drift_res={:.3} Hz/s",
            fb.n_time,
            fb.duration_seconds(),
            fb.drift_resolution_hz_s()
        );
    }

    #[cfg(feature = "hdf5-export")]
    #[test]
    fn test_read_coarse_channel_if_available() {
        let data_dir = Path::new("data/bl_6equj5_gbt");
        let obs = bl_6equj5_observations();
        let first = observation_file_path(&obs[0], data_dir);
        if !first.exists() {
            eprintln!("Skipping: {} not available", first.display());
            return;
        }
        let fb = open_filterbank(&first).expect("Failed to open filterbank");
        let data = read_coarse_channel(&fb, 0).expect("Failed to read coarse channel 0");
        let expected_len = fb.n_time * fb.header.nfpc as usize;
        assert_eq!(data.len(), expected_len);
        // Verify no NaN or Inf
        assert!(
            data.iter().all(|v| v.is_finite()),
            "Data should contain no NaN or Inf"
        );
        eprintln!("Coarse channel 0: {} values, range [{:.1}, {:.1}]",
            data.len(),
            data.iter().cloned().fold(f32::INFINITY, f32::min),
            data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        );
    }
}
