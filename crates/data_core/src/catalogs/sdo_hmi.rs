//! SDO/HMI SHARP vector magnetogram catalog module.
//!
//! Provides access to Space-weather HMI Active Region Patches (SHARP) data
//! from the Joint Science Operations Center (JSOC) at Stanford.
//!
//! Two data access modes:
//! 1. **Keyword time series** (this module): 16 scalar SHARP parameters at
//!    12-min cadence per active region (HARPNUM). No FITS parsing needed --
//!    JSOC returns JSON via HTTP GET. This gives a natural 16-channel
//!    embedding for CD analysis of pre-flare evolution.
//!
//! 2. **Image data** (future): Full 2D vector magnetogram arrays (Bp, Br, Bt)
//!    in FITS IMAGE format. Requires FITS image reader (fitsio or fitsrs).
//!
//! JSOC API: http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info
//! SHARP docs: http://jsoc.stanford.edu/doc/data/hmi/sharp/sharp.htm
//!
//! Reference: Bobra et al. (2014), "The Helioseismic and Magnetic Imager (HMI)
//! Vector Magnetic Field Pipeline: SHARPs -- Space-Weather HMI Active Region
//! Patches", Solar Physics 289, 3549-3578.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Number of SHARP keyword channels (excluding T_REC timestamp).
pub const SHARP_CHANNELS: usize = 15;

/// SHARP keyword names in canonical order (matches JSOC query order).
pub const SHARP_KEYWORD_NAMES: [&str; SHARP_CHANNELS] = [
    "USFLUX",   // Total unsigned flux (Mx)
    "MEANGBT",  // Mean gradient of total field (G/Mm)
    "MEANJZH",  // Mean current helicity (G^2/m)
    "TOTUSJH",  // Total unsigned current helicity (G^2/m * Mm^2)
    "MEANALP",  // Mean twist parameter alpha (1/Mm)
    "MEANGAM",  // Mean inclination angle (degrees)
    "MEANGBZ",  // Mean gradient of vertical field (G/Mm)
    "MEANSHR",  // Mean shear angle (degrees)
    "SHRGT45",  // Fraction of pixels with shear > 45 deg
    "AREA_ACR", // Active region area (micro-hemispheres)
    "ABSNJZH",  // Absolute value of net current helicity (G^2/m)
    "SAVNCPP",  // Sum of net current per polarity (A)
    "MEANPOT",  // Mean photospheric excess magnetic energy (ergs/cm^3)
    "TOTPOT",   // Total photospheric excess magnetic energy (ergs/cm)
    "R_VALUE",  // Log of unsigned flux near polarity inversion line (Mx)
];

/// A single SHARP keyword time step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharpRecord {
    /// JSOC timestamp (e.g. "2017.09.06_08:36:00_TAI")
    pub t_rec: String,
    /// HARPNUM (active region patch number)
    pub harpnum: u32,
    /// 15 SHARP keyword values in canonical order.
    /// NaN for missing/invalid values.
    pub keywords: [f64; SHARP_CHANNELS],
}

/// A complete SHARP time series for one active region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharpTimeSeries {
    pub harpnum: u32,
    pub records: Vec<SharpRecord>,
    pub cadence_minutes: f64,
}

/// Build the JSOC query URL for SHARP keywords.
///
/// Returns a URL that fetches the 16 SHARP keyword time series as JSON.
/// Example: HARPNUM 7115, around the X9.3 flare on 2017-09-06.
pub fn jsoc_sharp_keywords_url(harpnum: u32, t_start: &str, t_end: &str) -> String {
    let keywords = std::iter::once("T_REC")
        .chain(SHARP_KEYWORD_NAMES.iter().copied())
        .collect::<Vec<_>>()
        .join(",");

    format!(
        "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?\
         ds=hmi.sharp_cea_720s[{}][{}-{}]&op=rs_list&key={}",
        harpnum, t_start, t_end, keywords
    )
}

/// Parse the JSOC JSON response into a `SharpTimeSeries`.
///
/// The JSON has structure:
/// ```json
/// {
///   "keywords": [
///     {"name": "T_REC", "values": ["2017.09.06_08:36:00_TAI", ...]},
///     {"name": "USFLUX", "values": ["4.542070e+22", ...]},
///     ...
///   ],
///   "count": 27
/// }
/// ```
pub fn parse_jsoc_sharp_json(json_str: &str, harpnum: u32) -> Result<SharpTimeSeries, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {}", e))?;

    let keywords_arr = parsed["keywords"]
        .as_array()
        .ok_or("Missing 'keywords' array")?;

    // Build a map: keyword_name -> Vec<String>
    let mut kw_map: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for kw in keywords_arr {
        let name = kw["name"]
            .as_str()
            .ok_or("Missing keyword name")?
            .to_string();
        let values: Vec<String> = kw["values"]
            .as_array()
            .ok_or_else(|| format!("Missing values for {}", name))?
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();
        kw_map.insert(name, values);
    }

    let t_rec_vals = kw_map.get("T_REC").ok_or("Missing T_REC keyword")?;
    let n = t_rec_vals.len();

    let mut records = Vec::with_capacity(n);
    for (i, t_rec) in t_rec_vals.iter().enumerate() {
        let mut keywords = [f64::NAN; SHARP_CHANNELS];
        for (ch, kw_name) in SHARP_KEYWORD_NAMES.iter().enumerate() {
            if let Some(s) = kw_map.get(*kw_name).and_then(|vals| vals.get(i)) {
                keywords[ch] = s.parse::<f64>().unwrap_or(f64::NAN);
            }
        }
        records.push(SharpRecord {
            t_rec: t_rec.clone(),
            harpnum,
            keywords,
        });
    }

    Ok(SharpTimeSeries {
        harpnum,
        records,
        cadence_minutes: 12.0, // 720s standard SHARP cadence
    })
}

/// Load a SHARP time series from a previously saved JSOC JSON file.
pub fn load_sharp_json(path: &Path, harpnum: u32) -> Result<SharpTimeSeries, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Read {}: {}", path.display(), e))?;
    parse_jsoc_sharp_json(&content, harpnum)
}

/// Extract a multi-channel embedding vector from a window of SHARP records.
///
/// Each record contributes `SHARP_CHANNELS` (15) values, normalized by the
/// window mean for each channel. A window of `steps` records at the given
/// `lag` gives dim = SHARP_CHANNELS * steps.
///
/// Returns None if any channel has all NaN or zero mean in the window.
pub fn sharp_embedding_vector(
    records: &[SharpRecord],
    start: usize,
    steps: usize,
    lag: usize,
) -> Option<Vec<f64>> {
    let dim = SHARP_CHANNELS * steps;
    let indices: Vec<usize> = (0..steps).map(|s| start + s * lag).collect();

    // Check bounds
    if indices.last().copied().unwrap_or(0) >= records.len() {
        return None;
    }

    // Compute per-channel means for normalization
    let mut means = [0.0f64; SHARP_CHANNELS];
    let mut counts = [0usize; SHARP_CHANNELS];
    for &idx in &indices {
        for ch in 0..SHARP_CHANNELS {
            let v = records[idx].keywords[ch];
            if v.is_finite() && v != 0.0 {
                means[ch] += v.abs();
                counts[ch] += 1;
            }
        }
    }
    for ch in 0..SHARP_CHANNELS {
        if counts[ch] == 0 {
            return None;
        }
        means[ch] /= counts[ch] as f64;
        if means[ch] <= 0.0 {
            return None;
        }
    }

    // Build normalized embedding vector
    let mut v = vec![0.0; dim];
    for (s, &idx) in indices.iter().enumerate() {
        for ch in 0..SHARP_CHANNELS {
            let val = records[idx].keywords[ch];
            v[s * SHARP_CHANNELS + ch] = if val.is_finite() {
                val / means[ch]
            } else {
                0.0
            };
        }
    }

    Some(v)
}

/// Build sliding-window embeddings from a SHARP time series.
///
/// Returns (embedded_vectors, record_indices) where each vector has
/// dim = SHARP_CHANNELS * steps, and record_indices[i] is the index
/// of the LAST record in the i-th window.
pub fn sharp_takens_embed(
    series: &SharpTimeSeries,
    steps: usize,
    lag: usize,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let window_len = (steps - 1) * lag + 1;
    let n = series.records.len();
    if n < window_len {
        return (Vec::new(), Vec::new());
    }

    let mut embedded = Vec::new();
    let mut indices = Vec::new();

    for start in 0..=(n - window_len) {
        if let Some(v) = sharp_embedding_vector(&series.records, start, steps, lag) {
            embedded.push(v);
            indices.push(start + (steps - 1) * lag);
        }
    }

    (embedded, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsoc_url_construction() {
        let url =
            jsoc_sharp_keywords_url(7115, "2017.09.06_00:00:00_TAI", "2017.09.06_23:59:59_TAI");
        assert!(url.contains("hmi.sharp_cea_720s"));
        assert!(url.contains("[7115]"));
        assert!(url.contains("USFLUX"));
        assert!(url.contains("R_VALUE"));
        assert!(url.contains("T_REC"));
    }

    #[test]
    fn test_parse_existing_sharp_json() {
        let path = std::path::Path::new("data/external/sdo_aia/hmi_sharp_7115_x93.json");
        if !path.exists() {
            return; // skip if data not available
        }
        let ts = load_sharp_json(path, 7115).unwrap();
        assert_eq!(ts.harpnum, 7115);
        assert!(ts.records.len() > 0, "Should have records");
        assert_eq!(
            ts.records.len(),
            27,
            "Expected 27 records for X9.3 flare window"
        );

        // Check first record has finite USFLUX
        let r0 = &ts.records[0];
        assert!(r0.keywords[0].is_finite(), "USFLUX should be finite");
        assert!(r0.keywords[0] > 1e20, "USFLUX should be ~4.5e22 Mx");

        // Check timestamp format
        assert!(r0.t_rec.contains("2017"), "Should be 2017 data");
    }

    #[test]
    fn test_sharp_embedding() {
        let path = std::path::Path::new("data/external/sdo_aia/hmi_sharp_7115_x93.json");
        if !path.exists() {
            return;
        }
        let ts = load_sharp_json(path, 7115).unwrap();

        // 2 steps x 15 channels = 30D embedding (pad to 32 for CD)
        let (embedded, _indices) = sharp_takens_embed(&ts, 2, 1);
        assert!(!embedded.is_empty());
        assert_eq!(embedded[0].len(), SHARP_CHANNELS * 2); // 30

        // All values should be finite after normalization
        for v in &embedded {
            for &x in v {
                assert!(x.is_finite(), "Embedding value should be finite");
            }
        }
    }

    #[test]
    fn test_keyword_names_count() {
        assert_eq!(SHARP_KEYWORD_NAMES.len(), SHARP_CHANNELS);
        assert_eq!(SHARP_CHANNELS, 15);
    }
}
