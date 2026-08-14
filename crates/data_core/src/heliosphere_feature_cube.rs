//! Shared heliosphere feature-cube types and memory-planning helpers.
//!
//! This module does not fetch data directly. It provides the normalized row
//! schema used by the executed feature-cube, algebra, and LBM benchmark bins,
//! along with the memory estimators for dense and sparse 3D runs.

use chrono::{DateTime, NaiveDate, Utc};
use gororoba_sparse_grid::{
    BrickGrid3d, BrickShape3d, IndirectBrickTableShape, LogicalGrid3d, OccupancyBitsetStats,
    estimate_metadata_footprint,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, str::FromStr};

pub const HELIOSPHERE_FEATURE_DIM: usize = 16;
pub const HELIOSPHERE_SIGNAL_DIM: usize = HELIOSPHERE_FEATURE_DIM - 1;
pub const HELIOSPHERE_SUPPORT_DIM: usize = 3;
pub const HELIOSPHERE_DYNAMIC_DIM: usize = HELIOSPHERE_SIGNAL_DIM - HELIOSPHERE_SUPPORT_DIM;
const EVENT_THRESHOLD_ON_MULTIPLIER: f64 = 4.5;
const EVENT_THRESHOLD_OFF_MULTIPLIER: f64 = 2.25;
pub const HELIOSPHERE_CHANNEL_NAMES: [&str; HELIOSPHERE_FEATURE_DIM] = [
    "r_au",
    "lat_deg",
    "lon_deg",
    "density_cm3",
    "speed_kms",
    "temperature_k",
    "bx",
    "by",
    "bz",
    "b_mag",
    "crs_flux",
    "spectral_mean",
    "spectral_peak",
    "map_flux_mean",
    "map_flux_std",
    "bias",
];
pub const HELIOSPHERE_DYNAMIC_CHANNEL_NAMES: [&str; HELIOSPHERE_DYNAMIC_DIM] = [
    "density_cm3",
    "speed_kms",
    "temperature_k",
    "bx",
    "by",
    "bz",
    "b_mag",
    "crs_flux",
    "spectral_mean",
    "spectral_peak",
    "map_flux_mean",
    "map_flux_std",
];
pub const HELIOSPHERE_INVARIANT_DIM: usize = 10;
pub const HELIOSPHERE_INVARIANT_CHANNEL_NAMES: [&str; HELIOSPHERE_INVARIANT_DIM] = [
    "delta_b_over_bmag",
    "delta_v_over_vmag",
    "delta_n_over_n",
    "delta_t_over_t",
    "plasma_beta",
    "alfven_speed_kms",
    "alfvenicity_residual",
    "dynamic_pressure_residual",
    "magnetic_shear",
    "compressibility_proxy",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeliosphereFeatureRow {
    pub window_name: String,
    pub mission: String,
    pub product: String,
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    /// Radial distance, in a frame the schema does not declare and the ingest
    /// paths do not agree on.
    ///
    /// Measured over `data/output/heliosphere/densified_feature_cube_v3.csv`,
    /// the column carries three incompatible quantities. Heliocentric AU for
    /// Voyager 1 (1.0 to 165.66), Voyager 2 (to 125.83), New Horizons, Cassini,
    /// Juno, Parker Solar Probe (0.09 to 0.87), Helios 1 and 2, Solar Orbiter,
    /// BepiColombo and Ulysses. Geocentric AU for IMP 8 (0.00126 to 0.002) and
    /// IBEX, both in Earth orbit, and for SOHO (0.0083 to 0.011) about its L1
    /// halo. And the constant 1.0 for ACE, OMNI, STEREO-A and WIND across every
    /// row, which is a fill value rather than a measurement -- Voyager 1 and 2
    /// carry it too for their pre-cruise epochs.
    ///
    /// This matters because `signal_channels` puts this and the two angles at
    /// indices 0 through 2 of the vector the Cayley-Dickson embedding consumes.
    /// `transform_feature_rows` normalizes per `(window, mission, product)`, so
    /// `Normalized`, `DifferencedNormalized` and both robust modes absorb the
    /// frame difference into each mission's own mean and standard deviation,
    /// and `normalize_channel` sends a zero-variance channel to 0.0, which
    /// neutralizes the constant fill. `Raw` mode passes all three through
    /// unchanged, so a cross-mission comparison in `Raw` partly measures which
    /// ingest convention a mission's loader happened to use.
    pub r_au: f64,
    /// Latitude in the same undeclared frame as `r_au`; NaN where the ingest
    /// path found no ephemeris variable, which `normalize_channel` maps to 0.0.
    pub lat_deg: f64,
    /// Longitude in the same undeclared frame as `r_au`.
    pub lon_deg: f64,
    pub density_cm3: f64,
    pub speed_kms: f64,
    pub temperature_k: f64,
    pub bx: f64,
    pub by: f64,
    pub bz: f64,
    pub b_mag: f64,
    pub crs_flux: f64,
    pub spectral_mean: f64,
    pub spectral_peak: f64,
    pub map_flux_mean: f64,
    pub map_flux_std: f64,
    #[serde(default)]
    pub event_score: Option<f64>,
    #[serde(default)]
    pub event_mask: Option<bool>,
    #[serde(default)]
    pub event_segment_id: Option<u32>,
}

/// One invariant/residual sample derived from a heliosphere feature row.
///
/// These channels are intended for physically constrained downstream work:
/// predictive evaluation, cross-mission transfer, and adaptive sparsification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeliosphereInvariantSample {
    pub window_name: String,
    pub mission: String,
    pub product: String,
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub timestamp_utc: String,
    pub channels: [f64; HELIOSPHERE_INVARIANT_DIM],
    pub uncertainty_scales: [f64; HELIOSPHERE_INVARIANT_DIM],
    pub weighted_channels: [f64; HELIOSPHERE_INVARIANT_DIM],
    pub b_field: [f64; 4],
    pub inherited_event_score: Option<f64>,
    pub inherited_event_mask: Option<bool>,
    pub inherited_event_segment_id: Option<u32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum HeliosphereTransformMode {
    Raw,
    Normalized,
    Differenced,
    DifferencedNormalized,
    RobustCentered,
    RobustDifferencedCentered,
}

impl FromStr for HeliosphereTransformMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "raw" => Ok(Self::Raw),
            "normalized" => Ok(Self::Normalized),
            "differenced" => Ok(Self::Differenced),
            "differenced-normalized" => Ok(Self::DifferencedNormalized),
            "robust-centered" => Ok(Self::RobustCentered),
            "robust-differenced-centered" => Ok(Self::RobustDifferencedCentered),
            other => Err(format!(
                "unsupported mode '{other}'; expected raw, normalized, differenced, differenced-normalized, robust-centered, or robust-differenced-centered"
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeliosphereTransformGroupStats {
    pub window_name: String,
    pub mission: String,
    pub product: String,
    pub row_count: usize,
    pub event_row_count: usize,
    pub event_coverage_fraction: f64,
    pub baseline: Option<f64>,
    pub spread: Option<f64>,
    pub threshold_on: Option<f64>,
    pub threshold_off: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeliosphereTransformResult {
    pub rows: Vec<HeliosphereFeatureRow>,
    pub groups: Vec<HeliosphereTransformGroupStats>,
}

impl HeliosphereFeatureRow {
    pub fn signal_channels(&self) -> [f64; HELIOSPHERE_SIGNAL_DIM] {
        [
            self.r_au,
            self.lat_deg,
            self.lon_deg,
            self.density_cm3,
            self.speed_kms,
            self.temperature_k,
            self.bx,
            self.by,
            self.bz,
            self.b_mag,
            self.crs_flux,
            self.spectral_mean,
            self.spectral_peak,
            self.map_flux_mean,
            self.map_flux_std,
        ]
    }

    pub fn set_signal_channels(&mut self, values: [f64; HELIOSPHERE_SIGNAL_DIM]) {
        self.r_au = values[0];
        self.lat_deg = values[1];
        self.lon_deg = values[2];
        self.density_cm3 = values[3];
        self.speed_kms = values[4];
        self.temperature_k = values[5];
        self.bx = values[6];
        self.by = values[7];
        self.bz = values[8];
        self.b_mag = values[9];
        self.crs_flux = values[10];
        self.spectral_mean = values[11];
        self.spectral_peak = values[12];
        self.map_flux_mean = values[13];
        self.map_flux_std = values[14];
    }

    pub fn dynamic_signal_channels(&self) -> [f64; HELIOSPHERE_DYNAMIC_DIM] {
        [
            self.density_cm3,
            self.speed_kms,
            self.temperature_k,
            self.bx,
            self.by,
            self.bz,
            self.b_mag,
            self.crs_flux,
            self.spectral_mean,
            self.spectral_peak,
            self.map_flux_mean,
            self.map_flux_std,
        ]
    }

    pub fn set_dynamic_signal_channels(&mut self, values: [f64; HELIOSPHERE_DYNAMIC_DIM]) {
        self.density_cm3 = values[0];
        self.speed_kms = values[1];
        self.temperature_k = values[2];
        self.bx = values[3];
        self.by = values[4];
        self.bz = values[5];
        self.b_mag = values[6];
        self.crs_flux = values[7];
        self.spectral_mean = values[8];
        self.spectral_peak = values[9];
        self.map_flux_mean = values[10];
        self.map_flux_std = values[11];
    }

    pub fn algebra_vector(&self) -> [f64; HELIOSPHERE_FEATURE_DIM] {
        fn clean(value: f64) -> f64 {
            if value.is_finite() { value } else { 0.0 }
        }
        [
            clean(self.r_au),
            clean(self.lat_deg),
            clean(self.lon_deg),
            clean(self.density_cm3),
            clean(self.speed_kms),
            clean(self.temperature_k),
            clean(self.bx),
            clean(self.by),
            clean(self.bz),
            clean(self.b_mag),
            clean(self.crs_flux),
            clean(self.spectral_mean),
            clean(self.spectral_peak),
            clean(self.map_flux_mean),
            clean(self.map_flux_std),
            1.0,
        ]
    }

    pub fn algebra_vector_dynamic_bias_free(&self) -> [f64; HELIOSPHERE_FEATURE_DIM] {
        fn clean(value: f64) -> f64 {
            if value.is_finite() { value } else { 0.0 }
        }
        [
            0.0,
            0.0,
            0.0,
            clean(self.density_cm3),
            clean(self.speed_kms),
            clean(self.temperature_k),
            clean(self.bx),
            clean(self.by),
            clean(self.bz),
            clean(self.b_mag),
            clean(self.crs_flux),
            clean(self.spectral_mean),
            clean(self.spectral_peak),
            clean(self.map_flux_mean),
            clean(self.map_flux_std),
            0.0,
        ]
    }

    pub fn signal_energy(&self) -> f64 {
        let vector = self.algebra_vector();
        vector.iter().map(|value| value * value).sum::<f64>().sqrt()
    }

    pub fn event_active(&self) -> bool {
        self.event_mask.unwrap_or(false)
    }
}

/// Convert a feature row timestamp into UTC.
pub fn heliosphere_row_datetime(row: &HeliosphereFeatureRow) -> Option<DateTime<Utc>> {
    let date = NaiveDate::from_yo_opt(row.year as i32, row.doy as u32)?;
    let naive = date.and_hms_opt(row.hour as u32, 0, 0)?;
    Some(DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc))
}

/// Compute invariant and residual channels per mission/product time series.
///
/// This keeps the executed cube schema unchanged while exposing a
/// physics-first channel family for predictive tests and adaptive
/// sparsification.
pub fn compute_invariant_samples(
    rows: &[HeliosphereFeatureRow],
) -> Vec<HeliosphereInvariantSample> {
    let mut grouped: BTreeMap<(String, String, String), Vec<HeliosphereFeatureRow>> =
        BTreeMap::new();
    for row in rows {
        grouped
            .entry((
                row.window_name.clone(),
                row.mission.clone(),
                row.product.clone(),
            ))
            .or_default()
            .push(row.clone());
    }

    let mut output = Vec::with_capacity(rows.len());
    for ((_window, _mission, _product), mut group) in grouped {
        group.sort_by_key(row_sort_key);
        let mut current = group
            .iter()
            .enumerate()
            .map(|(idx, row)| invariant_channels_for_row(&group, idx, row))
            .collect::<Vec<_>>();

        let mut scales = [1.0_f64; HELIOSPHERE_INVARIANT_DIM];
        for channel_idx in 0..HELIOSPHERE_INVARIANT_DIM {
            let column = current
                .iter()
                .map(|value| value[channel_idx])
                .collect::<Vec<_>>();
            let median = finite_median(&column);
            let mad = finite_mad(&column, median);
            let mut scale = 1.4826 * mad;
            if !scale.is_finite() || scale <= 0.0 {
                scale = finite_std(&column, finite_mean(&column));
            }
            if !scale.is_finite() || scale <= 0.0 {
                scale = 1.0;
            }
            scales[channel_idx] = scale;
        }

        for (row, channels) in group.into_iter().zip(current.drain(..)) {
            let mut weighted = [0.0_f64; HELIOSPHERE_INVARIANT_DIM];
            for idx in 0..HELIOSPHERE_INVARIANT_DIM {
                let value = if channels[idx].is_finite() {
                    channels[idx]
                } else {
                    0.0
                };
                weighted[idx] = value / scales[idx];
            }
            let timestamp_utc = heliosphere_row_datetime(&row)
                .map(|value| value.to_rfc3339())
                .unwrap_or_default();
            output.push(HeliosphereInvariantSample {
                window_name: row.window_name,
                mission: row.mission,
                product: row.product,
                year: row.year,
                doy: row.doy,
                hour: row.hour,
                timestamp_utc,
                channels,
                uncertainty_scales: scales,
                weighted_channels: weighted,
                b_field: [
                    finite_or_zero(row.bx),
                    finite_or_zero(row.by),
                    finite_or_zero(row.bz),
                    finite_or_zero(row.b_mag),
                ],
                inherited_event_score: row.event_score,
                inherited_event_mask: row.event_mask,
                inherited_event_segment_id: row.event_segment_id,
            });
        }
    }
    output.sort_by(|a, b| {
        (
            a.window_name.as_str(),
            a.mission.as_str(),
            a.product.as_str(),
            a.year,
            a.doy,
            a.hour,
        )
            .cmp(&(
                b.window_name.as_str(),
                b.mission.as_str(),
                b.product.as_str(),
                b.year,
                b.doy,
                b.hour,
            ))
    });
    output
}

fn invariant_channels_for_row(
    group: &[HeliosphereFeatureRow],
    idx: usize,
    row: &HeliosphereFeatureRow,
) -> [f64; HELIOSPHERE_INVARIANT_DIM] {
    let previous = idx.checked_sub(1).and_then(|prev_idx| group.get(prev_idx));
    let current_b_mag = finite_b_mag(row);
    let previous_b_mag = previous.map(finite_b_mag).unwrap_or(current_b_mag);
    let delta_b_vec = previous.map(|prev| {
        let dx = finite_or_zero(row.bx) - finite_or_zero(prev.bx);
        let dy = finite_or_zero(row.by) - finite_or_zero(prev.by);
        let dz = finite_or_zero(row.bz) - finite_or_zero(prev.bz);
        (dx * dx + dy * dy + dz * dz).sqrt()
    });
    let delta_b_over_bmag = ratio(delta_b_vec.unwrap_or(0.0), current_b_mag);
    let delta_v_over_vmag = previous
        .map(|prev| {
            ratio(
                (finite_or_zero(row.speed_kms) - finite_or_zero(prev.speed_kms)).abs(),
                row.speed_kms,
            )
        })
        .unwrap_or(0.0);
    let delta_n_over_n = previous
        .map(|prev| {
            ratio(
                (finite_or_zero(row.density_cm3) - finite_or_zero(prev.density_cm3)).abs(),
                row.density_cm3,
            )
        })
        .unwrap_or(0.0);
    let delta_t_over_t = previous
        .map(|prev| {
            ratio(
                (finite_or_zero(row.temperature_k) - finite_or_zero(prev.temperature_k)).abs(),
                row.temperature_k,
            )
        })
        .unwrap_or(0.0);
    let plasma_beta = compute_plasma_beta(row.density_cm3, row.temperature_k, current_b_mag);
    let alfven_speed = compute_alfven_speed_kms(row.density_cm3, current_b_mag);
    let alfvenicity_residual = ratio(
        (finite_or_zero(row.speed_kms) - alfven_speed).abs(),
        row.speed_kms,
    );
    let dynamic_pressure = compute_dynamic_pressure_npa(row.density_cm3, row.speed_kms);
    let previous_dynamic_pressure = previous
        .map(|prev| compute_dynamic_pressure_npa(prev.density_cm3, prev.speed_kms))
        .unwrap_or(dynamic_pressure);
    let dynamic_pressure_residual = ratio(
        (dynamic_pressure - previous_dynamic_pressure).abs(),
        dynamic_pressure,
    );
    let magnetic_shear = ratio(delta_b_vec.unwrap_or(0.0), previous_b_mag);
    let compressibility_proxy = 0.5 * (delta_n_over_n + delta_t_over_t);
    [
        delta_b_over_bmag,
        delta_v_over_vmag,
        delta_n_over_n,
        delta_t_over_t,
        plasma_beta,
        alfven_speed,
        alfvenicity_residual,
        dynamic_pressure_residual,
        magnetic_shear,
        compressibility_proxy,
    ]
}

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() { value } else { 0.0 }
}

fn finite_b_mag(row: &HeliosphereFeatureRow) -> f64 {
    if row.b_mag.is_finite() && row.b_mag > 0.0 {
        row.b_mag
    } else {
        let bx = finite_or_zero(row.bx);
        let by = finite_or_zero(row.by);
        let bz = finite_or_zero(row.bz);
        let norm = (bx * bx + by * by + bz * bz).sqrt();
        if norm > 0.0 { norm } else { 1e-6 }
    }
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    let denom = denominator.abs().max(1e-6);
    numerator / denom
}

fn compute_dynamic_pressure_npa(density_cm3: f64, speed_kms: f64) -> f64 {
    if !density_cm3.is_finite() || !speed_kms.is_finite() || density_cm3 <= 0.0 || speed_kms <= 0.0
    {
        return 0.0;
    }
    1.6726219e-6 * density_cm3 * speed_kms * speed_kms
}

fn compute_alfven_speed_kms(density_cm3: f64, b_mag_nt: f64) -> f64 {
    if !density_cm3.is_finite() || !b_mag_nt.is_finite() || density_cm3 <= 0.0 || b_mag_nt <= 0.0 {
        return 0.0;
    }
    21.812 * b_mag_nt / density_cm3.sqrt()
}

fn compute_plasma_beta(density_cm3: f64, temperature_k: f64, b_mag_nt: f64) -> f64 {
    if !density_cm3.is_finite()
        || !temperature_k.is_finite()
        || !b_mag_nt.is_finite()
        || density_cm3 <= 0.0
        || temperature_k <= 0.0
        || b_mag_nt <= 0.0
    {
        return 0.0;
    }
    let number_density_m3 = density_cm3 * 1.0e6;
    let magnetic_field_t = b_mag_nt * 1.0e-9;
    let thermal_pressure = number_density_m3 * 1.380649e-23 * temperature_k;
    let magnetic_pressure =
        magnetic_field_t * magnetic_field_t / (2.0 * std::f64::consts::PI * 4.0e-7);
    if magnetic_pressure <= 0.0 {
        0.0
    } else {
        thermal_pressure / magnetic_pressure
    }
}

fn finite_mean(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return 0.0;
    }
    finite.iter().sum::<f64>() / finite.len() as f64
}

fn finite_std(values: &[f64], mean: f64) -> f64 {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.len() < 2 {
        return 0.0;
    }
    let var = finite
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f64>()
        / finite.len() as f64;
    var.sqrt()
}

fn finite_median(values: &[f64]) -> f64 {
    let mut finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return 0.0;
    }
    finite.sort_by(|a, b| a.total_cmp(b));
    let mid = finite.len() / 2;
    if finite.len().is_multiple_of(2) {
        (finite[mid - 1] + finite[mid]) * 0.5
    } else {
        finite[mid]
    }
}

fn finite_mad(values: &[f64], median: f64) -> f64 {
    let deviations: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| (value - median).abs())
        .collect();
    finite_median(&deviations)
}

fn normalize_channel(value: f64, mean: f64, std: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    if !mean.is_finite() || std <= 0.0 || !std.is_finite() {
        return 0.0;
    }
    (value - mean) / std
}

fn row_sort_key(row: &HeliosphereFeatureRow) -> (u16, u16, u8) {
    (row.year, row.doy, row.hour)
}

pub fn transform_feature_rows(
    rows: &[HeliosphereFeatureRow],
    mode: HeliosphereTransformMode,
) -> Vec<HeliosphereFeatureRow> {
    transform_feature_rows_with_stats(rows, mode).rows
}

pub fn transform_feature_rows_with_stats(
    rows: &[HeliosphereFeatureRow],
    mode: HeliosphereTransformMode,
) -> HeliosphereTransformResult {
    let mut grouped: BTreeMap<(String, String, String), Vec<HeliosphereFeatureRow>> =
        BTreeMap::new();
    for row in rows {
        grouped
            .entry((
                row.window_name.clone(),
                row.mission.clone(),
                row.product.clone(),
            ))
            .or_default()
            .push(row.clone());
    }

    let mut output = Vec::with_capacity(rows.len());
    let mut stats = Vec::with_capacity(grouped.len());
    for ((_window, _mission, _product), mut group) in grouped {
        group.sort_by_key(row_sort_key);
        let (base_rows, group_stats) = match mode {
            HeliosphereTransformMode::Raw => (group.clone(), summarize_group_stats(&group, None)),
            HeliosphereTransformMode::Differenced => {
                (difference_rows(&group), summarize_group_stats(&group, None))
            }
            HeliosphereTransformMode::Normalized => {
                (normalize_rows(&group), summarize_group_stats(&group, None))
            }
            HeliosphereTransformMode::DifferencedNormalized => (
                normalize_rows(&difference_rows(&group)),
                summarize_group_stats(&group, None),
            ),
            HeliosphereTransformMode::RobustCentered => robust_transform_rows(&group, false),
            HeliosphereTransformMode::RobustDifferencedCentered => {
                robust_transform_rows(&group, true)
            }
        };
        output.extend(base_rows);
        stats.push(group_stats);
    }
    output.sort_by(|a, b| {
        (
            a.window_name.as_str(),
            a.mission.as_str(),
            a.product.as_str(),
            row_sort_key(a),
        )
            .cmp(&(
                b.window_name.as_str(),
                b.mission.as_str(),
                b.product.as_str(),
                row_sort_key(b),
            ))
    });
    stats.sort_by(|a, b| {
        (
            a.window_name.as_str(),
            a.mission.as_str(),
            a.product.as_str(),
        )
            .cmp(&(
                b.window_name.as_str(),
                b.mission.as_str(),
                b.product.as_str(),
            ))
    });
    HeliosphereTransformResult {
        rows: output,
        groups: stats,
    }
}

fn normalize_rows(rows: &[HeliosphereFeatureRow]) -> Vec<HeliosphereFeatureRow> {
    let mut columns = (0..HELIOSPHERE_SIGNAL_DIM)
        .map(|_| Vec::with_capacity(rows.len()))
        .collect::<Vec<_>>();
    for row in rows {
        let values = row.signal_channels();
        for idx in 0..HELIOSPHERE_SIGNAL_DIM {
            columns[idx].push(values[idx]);
        }
    }
    let means = columns
        .iter()
        .map(|col| finite_mean(col))
        .collect::<Vec<_>>();
    let stds = columns
        .iter()
        .zip(means.iter().copied())
        .map(|(col, mean)| finite_std(col, mean))
        .collect::<Vec<_>>();

    rows.iter()
        .cloned()
        .map(|mut row| {
            let values = row.signal_channels();
            let mut out = [0.0_f64; HELIOSPHERE_SIGNAL_DIM];
            for idx in 0..HELIOSPHERE_SIGNAL_DIM {
                out[idx] = normalize_channel(values[idx], means[idx], stds[idx]);
            }
            row.set_signal_channels(out);
            row
        })
        .collect()
}

fn difference_rows(rows: &[HeliosphereFeatureRow]) -> Vec<HeliosphereFeatureRow> {
    let mut previous: Option<[f64; HELIOSPHERE_SIGNAL_DIM]> = None;
    let mut output = Vec::with_capacity(rows.len());
    for row in rows.iter().cloned() {
        let current = row.signal_channels();
        let mut diff = [0.0_f64; HELIOSPHERE_SIGNAL_DIM];
        if let Some(prev) = previous {
            for idx in 0..HELIOSPHERE_SIGNAL_DIM {
                let lhs = if current[idx].is_finite() {
                    current[idx]
                } else {
                    0.0
                };
                let rhs = if prev[idx].is_finite() {
                    prev[idx]
                } else {
                    0.0
                };
                diff[idx] = lhs - rhs;
            }
        }
        let mut transformed = row;
        transformed.set_signal_channels(diff);
        output.push(transformed);
        previous = Some(current);
    }
    output
}

fn summarize_group_stats(
    rows: &[HeliosphereFeatureRow],
    thresholds: Option<(f64, f64, f64, f64)>,
) -> HeliosphereTransformGroupStats {
    let sample = rows.first().cloned().unwrap_or(HeliosphereFeatureRow {
        window_name: String::new(),
        mission: String::new(),
        product: String::new(),
        year: 0,
        doy: 0,
        hour: 0,
        r_au: 0.0,
        lat_deg: 0.0,
        lon_deg: 0.0,
        density_cm3: 0.0,
        speed_kms: 0.0,
        temperature_k: 0.0,
        bx: 0.0,
        by: 0.0,
        bz: 0.0,
        b_mag: 0.0,
        crs_flux: 0.0,
        spectral_mean: 0.0,
        spectral_peak: 0.0,
        map_flux_mean: 0.0,
        map_flux_std: 0.0,
        event_score: None,
        event_mask: None,
        event_segment_id: None,
    });
    let event_row_count = rows.iter().filter(|row| row.event_active()).count();
    HeliosphereTransformGroupStats {
        window_name: sample.window_name,
        mission: sample.mission,
        product: sample.product,
        row_count: rows.len(),
        event_row_count,
        event_coverage_fraction: if rows.is_empty() {
            0.0
        } else {
            event_row_count as f64 / rows.len() as f64
        },
        baseline: thresholds.map(|value| value.0),
        spread: thresholds.map(|value| value.1),
        threshold_on: thresholds.map(|value| value.2),
        threshold_off: thresholds.map(|value| value.3),
    }
}

fn robust_transform_rows(
    rows: &[HeliosphereFeatureRow],
    differenced: bool,
) -> (Vec<HeliosphereFeatureRow>, HeliosphereTransformGroupStats) {
    let mut dynamic_series = Vec::with_capacity(rows.len());
    let mut previous: Option<[f64; HELIOSPHERE_DYNAMIC_DIM]> = None;
    for row in rows {
        let current = row.dynamic_signal_channels();
        let dynamic = if differenced {
            let mut diff = [0.0_f64; HELIOSPHERE_DYNAMIC_DIM];
            if let Some(prev) = previous {
                for idx in 0..HELIOSPHERE_DYNAMIC_DIM {
                    let lhs = if current[idx].is_finite() {
                        current[idx]
                    } else {
                        0.0
                    };
                    let rhs = if prev[idx].is_finite() {
                        prev[idx]
                    } else {
                        0.0
                    };
                    diff[idx] = lhs - rhs;
                }
            }
            diff
        } else {
            current.map(|value| if value.is_finite() { value } else { 0.0 })
        };
        dynamic_series.push(dynamic);
        previous = Some(current);
    }

    let mut medians = [0.0_f64; HELIOSPHERE_DYNAMIC_DIM];
    let mut scales = [1.0_f64; HELIOSPHERE_DYNAMIC_DIM];
    for idx in 0..HELIOSPHERE_DYNAMIC_DIM {
        let column = dynamic_series
            .iter()
            .map(|row| row[idx])
            .collect::<Vec<_>>();
        let median = finite_median(&column);
        let mad = finite_mad(&column, median);
        let mut scale = 1.4826 * mad;
        if !scale.is_finite() || scale <= 0.0 {
            scale = finite_std(&column, finite_mean(&column));
        }
        if !scale.is_finite() || scale <= 0.0 {
            scale = 1.0;
        }
        medians[idx] = median;
        scales[idx] = scale;
    }

    let mut transformed = rows.to_vec();
    let mut event_scores = Vec::with_capacity(rows.len());
    for (row, dynamic) in transformed.iter_mut().zip(dynamic_series.iter()) {
        let mut out = [0.0_f64; HELIOSPHERE_DYNAMIC_DIM];
        for idx in 0..HELIOSPHERE_DYNAMIC_DIM {
            let centered = (dynamic[idx] - medians[idx]) / scales[idx];
            out[idx] = centered.clamp(-8.0, 8.0);
        }
        row.set_dynamic_signal_channels(out);
        event_scores.push(rms_energy(&out));
    }

    let smoothed_scores = median_filter3(&event_scores);
    let thresholds = compute_event_thresholds(&smoothed_scores);
    let event_mask = if transformed.len() < 4 {
        vec![true; transformed.len()]
    } else {
        build_event_mask(&smoothed_scores, thresholds.2, thresholds.3)
    };
    let event_segment_ids = build_event_segment_ids(&event_mask);
    for idx in 0..transformed.len() {
        transformed[idx].event_score = Some(smoothed_scores[idx]);
        transformed[idx].event_mask = Some(event_mask[idx]);
        transformed[idx].event_segment_id = event_segment_ids[idx];
    }

    (
        transformed.clone(),
        summarize_group_stats(&transformed, Some(thresholds)),
    )
}

fn rms_energy(values: &[f64; HELIOSPHERE_DYNAMIC_DIM]) -> f64 {
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for value in values {
        if value.is_finite() {
            sum_sq += value * value;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum_sq / count as f64).sqrt()
    }
}

fn median_filter3(values: &[f64]) -> Vec<f64> {
    let mut output = Vec::with_capacity(values.len());
    for idx in 0..values.len() {
        let start = idx.saturating_sub(1);
        let end = (idx + 2).min(values.len());
        output.push(finite_median(&values[start..end]));
    }
    output
}

fn compute_event_thresholds(values: &[f64]) -> (f64, f64, f64, f64) {
    let baseline = finite_median(values);
    let mad = finite_mad(values, baseline);
    let mut spread = 1.4826 * mad;
    if !spread.is_finite() || spread <= 0.0 {
        spread = finite_std(values, finite_mean(values));
    }
    if !spread.is_finite() || spread <= 0.0 {
        spread = 1.0;
    }
    let threshold_on = baseline + EVENT_THRESHOLD_ON_MULTIPLIER * spread;
    let threshold_off = baseline + EVENT_THRESHOLD_OFF_MULTIPLIER * spread;
    (baseline, spread, threshold_on, threshold_off)
}

fn build_event_mask(values: &[f64], threshold_on: f64, threshold_off: f64) -> Vec<bool> {
    let mut mask = Vec::with_capacity(values.len());
    let mut active = false;
    for value in values {
        let finite = if value.is_finite() { *value } else { 0.0 };
        if !active && finite >= threshold_on {
            active = true;
        } else if active && finite < threshold_off {
            active = false;
        }
        mask.push(active);
    }
    let mut dilated = mask.clone();
    for idx in 0..mask.len() {
        if mask[idx] {
            if idx > 0 {
                dilated[idx - 1] = true;
            }
            dilated[idx] = true;
            if idx + 1 < mask.len() {
                dilated[idx + 1] = true;
            }
        }
    }

    let mut merged = dilated.clone();
    let mut idx = 0usize;
    while idx < merged.len() {
        if merged[idx] {
            idx += 1;
            continue;
        }
        let start = idx;
        while idx < merged.len() && !merged[idx] {
            idx += 1;
        }
        let end = idx;
        let gap = end - start;
        let left_active = start > 0 && merged[start - 1];
        let right_active = end < merged.len() && merged[end];
        if gap <= 1 && left_active && right_active {
            for value in &mut merged[start..end] {
                *value = true;
            }
        }
    }
    merged
}

fn build_event_segment_ids(mask: &[bool]) -> Vec<Option<u32>> {
    let mut out = vec![None; mask.len()];
    let mut current_segment = 0u32;
    let mut in_segment = false;
    for (idx, active) in mask.iter().copied().enumerate() {
        if active {
            if !in_segment {
                current_segment += 1;
                in_segment = true;
            }
            out[idx] = Some(current_segment);
        } else {
            in_segment = false;
        }
    }
    out
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeliosphereFeatureCubeManifest {
    pub window_name: String,
    pub generated_at_utc: String,
    pub temporal_start_utc: Option<String>,
    pub temporal_end_utc: Option<String>,
    pub row_count: usize,
    pub missions: Vec<String>,
    pub products: Vec<String>,
    pub channel_names: Vec<String>,
    pub source_paths: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeliosphereFeatureCube {
    pub manifest: HeliosphereFeatureCubeManifest,
    pub rows: Vec<HeliosphereFeatureRow>,
}

/// Sparse-memory sizing estimate for a brick-mapped D3Q19 LBM domain.
///
/// The planner assumes:
/// - an `8^3` active brick core
/// - a `10^3` halo-loaded shared-memory tile
/// - GPU storage in SoA order (`[19][1000]`) for coalesced reads
/// - CPU fallback storage in AoS order (`[1000][19]`) for cache-friendly
///   scalar or SIMD stepping
///
/// These values are used by higher-level execution planning to decide whether
/// a window should stay fully resident in VRAM, run as cache-aware temporal
/// tiles, or fall back to managed memory / CPU execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMemoryPlan {
    /// Cubic grid edge length.
    pub grid_size: usize,
    /// Active brick edge in cells. The current sparse solver uses `8`.
    pub brick_edge: usize,
    /// Shared-memory halo tile edge in cells. The current tiled path uses `10`.
    pub halo_edge: usize,
    /// Occupancy fraction used to estimate active bricks.
    pub active_fraction: f64,
    /// Total dense-cell count in the logical domain.
    pub total_cells: u64,
    /// Total number of `8^3` bricks in the logical domain.
    pub total_bricks: u64,
    /// Estimated number of active bricks after sparsification.
    pub active_bricks: u64,
    /// Total active core cells (`active_bricks * 8^3`).
    pub active_core_cells: u64,
    /// Dense FP32 ping-pong footprint in GiB.
    pub dense_fp32_pingpong_gib: f64,
    /// Dense BF16 ping-pong footprint in GiB.
    pub dense_bf16_pingpong_gib: f64,
    /// Sparse FP32 A-A footprint in GiB.
    pub sparse_fp32_aa_gib: f64,
    /// Sparse BF16 A-A footprint in GiB.
    pub sparse_bf16_aa_projected_gib: f64,
    /// Occupancy bitset footprint in MiB.
    pub occupancy_bitset_mib: f64,
    /// Indirect brick table footprint in MiB.
    pub indirect_table_mib: f64,
    /// Active-brick ID list footprint in MiB.
    pub active_brick_id_mib: f64,
    /// Shared-memory tile bytes for BF16 D3Q19 halo execution.
    pub shared_tile_bytes_bf16: usize,
    /// Preferred GPU shared-memory tile layout.
    pub shared_tile_layout_gpu: String,
    /// Preferred CPU cache-local tile layout.
    pub shared_tile_layout_cpu: String,
}

/// Hardware envelope used to choose a sparse execution mode.
///
/// This struct intentionally separates:
/// - device-local facts such as VRAM, L2, shared memory, and managed-memory
///   support
/// - host-local facts such as safe CPU L3 working-set size
///
/// The CPU L3 cache is not treated as a shared GPU cache. It only informs
/// host-side compaction, event scoring, temporal tiling, and managed-memory
/// orchestration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct SparseHardwareEnvelope {
    /// Device-local memory budget in bytes.
    pub cuda_vram_budget_bytes: Option<usize>,
    /// GPU L2 cache size in bytes.
    pub cuda_l2_bytes: Option<usize>,
    /// Shared memory available per block in bytes.
    pub cuda_shared_mem_per_block: Option<usize>,
    /// Whether CUDA managed memory is available.
    pub cuda_managed_memory: Option<bool>,
    /// Whether CUDA concurrent managed access is available.
    pub cuda_concurrent_managed_access: Option<bool>,
    /// Safe host L3 working-set size in bytes.
    pub cpu_l3_safe_working_set_bytes: Option<usize>,
    /// Whether the GPU is a good match for sparse shared-tile kernels.
    pub prefer_sparse_tile: bool,
}

/// Recommended sparse execution mode for a planned domain.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SparseExecutionMode {
    /// The full sparse domain fits in VRAM and should stay device resident.
    VramResidentSparseSoA,
    /// The full sparse domain fits in VRAM and the metadata/tile footprints
    /// align well with the GPU cache and shared-memory geometry.
    CacheAwareTiledSparseSoA,
    /// The full logical window is too large, but per-segment temporal tiles fit
    /// in VRAM while preserving the full cube artifact on disk.
    TemporalTiledSparseSoA,
    /// Managed / unified memory with explicit prefetch is the required fallback.
    ///
    /// This is treated as slower than a VRAM-resident path and should not be the
    /// primary fast path on Ada-class devices.
    ManagedPrefetchFallback,
    /// CPU fallback using cache-friendly AoS tiles.
    CpuAosFallback,
}

/// Fully annotated sparse execution recommendation for a domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseExecutionPlan {
    /// Base memory-size estimate for the sparse domain.
    pub memory: SparseMemoryPlan,
    /// Recommended execution mode.
    pub mode: SparseExecutionMode,
    /// Whether the whole sparse BF16 A-A domain fits the declared VRAM budget.
    pub fits_vram_budget: Option<bool>,
    /// Whether the sparse metadata hotset fits in GPU L2.
    pub metadata_fits_gpu_l2: Option<bool>,
    /// Whether a single host orchestration tile fits in safe CPU L3.
    pub host_tile_fits_cpu_l3: Option<bool>,
    /// Total sparse metadata hotset size in MiB.
    pub metadata_hotset_mib: f64,
    /// Host-side orchestration target in MiB.
    pub host_orchestration_mib: f64,
    /// Recommended number of temporal tiles.
    pub recommended_temporal_tiles: usize,
    /// Peak BF16 A-A tile footprint in GiB when temporal tiling is used.
    pub peak_temporal_tile_gib: f64,
    /// Whether managed-memory fallback is supported by the device/runtime.
    pub managed_prefetch_supported: bool,
    /// Whether managed-memory fallback is recommended for this domain.
    pub managed_prefetch_recommended: bool,
    /// Whether low-level VMM is worth considering if managed-memory fallback is
    /// still too large.
    pub vmm_candidate: bool,
    /// BAR/ReBAR note to make the fallback semantics explicit in reports.
    pub rebar_note: String,
    /// Human-readable planner notes.
    pub notes: Vec<String>,
}

/// Estimate the raw sparse-memory footprint for a grid and active fraction.
pub fn estimate_sparse_memory_plan(grid_size: usize, active_fraction: f64) -> SparseMemoryPlan {
    let brick_shape = BrickShape3d {
        core_edge_cells: 8,
        halo_edge_cells: 10,
    };
    let logical_grid = LogicalGrid3d {
        nx: grid_size as u32,
        ny: grid_size as u32,
        nz: grid_size as u32,
    };
    let brick_grid = BrickGrid3d::from_logical_grid(logical_grid, brick_shape);
    let active_fraction = active_fraction.clamp(0.001, 1.0);
    let total_cells = logical_grid.cell_count();
    let total_bricks = brick_grid.total_bricks();
    let active_bricks = ((total_bricks as f64) * active_fraction).ceil() as u64;
    let occupancy = OccupancyBitsetStats {
        total_bricks,
        active_bricks,
    };
    let indirect_table = IndirectBrickTableShape {
        entry_count: total_bricks,
        bytes_per_entry: 4,
    };
    let metadata = estimate_metadata_footprint(occupancy, indirect_table, 4);
    let active_core_cells = active_bricks * brick_shape.core_cell_count();

    let dense_fp32_pingpong_bytes = total_cells as f64 * 184.0;
    let dense_bf16_pingpong_bytes = total_cells as f64 * 108.0;
    let sparse_fp32_aa_bytes = active_core_cells as f64 * 108.0;
    let sparse_bf16_aa_projected_bytes = active_core_cells as f64 * 70.0;

    SparseMemoryPlan {
        grid_size,
        brick_edge: brick_shape.core_edge_cells as usize,
        halo_edge: brick_shape.halo_edge_cells as usize,
        active_fraction,
        total_cells,
        total_bricks,
        active_bricks,
        active_core_cells,
        dense_fp32_pingpong_gib: dense_fp32_pingpong_bytes / 1024.0_f64.powi(3),
        dense_bf16_pingpong_gib: dense_bf16_pingpong_bytes / 1024.0_f64.powi(3),
        sparse_fp32_aa_gib: sparse_fp32_aa_bytes / 1024.0_f64.powi(3),
        sparse_bf16_aa_projected_gib: sparse_bf16_aa_projected_bytes / 1024.0_f64.powi(3),
        occupancy_bitset_mib: metadata.occupancy_bitset_bytes as f64 / 1024.0_f64.powi(2),
        indirect_table_mib: metadata.indirect_table_bytes as f64 / 1024.0_f64.powi(2),
        active_brick_id_mib: metadata.active_brick_id_bytes as f64 / 1024.0_f64.powi(2),
        shared_tile_bytes_bf16: (brick_shape.halo_cell_count() * 19 * 2) as usize,
        shared_tile_layout_gpu: "[19][1000]".to_string(),
        shared_tile_layout_cpu: "[1000][19]".to_string(),
    }
}

/// Estimate an execution strategy for a sparse heliosphere LBM window.
///
/// This planner is intentionally conservative:
/// - ReBAR/BAR1 is not treated as an extension of VRAM.
/// - GPU execution remains SoA-first.
/// - CPU fallback remains AoS-first.
/// - Managed memory is modeled as a slower overflow path with explicit
///   prefetch/tiling, not a peer of VRAM-resident execution.
///
/// `temporal_tile_fraction` lets callers preserve the full cube artifact while
/// estimating the peak per-tile footprint needed for sequential execution.
pub fn estimate_sparse_execution_plan(
    grid_size: usize,
    active_fraction: f64,
    temporal_tile_fraction: Option<f64>,
    hardware: SparseHardwareEnvelope,
) -> SparseExecutionPlan {
    let memory = estimate_sparse_memory_plan(grid_size, active_fraction);
    let metadata_hotset_mib =
        memory.occupancy_bitset_mib + memory.indirect_table_mib + memory.active_brick_id_mib;
    let host_orchestration_mib =
        metadata_hotset_mib + (memory.shared_tile_bytes_bf16 as f64 / 1024.0_f64.powi(2)).max(1.0);
    let metadata_fits_gpu_l2 = hardware
        .cuda_l2_bytes
        .map(|bytes| metadata_hotset_mib <= bytes as f64 / 1024.0_f64.powi(2));
    let host_tile_fits_cpu_l3 = hardware
        .cpu_l3_safe_working_set_bytes
        .map(|bytes| host_orchestration_mib <= bytes as f64 / 1024.0_f64.powi(2));
    let fits_vram_budget = hardware
        .cuda_vram_budget_bytes
        .map(|bytes| memory.sparse_bf16_aa_projected_gib <= bytes as f64 / 1024.0_f64.powi(3));
    let temporal_tile_fraction = temporal_tile_fraction
        .unwrap_or(active_fraction)
        .clamp(0.000_001, 1.0);
    let peak_temporal_tile = estimate_sparse_memory_plan(grid_size, temporal_tile_fraction);
    let recommended_temporal_tiles = if temporal_tile_fraction >= active_fraction {
        1
    } else {
        (active_fraction / temporal_tile_fraction).ceil().max(1.0) as usize
    };
    let managed_prefetch_supported = hardware.cuda_managed_memory.unwrap_or(false)
        && hardware.cuda_concurrent_managed_access.unwrap_or(false);

    let mode = match fits_vram_budget {
        Some(true) if hardware.prefer_sparse_tile && metadata_fits_gpu_l2.unwrap_or(false) => {
            SparseExecutionMode::CacheAwareTiledSparseSoA
        }
        Some(true) => SparseExecutionMode::VramResidentSparseSoA,
        Some(false)
            if hardware.cuda_vram_budget_bytes.is_some()
                && peak_temporal_tile.sparse_bf16_aa_projected_gib
                    <= hardware.cuda_vram_budget_bytes.unwrap_or_default() as f64
                        / 1024.0_f64.powi(3) =>
        {
            SparseExecutionMode::TemporalTiledSparseSoA
        }
        Some(false) if managed_prefetch_supported => SparseExecutionMode::ManagedPrefetchFallback,
        _ => SparseExecutionMode::CpuAosFallback,
    };

    let mut notes = Vec::new();
    if metadata_fits_gpu_l2 == Some(true) {
        notes.push("sparse metadata hotset fits GPU L2".to_string());
    } else if metadata_fits_gpu_l2 == Some(false) {
        notes.push("sparse metadata hotset exceeds GPU L2; favor temporal tiling".to_string());
    }
    if host_tile_fits_cpu_l3 == Some(true) {
        notes.push("host orchestration tile fits safe CPU L3 working set".to_string());
    }
    if matches!(mode, SparseExecutionMode::ManagedPrefetchFallback) {
        notes.push(
            "managed-memory overflow is available, but should be treated as slower than VRAM"
                .to_string(),
        );
    }
    if matches!(mode, SparseExecutionMode::TemporalTiledSparseSoA) {
        notes.push(format!(
            "full window can be preserved while processing roughly {} temporal tiles",
            recommended_temporal_tiles
        ));
    }

    SparseExecutionPlan {
        memory,
        mode,
        fits_vram_budget,
        metadata_fits_gpu_l2,
        host_tile_fits_cpu_l3,
        metadata_hotset_mib,
        host_orchestration_mib,
        recommended_temporal_tiles,
        peak_temporal_tile_gib: peak_temporal_tile.sparse_bf16_aa_projected_gib,
        managed_prefetch_supported,
        managed_prefetch_recommended: matches!(mode, SparseExecutionMode::ManagedPrefetchFallback),
        vmm_candidate: matches!(mode, SparseExecutionMode::ManagedPrefetchFallback),
        rebar_note:
            "BAR/ReBAR is not counted as VRAM capacity; use managed memory or temporal tiling for overflow".to_string(),
        notes,
    }
}

/// Build magnetic Takens embedding vectors of dimension `dim` from a
/// time-ordered sequence of [`HeliosphereFeatureRow`].
///
/// `dim` must be a positive multiple of 4 and a power of 2 (16, 32, 64, ...).
/// `lag_steps` controls the temporal spacing between samples in the delay
/// window (1 = consecutive hourly steps, 2 = every other hour, etc.).
///
/// The sliding window spans `(dim/4 - 1) * lag_steps + 1` rows.
/// Each window produces one embedded vector with 4 channels per time step:
/// `Bx/mean_B, By/mean_B, Bz/mean_B, (|B| - mean_B)/mean_B`.
///
/// Returns `(embedded_vectors, metadata_indices)` where `metadata_indices[k]`
/// is the row index of the *last* sample in the window that produced
/// `embedded_vectors[k]`. Callers use `metadata_indices[k]` to retrieve
/// spatial tags (r_au, lat_deg, etc.) from the original row slice.
pub fn magnetic_takens_embed(
    rows: &[HeliosphereFeatureRow],
    dim: usize,
    lag_steps: usize,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    assert!(
        dim >= 4 && dim.is_power_of_two(),
        "dim must be a power-of-2 >= 4, got {dim}"
    );
    assert!(lag_steps >= 1, "lag_steps must be >= 1, got {lag_steps}");
    let channels: usize = 4;
    let steps = dim / channels;
    let window_rows = (steps - 1) * lag_steps + 1;

    let mut embedded = Vec::new();
    let mut indices = Vec::new();

    if rows.len() < window_rows {
        return (embedded, indices);
    }

    for w_start in 0..=(rows.len() - window_rows) {
        let sample_indices: Vec<usize> = (0..steps).map(|s| w_start + s * lag_steps).collect();

        let sum_b: f64 = sample_indices.iter().map(|&i| rows[i].b_mag).sum();
        let local_mean_b = sum_b / steps as f64;
        if local_mean_b <= 0.0 {
            continue;
        }

        let mut v = vec![0.0; dim];
        for (s, &ri) in sample_indices.iter().enumerate() {
            v[s * channels] = rows[ri].bx / local_mean_b;
            v[s * channels + 1] = rows[ri].by / local_mean_b;
            v[s * channels + 2] = rows[ri].bz / local_mean_b;
            v[s * channels + 3] = (rows[ri].b_mag - local_mean_b) / local_mean_b;
        }
        embedded.push(v);
        indices.push(*sample_indices.last().unwrap());
    }

    (embedded, indices)
}

/// Build mixed magnetic+plasma Takens embedding (32D = 4 steps x 8 channels).
///
/// Channels per step: Bx/mean_B, By/mean_B, Bz/mean_B, (|B|-mean_B)/mean_B,
/// n_p/mean_n, v_sw/mean_v, T_p/mean_T, v_A/mean_vA
/// where v_A = |B| * 21.81 / sqrt(n_p) [nT, cm^-3 -> km/s].
///
/// **Missingness policy**: NO zero-filling. Only rows with ALL 8 channels
/// finite and positive are eligible. Returns `(vectors, indices, eligible_mask)`
/// where `eligible_mask[i]` is true if row `i` was usable.
pub fn magnetic_plasma_takens_embed(
    rows: &[HeliosphereFeatureRow],
    lag_steps: usize,
) -> (Vec<Vec<f64>>, Vec<usize>, Vec<bool>) {
    let channels: usize = 8;
    let steps: usize = 4;
    let dim = channels * steps; // 32
    let window_rows = (steps - 1) * lag_steps + 1;

    let mut embedded = Vec::new();
    let mut indices = Vec::new();

    // Build eligibility mask: all 8 channels must be finite and positive
    let eligible: Vec<bool> = rows
        .iter()
        .map(|r| {
            let va = if r.density_cm3 > 0.0 {
                r.b_mag * 21.81 / r.density_cm3.sqrt()
            } else {
                f64::NAN
            };
            r.bx.is_finite()
                && r.by.is_finite()
                && r.bz.is_finite()
                && r.b_mag.is_finite()
                && r.b_mag > 0.0
                && r.density_cm3.is_finite()
                && r.density_cm3 > 0.0
                && r.speed_kms.is_finite()
                && r.speed_kms > 0.0
                && r.temperature_k.is_finite()
                && r.temperature_k > 0.0
                && va.is_finite()
        })
        .collect();

    if rows.len() < window_rows {
        return (embedded, indices, eligible);
    }

    for w_start in 0..=(rows.len() - window_rows) {
        let sample_indices: Vec<usize> = (0..steps).map(|s| w_start + s * lag_steps).collect();

        // ALL samples in window must be eligible
        if !sample_indices.iter().all(|&i| eligible[i]) {
            continue;
        }

        // Compute local means for normalization
        let mean_b: f64 = sample_indices.iter().map(|&i| rows[i].b_mag).sum::<f64>() / steps as f64;
        let mean_n: f64 = sample_indices
            .iter()
            .map(|&i| rows[i].density_cm3)
            .sum::<f64>()
            / steps as f64;
        let mean_v: f64 = sample_indices
            .iter()
            .map(|&i| rows[i].speed_kms)
            .sum::<f64>()
            / steps as f64;
        let mean_t: f64 = sample_indices
            .iter()
            .map(|&i| rows[i].temperature_k)
            .sum::<f64>()
            / steps as f64;
        let alfven_speeds: Vec<f64> = sample_indices
            .iter()
            .map(|&i| rows[i].b_mag * 21.81 / rows[i].density_cm3.sqrt())
            .collect();
        let mean_va: f64 = alfven_speeds.iter().sum::<f64>() / steps as f64;

        if mean_b <= 0.0 || mean_n <= 0.0 || mean_v <= 0.0 || mean_t <= 0.0 || mean_va <= 0.0 {
            continue;
        }

        let mut v = vec![0.0; dim];
        for (s, &ri) in sample_indices.iter().enumerate() {
            let r = &rows[ri];
            v[s * channels] = r.bx / mean_b;
            v[s * channels + 1] = r.by / mean_b;
            v[s * channels + 2] = r.bz / mean_b;
            v[s * channels + 3] = (r.b_mag - mean_b) / mean_b;
            v[s * channels + 4] = r.density_cm3 / mean_n;
            v[s * channels + 5] = r.speed_kms / mean_v;
            v[s * channels + 6] = r.temperature_k / mean_t;
            v[s * channels + 7] = alfven_speeds[s] / mean_va;
        }
        embedded.push(v);
        indices.push(*sample_indices.last().unwrap());
    }

    (embedded, indices, eligible)
}

/// Build dimension-parameterized mixed magnetic+plasma Takens embedding.
///
/// 8 channels per step: Bx/mean_B, By/mean_B, Bz/mean_B, (|B|-mean_B)/mean_B,
/// n_p/mean_n, v_sw/mean_v, T_p/mean_T, v_A/mean_vA.
///
/// dim must be a multiple of 8 and a power of 2. steps = dim / 8.
///
/// At dim=32: same as magnetic_plasma_takens_embed (4 steps x 8 channels).
/// At dim=64: 8 steps x 8 channels = wider temporal window.
/// At dim=128: 16 steps x 8 channels = 16 time delays with 8 independent channels.
///
/// The key advantage over 4-channel delay embedding: 8 genuinely independent
/// channels means the SVD effective rank scales as ~8*steps instead of ~4*min(steps, T_corr).
pub fn plasma_takens_embed_dim(
    rows: &[HeliosphereFeatureRow],
    dim: usize,
    lag_steps: usize,
) -> (Vec<Vec<f64>>, Vec<usize>, Vec<bool>) {
    let channels: usize = 8;
    assert!(
        dim >= 8 && dim.is_power_of_two() && dim.is_multiple_of(channels),
        "dim must be a power-of-2 multiple of 8, got {dim}"
    );
    let steps = dim / channels;
    let window_rows = (steps - 1) * lag_steps + 1;

    let mut embedded = Vec::new();
    let mut indices = Vec::new();

    let eligible: Vec<bool> = rows
        .iter()
        .map(|r| {
            let va = if r.density_cm3 > 0.0 {
                r.b_mag * 21.81 / r.density_cm3.sqrt()
            } else {
                f64::NAN
            };
            r.bx.is_finite()
                && r.by.is_finite()
                && r.bz.is_finite()
                && r.b_mag.is_finite()
                && r.b_mag > 0.0
                && r.density_cm3.is_finite()
                && r.density_cm3 > 0.0
                && r.speed_kms.is_finite()
                && r.speed_kms > 0.0
                && r.temperature_k.is_finite()
                && r.temperature_k > 0.0
                && va.is_finite()
        })
        .collect();

    if rows.len() < window_rows {
        return (embedded, indices, eligible);
    }

    for w_start in 0..=(rows.len() - window_rows) {
        let sample_indices: Vec<usize> = (0..steps).map(|s| w_start + s * lag_steps).collect();

        if !sample_indices.iter().all(|&i| eligible[i]) {
            continue;
        }

        let mean_b: f64 = sample_indices.iter().map(|&i| rows[i].b_mag).sum::<f64>() / steps as f64;
        let mean_n: f64 = sample_indices
            .iter()
            .map(|&i| rows[i].density_cm3)
            .sum::<f64>()
            / steps as f64;
        let mean_v: f64 = sample_indices
            .iter()
            .map(|&i| rows[i].speed_kms)
            .sum::<f64>()
            / steps as f64;
        let mean_t: f64 = sample_indices
            .iter()
            .map(|&i| rows[i].temperature_k)
            .sum::<f64>()
            / steps as f64;
        let alfven_speeds: Vec<f64> = sample_indices
            .iter()
            .map(|&i| rows[i].b_mag * 21.81 / rows[i].density_cm3.sqrt())
            .collect();
        let mean_va: f64 = alfven_speeds.iter().sum::<f64>() / steps as f64;

        if mean_b <= 0.0 || mean_n <= 0.0 || mean_v <= 0.0 || mean_t <= 0.0 || mean_va <= 0.0 {
            continue;
        }

        let mut v = vec![0.0; dim];
        for (s, &ri) in sample_indices.iter().enumerate() {
            let r = &rows[ri];
            v[s * channels] = r.bx / mean_b;
            v[s * channels + 1] = r.by / mean_b;
            v[s * channels + 2] = r.bz / mean_b;
            v[s * channels + 3] = (r.b_mag - mean_b) / mean_b;
            v[s * channels + 4] = r.density_cm3 / mean_n;
            v[s * channels + 5] = r.speed_kms / mean_v;
            v[s * channels + 6] = r.temperature_k / mean_t;
            v[s * channels + 7] = alfven_speeds[s] / mean_va;
        }
        embedded.push(v);
        indices.push(*sample_indices.last().unwrap());
    }

    (embedded, indices, eligible)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row(
        hour: u8,
        density_cm3: f64,
        speed_kms: f64,
        temperature_k: f64,
    ) -> HeliosphereFeatureRow {
        HeliosphereFeatureRow {
            window_name: "w".to_string(),
            mission: "m".to_string(),
            product: "p".to_string(),
            year: 2020,
            doy: 1,
            hour,
            r_au: 1.0 + hour as f64,
            lat_deg: 2.0,
            lon_deg: 3.0,
            density_cm3,
            speed_kms,
            temperature_k,
            bx: 0.1 * hour as f64,
            by: 0.0,
            bz: 0.0,
            b_mag: 0.1 * hour as f64,
            crs_flux: 0.0,
            spectral_mean: 0.0,
            spectral_peak: 0.0,
            map_flux_mean: 0.0,
            map_flux_std: 0.0,
            event_score: None,
            event_mask: None,
            event_segment_id: None,
        }
    }

    #[test]
    fn test_algebra_vector_dimension() {
        let row = HeliosphereFeatureRow {
            window_name: "fleet2016".to_string(),
            mission: "Voyager 1".to_string(),
            product: "Merged".to_string(),
            year: 2016,
            doy: 1,
            hour: 0,
            r_au: 135.0,
            lat_deg: 34.0,
            lon_deg: 120.0,
            density_cm3: 0.002,
            speed_kms: 380.0,
            temperature_k: 12000.0,
            bx: 0.1,
            by: -0.2,
            bz: 0.05,
            b_mag: 0.23,
            crs_flux: 1.2,
            spectral_mean: 0.8,
            spectral_peak: 1.1,
            map_flux_mean: f64::NAN,
            map_flux_std: f64::NAN,
            event_score: None,
            event_mask: None,
            event_segment_id: None,
        };
        let vector = row.algebra_vector();
        assert_eq!(vector.len(), HELIOSPHERE_FEATURE_DIM);
        assert_eq!(vector[15], 1.0);
    }

    #[test]
    fn test_sparse_memory_plan_1024() {
        let plan = estimate_sparse_memory_plan(1024, 0.0575);
        assert_eq!(plan.grid_size, 1024);
        assert_eq!(plan.brick_edge, 8);
        assert_eq!(plan.halo_edge, 10);
        assert_eq!(plan.shared_tile_bytes_bf16, 38_000);
        assert!(plan.dense_fp32_pingpong_gib > plan.sparse_fp32_aa_gib);
        assert!(plan.sparse_fp32_aa_gib > plan.sparse_bf16_aa_projected_gib);
    }

    #[test]
    fn test_sparse_execution_plan_prefers_cache_aware_vram_path() {
        let plan = estimate_sparse_execution_plan(
            1024,
            0.07,
            Some(0.03),
            SparseHardwareEnvelope {
                cuda_vram_budget_bytes: Some(12 * 1024 * 1024 * 1024),
                cuda_l2_bytes: Some(48 * 1024 * 1024),
                cuda_shared_mem_per_block: Some(48 * 1024),
                cuda_managed_memory: Some(true),
                cuda_concurrent_managed_access: Some(true),
                cpu_l3_safe_working_set_bytes: Some(96 * 1024 * 1024),
                prefer_sparse_tile: true,
            },
        );
        assert!(matches!(
            plan.mode,
            SparseExecutionMode::CacheAwareTiledSparseSoA
                | SparseExecutionMode::VramResidentSparseSoA
        ));
        assert_eq!(plan.fits_vram_budget, Some(true));
    }

    #[test]
    fn test_sparse_execution_plan_uses_temporal_tiling_before_managed_fallback() {
        let plan = estimate_sparse_execution_plan(
            1024,
            0.20,
            Some(0.08),
            SparseHardwareEnvelope {
                cuda_vram_budget_bytes: Some(12 * 1024 * 1024 * 1024),
                cuda_l2_bytes: Some(48 * 1024 * 1024),
                cuda_shared_mem_per_block: Some(48 * 1024),
                cuda_managed_memory: Some(true),
                cuda_concurrent_managed_access: Some(true),
                cpu_l3_safe_working_set_bytes: Some(96 * 1024 * 1024),
                prefer_sparse_tile: true,
            },
        );
        assert_eq!(plan.mode, SparseExecutionMode::TemporalTiledSparseSoA);
        assert!(plan.peak_temporal_tile_gib < plan.memory.sparse_bf16_aa_projected_gib);
    }

    #[test]
    fn test_transform_feature_rows_differenced_keeps_row_count() {
        let rows = vec![
            HeliosphereFeatureRow {
                window_name: "w".to_string(),
                mission: "m".to_string(),
                product: "p".to_string(),
                year: 2020,
                doy: 1,
                hour: 0,
                r_au: 1.0,
                lat_deg: 2.0,
                lon_deg: 3.0,
                density_cm3: 4.0,
                speed_kms: 5.0,
                temperature_k: 6.0,
                bx: 7.0,
                by: 8.0,
                bz: 9.0,
                b_mag: 10.0,
                crs_flux: 11.0,
                spectral_mean: 12.0,
                spectral_peak: 13.0,
                map_flux_mean: 14.0,
                map_flux_std: 15.0,
                event_score: None,
                event_mask: None,
                event_segment_id: None,
            },
            HeliosphereFeatureRow {
                window_name: "w".to_string(),
                mission: "m".to_string(),
                product: "p".to_string(),
                year: 2020,
                doy: 1,
                hour: 1,
                r_au: 2.0,
                lat_deg: 4.0,
                lon_deg: 6.0,
                density_cm3: 8.0,
                speed_kms: 10.0,
                temperature_k: 12.0,
                bx: 14.0,
                by: 16.0,
                bz: 18.0,
                b_mag: 20.0,
                crs_flux: 22.0,
                spectral_mean: 24.0,
                spectral_peak: 26.0,
                map_flux_mean: 28.0,
                map_flux_std: 30.0,
                event_score: None,
                event_mask: None,
                event_segment_id: None,
            },
        ];
        let transformed = transform_feature_rows(&rows, HeliosphereTransformMode::Differenced);
        assert_eq!(transformed.len(), 2);
        assert_eq!(transformed[0].density_cm3, 0.0);
        assert_eq!(transformed[1].density_cm3, 4.0);
        assert_eq!(transformed[1].map_flux_std, 15.0);
    }

    #[test]
    fn test_transform_feature_rows_normalized_zero_mean_for_constant_channel() {
        let rows = vec![
            HeliosphereFeatureRow {
                window_name: "w".to_string(),
                mission: "m".to_string(),
                product: "p".to_string(),
                year: 2020,
                doy: 1,
                hour: 0,
                r_au: 1.0,
                lat_deg: 0.0,
                lon_deg: 0.0,
                density_cm3: 5.0,
                speed_kms: 10.0,
                temperature_k: 100.0,
                bx: 0.0,
                by: 0.0,
                bz: 0.0,
                b_mag: 0.0,
                crs_flux: 0.0,
                spectral_mean: 0.0,
                spectral_peak: 0.0,
                map_flux_mean: 0.0,
                map_flux_std: 0.0,
                event_score: None,
                event_mask: None,
                event_segment_id: None,
            },
            HeliosphereFeatureRow {
                window_name: "w".to_string(),
                mission: "m".to_string(),
                product: "p".to_string(),
                year: 2020,
                doy: 1,
                hour: 1,
                r_au: 3.0,
                lat_deg: 0.0,
                lon_deg: 0.0,
                density_cm3: 5.0,
                speed_kms: 14.0,
                temperature_k: 120.0,
                bx: 0.0,
                by: 0.0,
                bz: 0.0,
                b_mag: 0.0,
                crs_flux: 0.0,
                spectral_mean: 0.0,
                spectral_peak: 0.0,
                map_flux_mean: 0.0,
                map_flux_std: 0.0,
                event_score: None,
                event_mask: None,
                event_segment_id: None,
            },
        ];
        let transformed = transform_feature_rows(&rows, HeliosphereTransformMode::Normalized);
        assert_eq!(transformed.len(), 2);
        assert_eq!(transformed[0].density_cm3, 0.0);
        assert_eq!(transformed[1].density_cm3, 0.0);
        assert!(transformed[0].r_au.is_finite());
    }

    #[test]
    fn test_robust_differenced_centered_marks_small_groups_active() {
        let rows = vec![
            sample_row(0, 1.0, 100.0, 1000.0),
            sample_row(1, 2.0, 110.0, 1010.0),
            sample_row(2, 3.0, 120.0, 1020.0),
        ];
        let transformed = transform_feature_rows_with_stats(
            &rows,
            HeliosphereTransformMode::RobustDifferencedCentered,
        );
        assert_eq!(transformed.rows.len(), 3);
        assert!(
            transformed
                .rows
                .iter()
                .all(HeliosphereFeatureRow::event_active)
        );
        assert_eq!(transformed.groups.len(), 1);
        assert_eq!(transformed.groups[0].event_row_count, 3);
        assert_eq!(transformed.rows[1].r_au, rows[1].r_au);
        assert_eq!(transformed.rows[1].lat_deg, rows[1].lat_deg);
        assert_eq!(transformed.rows[1].lon_deg, rows[1].lon_deg);
    }

    #[test]
    fn test_robust_centered_constant_dynamic_channels_stay_finite() {
        let rows = vec![
            sample_row(0, 5.0, 10.0, 100.0),
            sample_row(1, 5.0, 10.0, 100.0),
            sample_row(2, 5.0, 10.0, 100.0),
            sample_row(3, 5.0, 10.0, 100.0),
        ];
        let transformed =
            transform_feature_rows_with_stats(&rows, HeliosphereTransformMode::RobustCentered);
        assert!(transformed.rows.iter().all(|row| {
            row.dynamic_signal_channels()
                .iter()
                .all(|value| value.is_finite())
        }));
        assert!(transformed.rows.iter().all(|row| {
            row.dynamic_signal_channels()
                .iter()
                .all(|value| value.abs() <= 8.0)
        }));
    }

    fn b_row(hour: u8, bx: f64, by: f64, bz: f64) -> HeliosphereFeatureRow {
        let b_mag = (bx * bx + by * by + bz * bz).sqrt();
        HeliosphereFeatureRow {
            window_name: "test".to_string(),
            mission: "Test".to_string(),
            product: "TestProd".to_string(),
            year: 2020,
            doy: 1,
            hour,
            r_au: 1.0 + hour as f64 * 0.1,
            lat_deg: 10.0,
            lon_deg: 0.0,
            density_cm3: 5.0,
            speed_kms: 400.0,
            temperature_k: 100000.0,
            bx,
            by,
            bz,
            b_mag,
            crs_flux: 0.0,
            spectral_mean: 0.0,
            spectral_peak: 0.0,
            map_flux_mean: 0.0,
            map_flux_std: 0.0,
            event_score: None,
            event_mask: None,
            event_segment_id: None,
        }
    }

    #[test]
    fn test_magnetic_takens_embed_dim16_basic() {
        // 6 rows -> 4-step windows -> 3 embedded vectors (at indices 3, 4, 5)
        let rows: Vec<HeliosphereFeatureRow> = (1..=6)
            .map(|h| b_row(h, h as f64 * 1.0, h as f64 * 0.5, h as f64 * 0.3))
            .collect();
        let (vecs, idx) = magnetic_takens_embed(&rows, 16, 1);
        assert_eq!(vecs.len(), 3);
        assert_eq!(idx.len(), 3);
        // Each vector should be 16 components
        assert!(vecs.iter().all(|v| v.len() == 16));
        // metadata_indices point to last row in each 4-step window
        assert_eq!(idx[0], 3); // window [0,1,2,3] -> last = 3
        assert_eq!(idx[1], 4);
        assert_eq!(idx[2], 5);

        // Verify normalization: first window rows 0-3
        let mean_b = (rows[0].b_mag + rows[1].b_mag + rows[2].b_mag + rows[3].b_mag) / 4.0;
        assert!(mean_b > 0.0);
        let expected_bx0 = rows[0].bx / mean_b;
        assert!((vecs[0][0] - expected_bx0).abs() < 1e-12);
    }

    #[test]
    fn test_magnetic_takens_embed_dim32() {
        // 10 rows -> 8-step windows -> 3 embedded vectors
        let rows: Vec<HeliosphereFeatureRow> = (1..=10)
            .map(|h| b_row(h, h as f64 * 1.0, h as f64 * 0.5, h as f64 * 0.3))
            .collect();
        let (vecs, idx) = magnetic_takens_embed(&rows, 32, 1);
        assert_eq!(vecs.len(), 3); // 10 - 8 + 1 = 3
        assert!(vecs.iter().all(|v| v.len() == 32));
        assert_eq!(idx[0], 7); // window [0..7] -> last = 7
        assert_eq!(idx[1], 8);
        assert_eq!(idx[2], 9);
    }

    #[test]
    fn test_magnetic_takens_embed_lag2() {
        // 10 rows, dim=16 (4 steps), lag=2 -> window spans rows 0,2,4,6
        // window_rows = (4-1)*2 + 1 = 7
        // sliding from w_start=0 to w_start=3 -> 4 embedded vectors
        let rows: Vec<HeliosphereFeatureRow> = (1..=10)
            .map(|h| b_row(h, h as f64 * 1.0, h as f64 * 0.5, h as f64 * 0.3))
            .collect();
        let (vecs, idx) = magnetic_takens_embed(&rows, 16, 2);
        assert_eq!(vecs.len(), 4);
        assert_eq!(idx[0], 6); // w_start=0: samples [0,2,4,6] -> last=6
        assert_eq!(idx[1], 7); // w_start=1: samples [1,3,5,7] -> last=7
    }

    #[test]
    fn test_magnetic_takens_embed_too_short() {
        let rows: Vec<HeliosphereFeatureRow> =
            (1..=3).map(|h| b_row(h, h as f64, 0.0, 0.0)).collect();
        let (vecs, _) = magnetic_takens_embed(&rows, 16, 1);
        assert!(vecs.is_empty()); // 3 rows < 4-step window
    }
}
