//! Shared heliosphere feature-cube types and memory-planning helpers.
//!
//! This module does not fetch data directly. It provides the normalized row
//! schema used by the executed feature-cube, algebra, and LBM benchmark bins,
//! along with the memory estimators for dense and sparse 3D runs.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::str::FromStr;

pub const HELIOSPHERE_FEATURE_DIM: usize = 16;
pub const HELIOSPHERE_SIGNAL_DIM: usize = HELIOSPHERE_FEATURE_DIM - 1;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeliosphereFeatureRow {
    pub window_name: String,
    pub mission: String,
    pub product: String,
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub r_au: f64,
    pub lat_deg: f64,
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
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum HeliosphereTransformMode {
    Raw,
    Normalized,
    Differenced,
    DifferencedNormalized,
}

impl FromStr for HeliosphereTransformMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "raw" => Ok(Self::Raw),
            "normalized" => Ok(Self::Normalized),
            "differenced" => Ok(Self::Differenced),
            "differenced-normalized" => Ok(Self::DifferencedNormalized),
            other => Err(format!(
                "unsupported mode '{other}'; expected raw, normalized, differenced, or differenced-normalized"
            )),
        }
    }
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

    pub fn signal_energy(&self) -> f64 {
        let vector = self.algebra_vector();
        vector.iter().map(|value| value * value).sum::<f64>().sqrt()
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
    if mode == HeliosphereTransformMode::Raw {
        return rows.to_vec();
    }

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

        let base_rows = if mode == HeliosphereTransformMode::Differenced {
            difference_rows(&group)
        } else if mode == HeliosphereTransformMode::Normalized {
            normalize_rows(&group)
        } else {
            normalize_rows(&difference_rows(&group))
        };
        output.extend(base_rows);
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
    output
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
    let means = columns.iter().map(|col| finite_mean(col)).collect::<Vec<_>>();
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
                let lhs = if current[idx].is_finite() { current[idx] } else { 0.0 };
                let rhs = if prev[idx].is_finite() { prev[idx] } else { 0.0 };
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

#[derive(Debug, Clone, Serialize)]
pub struct SparseMemoryPlan {
    pub grid_size: usize,
    pub brick_edge: usize,
    pub halo_edge: usize,
    pub active_fraction: f64,
    pub total_cells: u64,
    pub total_bricks: u64,
    pub active_bricks: u64,
    pub active_core_cells: u64,
    pub dense_fp32_pingpong_gib: f64,
    pub dense_bf16_pingpong_gib: f64,
    pub sparse_fp32_aa_gib: f64,
    pub sparse_bf16_aa_projected_gib: f64,
    pub occupancy_bitset_mib: f64,
    pub indirect_table_mib: f64,
    pub active_brick_id_mib: f64,
    pub shared_tile_bytes_bf16: usize,
    pub shared_tile_layout_gpu: String,
    pub shared_tile_layout_cpu: String,
}

pub fn estimate_sparse_memory_plan(grid_size: usize, active_fraction: f64) -> SparseMemoryPlan {
    let brick_edge = 8u64;
    let halo_edge = 10u64;
    let grid = grid_size as u64;
    let total_cells = grid * grid * grid;
    let bricks_per_axis = grid.div_ceil(brick_edge);
    let total_bricks = bricks_per_axis * bricks_per_axis * bricks_per_axis;
    let active_fraction = active_fraction.clamp(0.001, 1.0);
    let active_bricks = ((total_bricks as f64) * active_fraction).ceil() as u64;
    let active_core_cells = active_bricks * brick_edge.pow(3);

    let dense_fp32_pingpong_bytes = total_cells as f64 * 184.0;
    let dense_bf16_pingpong_bytes = total_cells as f64 * 108.0;
    let sparse_fp32_aa_bytes = active_core_cells as f64 * 108.0;
    let sparse_bf16_aa_projected_bytes = active_core_cells as f64 * 70.0;

    SparseMemoryPlan {
        grid_size,
        brick_edge: brick_edge as usize,
        halo_edge: halo_edge as usize,
        active_fraction,
        total_cells,
        total_bricks,
        active_bricks,
        active_core_cells,
        dense_fp32_pingpong_gib: dense_fp32_pingpong_bytes / 1024.0_f64.powi(3),
        dense_bf16_pingpong_gib: dense_bf16_pingpong_bytes / 1024.0_f64.powi(3),
        sparse_fp32_aa_gib: sparse_fp32_aa_bytes / 1024.0_f64.powi(3),
        sparse_bf16_aa_projected_gib: sparse_bf16_aa_projected_bytes / 1024.0_f64.powi(3),
        occupancy_bitset_mib: (total_bricks as f64 / 8.0) / 1024.0_f64.powi(2),
        indirect_table_mib: (total_bricks as f64 * 4.0) / 1024.0_f64.powi(2),
        active_brick_id_mib: (active_bricks as f64 * 4.0) / 1024.0_f64.powi(2),
        shared_tile_bytes_bf16: (halo_edge.pow(3) * 19 * 2) as usize,
        shared_tile_layout_gpu: "[19][1000]".to_string(),
        shared_tile_layout_cpu: "[1000][19]".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            },
        ];
        let transformed = transform_feature_rows(&rows, HeliosphereTransformMode::Normalized);
        assert_eq!(transformed.len(), 2);
        assert_eq!(transformed[0].density_cm3, 0.0);
        assert_eq!(transformed[1].density_cm3, 0.0);
        assert!(transformed[0].r_au.is_finite());
    }
}
