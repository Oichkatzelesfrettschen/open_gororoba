//! Public summary record types for the heliosphere evaluation suite.
//!
//! These types are emitted (via Serialize/Deserialize) into the
//! reproducible artifact lanes consumed by the predictive, invariance,
//! and sparse-preservation experiment binaries. Keep field shapes
//! stable; downstream readers depend on them.

use data_core::HELIOSPHERE_INVARIANT_DIM;
use serde::{Deserialize, Serialize};

use super::DESCRIPTOR_DIM;

/// Stable row key used to join raw, transformed, and invariant views.
pub type RowKey = (String, String, String, u16, u16, u8);

/// One invariant sample augmented with official-label and descriptor context.
#[derive(Debug, Clone, Serialize)]
pub struct LabeledInvariantSample {
    pub key: RowKey,
    pub window_name: String,
    pub mission: String,
    pub product: String,
    pub timestamp_utc: String,
    pub label_positive: bool,
    pub scalar_event_score: f64,
    pub channels: [f64; HELIOSPHERE_INVARIANT_DIM],
    pub uncertainty_scales: [f64; HELIOSPHERE_INVARIANT_DIM],
    pub weighted_channels: [f64; HELIOSPHERE_INVARIANT_DIM],
    pub b_field: [f64; 4],
    pub descriptor_channels: [f64; DESCRIPTOR_DIM],
}

/// Scalar evaluation metrics for one predictive lane.
#[derive(Debug, Clone, Serialize)]
pub struct BinaryMetrics {
    pub feature_mode: String,
    pub name: String,
    pub threshold: f64,
    pub positive_rows: usize,
    pub negative_rows: usize,
    pub predicted_positive_rows: usize,
    pub auprc: f64,
    pub auroc: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub false_alert_rate: f64,
    pub median_lead_time_hours: Option<f64>,
}

/// Chronological train/validation/test split sizes for one mission.
#[derive(Debug, Clone, Serialize)]
pub struct MissionSplitSummary {
    pub mission: String,
    pub train_rows: usize,
    pub validation_rows: usize,
    pub test_rows: usize,
}

/// Cross-mission descriptor stability summary.
#[derive(Debug, Clone, Serialize)]
pub struct MissionInvarianceSummary {
    pub feature_mode: String,
    pub mission: String,
    pub positive_rows: usize,
    pub negative_rows: usize,
    pub positive_mean_weighted_norm: f64,
    pub negative_mean_weighted_norm: f64,
    pub positive_descriptor_mean: f64,
    pub leave_one_mission_out_cosine: f64,
    pub blocking_channels: Vec<String>,
}

/// Sparse-preservation metrics comparing two event masks.
#[derive(Debug, Clone, Serialize)]
pub struct SparseMaskSummary {
    pub name: String,
    pub active_rows: usize,
    pub active_fraction: f64,
    pub occupancy_tiles_active: usize,
    pub occupancy_tiles_total: usize,
    pub occupancy_tile_fraction: f64,
    pub event_label_recall: f64,
    pub event_label_precision: f64,
    pub density_mean: f64,
    pub speed_mean: f64,
    pub temperature_mean: f64,
    pub bmag_mean: f64,
    pub median_lead_time_hours: Option<f64>,
}

/// Predictive robustness row used to challenge a recorded falsification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualPredictiveSummary {
    pub view_mode: String,
    pub normalization_strategy: String,
    pub descriptor_profile: String,
    pub threshold: f64,
    pub positive_rows: usize,
    pub negative_rows: usize,
    pub predicted_positive_rows: usize,
    pub auprc: f64,
    pub auroc: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub false_alert_rate: f64,
    pub median_lead_time_hours: Option<f64>,
}

/// Sparse-policy robustness row used to challenge a recorded falsification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualSparseSummary {
    pub normalization_strategy: String,
    pub descriptor_profile: String,
    pub active_rows: usize,
    pub active_fraction: f64,
    pub occupancy_tiles_active: usize,
    pub occupancy_tiles_total: usize,
    pub occupancy_tile_fraction: f64,
    pub event_label_recall: f64,
    pub event_label_precision: f64,
    pub sparse_bf16_aa_projected_gib: f64,
    pub median_lead_time_hours: Option<f64>,
}

/// One seeded sparse-policy evaluation row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeededSparsePolicySummary {
    pub split_seed: u64,
    pub normalization_strategy: String,
    pub descriptor_profile: String,
    pub active_rows: usize,
    pub active_fraction: f64,
    pub occupancy_tiles_active: usize,
    pub occupancy_tiles_total: usize,
    pub occupancy_tile_fraction: f64,
    pub event_label_recall: f64,
    pub event_label_precision: f64,
    pub sparse_bf16_aa_projected_gib: f64,
    pub median_lead_time_hours: Option<f64>,
}

/// Configuration for one sparse-policy transfer evaluation.
#[derive(Debug, Clone)]
pub struct SparsePolicyTransferSpec<'a> {
    pub horizon_hours: i64,
    pub grid: usize,
    pub split_seed: u64,
    pub normalization_strategy: &'a str,
    pub descriptor_profile: &'a str,
}

/// Cached label-joined context for sparse-policy fitting or transfer.
#[derive(Debug, Clone)]
pub struct SparsePolicyDatasetContext {
    pub(super) positive_sample_count: usize,
    pub(super) samples: Vec<LabeledInvariantSample>,
}

impl SparsePolicyDatasetContext {
    /// Number of positive labeled samples available in this prepared context.
    pub fn positive_sample_count(&self) -> usize {
        self.positive_sample_count
    }
}

/// Label-coverage summary for one mission/product lane.
#[derive(Debug, Clone, Serialize)]
pub struct LabelCoverageRow {
    pub mission: String,
    pub product: String,
    pub row_count: usize,
    pub source_families: Vec<String>,
    pub positive_window_count_6h: usize,
    pub positive_window_count_12h: usize,
    pub positive_window_count_24h: usize,
    pub positive_row_count_6h: usize,
    pub positive_row_count_12h: usize,
    pub positive_row_count_24h: usize,
    pub forecast_residual_count: usize,
    pub coverage_status: String,
    pub blocked_reasons: Vec<String>,
}
