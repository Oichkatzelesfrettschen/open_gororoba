//! Occupancy-tile helpers for heliosphere sparse-policy summaries.
//!
//! A "tile" buckets each `LabeledInvariantSample` by
//! `(window, mission, product, year, doy, 6h-hour-bucket)` so we can
//! summarize spatial/temporal coverage of an active mask independently
//! of the per-row count.
//!
//! Functions exposed at `pub(super)` scope:
//!   * `occupancy_tile_key`           -- compute the bucket key
//!   * `label_index`                  -- RowKey -> positive-label map
//!   * `occupancy_tile_totals`        -- total distinct tiles in a sample set
//!   * `occupancy_tile_stats_from_mask`     -- (active, total, fraction) for a bool mask
//!   * `occupancy_tile_fraction_for_scores` -- active-fraction for a score+threshold mask

use std::collections::{BTreeMap, BTreeSet};

use super::public_types::{LabeledInvariantSample, RowKey};
use super::stats::ratio_usize;

pub(super) type OccupancyTileKey = (String, String, String, u16, u16, u8, Option<u32>);

pub(super) fn occupancy_tile_key(sample: &LabeledInvariantSample) -> OccupancyTileKey {
    let hour_bucket = (sample.key.5 / 6) * 6;
    (
        sample.window_name.clone(),
        sample.mission.clone(),
        sample.product.clone(),
        sample.key.3,
        sample.key.4,
        hour_bucket,
        None,
    )
}

pub(super) fn label_index(samples: &[LabeledInvariantSample]) -> BTreeMap<RowKey, bool> {
    samples
        .iter()
        .map(|sample| (sample.key.clone(), sample.label_positive))
        .collect()
}

pub(super) fn occupancy_tile_totals(samples: &[LabeledInvariantSample]) -> usize {
    samples
        .iter()
        .map(occupancy_tile_key)
        .collect::<BTreeSet<_>>()
        .len()
}

pub(super) fn occupancy_tile_stats_from_mask(
    samples: &[LabeledInvariantSample],
    active_index: &BTreeMap<RowKey, bool>,
) -> (usize, usize, f64) {
    let total_tiles = occupancy_tile_totals(samples);
    let mut active_tiles = BTreeSet::new();
    for sample in samples {
        if *active_index.get(&sample.key).unwrap_or(&false) {
            active_tiles.insert(occupancy_tile_key(sample));
        }
    }
    let active_count = active_tiles.len();
    (
        active_count,
        total_tiles,
        ratio_usize(active_count, total_tiles.max(1)),
    )
}

pub(super) fn occupancy_tile_fraction_for_scores(
    samples: &[&LabeledInvariantSample],
    scores: &[f64],
    threshold: f64,
) -> f64 {
    let total_tiles = samples
        .iter()
        .map(|sample| occupancy_tile_key(sample))
        .collect::<BTreeSet<_>>()
        .len();
    let mut active_tiles = BTreeSet::new();
    for (sample, score) in samples.iter().zip(scores.iter()) {
        if score.is_finite() && *score >= threshold {
            active_tiles.insert(occupancy_tile_key(sample));
        }
    }
    ratio_usize(active_tiles.len(), total_tiles.max(1))
}
