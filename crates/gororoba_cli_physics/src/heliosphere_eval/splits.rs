//! Train/validation/test split helpers for labeled invariant samples.
//!
//! `mission_splits` reports per-mission row counts using a deterministic
//! 70/15/15 chronological cut. `split_samples` and
//! `split_samples_with_seed` build the actual borrowed reference splits;
//! a non-zero seed reorders within-mission groups via
//! `seeded_split_rank` while still tie-breaking on timestamp for
//! reproducibility.
//!
//! `SampleSplits` is the shared output struct (3 borrowed slices).

use std::collections::BTreeMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use super::public_types::{LabeledInvariantSample, MissionSplitSummary};

#[derive(Debug, Clone)]
pub(super) struct SampleSplits<'a> {
    pub(super) train: Vec<&'a LabeledInvariantSample>,
    pub(super) validation: Vec<&'a LabeledInvariantSample>,
    pub(super) test: Vec<&'a LabeledInvariantSample>,
}

pub(super) fn mission_splits(samples: &[LabeledInvariantSample]) -> Vec<MissionSplitSummary> {
    let mut grouped: BTreeMap<String, Vec<&LabeledInvariantSample>> = BTreeMap::new();
    for sample in samples {
        grouped
            .entry(sample.mission.clone())
            .or_default()
            .push(sample);
    }
    grouped
        .into_iter()
        .map(|(mission, mut group)| {
            group.sort_by_key(|sample| sample.timestamp_utc.clone());
            let n = group.len();
            let train_end = ((n as f64) * 0.70).round() as usize;
            let val_end = ((n as f64) * 0.85).round() as usize;
            MissionSplitSummary {
                mission,
                train_rows: train_end.min(n),
                validation_rows: val_end
                    .saturating_sub(train_end)
                    .min(n.saturating_sub(train_end)),
                test_rows: n.saturating_sub(val_end),
            }
        })
        .collect()
}

pub(super) fn split_samples(samples: &[LabeledInvariantSample]) -> SampleSplits<'_> {
    split_samples_with_seed(samples, 0)
}

pub(super) fn split_samples_with_seed(
    samples: &[LabeledInvariantSample],
    split_seed: u64,
) -> SampleSplits<'_> {
    let mut grouped: BTreeMap<String, Vec<&LabeledInvariantSample>> = BTreeMap::new();
    for sample in samples {
        grouped
            .entry(sample.mission.clone())
            .or_default()
            .push(sample);
    }
    let mut train = Vec::new();
    let mut validation = Vec::new();
    let mut test = Vec::new();
    for mut group in grouped.into_values() {
        if split_seed == 0 {
            group.sort_by_key(|sample| sample.timestamp_utc.clone());
        } else {
            group.sort_by(|a, b| {
                seeded_split_rank(a, split_seed)
                    .cmp(&seeded_split_rank(b, split_seed))
                    .then_with(|| a.timestamp_utc.cmp(&b.timestamp_utc))
            });
        }
        let n = group.len();
        let train_end = ((n as f64) * 0.70).round() as usize;
        let val_end = ((n as f64) * 0.85).round() as usize;
        train.extend(group.iter().take(train_end).copied());
        validation.extend(
            group
                .iter()
                .skip(train_end)
                .take(val_end.saturating_sub(train_end))
                .copied(),
        );
        test.extend(group.iter().skip(val_end).copied());
    }
    SampleSplits {
        train,
        validation,
        test,
    }
}

pub(super) fn seeded_split_rank(sample: &LabeledInvariantSample, split_seed: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    split_seed.hash(&mut hasher);
    sample.key.hash(&mut hasher);
    sample.timestamp_utc.hash(&mut hasher);
    hasher.finish()
}
