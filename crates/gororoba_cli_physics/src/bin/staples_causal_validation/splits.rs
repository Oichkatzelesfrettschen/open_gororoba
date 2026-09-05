//! Immutable chronological memberships and paired daily sampling.

use anyhow::{Result, ensure};
use rand::RngExt;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;

use super::{Config, admission::Dataset};

pub(super) fn validate(config: &Config) -> Result<()> {
    let training: BTreeSet<_> = config.training_years.iter().copied().collect();
    let validation: BTreeSet<_> = config.validation_years.iter().copied().collect();
    let final_years: BTreeSet<_> = config.final_years.iter().copied().collect();
    ensure!(
        !training.is_empty() && !validation.is_empty() && !final_years.is_empty(),
        "empty epoch partition"
    );
    ensure!(
        training.is_disjoint(&validation)
            && training.is_disjoint(&final_years)
            && validation.is_disjoint(&final_years),
        "epoch partitions overlap"
    );
    ensure!(
        training.last() < validation.first() && validation.last() < final_years.first(),
        "epoch partitions violate chronology"
    );
    Ok(())
}

pub(super) fn training_rows(data: &Dataset, config: &Config) -> Vec<u32> {
    data.rows
        .iter()
        .enumerate()
        .filter_map(|(index, row)| {
            config
                .training_years
                .contains(&row.year)
                .then_some(index as u32)
        })
        .collect()
}

pub(super) fn draw_counts(file_count: usize, random: &mut ChaCha8Rng) -> Vec<u32> {
    let mut counts = vec![0; file_count];
    for _ in 0..file_count {
        counts[random.random_range(0..file_count)] += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    #[test]
    fn chronology_rejects_overlap_and_future_training() {
        let mut config = crate::test_config();
        super::validate(&config).unwrap();
        config.training_years.push(2015);
        assert!(super::validate(&config).is_err());
    }
}
