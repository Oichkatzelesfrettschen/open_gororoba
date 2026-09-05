//! Seeded file partitions and paired whole-file sampling.

use gororoba_cli_physics::staple_calibration::PreparedDataset;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub(super) fn folds(file_ids: &[u16], seed: u64) -> Vec<Vec<u16>> {
    let mut shuffled = file_ids.to_vec();
    shuffled.sort_unstable();
    let mut random = ChaCha8Rng::seed_from_u64(seed);
    for index in (1..shuffled.len()).rev() {
        let other = random.random_range(0..=index);
        shuffled.swap(index, other);
    }
    let mut output = vec![Vec::new(); 5];
    for (index, file) in shuffled.into_iter().enumerate() {
        output[index % 5].push(file);
    }
    for fold in &mut output {
        fold.sort_unstable();
    }
    output
}

pub(super) fn draw_counts(ids: &[u16], file_count: usize, random: &mut ChaCha8Rng) -> Vec<u32> {
    let mut counts = vec![0; file_count];
    for _ in ids {
        let selected = random.random_range(0..ids.len());
        counts[usize::from(ids[selected])] += 1;
    }
    counts
}

pub(super) fn unit_counts(ids: &[u16], file_count: usize) -> Vec<u32> {
    let mut counts = vec![0; file_count];
    for &id in ids {
        counts[usize::from(id)] = 1;
    }
    counts
}

pub(super) fn selected_rows(data: &PreparedDataset, counts: &[u32]) -> Vec<u32> {
    data.file_index
        .iter()
        .enumerate()
        .flat_map(|(row, &file)| {
            std::iter::repeat_n(row as u32, counts[usize::from(file)] as usize)
        })
        .collect()
}
