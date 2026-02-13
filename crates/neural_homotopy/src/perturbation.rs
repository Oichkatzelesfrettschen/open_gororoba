//! Perturbation dataset for robustness testing of neural homotopy search.
//!
//! Adds controlled noise to the sedenion multiplication table to test
//! whether the A-infinity correction is robust to small perturbations
//! in the algebraic structure.

use crate::training_data::{build_sedenion_table, MultiplicationSample, SEDENION_DIM};

/// Perturb the sedenion multiplication table by randomly flipping
/// product signs and/or changing product basis indices.
///
/// # Arguments
/// * `noise_level` - Probability of each entry being perturbed (0.0 to 1.0)
/// * `seed` - Random seed for reproducibility
///
/// Returns a modified table where some entries have been randomly changed.
pub fn perturbed_sedenion_table(
    noise_level: f64,
    seed: u64,
) -> SedenionMulTable {
    let mut table = build_sedenion_table();
    let mut state = seed.wrapping_add(1);

    let mut next_rand = || -> f64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64) / (u64::MAX as f64)
    };

    for row in &mut table {
        for cell in row.iter_mut() {
            if next_rand() < noise_level {
                // Choose perturbation type
                let choice = next_rand();
                if choice < 0.5 {
                    // Flip sign
                    cell.1 = -cell.1;
                } else {
                    // Shift basis by 1 (modular)
                    cell.0 = (cell.0 + 1) % SEDENION_DIM;
                }
            }
        }
    }

    table
}

/// A sedenion multiplication table: 16x16 array of (basis_index, sign) pairs.
pub type SedenionMulTable = [[(usize, i32); SEDENION_DIM]; SEDENION_DIM];

/// Collection of perturbed multiplication tables for robustness testing.
#[derive(Debug, Clone)]
pub struct PerturbationDataset {
    /// Original unperturbed training samples.
    pub original: Vec<MultiplicationSample>,
    /// Perturbed tables at various noise levels.
    pub perturbed_tables: Vec<(f64, SedenionMulTable)>,
}

impl PerturbationDataset {
    /// Build a dataset with perturbations at the specified noise levels.
    ///
    /// # Arguments
    /// * `noise_levels` - Noise levels to generate (e.g., [0.01, 0.05, 0.10])
    /// * `base_seed` - Base random seed (each level uses `base_seed + level_index`)
    pub fn build(noise_levels: &[f64], base_seed: u64) -> Self {
        let original = crate::training_data::multiplication_samples();
        let perturbed_tables = noise_levels
            .iter()
            .enumerate()
            .map(|(idx, &level)| {
                let seed = base_seed.wrapping_add(idx as u64);
                (level, perturbed_sedenion_table(level, seed))
            })
            .collect();

        Self {
            original,
            perturbed_tables,
        }
    }

    /// Number of perturbed variants.
    pub fn n_variants(&self) -> usize {
        self.perturbed_tables.len()
    }

    /// Extract samples from a perturbed table.
    pub fn samples_at(&self, variant_index: usize) -> Vec<MultiplicationSample> {
        let (_, ref table) = self.perturbed_tables[variant_index];
        let mut out = Vec::with_capacity(SEDENION_DIM * SEDENION_DIM);
        for (i, row) in table.iter().enumerate() {
            for (j, &(basis, sign)) in row.iter().enumerate() {
                out.push(MultiplicationSample {
                    lhs: i,
                    rhs: j,
                    product_basis: basis,
                    product_sign: sign,
                });
            }
        }
        out
    }

    /// Count how many entries differ from the original at each noise level.
    pub fn difference_counts(&self) -> Vec<(f64, usize)> {
        let orig_table = build_sedenion_table();
        self.perturbed_tables
            .iter()
            .map(|(level, table)| {
                let mut diffs = 0;
                for i in 0..SEDENION_DIM {
                    for j in 0..SEDENION_DIM {
                        if table[i][j] != orig_table[i][j] {
                            diffs += 1;
                        }
                    }
                }
                (*level, diffs)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_noise_preserves_table() {
        let original = build_sedenion_table();
        let perturbed = perturbed_sedenion_table(0.0, 42);
        for i in 0..SEDENION_DIM {
            for j in 0..SEDENION_DIM {
                assert_eq!(
                    original[i][j], perturbed[i][j],
                    "Zero noise should preserve table at ({},{})",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_high_noise_changes_table() {
        let original = build_sedenion_table();
        let perturbed = perturbed_sedenion_table(0.5, 42);
        let mut diffs = 0;
        for i in 0..SEDENION_DIM {
            for j in 0..SEDENION_DIM {
                if original[i][j] != perturbed[i][j] {
                    diffs += 1;
                }
            }
        }
        assert!(
            diffs > 0,
            "50% noise should change at least some entries"
        );
    }

    #[test]
    fn test_perturbation_reproducible() {
        let a = perturbed_sedenion_table(0.1, 42);
        let b = perturbed_sedenion_table(0.1, 42);
        for i in 0..SEDENION_DIM {
            for j in 0..SEDENION_DIM {
                assert_eq!(a[i][j], b[i][j], "Same seed should give same result");
            }
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let a = perturbed_sedenion_table(0.3, 42);
        let b = perturbed_sedenion_table(0.3, 99);
        let mut diffs = 0;
        for i in 0..SEDENION_DIM {
            for j in 0..SEDENION_DIM {
                if a[i][j] != b[i][j] {
                    diffs += 1;
                }
            }
        }
        assert!(diffs > 0, "Different seeds should produce different tables");
    }

    #[test]
    fn test_perturbation_dataset_build() {
        let ds = PerturbationDataset::build(&[0.01, 0.05, 0.10], 42);
        assert_eq!(ds.n_variants(), 3);
        assert_eq!(ds.original.len(), SEDENION_DIM * SEDENION_DIM);
    }

    #[test]
    fn test_perturbation_dataset_samples() {
        let ds = PerturbationDataset::build(&[0.05], 42);
        let samples = ds.samples_at(0);
        assert_eq!(samples.len(), SEDENION_DIM * SEDENION_DIM);
    }

    #[test]
    fn test_difference_counts_monotonic() {
        let ds = PerturbationDataset::build(&[0.01, 0.10, 0.50], 42);
        let counts = ds.difference_counts();
        // Higher noise should generally produce more differences
        // (probabilistic, but with these levels it should hold)
        assert!(
            counts[2].1 >= counts[0].1,
            "50% noise ({}) should produce >= differences than 1% noise ({})",
            counts[2].1,
            counts[0].1
        );
    }
}
