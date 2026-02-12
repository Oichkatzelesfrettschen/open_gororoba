//! Anti-diagonal parity filter layer.

use crate::traits::ParityLayer;

/// Parity filter mapping word popcount parity to signed edges.
#[derive(Debug, Clone, Copy, Default)]
pub struct AntiDiagonalParityFilter;

impl ParityLayer for AntiDiagonalParityFilter {
    fn compute_signs(&self, words: &[u64]) -> Vec<i32> {
        words
            .iter()
            .map(|w| if w.count_ones() % 2 == 0 { 1 } else { -1 })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ParityLayer;

    #[test]
    fn test_parity_filter_emits_signs() {
        let filter = AntiDiagonalParityFilter;
        let signs = filter.compute_signs(&[0, 1, 3, 4]);
        assert_eq!(signs.len(), 4);
        assert_eq!(signs[0], 1); // popcount(0)=0
        assert_eq!(signs[1], -1); // popcount(1)=1
    }
}
