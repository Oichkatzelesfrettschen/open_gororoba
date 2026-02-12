//! Minimal Patricia-style prefix index for 64-bit keys.

use std::collections::HashMap;

/// Sparse prefix index over integer keys.
#[derive(Debug, Clone, Default)]
pub struct PatriciaIndex {
    counts: HashMap<u64, usize>,
}

impl PatriciaIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: u64) {
        *self.counts.entry(key).or_insert(0) += 1;
    }

    pub fn frequency(&self, key: u64) -> usize {
        self.counts.get(&key).copied().unwrap_or(0)
    }

    /// Longest shared prefix (MSB-first) in bits.
    pub fn shared_prefix_bits(a: u64, b: u64) -> u32 {
        let x = a ^ b;
        if x == 0 {
            64
        } else {
            x.leading_zeros()
        }
    }

    /// Aggregate multiplicities by prefix length.
    pub fn prefix_histogram(&self, anchor: u64) -> HashMap<u32, usize> {
        let mut out = HashMap::new();
        for (&k, &c) in &self.counts {
            let prefix = Self::shared_prefix_bits(anchor, k);
            *out.entry(prefix).or_insert(0) += c;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_frequency() {
        let mut idx = PatriciaIndex::new();
        idx.insert(7);
        idx.insert(7);
        assert_eq!(idx.frequency(7), 2);
    }

    #[test]
    fn test_shared_prefix_bits() {
        let a = 0b1011_0000_u64;
        let b = 0b1011_1111_u64;
        assert!(PatriciaIndex::shared_prefix_bits(a, b) >= 4);
    }
}
