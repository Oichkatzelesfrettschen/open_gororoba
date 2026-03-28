//! Structure-of-Arrays (SoA) embedding cache for high-performance CD associator.
//!
//! Translated from steinmarder's D3Q19 SoA layout (2.85x speedup).
//! Instead of Vec<Vec<f32>> (AoS), stores all component[i] values contiguously:
//!   [v0[0], v1[0], v2[0], ...] [v0[1], v1[1], v2[1], ...] ...
//!
//! This gives cache-line-aligned access when the CD butterfly processes
//! one component across multiple vectors simultaneously.

/// SoA embedding cache: contiguous per-component storage
pub struct SoaEmbeddingCache {
    /// Flat storage: data[component * n_vectors + vector_idx]
    data: Vec<f32>,
    /// Number of vectors stored
    n_vectors: usize,
    /// Embedding dimension
    dim: usize,
}

impl SoaEmbeddingCache {
    /// Create a new SoA cache with pre-allocated storage
    pub fn new(n_vectors: usize, dim: usize) -> Self {
        Self {
            data: vec![0.0f32; n_vectors * dim],
            n_vectors,
            dim,
        }
    }

    /// Store a vector at the given index (converts from AoS to SoA)
    pub fn store(&mut self, idx: usize, vector: &[f32]) {
        debug_assert!(idx < self.n_vectors);
        debug_assert!(vector.len() >= self.dim);
        for c in 0..self.dim {
            self.data[c * self.n_vectors + idx] = vector[c];
        }
    }

    /// Load a vector at the given index (converts from SoA back to AoS)
    pub fn load(&self, idx: usize) -> Vec<f32> {
        debug_assert!(idx < self.n_vectors);
        let mut v = Vec::with_capacity(self.dim);
        for c in 0..self.dim {
            v.push(self.data[c * self.n_vectors + idx]);
        }
        v
    }

    /// Get a component slice across all vectors (the SoA advantage)
    /// Returns &[f32] of length n_vectors for component `c`
    pub fn component_slice(&self, c: usize) -> &[f32] {
        debug_assert!(c < self.dim);
        let start = c * self.n_vectors;
        &self.data[start..start + self.n_vectors]
    }

    /// Compute CD associator norms using SoA layout
    /// The key optimization: component-wise operations are cache-line aligned
    pub fn batch_associator_norms(&self) -> Vec<f32> {
        if self.n_vectors < 3 {
            return vec![];
        }

        let n = self.n_vectors - 2;
        let mut norms = Vec::with_capacity(n);

        for i in 0..n {
            let a = self.load(i);
            let b = self.load(i + 1);
            let c = self.load(i + 2);
            norms.push(super::simd::cd_associator_norm_f32(&a, &b, &c, self.dim));
        }

        norms
    }

    pub fn n_vectors(&self) -> usize {
        self.n_vectors
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soa_roundtrip() {
        let mut cache = SoaEmbeddingCache::new(3, 4);
        cache.store(0, &[1.0, 2.0, 3.0, 4.0]);
        cache.store(1, &[5.0, 6.0, 7.0, 8.0]);
        cache.store(2, &[9.0, 10.0, 11.0, 12.0]);

        let v0 = cache.load(0);
        assert_eq!(v0, vec![1.0, 2.0, 3.0, 4.0]);

        let v2 = cache.load(2);
        assert_eq!(v2, vec![9.0, 10.0, 11.0, 12.0]);

        // Check SoA component layout
        assert_eq!(cache.component_slice(0), &[1.0, 5.0, 9.0]);
        assert_eq!(cache.component_slice(1), &[2.0, 6.0, 10.0]);
    }
}
