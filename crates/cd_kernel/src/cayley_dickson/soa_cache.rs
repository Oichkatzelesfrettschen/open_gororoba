//! Structure-of-Arrays (SoA) embedding cache for high-performance CD associator.
//!
//! Translated from steinmarder's D3Q19 SoA layout (2.85x speedup).
//! Instead of `Vec<Vec<f32>`> (AoS), stores all component`[i]` values contiguously:
//!   `[v0[0]`, v1`[0]`, v2`[0]`, ...] `[v0[1]`, v1`[1]`, v2`[1]`, ...] ...
//!
//! This gives cache-line-aligned access when the CD butterfly processes
//! one component across multiple vectors simultaneously.

/// SoA embedding cache: contiguous per-component storage
pub struct SoaEmbeddingCache {
    /// Flat storage: data`[component * n_vectors + vector_idx]`
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
        for (c, &val) in vector[..self.dim].iter().enumerate() {
            self.data[c * self.n_vectors + idx] = val;
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
    /// Returns &`[f32]` of length n_vectors for component `c`
    pub fn component_slice(&self, c: usize) -> &[f32] {
        debug_assert!(c < self.dim);
        let start = c * self.n_vectors;
        &self.data[start..start + self.n_vectors]
    }

    /// Compute CD associator norms using SoA layout with workspace reuse.
    /// The key optimization: component-wise operations are cache-line aligned,
    /// and the AssociatorWorkspace reuses 3 buffers across all triplets.
    pub fn batch_associator_norms(&self) -> Vec<f32> {
        if self.n_vectors < 3 {
            return vec![];
        }

        let n = self.n_vectors - 2;
        let mut norms = Vec::with_capacity(n);
        let mut ws = super::fast_associator::AssociatorWorkspace::new(self.dim);

        for i in 0..n {
            let a = self.load(i);
            let b = self.load(i + 1);
            let c = self.load(i + 2);
            norms.push(ws.associator_norm(&a, &b, &c));
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

/// Ring-buffer embedding cache: O(3*dim) memory for sliding window associator.
/// Translated from steinmarder's A-A streaming single-buffer parity pattern.
/// Instead of storing ALL embeddings (O(N*dim)), keeps only the 3-element
/// sliding window needed for the associator triplet (a, b, c).
///
/// Uses AssociatorWorkspace internally: the ring buffer holds 3 embedding slots
/// and the workspace holds 3 computation buffers, for a total of 6*dim f32 memory.
pub struct RingEmbeddingCache {
    /// Three slots for the sliding window `[oldest, middle, newest]`
    slots: [Vec<f32>; 3],
    /// Reusable workspace for associator computation
    ws: super::fast_associator::AssociatorWorkspace,
    /// Embedding dimension
    dim: usize,
    /// Current write position (0, 1, 2, wraps around)
    write_pos: usize,
    /// Number of vectors pushed so far
    count: usize,
}

impl RingEmbeddingCache {
    pub fn new(dim: usize) -> Self {
        Self {
            slots: [vec![0.0f32; dim], vec![0.0f32; dim], vec![0.0f32; dim]],
            ws: super::fast_associator::AssociatorWorkspace::new(dim),
            dim,
            write_pos: 0,
            count: 0,
        }
    }

    /// Push a new embedding vector into the ring buffer.
    /// Returns Some(associator_norm) if we have a complete triplet.
    /// Uses the reusable AssociatorWorkspace to avoid per-call allocation.
    pub fn push_and_compute(&mut self, vector: &[f32]) -> Option<f32> {
        debug_assert!(vector.len() >= self.dim);
        self.slots[self.write_pos][..self.dim].copy_from_slice(&vector[..self.dim]);
        self.write_pos = (self.write_pos + 1) % 3;
        self.count += 1;

        if self.count >= 3 {
            // The three slots contain [newest-2, newest-1, newest] but in ring order.
            // Oldest is at write_pos (just overwritten by the PREVIOUS push).
            let a_idx = self.write_pos; // oldest (just wrapped past)
            let b_idx = (self.write_pos + 1) % 3;
            let c_idx = (self.write_pos + 2) % 3;
            Some(self.ws.associator_norm(
                &self.slots[a_idx],
                &self.slots[b_idx],
                &self.slots[c_idx],
            ))
        } else {
            None
        }
    }

    /// Stream through a sequence of embeddings, computing associator norms on the fly.
    /// Memory usage: O(3 * dim) regardless of sequence length.
    pub fn stream_norms(&mut self, embeddings: impl Iterator<Item = Vec<f32>>) -> Vec<f32> {
        let mut norms = Vec::new();
        for v in embeddings {
            if let Some(norm) = self.push_and_compute(&v) {
                norms.push(norm);
            }
        }
        norms
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn count(&self) -> usize {
        self.count
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

    #[test]
    fn test_ring_buffer_streaming() {
        let dim = 8; // octonion dimension
        let mut ring = RingEmbeddingCache::new(dim);

        // Push 5 vectors, should get 3 norms (after first triplet completes)
        let vecs: Vec<Vec<f32>> = (0..5)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[0] = 1.0;
                v[i % dim] = (i as f32 + 1.0) * 0.1;
                // Normalize
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in v.iter_mut() {
                    *x /= norm;
                }
                v
            })
            .collect();

        let norms = ring.stream_norms(vecs.into_iter());
        assert_eq!(norms.len(), 3); // 5 vectors -> 3 triplets
        assert_eq!(ring.count(), 5);
        // All norms should be non-negative
        for n in &norms {
            assert!(*n >= 0.0, "Norm should be non-negative, got {}", n);
        }
    }

    #[test]
    fn test_ring_matches_batch() {
        // Verify ring buffer gives same results as batch computation
        let dim = 16; // sedenion
        let vecs: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                for (j, slot) in v.iter_mut().enumerate() {
                    *slot = ((i * 7 + j * 3) as f32).sin();
                }
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in v.iter_mut() {
                    *x /= norm;
                }
                v
            })
            .collect();

        // Ring buffer path
        let mut ring = RingEmbeddingCache::new(dim);
        let ring_norms = ring.stream_norms(vecs.clone().into_iter());

        // Batch path
        let batch_norms =
            crate::cayley_dickson::simd::batch_sliding_associator_norms_f32(&vecs, dim);

        assert_eq!(ring_norms.len(), batch_norms.len());
        for (r, b) in ring_norms.iter().zip(batch_norms.iter()) {
            assert!(
                (r - b).abs() < 1e-5,
                "Ring {} != Batch {} (diff {})",
                r,
                b,
                (r - b).abs()
            );
        }
    }
}
