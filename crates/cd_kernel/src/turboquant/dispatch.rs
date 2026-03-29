//! Runtime CPU feature dispatch for TurboQuant hot paths.
//!
//! Detects available SIMD instruction sets at runtime and selects the
//! fastest quantization path.  Pattern from steinmarder's nerf_simd.c
//! which does CPUID-based dispatch between scalar, AVX2, and AVX-512.
//!
//! On x86_64:
//! - AVX2 + FMA: use wide::f32x8 SIMD codebook (default on modern CPUs)
//! - AVX-512: future path for 16-wide operations
//! - Scalar fallback: always available
//!
//! On other architectures: scalar only (NEON future work).

use super::simd_codebook::{self, SimdBoundaries};

/// Detected CPU SIMD capability level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdLevel {
    /// No SIMD: scalar boundary search.
    Scalar,
    /// AVX2 + FMA: f32x8 operations (256-bit).
    Avx2,
    /// AVX-512F: f32x16 operations (512-bit). Reserved for future.
    Avx512,
}

impl std::fmt::Display for SimdLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimdLevel::Scalar => write!(f, "scalar"),
            SimdLevel::Avx2 => write!(f, "avx2"),
            SimdLevel::Avx512 => write!(f, "avx512"),
        }
    }
}

/// Detect the highest available SIMD level at runtime.
///
/// Uses `std::arch::is_x86_feature_detected!` on x86_64.
/// Returns `Scalar` on non-x86 architectures.
pub fn detect_simd_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return SimdLevel::Avx512;
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            return SimdLevel::Avx2;
        }
    }
    SimdLevel::Scalar
}

/// Dispatched quantizer: selects the fastest path based on runtime SIMD level.
///
/// Wraps a `SimdBoundaries` for SIMD paths and raw boundaries for scalar fallback.
pub struct DispatchedQuantizer {
    simd: SimdBoundaries,
    boundaries: Vec<f32>,
    centroids: Vec<f32>,
    level: SimdLevel,
    bits: u32,
}

impl DispatchedQuantizer {
    /// Create a dispatched quantizer from a Lloyd-Max codebook.
    ///
    /// Automatically detects SIMD level and prepares the appropriate
    /// acceleration structures.
    pub fn new(codebook: &crate::lloyd_max::LloydMaxCodebook, bits: u32) -> Self {
        let level = detect_simd_level();
        let simd = SimdBoundaries::from_boundaries(&codebook.boundaries, bits);
        DispatchedQuantizer {
            simd,
            boundaries: codebook.boundaries.clone(),
            centroids: codebook.centroids.clone(),
            level,
            bits,
        }
    }

    /// Create with an explicit SIMD level (for benchmarking or testing).
    pub fn with_level(
        codebook: &crate::lloyd_max::LloydMaxCodebook,
        bits: u32,
        level: SimdLevel,
    ) -> Self {
        let simd = SimdBoundaries::from_boundaries(&codebook.boundaries, bits);
        DispatchedQuantizer {
            simd,
            boundaries: codebook.boundaries.clone(),
            centroids: codebook.centroids.clone(),
            level,
            bits,
        }
    }

    /// Quantize a batch of values using the best available path.
    pub fn quantize(&self, values: &[f32], out: &mut [u8]) {
        match self.level {
            SimdLevel::Avx2 | SimdLevel::Avx512 => {
                // Use SIMD boundary comparison via wide::f32x8
                self.simd.quantize_batch(values, out);
            }
            SimdLevel::Scalar => {
                // Scalar boundary search fallback
                for (i, &v) in values.iter().enumerate() {
                    out[i] = simd_codebook::quantize_scalar_boundary(v, &self.boundaries);
                }
            }
        }
    }

    /// Dequantize a batch of indices to centroid values.
    pub fn dequantize(&self, indices: &[u8], out: &mut [f32]) {
        simd_codebook::dequantize_batch(indices, &self.centroids, out);
    }

    /// Get the detected SIMD level.
    pub fn simd_level(&self) -> SimdLevel {
        self.level
    }

    pub fn bits(&self) -> u32 {
        self.bits
    }

    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    pub fn boundaries(&self) -> &[f32] {
        &self.boundaries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lloyd_max;

    #[test]
    fn test_detect_simd_level() {
        let level = detect_simd_level();
        // On any modern x86_64 machine, we should have at least AVX2
        #[cfg(target_arch = "x86_64")]
        {
            assert!(
                level == SimdLevel::Avx2 || level == SimdLevel::Avx512,
                "Expected AVX2 or AVX512, got {:?}",
                level
            );
        }
        println!("Detected SIMD level: {}", level);
    }

    #[test]
    fn test_dispatched_quantizer_correctness() {
        let cb = lloyd_max::get_codebook(128, 3);

        // Test all three dispatch levels produce identical results
        let scalar_q = DispatchedQuantizer::with_level(&cb, 3, SimdLevel::Scalar);
        let avx2_q = DispatchedQuantizer::with_level(&cb, 3, SimdLevel::Avx2);

        let sigma = 1.0 / (128.0f32).sqrt();
        let values: Vec<f32> = (0..512)
            .map(|i| ((i as f32 * 0.618) % 1.0 - 0.5) * 7.0 * sigma)
            .collect();

        let mut scalar_out = vec![0u8; 512];
        let mut avx2_out = vec![0u8; 512];
        scalar_q.quantize(&values, &mut scalar_out);
        avx2_q.quantize(&values, &mut avx2_out);

        assert_eq!(scalar_out, avx2_out, "Scalar and AVX2 paths produce different results");
    }

    #[test]
    fn test_dispatched_roundtrip() {
        let cb = lloyd_max::get_codebook(128, 3);
        let q = DispatchedQuantizer::new(&cb, 3);

        let values: Vec<f32> = (0..128).map(|i| ((i as f32 * 0.1).sin()) * 0.1).collect();
        let mut indices = vec![0u8; 128];
        let mut reconstructed = vec![0.0f32; 128];

        q.quantize(&values, &mut indices);
        q.dequantize(&indices, &mut reconstructed);

        // Each reconstructed value should be a valid centroid
        for &r in &reconstructed {
            assert!(cb.centroids.contains(&r));
        }
    }

    #[test]
    fn test_dispatched_display() {
        let level = detect_simd_level();
        let s = format!("{}", level);
        assert!(!s.is_empty());
        assert!(s == "scalar" || s == "avx2" || s == "avx512");
    }
}
