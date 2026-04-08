//! Reusable workspace for TurboQuant hot-loop allocation elimination.
//!
//! The quantize/dequantize pipeline needs scratch buffers for:
//! - Rotation (d elements for input, d for scratch, d for output)
//! - Quantized indices (d elements, u8)
//! - Residual computation (d elements)
//!
//! Without a workspace, each call allocates 3-5 Vec<f64> on the heap.
//! With dyn-stack, these are allocated ONCE and reused across calls.
//! With aligned-vec, the buffers are SIMD-aligned for f64x4/f32x8 ops.

use dyn_stack::{PodBuffer, StackReq};

/// Pre-allocated workspace for TurboQuant operations.
///
/// Eliminates heap allocation in the quantize/dequantize hot loop.
/// Create once, reuse across all calls via the scratch buffer.
pub struct TurboQuantWorkspace {
    /// Raw byte buffer for scratch space.
    scratch: Vec<f64>,
    d: usize,
}

impl TurboQuantWorkspace {
    /// Create a workspace for dimension d.
    ///
    /// Pre-allocates 4*d f64 values of scratch space (reusable).
    pub fn new(d: usize) -> Self {
        TurboQuantWorkspace {
            scratch: vec![0.0f64; 4 * d],
            d,
        }
    }

    /// Get the scratch buffer as a mutable slice.
    ///
    /// The caller uses this for rotation, quantization, and residual
    /// computation. The same buffer is reused across calls.
    pub fn scratch(&mut self) -> &mut [f64] {
        &mut self.scratch
    }

    /// Dimension this workspace was created for.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Get a dyn-stack PodStack from a separate allocation.
    ///
    /// For operations that need dyn-stack's structured allocation.
    pub fn pod_stack(d: usize) -> (PodBuffer, StackReq) {
        let req = StackReq::new::<f64>(4 * d);
        (PodBuffer::new(req), req)
    }
}

/// Aligned workspace using aligned-vec for guaranteed SIMD alignment.
///
/// For hot paths where SIMD alignment matters (WHT butterfly, codebook lookup).
pub struct AlignedWorkspace {
    /// 64-byte aligned f64 buffer (fits AVX-512).
    pub f64_buf: aligned_vec::AVec<f64>,
    /// 64-byte aligned f32 buffer.
    pub f32_buf: aligned_vec::AVec<f32>,
    /// 64-byte aligned u8 buffer (for indices).
    pub u8_buf: aligned_vec::AVec<u8>,
}

impl AlignedWorkspace {
    /// Create an aligned workspace for dimension d.
    pub fn new(d: usize) -> Self {
        AlignedWorkspace {
            f64_buf: aligned_vec::avec![0.0f64; 4 * d],
            f32_buf: aligned_vec::avec![0.0f32; 2 * d],
            u8_buf: aligned_vec::avec![0u8; d],
        }
    }

    /// Get f64 scratch as a mutable slice.
    pub fn f64_scratch(&mut self) -> &mut [f64] {
        &mut self.f64_buf
    }

    /// Get f32 scratch as a mutable slice.
    pub fn f32_scratch(&mut self) -> &mut [f32] {
        &mut self.f32_buf
    }

    /// Get u8 scratch as a mutable slice.
    pub fn u8_scratch(&mut self) -> &mut [u8] {
        &mut self.u8_buf
    }

    /// Check alignment (should be >= 32 for AVX2, >= 64 for AVX-512).
    pub fn alignment(&self) -> usize {
        let ptr = self.f64_buf.as_ptr() as usize;
        // Find the largest power of 2 that divides the address
        if ptr == 0 {
            return 0;
        }
        1 << ptr.trailing_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turboquant_workspace() {
        let mut ws = TurboQuantWorkspace::new(128);
        assert_eq!(ws.dim(), 128);

        // Scratch should be usable and pre-allocated
        let scratch = ws.scratch();
        assert!(scratch.len() >= 128 * 4);
        scratch[50] = std::f64::consts::PI;
        assert!((scratch[50] - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_aligned_workspace() {
        let mut aws = AlignedWorkspace::new(128);

        // Check sizes
        assert_eq!(aws.f64_scratch().len(), 512); // 4 * 128
        assert_eq!(aws.f32_scratch().len(), 256); // 2 * 128
        assert_eq!(aws.u8_scratch().len(), 128);

        // Check alignment (at least 16-byte for SSE)
        let align = aws.alignment();
        assert!(align >= 16, "Alignment {} < 16", align);
        println!("AlignedWorkspace alignment: {} bytes", align);
    }

    #[test]
    fn test_workspace_reuse() {
        let mut ws = TurboQuantWorkspace::new(64);

        // First use
        {
            let scratch = ws.scratch();
            for (i, slot) in scratch.iter_mut().enumerate().take(64) {
                *slot = i as f64;
            }
            assert_eq!(scratch[0], 0.0);
            assert_eq!(scratch[63], 63.0);
        }

        // Second use (reuses same memory, no new allocation)
        {
            let scratch = ws.scratch();
            for (i, slot) in scratch.iter_mut().enumerate().take(64) {
                *slot = (i as f64) * 2.0;
            }
            assert_eq!(scratch[0], 0.0);
            assert_eq!(scratch[63], 126.0);
        }
    }
}
