//! Complementary compute stack: role assignment for each crate.
//!
//! The open_gororoba compute stack is a strictly non-overlapping ensemble:
//!
//! | Crate           | Role                              | Used Where                    |
//! |-----------------|-----------------------------------|-------------------------------|
//! | **wide**        | Named SIMD types (f64x4, f32x8)   | CD algebra, LBM, codebook     |
//! | **pulp**        | Runtime SIMD dispatch + autovec   | Loop autovectorization         |
//! | **simsimd**     | HW-accel distance functions       | Spatial, similarity            |
//! | **rayon**       | Data-parallel thread pool         | Batch processing, LBM grid    |
//! | **crossbeam**   | CachePadded, Backoff, scoped      | False-sharing, sync primitives |
//! | **aligned-vec** | AVec: aligned allocation          | SIMD buffer feeding            |
//! | **dyn-stack**   | Reusable scratch workspace        | Temp arrays without alloc      |
//! | **bytemuck**    | POD slice casting                 | Zero-copy between repr(C) types|
//! | **smallvec**    | Inline small buffers              | Hot-path locality              |
//! | **criterion**   | Stats microbenchmarks             | Development tuning             |
//! | **iai-callgrind**| Instruction-count benchmarks     | CI regression detection        |
//!
//! # Why each is strictly complementary
//!
//! - wide vs pulp: wide provides NAMED types (f64x4), pulp provides DISPATCH.
//!   You can use wide::f64x4 inside a pulp::Arch::dispatch() closure.
//! - simsimd vs wide: simsimd does L2/cosine/dot with HW-specific kernels,
//!   wide does arbitrary SIMD ops. No overlap.
//! - rayon vs crossbeam: rayon does parallel iteration, crossbeam adds
//!   CachePadded (false-sharing prevention) and Backoff (spin-wait).
//! - aligned-vec vs smallvec: aligned-vec guarantees SIMD alignment,
//!   smallvec optimizes for small inline buffers. Different allocator strategies.
//! - dyn-stack vs Vec: dyn-stack reuses scratch memory across calls,
//!   Vec allocates fresh each time. dyn-stack eliminates alloc churn in hot loops.
//! - bytemuck vs zerocopy: bytemuck is lean (cast_slice), zerocopy is rich
//!   (derive macros). We chose bytemuck for minimal footprint.

/// Aligned scratch buffer for SIMD operations.
///
/// Uses aligned-vec to guarantee alignment for AVX2 (32-byte) or
/// AVX-512 (64-byte) operations.
pub fn aligned_scratch_f64(n: usize) -> aligned_vec::AVec<f64> {
    aligned_vec::avec![0.0f64; n]
}

/// Aligned scratch buffer for f32 SIMD operations.
pub fn aligned_scratch_f32(n: usize) -> aligned_vec::AVec<f32> {
    aligned_vec::avec![0.0f32; n]
}

/// Cast a slice of f64 to bytes via bytemuck (zero-copy).
pub fn as_bytes(data: &[f64]) -> &[u8] {
    bytemuck::cast_slice(data)
}

/// Cast bytes back to f64 slice via bytemuck (zero-copy).
///
/// # Safety
/// The input must be properly aligned and have length divisible by 8.
pub fn from_bytes(data: &[u8]) -> &[f64] {
    bytemuck::cast_slice(data)
}

/// Create a SmallVec for hot-path inline storage.
///
/// For d <= 16 (quaternion through sedenion), the entire vector fits
/// inline without heap allocation.
pub type SmallVecD<T> = smallvec::SmallVec<[T; 16]>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_scratch() {
        let buf = aligned_scratch_f64(128);
        assert_eq!(buf.len(), 128);
        // Check alignment (at least 16-byte for SSE, 32-byte for AVX2)
        let ptr = buf.as_ptr() as usize;
        assert_eq!(ptr % 16, 0, "Not 16-byte aligned");
    }

    #[test]
    fn test_bytemuck_roundtrip() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = as_bytes(&data);
        assert_eq!(bytes.len(), 32); // 4 * 8 bytes
        let recovered = from_bytes(bytes);
        assert_eq!(recovered, &data[..]);
    }

    #[test]
    fn test_smallvec_inline() {
        let mut sv: SmallVecD<f64> = SmallVecD::new();
        for i in 0..16 {
            sv.push(i as f64);
        }
        // Should be inline (no heap) for <= 16 elements
        assert!(!sv.spilled(), "SmallVec should be inline for 16 elements");

        // Adding one more should spill to heap
        sv.push(16.0);
        assert!(sv.spilled(), "SmallVec should spill after 16 elements");
    }

    #[test]
    fn test_crossbeam_cache_padded() {
        use crossbeam_utils::CachePadded;

        // CachePadded prevents false sharing between threads
        let padded = CachePadded::new(42u64);
        assert_eq!(*padded, 42);

        // Size should be at least one cache line (64 bytes)
        assert!(
            std::mem::size_of::<CachePadded<u64>>() >= 64,
            "CachePadded should be >= 64 bytes, got {}",
            std::mem::size_of::<CachePadded<u64>>()
        );
    }

    #[test]
    fn test_dyn_stack_scratch() {
        use dyn_stack::{PodBuffer, PodStack, StackReq};

        // Allocate a reusable scratch buffer
        let req = StackReq::new::<f64>(128);
        let mut buf = PodBuffer::new(req);
        let stack = PodStack::new(&mut buf);

        // Use the scratch buffer -- make_with takes |index: usize| -> T
        let (scratch, _) = stack.make_with(128, |i| i as f64 * 0.1);
        assert_eq!(scratch.len(), 128);
        assert!((scratch[10] - 1.0).abs() < 1e-10);
    }
}
